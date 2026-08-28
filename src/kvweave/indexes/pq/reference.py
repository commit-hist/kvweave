"""Independent readable product-quantization reference operations.

This module implements standard product quantization from its mathematical
description. It contains no source copied or adapted from PQCache.
"""

from dataclasses import dataclass

import torch

from kvweave.core.types import validate_keys
from kvweave.profiling import profile_component


def _validate_positive_integer(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _validate_finite(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _squared_distances(
    samples: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Return squared Euclidean distances with shape ``[N, C]``."""
    differences = samples.unsqueeze(1) - centroids.unsqueeze(0)
    return (differences * differences).sum(dim=-1)


@dataclass(frozen=True)
class KMeansResult:
    """Result of the small reference Lloyd-style K-means routine."""

    centroids: torch.Tensor
    assignments: torch.Tensor
    iterations: int
    empty_cluster_reinitializations: int


def train_kmeans(
    samples: torch.Tensor,
    *,
    num_centroids: int,
    max_iterations: int,
    seed: int,
) -> KMeansResult:
    """Cluster ``samples`` shaped ``[N, Dsub]`` deterministically.

    Initial centroids are distinct sample rows selected by a seeded CPU
    permutation. If an update produces an empty cluster, that centroid is
    explicitly reinitialized to a distinct sample with the largest current
    quantization error; ties prefer the lowest sample index. Iterations are
    bounded by ``max_iterations``.
    """
    if not isinstance(samples, torch.Tensor):
        raise TypeError("samples must be a torch.Tensor")
    if samples.ndim != 2:
        raise ValueError("samples must have shape [N, Dsub]")
    if any(size <= 0 for size in samples.shape):
        raise ValueError("sample dimensions N and Dsub must be positive")
    if not torch.is_floating_point(samples):
        raise TypeError("samples must use a floating-point dtype")
    _validate_finite(samples, name="samples")
    _validate_positive_integer(num_centroids, name="num_centroids")
    _validate_positive_integer(max_iterations, name="max_iterations")
    _validate_seed(seed)
    if num_centroids > samples.shape[0]:
        raise ValueError("num_centroids cannot exceed the number of samples")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_indices = torch.randperm(samples.shape[0], generator=generator)[
        :num_centroids
    ].to(samples.device)
    working_samples = samples.to(_working_dtype(samples.dtype))
    centroids = working_samples.index_select(0, initial_indices).clone()
    empty_cluster_reinitializations = 0
    iterations = 0

    for iteration in range(max_iterations):
        distances = _squared_distances(working_samples, centroids)
        assignments = distances.argmin(dim=-1)
        assigned_distances = torch.gather(
            distances,
            dim=1,
            index=assignments.unsqueeze(-1),
        ).squeeze(-1)
        # Stable sorting makes equal-error reinitialization prefer the lower
        # sample ID, keeping the reference behavior reproducible.
        reinitialization_order = torch.argsort(
            assigned_distances,
            descending=True,
            stable=True,
        ).tolist()
        reinitialization_cursor = 0
        used_reinitialization_samples: set[int] = set()
        updated_centroids = torch.empty_like(centroids)

        for centroid_id in range(num_centroids):
            members = working_samples[assignments == centroid_id]
            if members.shape[0] > 0:
                updated_centroids[centroid_id] = members.mean(dim=0)
                continue

            while (
                reinitialization_order[reinitialization_cursor]
                in used_reinitialization_samples
            ):
                reinitialization_cursor += 1
            sample_id = reinitialization_order[reinitialization_cursor]
            reinitialization_cursor += 1
            used_reinitialization_samples.add(sample_id)
            updated_centroids[centroid_id] = working_samples[sample_id]
            empty_cluster_reinitializations += 1

        iterations = iteration + 1
        converged = torch.equal(updated_centroids, centroids)
        centroids = updated_centroids
        if converged:
            break

    final_assignments = _squared_distances(
        working_samples,
        centroids,
    ).argmin(dim=-1)
    return KMeansResult(
        centroids=centroids.to(samples.dtype),
        assignments=final_assignments,
        iterations=iterations,
        empty_cluster_reinitializations=empty_cluster_reinitializations,
    )


def _validate_pq_configuration(
    keys: torch.Tensor,
    *,
    num_subspaces: int,
    num_centroids: int,
    max_iterations: int,
    seed: int,
) -> None:
    validate_keys(keys)
    _validate_finite(keys, name="keys")
    _validate_positive_integer(num_subspaces, name="num_subspaces")
    _validate_positive_integer(num_centroids, name="num_centroids")
    _validate_positive_integer(max_iterations, name="max_iterations")
    _validate_seed(seed)
    if keys.shape[-1] % num_subspaces != 0:
        raise ValueError("key dimension D must be divisible by num_subspaces")
    if num_centroids > keys.shape[2]:
        raise ValueError("num_centroids cannot exceed sequence length S")


def train_codebooks(
    keys: torch.Tensor,
    *,
    num_subspaces: int,
    num_centroids: int,
    max_iterations: int,
    seed: int,
) -> torch.Tensor:
    """Train codebooks shaped ``[B, Hkv, M, C, Dsub]``.

    Each batch item, KV head, and subspace is clustered independently. The
    caller-provided seed is offset by that group's row-major ID.
    """
    _validate_pq_configuration(
        keys,
        num_subspaces=num_subspaces,
        num_centroids=num_centroids,
        max_iterations=max_iterations,
        seed=seed,
    )
    batch_size, kv_heads, sequence_length, head_dim = keys.shape
    subspace_dimension = head_dim // num_subspaces
    with profile_component("pq.init.subspace_split"):
        subvectors = keys.reshape(
            batch_size,
            kv_heads,
            sequence_length,
            num_subspaces,
            subspace_dimension,
        )
    with profile_component("pq.init.codebook_allocation"):
        codebooks = keys.new_empty(
            batch_size,
            kv_heads,
            num_subspaces,
            num_centroids,
            subspace_dimension,
        )

    group_id = 0
    for batch_id in range(batch_size):
        for head_id in range(kv_heads):
            for subspace_id in range(num_subspaces):
                with profile_component("pq.init.kmeans_training"):
                    result = train_kmeans(
                        subvectors[batch_id, head_id, :, subspace_id, :],
                        num_centroids=num_centroids,
                        max_iterations=max_iterations,
                        seed=seed + group_id,
                    )
                    codebooks[batch_id, head_id, subspace_id] = result.centroids
                group_id += 1
    return codebooks


def _validate_codebooks(codebooks: torch.Tensor) -> None:
    if not isinstance(codebooks, torch.Tensor):
        raise TypeError("codebooks must be a torch.Tensor")
    if codebooks.ndim != 5:
        raise ValueError("codebooks must have shape [B, Hkv, M, C, Dsub]")
    if any(size <= 0 for size in codebooks.shape):
        raise ValueError("codebook dimensions must be positive")
    if not torch.is_floating_point(codebooks):
        raise TypeError("codebooks must use a floating-point dtype")
    _validate_finite(codebooks, name="codebooks")


def encode_keys(keys: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Encode keys as centroid IDs shaped ``[B, Hkv, S, M]``."""
    validate_keys(keys)
    _validate_finite(keys, name="keys")
    _validate_codebooks(codebooks)
    batch_size, kv_heads, sequence_length, head_dim = keys.shape
    codebook_batch, codebook_heads, num_subspaces, _, subspace_dimension = (
        codebooks.shape
    )
    if (codebook_batch, codebook_heads) != (batch_size, kv_heads):
        raise ValueError("codebooks must match key batch and KV-head dimensions")
    if num_subspaces * subspace_dimension != head_dim:
        raise ValueError("codebook subspaces must reconstruct key dimension D")
    if codebooks.dtype != keys.dtype:
        raise ValueError("keys and codebooks must have the same dtype")
    if codebooks.device != keys.device:
        raise ValueError("keys and codebooks must be on the same device")

    with profile_component("pq.encode.subspace_split"):
        subvectors = keys.reshape(
            batch_size,
            kv_heads,
            sequence_length,
            num_subspaces,
            subspace_dimension,
        )
        working_subvectors = subvectors.to(_working_dtype(keys.dtype))
        working_codebooks = codebooks.to(_working_dtype(codebooks.dtype))
    with profile_component("pq.encode.centroid_distance"):
        differences = working_subvectors.unsqueeze(-2) - working_codebooks.unsqueeze(2)
        distances = (differences * differences).sum(dim=-1)
    with profile_component("pq.encode.centroid_assignment"):
        codes = distances.argmin(dim=-1).to(torch.int64)
    return codes


@dataclass(frozen=True)
class PQMetadata:
    """Minimum trained state required for PQ token scoring.

    ``codebooks`` has shape ``[B, Hkv, M, C, Dsub]`` and ``codes`` has shape
    ``[B, Hkv, S, M]``. The seed and iteration limit preserve construction
    provenance; searching needs only the two tensors.
    """

    codebooks: torch.Tensor
    codes: torch.Tensor
    seed: int
    max_iterations: int

    def __post_init__(self) -> None:
        _validate_codebooks(self.codebooks)
        _validate_seed(self.seed)
        _validate_positive_integer(self.max_iterations, name="max_iterations")
        if not isinstance(self.codes, torch.Tensor):
            raise TypeError("codes must be a torch.Tensor")
        if self.codes.ndim != 4:
            raise ValueError("codes must have shape [B, Hkv, S, M]")
        if any(size <= 0 for size in self.codes.shape):
            raise ValueError("code dimensions must be positive")
        if self.codes.dtype != torch.int64:
            raise TypeError("codes must use torch.int64")
        if self.codes.device != self.codebooks.device:
            raise ValueError("codes and codebooks must be on the same device")
        if self.codes.shape[:2] != self.codebooks.shape[:2]:
            raise ValueError("codes and codebooks must share B and Hkv")
        if self.codes.shape[-1] != self.codebooks.shape[2]:
            raise ValueError("codes and codebooks must use the same M")
        if torch.any(self.codes < 0).item():
            raise ValueError("codes must be non-negative")
        if torch.any(self.codes >= self.codebooks.shape[3]).item():
            raise ValueError("code exceeds the codebook centroid range")

    @property
    def num_subspaces(self) -> int:
        return self.codebooks.shape[2]

    @property
    def num_centroids(self) -> int:
        return self.codebooks.shape[3]

    @property
    def subspace_dimension(self) -> int:
        return self.codebooks.shape[4]

    @property
    def sequence_length(self) -> int:
        return self.codes.shape[2]

    @property
    def head_dimension(self) -> int:
        return self.num_subspaces * self.subspace_dimension


def build_pq_metadata(
    keys: torch.Tensor,
    *,
    num_subspaces: int,
    num_centroids: int,
    max_iterations: int,
    seed: int,
) -> PQMetadata:
    """Train codebooks and encode canonical keys."""
    codebooks = train_codebooks(
        keys,
        num_subspaces=num_subspaces,
        num_centroids=num_centroids,
        max_iterations=max_iterations,
        seed=seed,
    )
    codes = encode_keys(keys, codebooks)
    with profile_component("pq.init.initial_code_storage"):
        metadata = PQMetadata(
            codebooks=codebooks,
            codes=codes,
            seed=seed,
            max_iterations=max_iterations,
        )
    return metadata


def append_pq_codes(
    metadata: PQMetadata,
    new_keys: torch.Tensor,
) -> PQMetadata:
    """Assign appended keys to frozen codebooks and extend PQ codes.

    ``new_keys`` uses canonical shape ``[B, Hkv, Snew, D]``. The pre-existing
    codebooks are intentionally reused rather than retrained; this is the
    standard PQ operation of encoding new database vectors against a trained
    quantizer. The returned metadata shares the immutable codebook tensor and
    appends code IDs in causal sequence order.
    """
    new_codes = encode_keys(new_keys, metadata.codebooks)
    with profile_component("pq.append.code_append"):
        appended_codes = torch.cat((metadata.codes, new_codes), dim=2)
        appended = PQMetadata(
            codebooks=metadata.codebooks,
            codes=appended_codes,
            seed=metadata.seed,
            max_iterations=metadata.max_iterations,
        )
    return appended


def reconstruct_keys(metadata: PQMetadata) -> torch.Tensor:
    """Reconstruct approximate keys shaped ``[B, Hkv, S, D]``."""
    reconstructed_subspaces: list[torch.Tensor] = []
    for subspace_id in range(metadata.num_subspaces):
        subspace_codebook = metadata.codebooks[:, :, subspace_id, :, :]
        subspace_codes = metadata.codes[:, :, :, subspace_id]
        gather_indices = subspace_codes.unsqueeze(-1).expand(
            *subspace_codes.shape,
            metadata.subspace_dimension,
        )
        reconstructed_subspaces.append(
            torch.gather(subspace_codebook, dim=2, index=gather_indices)
        )
    return torch.cat(reconstructed_subspaces, dim=-1)


def _validate_query(query: torch.Tensor, metadata: PQMetadata) -> None:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if query.ndim != 3:
        raise ValueError("query must have shape [B, Hkv, D]")
    expected_shape = (
        metadata.codebooks.shape[0],
        metadata.codebooks.shape[1],
        metadata.head_dimension,
    )
    if query.shape != expected_shape:
        raise ValueError(f"query must have shape {expected_shape} to match PQ metadata")
    if query.dtype != metadata.codebooks.dtype:
        raise ValueError("query and codebooks must have the same dtype")
    if query.device != metadata.codebooks.device:
        raise ValueError("query and codebooks must be on the same device")
    _validate_finite(query, name="query")


def query_lookup_tables(
    query: torch.Tensor,
    metadata: PQMetadata,
) -> torch.Tensor:
    """Return query-to-centroid dot products ``[B, Hkv, M, C]``."""
    _validate_query(query, metadata)
    with profile_component("pq.search.query_split"):
        query_subspaces = query.reshape(
            query.shape[0],
            query.shape[1],
            metadata.num_subspaces,
            metadata.subspace_dimension,
        )
    with profile_component("pq.search.query_centroid_dot_products"):
        lookup_tables = torch.einsum(
            "bhmd,bhmcd->bhmc",
            query_subspaces,
            metadata.codebooks,
        )
    return lookup_tables


def score_pq_codes(
    query: torch.Tensor,
    metadata: PQMetadata,
) -> torch.Tensor:
    """Approximate raw query-key dot products with shape ``[B, Hkv, S]``."""
    lookup_tables = query_lookup_tables(query, metadata)
    with profile_component("pq.search.score_allocation"):
        scores = query.new_zeros(
            query.shape[0],
            query.shape[1],
            metadata.sequence_length,
        )
    for subspace_id in range(metadata.num_subspaces):
        with profile_component("pq.search.code_lookup"):
            subspace_scores = torch.gather(
                lookup_tables[:, :, subspace_id, :],
                dim=-1,
                index=metadata.codes[:, :, :, subspace_id],
            )
        with profile_component("pq.search.subspace_summation"):
            scores += subspace_scores
    return scores
