"""Product-quantized token retrieval behind the common KVIndex contract."""

import torch

from kvdb.core.types import Selection, validate_budget
from kvdb.indexes.pq.reference import (
    PQMetadata,
    _validate_positive_integer,
    _validate_seed,
    append_pq_codes,
    build_pq_metadata,
    score_pq_codes,
)


class PQIndex:
    """Deterministic reference product-quantization token index."""

    def __init__(
        self,
        *,
        num_subspaces: int,
        num_centroids: int,
        max_iterations: int = 20,
        seed: int = 0,
    ) -> None:
        _validate_positive_integer(num_subspaces, name="num_subspaces")
        _validate_positive_integer(num_centroids, name="num_centroids")
        _validate_positive_integer(max_iterations, name="max_iterations")
        _validate_seed(seed)
        self.num_subspaces = num_subspaces
        self.num_centroids = num_centroids
        self.max_iterations = max_iterations
        self.seed = seed
        self._metadata: PQMetadata | None = None

    @property
    def metadata(self) -> PQMetadata:
        """Return built PQ state, or fail if ``build`` has not run."""
        if self._metadata is None:
            raise RuntimeError("build must be called before accessing metadata")
        return self._metadata

    def build(self, keys: torch.Tensor) -> None:
        """Train codebooks and encode canonical keys ``[B, Hkv, S, D]``."""
        with torch.no_grad():
            self._metadata = build_pq_metadata(
                keys,
                num_subspaces=self.num_subspaces,
                num_centroids=self.num_centroids,
                max_iterations=self.max_iterations,
                seed=self.seed,
            )

    def search(self, query: torch.Tensor, budget: int) -> Selection:
        """Return approximate raw-dot-product Top-K token candidates."""
        metadata = self.metadata
        validate_budget(budget, metadata.sequence_length)
        with torch.no_grad():
            scores = score_pq_codes(query, metadata)
            ranked_indices = torch.argsort(
                scores,
                dim=-1,
                descending=True,
                stable=True,
            )
            top_indices = ranked_indices[..., :budget]
            top_scores = torch.gather(scores, dim=-1, index=top_indices)
        return Selection(indices=top_indices, scores=top_scores)

    def append(self, new_keys: torch.Tensor) -> None:
        """Encode new causal keys against the existing frozen codebooks."""
        with torch.no_grad():
            self._metadata = append_pq_codes(self.metadata, new_keys)
