import pytest
import torch

from kvdb.indexes.pq import (
    PQIndex,
    build_pq_metadata,
    encode_keys,
    query_lookup_tables,
    reconstruct_keys,
    score_pq_codes,
    train_codebooks,
    train_kmeans,
)


def make_values(
    shape: tuple[int, int, int, int],
    value_kind: str,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(401 + sum(shape))
    values = torch.randn(*shape, generator=generator)
    if value_kind == "positive":
        return values.abs() + 0.125
    if value_kind == "negative":
        return -(values.abs() + 0.125)
    return values


def slow_score_oracle(
    query: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Deliberately loop over every encoded subvector and centroid ID."""
    batch_size, kv_heads, sequence_length, num_subspaces = codes.shape
    scores = query.new_zeros(batch_size, kv_heads, sequence_length)
    subspace_dimension = codebooks.shape[-1]
    for batch_id in range(batch_size):
        for head_id in range(kv_heads):
            for token_id in range(sequence_length):
                for subspace_id in range(num_subspaces):
                    start = subspace_id * subspace_dimension
                    stop = start + subspace_dimension
                    centroid_id = codes[
                        batch_id,
                        head_id,
                        token_id,
                        subspace_id,
                    ].item()
                    scores[batch_id, head_id, token_id] += torch.dot(
                        query[batch_id, head_id, start:stop],
                        codebooks[
                            batch_id,
                            head_id,
                            subspace_id,
                            centroid_id,
                        ],
                    )
    return scores


def test_train_kmeans_is_deterministic() -> None:
    samples = torch.randn(17, 3, generator=torch.Generator().manual_seed(7))

    first = train_kmeans(
        samples,
        num_centroids=4,
        max_iterations=9,
        seed=19,
    )
    second = train_kmeans(
        samples,
        num_centroids=4,
        max_iterations=9,
        seed=19,
    )

    torch.testing.assert_close(first.centroids, second.centroids)
    torch.testing.assert_close(first.assignments, second.assignments)
    assert first.iterations == second.iterations
    assert (
        first.empty_cluster_reinitializations == second.empty_cluster_reinitializations
    )


def test_train_kmeans_handles_empty_clusters_explicitly() -> None:
    samples = torch.ones(6, 2)

    result = train_kmeans(
        samples,
        num_centroids=3,
        max_iterations=4,
        seed=0,
    )

    assert result.empty_cluster_reinitializations > 0
    assert result.iterations <= 4
    assert torch.isfinite(result.centroids).all().item()
    torch.testing.assert_close(result.centroids, torch.ones(3, 2))


@pytest.mark.parametrize(
    ("samples", "num_centroids", "max_iterations", "seed", "error_type"),
    [
        (torch.randn(4, 2, 1), 2, 3, 0, ValueError),
        (torch.ones(4, 2, dtype=torch.int64), 2, 3, 0, TypeError),
        (torch.randn(4, 2), 0, 3, 0, ValueError),
        (torch.randn(4, 2), 5, 3, 0, ValueError),
        (torch.randn(4, 2), 2, 0, 0, ValueError),
        (torch.randn(4, 2), 2, 3, -1, ValueError),
    ],
)
def test_train_kmeans_rejects_invalid_configuration(
    samples: torch.Tensor,
    num_centroids: int,
    max_iterations: int,
    seed: int,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        train_kmeans(
            samples,
            num_centroids=num_centroids,
            max_iterations=max_iterations,
            seed=seed,
        )


@pytest.mark.parametrize("value_kind", ["positive", "negative", "mixed"])
def test_codebook_and_code_shapes_for_multiple_batches_and_heads(
    value_kind: str,
) -> None:
    keys = make_values((2, 3, 9, 8), value_kind)

    metadata = build_pq_metadata(
        keys,
        num_subspaces=4,
        num_centroids=3,
        max_iterations=6,
        seed=23,
    )

    assert metadata.codebooks.shape == (2, 3, 4, 3, 2)
    assert metadata.codes.shape == (2, 3, 9, 4)
    assert metadata.codes.dtype == torch.int64
    assert metadata.num_subspaces == 4
    assert metadata.num_centroids == 3
    assert metadata.subspace_dimension == 2
    assert metadata.sequence_length == 9
    assert metadata.head_dimension == 8
    assert torch.all(metadata.codes >= 0).item()
    assert torch.all(metadata.codes < 3).item()


def test_invalid_dimension_subspace_configuration_is_rejected() -> None:
    index = PQIndex(
        num_subspaces=3,
        num_centroids=2,
        max_iterations=4,
        seed=0,
    )

    with pytest.raises(ValueError, match="D must be divisible"):
        index.build(torch.randn(1, 1, 5, 8))


def test_num_centroids_cannot_exceed_sequence_length() -> None:
    index = PQIndex(num_subspaces=2, num_centroids=6)

    with pytest.raises(ValueError, match="sequence length"):
        index.build(torch.randn(1, 1, 5, 4))


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"num_subspaces": 0, "num_centroids": 2}, ValueError),
        ({"num_subspaces": True, "num_centroids": 2}, TypeError),
        ({"num_subspaces": 2, "num_centroids": 0}, ValueError),
        ({"num_subspaces": 2, "num_centroids": 2, "max_iterations": 0}, ValueError),
        ({"num_subspaces": 2, "num_centroids": 2, "seed": -1}, ValueError),
    ],
)
def test_index_rejects_invalid_constructor_configuration(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        PQIndex(**kwargs)  # type: ignore[arg-type]


def test_codebook_training_and_encoding_are_deterministic() -> None:
    keys = torch.randn(2, 2, 13, 8, generator=torch.Generator().manual_seed(31))
    configuration = {
        "num_subspaces": 2,
        "num_centroids": 4,
        "max_iterations": 7,
        "seed": 29,
    }

    first_codebooks = train_codebooks(keys, **configuration)
    second_codebooks = train_codebooks(keys, **configuration)
    first_codes = encode_keys(keys, first_codebooks)
    second_codes = encode_keys(keys, second_codebooks)

    torch.testing.assert_close(first_codebooks, second_codebooks)
    torch.testing.assert_close(first_codes, second_codes)


def test_lookup_table_scoring_matches_slow_loop_and_reconstruction_oracles() -> None:
    generator = torch.Generator().manual_seed(53)
    keys = torch.randn(2, 2, 7, 6, generator=generator)
    query = torch.randn(2, 2, 6, generator=generator)
    metadata = build_pq_metadata(
        keys,
        num_subspaces=3,
        num_centroids=3,
        max_iterations=8,
        seed=11,
    )

    lookup_tables = query_lookup_tables(query, metadata)
    scores = score_pq_codes(query, metadata)
    slow_scores = slow_score_oracle(query, metadata.codebooks, metadata.codes)
    reconstructed_scores = torch.einsum(
        "bhd,bhsd->bhs",
        query,
        reconstruct_keys(metadata),
    )

    assert lookup_tables.shape == (2, 2, 3, 3)
    assert scores.shape == (2, 2, 7)
    torch.testing.assert_close(scores, slow_scores)
    torch.testing.assert_close(scores, reconstructed_scores)


def test_retrieval_is_deterministic_with_exact_shape() -> None:
    generator = torch.Generator().manual_seed(71)
    keys = torch.randn(2, 3, 12, 8, generator=generator)
    query = torch.randn(2, 3, 8, generator=generator)
    index = PQIndex(
        num_subspaces=4,
        num_centroids=4,
        max_iterations=8,
        seed=37,
    )
    index.build(keys)

    first = index.search(query, budget=5)
    second = index.search(query, budget=5)

    assert first.indices.shape == (2, 3, 5)
    assert first.scores is not None
    assert first.scores.shape == (2, 3, 5)
    assert first.valid_mask is None
    torch.testing.assert_close(first.indices, second.indices)
    torch.testing.assert_close(first.scores, second.scores)


def test_full_budget_covers_every_token_exactly_once() -> None:
    sequence_length = 11
    keys = torch.randn(2, 3, sequence_length, 8)
    query = torch.randn(2, 3, 8)
    index = PQIndex(num_subspaces=2, num_centroids=4, seed=5)
    index.build(keys)

    selection = index.search(query, budget=sequence_length)

    expected = torch.arange(sequence_length).expand(2, 3, sequence_length)
    torch.testing.assert_close(selection.indices.sort(dim=-1).values, expected)
    assert selection.valid_mask is None


def test_search_requires_build_and_rejects_invalid_query_and_budget() -> None:
    with pytest.raises(RuntimeError, match="build must be called"):
        PQIndex(num_subspaces=2, num_centroids=2).search(
            torch.randn(1, 1, 4),
            budget=1,
        )

    index = PQIndex(num_subspaces=2, num_centroids=2)
    index.build(torch.randn(1, 2, 5, 4))
    with pytest.raises(ValueError, match="query must have shape"):
        index.search(torch.randn(1, 1, 4), budget=2)
    with pytest.raises(ValueError, match="budget"):
        index.search(torch.randn(1, 2, 4), budget=6)
