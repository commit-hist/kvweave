import torch

from kvweave import BruteForceIndex, KVCache, PQIndex, TensorStorage
from kvweave.indexes.pq import build_pq_metadata, reconstruct_keys
from kvweave.indexes.quest.reference import candidate_recall, compare_attention


def reconstruction_relative_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    return (
        torch.linalg.vector_norm(reconstructed - original)
        / torch.linalg.vector_norm(original)
    ).item()


def test_more_centroids_reduce_reconstruction_error_on_fixed_data() -> None:
    keys = torch.tensor(
        [
            [
                [
                    [-4.0, -3.0, -2.0, -1.0],
                    [-1.0, 2.0, -3.0, 4.0],
                    [2.0, -1.0, 4.0, -3.0],
                    [5.0, 4.0, 3.0, 2.0],
                ]
            ]
        ]
    )
    low_capacity = build_pq_metadata(
        keys,
        num_subspaces=2,
        num_centroids=1,
        max_iterations=8,
        seed=3,
    )
    high_capacity = build_pq_metadata(
        keys,
        num_subspaces=2,
        num_centroids=4,
        max_iterations=8,
        seed=3,
    )

    low_error = reconstruction_relative_error(
        keys,
        reconstruct_keys(low_capacity),
    )
    high_error = reconstruction_relative_error(
        keys,
        reconstruct_keys(high_capacity),
    )

    assert high_error < low_error
    torch.testing.assert_close(reconstruct_keys(high_capacity), keys)


def test_pq_candidates_have_well_formed_recall_against_exact_topk() -> None:
    generator = torch.Generator().manual_seed(101)
    keys = torch.randn(2, 3, 32, 8, generator=generator)
    query = torch.randn(2, 3, 8, generator=generator)
    budget = 8
    pq = PQIndex(
        num_subspaces=4,
        num_centroids=8,
        max_iterations=10,
        seed=13,
    )
    exact = BruteForceIndex()
    pq.build(keys)
    exact.build(keys)

    recall = candidate_recall(
        pq.search(query, budget),
        exact.search(query, budget),
    )

    assert recall.shape == (2, 3)
    assert torch.all((recall >= 0.0) & (recall <= 1.0)).item()
    assert recall.mean().item() > 0.0


def test_pq_runs_through_complete_cache_storage_and_attention_path() -> None:
    generator = torch.Generator().manual_seed(151)
    keys = torch.randn(2, 3, 17, 8, generator=generator)
    values = torch.randn(2, 3, 17, 8, generator=generator)
    query = torch.randn(2, 3, 8, generator=generator)
    cache = KVCache(
        index=PQIndex(
            num_subspaces=4,
            num_centroids=4,
            max_iterations=8,
            seed=17,
        ),
        storage=TensorStorage(),
    )
    cache.build(keys, values)

    retrieved = cache.retrieve(query, budget=6)
    comparison = compare_attention(query, keys, values, retrieved)

    assert retrieved.keys.shape == (2, 3, 6, 8)
    assert retrieved.values.shape == retrieved.keys.shape
    assert retrieved.valid_mask is None
    assert comparison.selected_output.shape == (2, 3, 8)
    assert 0.0 < comparison.selected_token_percentage < 100.0
    assert torch.isfinite(comparison.selected_output).all().item()
    assert comparison.relative_output_error >= 0.0


def test_full_budget_reproduces_full_attention_through_shared_path() -> None:
    generator = torch.Generator().manual_seed(181)
    sequence_length = 19
    keys = torch.randn(2, 2, sequence_length, 8, generator=generator)
    values = torch.randn(2, 2, sequence_length, 8, generator=generator)
    query = torch.randn(2, 2, 8, generator=generator)
    cache = KVCache(
        index=PQIndex(
            num_subspaces=2,
            num_centroids=4,
            max_iterations=8,
            seed=19,
        ),
        storage=TensorStorage(),
    )
    cache.build(keys, values)

    retrieved = cache.retrieve(query, budget=sequence_length)
    comparison = compare_attention(query, keys, values, retrieved)

    assert retrieved.valid_mask is None
    assert comparison.selected_token_percentage == 100.0
    torch.testing.assert_close(
        comparison.selected_output,
        comparison.full_output,
        rtol=1e-5,
        atol=1e-6,
    )
    assert comparison.relative_output_error < 1e-6
