"""Exact oracle equivalence, including adversarial PQ threshold ties."""

from contextvars import copy_context
import itertools
import math

import pytest
import torch

from kvweave import PQIndex
from kvweave.indexes.pq.reference import score_pq_codes
from kvweave.indexes.pq.selection import (
    PQRankingMode,
    deterministic_partial_selection,
    full_stable_ranking,
    pq_ranking_mode,
    rank_pq_candidates,
)


def assert_exact_selection(scores: torch.Tensor, budget: int) -> None:
    expected = full_stable_ranking(scores)[..., :budget]
    actual = deterministic_partial_selection(scores, budget)
    assert torch.equal(actual, expected)
    actual_scores = scores.gather(-1, actual)
    expected_scores = scores.gather(-1, expected)
    # Byte comparison also checks signed zero and preserves NaN payloads.
    assert torch.equal(
        actual_scores.contiguous().view(torch.uint8),
        expected_scores.contiguous().view(torch.uint8),
    )
    assert actual.shape == (*scores.shape[:-1], budget)
    assert actual.dtype == torch.int64


@pytest.mark.parametrize("length", range(1, 7))
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_exhaustive_small_scores(length: int, dtype: torch.dtype) -> None:
    rows = torch.tensor(
        list(itertools.product((-1.0, 0.0, 1.0), repeat=length)), dtype=dtype
    )
    # Exhaust every score vector and every legal K, with independently varying heads.
    scores = torch.stack((rows, rows.flip(-1)), dim=1)
    for budget in range(1, length + 1):
        assert_exact_selection(scores, budget)


@pytest.mark.parametrize(
    "values",
    [
        [1, 1, 1, 1, 1],
        [5, 2, 5, 2, 5, 2],
        [4] + [3] * 200 + [5, -1],
        [0.0, -0.0, 1.0, -0.0, 0.0, -1.0],
        [-5, -3, -1, -2, -4],
        [5, 3, 1, 2, 4],
        [float("inf"), 0, float("-inf"), float("inf"), float("-inf")],
        [float("nan"), 1, float("nan"), -1, 0],
    ],
)
def test_adversarial_ties_and_special_values(values: list[float]) -> None:
    scores = torch.tensor(values).reshape(1, 1, -1)
    for budget in range(1, len(values) + 1):
        assert_exact_selection(scores, budget)


def test_explicit_tie_order() -> None:
    assert deterministic_partial_selection(torch.ones(1, 1, 5), 3).tolist() == [
        [[0, 1, 2]]
    ]
    scores = torch.tensor([5.0, 2.0, 5.0, 2.0, 5.0, 2.0]).reshape(1, 1, 6)
    for budget in range(1, 7):
        assert deterministic_partial_selection(scores, budget).tolist() == [
            [[0, 2, 4, 1, 3, 5][:budget]]
        ]


@pytest.mark.parametrize("budget", [0, -1, 3, True, 1.5])
def test_invalid_budget_matches_search_contract(budget: int) -> None:
    with pytest.raises((ValueError, TypeError)):
        deterministic_partial_selection(torch.ones(1, 1, 2), budget)


def test_noncontiguous_scores_and_input_preservation() -> None:
    generator = torch.Generator().manual_seed(42)
    scores = (
        torch.randint(-3, 4, (2, 17, 3), generator=generator).float().transpose(1, 2)
    )
    original = scores.clone()
    for budget in (1, 8, 16, 17):
        assert_exact_selection(scores, budget)
    assert torch.equal(scores, original)


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (2, 3, 33, 16), (1, 16, 128, 64)])
@pytest.mark.parametrize("configuration", [(1, 2), (2, 4), (4, 8)])
def test_reconstructed_pq_scores_and_search_results(
    shape: tuple[int, ...], configuration: tuple[int, int]
) -> None:
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(shape, generator=generator)
    index = PQIndex(
        num_subspaces=configuration[0],
        num_centroids=configuration[1],
        max_iterations=8,
        seed=0,
    )
    index.build(keys)
    for _ in range(3):
        query = torch.randn((*shape[:2], shape[-1]), generator=generator)
        scores = score_pq_codes(query, index.metadata)
        for fraction in (0.125, 0.25, 0.5, 1.0):
            budget = math.ceil(shape[2] * fraction)
            assert_exact_selection(scores, budget)
            with pq_ranking_mode(PQRankingMode.FULL_SORT):
                expected = index.search(query, budget)
            with pq_ranking_mode(PQRankingMode.PARTIAL):
                actual = index.search(query, budget)
            assert torch.equal(actual.indices, expected.indices)
            assert torch.equal(actual.scores, expected.scores)
            assert actual.indices.shape == (*shape[:2], budget)
            assert actual.valid_mask is expected.valid_mask is None
            assert torch.equal(actual.valid_token_counts, expected.valid_token_counts)
            assert torch.all(actual.valid_token_counts == budget)
    with pytest.raises(ValueError):
        index.search(query, shape[2] + 1)


def test_oracle_mode_is_nested_and_context_local() -> None:
    scores = torch.ones(1, 1, 4)
    assert rank_pq_candidates(scores, 2).shape[-1] == 4
    with pq_ranking_mode(PQRankingMode.FULL_SORT):
        assert rank_pq_candidates(scores, 2).shape[-1] == 4
        with pytest.raises(RuntimeError), pq_ranking_mode(PQRankingMode.PARTIAL):
            assert rank_pq_candidates(scores, 2).shape[-1] == 2
            raise RuntimeError("exercise restoration")
        assert rank_pq_candidates(scores, 2).shape[-1] == 4
        isolated = copy_context()
    assert isolated.run(rank_pq_candidates, scores, 2).shape[-1] == 4
    assert rank_pq_candidates(scores, 2).shape[-1] == 4
    with pytest.raises(TypeError), pq_ranking_mode("full"):
        pass
