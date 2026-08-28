import pytest
import torch

from kvweave.indexes.brute_force import BruteForceIndex


def test_exact_topk_indices_and_scores() -> None:
    keys = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [-1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [0.0, -1.0]],
            ]
        ]
    )
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    index = BruteForceIndex()
    index.build(keys)

    selection = index.search(query, budget=2)

    expected_indices = torch.tensor([[[2, 0], [2, 0]]])
    expected_scores = torch.tensor([[[2.0, 1.0], [2.0, 1.0]]])
    torch.testing.assert_close(selection.indices, expected_indices)
    assert selection.scores is not None
    torch.testing.assert_close(selection.scores, expected_scores)


def test_topk_matches_direct_dot_product_for_every_batch_and_head() -> None:
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(2, 3, 11, 5, generator=generator)
    query = torch.randn(2, 3, 5, generator=generator)
    index = BruteForceIndex()
    index.build(keys)

    selection = index.search(query, budget=4)

    exact_scores = torch.einsum("bhd,bhsd->bhs", query, keys)
    expected_scores, expected_indices = exact_scores.topk(4, dim=-1)
    assert selection.indices.shape == (2, 3, 4)
    assert selection.scores is not None
    torch.testing.assert_close(selection.indices, expected_indices)
    torch.testing.assert_close(selection.scores, expected_scores)


def test_search_is_deterministic_for_fixed_inputs() -> None:
    generator = torch.Generator().manual_seed(23)
    keys = torch.randn(1, 2, 13, 4, generator=generator)
    query = torch.randn(1, 2, 4, generator=generator)
    index = BruteForceIndex()
    index.build(keys)

    first = index.search(query, budget=5)
    second = index.search(query, budget=5)

    torch.testing.assert_close(first.indices, second.indices)
    assert first.scores is not None and second.scores is not None
    torch.testing.assert_close(first.scores, second.scores)


def test_search_requires_build() -> None:
    with pytest.raises(RuntimeError, match="build must be called"):
        BruteForceIndex().search(torch.randn(1, 1, 4), budget=1)


@pytest.mark.parametrize("budget", [0, -1, 5])
def test_search_rejects_invalid_budget(budget: int) -> None:
    index = BruteForceIndex()
    index.build(torch.randn(1, 1, 4, 3))

    with pytest.raises(ValueError):
        index.search(torch.randn(1, 1, 3), budget=budget)


def test_search_rejects_boolean_budget() -> None:
    index = BruteForceIndex()
    index.build(torch.randn(1, 1, 4, 3))

    with pytest.raises(TypeError):
        index.search(torch.randn(1, 1, 3), budget=True)


@pytest.mark.parametrize(
    "query",
    [
        torch.randn(1, 1, 1, 3),
        torch.randn(2, 1, 3),
        torch.randn(1, 2, 3),
        torch.randn(1, 1, 4),
        torch.randn(1, 1, 3, dtype=torch.float64),
    ],
)
def test_search_rejects_invalid_query(query: torch.Tensor) -> None:
    index = BruteForceIndex()
    index.build(torch.randn(1, 1, 4, 3))

    with pytest.raises((TypeError, ValueError)):
        index.search(query, budget=1)
