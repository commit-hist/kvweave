import pytest
import torch

from kvweave import BruteForceIndex, KVCache, RetrievedKV, TensorStorage
from kvweave.core.types import Selection
from kvweave.indexes.quest import QuestIndex, build_page_metadata, score_pages
from kvweave.metrics.reference import (
    candidate_recall,
    compare_attention,
    selected_attention,
)


def _adversarial_keys() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [
                    [8.0, -7.0, 0.5, -0.25],
                    [-9.0, 6.0, -0.75, 0.5],
                    [4.0, 3.0, -5.0, -6.0],
                    [-2.0, -1.0, 7.0, 8.0],
                    [10.0, -11.0, 12.0, -13.0],
                ]
            ]
        ]
    )


@pytest.mark.parametrize("query_kind", ["random", "positive", "negative", "mixed"])
@pytest.mark.parametrize("keys_kind", ["random", "adversarial"])
def test_page_score_is_upper_bound_for_every_token_including_partial_page(
    query_kind: str,
    keys_kind: str,
) -> None:
    generator = torch.Generator().manual_seed(907)
    if keys_kind == "random":
        keys = torch.randn(2, 3, 7, 4, generator=generator)
    else:
        keys = _adversarial_keys()

    query = torch.randn(
        keys.shape[0],
        keys.shape[1],
        keys.shape[-1],
        generator=generator,
    )
    if query_kind == "positive":
        query = query.abs() + 0.1
    elif query_kind == "negative":
        query = -(query.abs() + 0.1)
    elif query_kind == "mixed":
        signs = torch.tensor([1.0, -1.0, 1.0, -1.0])
        query = (query.abs() + 0.1) * signs

    page_size = 3
    metadata = build_page_metadata(keys, page_size)
    upper_bounds = score_pages(query, metadata)

    for page_id in range(metadata.num_pages):
        start = page_id * page_size
        stop = min(start + page_size, keys.shape[2])
        exact_scores = torch.einsum(
            "bhd,bhtd->bht",
            query,
            keys[:, :, start:stop],
        )
        bound = upper_bounds[:, :, page_id].unsqueeze(-1)
        tolerance = 1e-5 + 1e-5 * exact_scores.abs()
        assert torch.all(bound >= exact_scores - tolerance).item()


@pytest.mark.parametrize("budget", [7, 8, 100])
def test_budget_at_least_sequence_length_recovers_every_token_once(
    budget: int,
) -> None:
    generator = torch.Generator().manual_seed(53)
    keys = torch.randn(2, 3, 7, 5, generator=generator)
    query = torch.randn(2, 3, 5, generator=generator)
    index = QuestIndex(page_size=3)
    index.build(keys)

    result = index.search_with_details(query, budget)

    assert result.num_pages_to_select == 3
    torch.testing.assert_close(
        result.actual_token_counts,
        torch.full((2, 3), 7, dtype=torch.int64),
    )
    assert result.selection.valid_mask is not None
    for batch_id in range(2):
        for head_id in range(3):
            valid = result.selection.valid_mask[batch_id, head_id]
            selected = result.selection.indices[batch_id, head_id][valid]
            torch.testing.assert_close(
                selected.sort().values,
                torch.arange(7),
            )
            assert selected.unique().numel() == 7


def test_candidate_recall_counts_exact_topk_contained_anywhere_in_candidates() -> None:
    candidates = Selection(indices=torch.tensor([[[4, 0, 2, 3]]]))
    exact = Selection(indices=torch.tensor([[[2, 1]]]))

    recall = candidate_recall(candidates, exact)

    torch.testing.assert_close(recall, torch.tensor([[0.5]], dtype=torch.float32))


def test_candidate_recall_ignores_masked_rectangular_placeholders() -> None:
    candidates = Selection(
        indices=torch.tensor([[[4, 0]]]),
        valid_mask=torch.tensor([[[True, False]]]),
    )
    exact = Selection(indices=torch.tensor([[[0]]]))

    torch.testing.assert_close(
        candidate_recall(candidates, exact),
        torch.tensor([[0.0]], dtype=torch.float32),
    )


def test_quest_candidate_recall_against_brute_force_uses_explicit_comparison_k() -> (
    None
):
    keys = torch.tensor([[[[9.0], [8.0], [1.0], [0.0], [7.0], [-10.0]]]])
    query = torch.ones(1, 1, 1)
    quest = QuestIndex(page_size=2)
    quest.build(keys)
    brute_force = BruteForceIndex()
    brute_force.build(keys)

    quest_result = quest.search_with_details(query, budget=3)
    exact_topk = brute_force.search(query, budget=3)
    recall = candidate_recall(quest_result.selection, exact_topk)

    assert quest_result.requested_token_budget == 3
    assert quest_result.num_pages_to_select == 2
    torch.testing.assert_close(quest_result.actual_token_counts, torch.tensor([[4]]))
    assert exact_topk.indices.shape[-1] == 3
    torch.testing.assert_close(recall, torch.tensor([[1.0]], dtype=torch.float32))


def test_full_budget_selected_attention_equals_full_attention() -> None:
    generator = torch.Generator().manual_seed(71)
    keys = torch.randn(2, 3, 7, 4, generator=generator)
    values = torch.randn(2, 3, 7, 4, generator=generator)
    query = torch.randn(2, 3, 4, generator=generator)
    quest = QuestIndex(page_size=3)
    cache = KVCache(index=quest, storage=TensorStorage())
    cache.build(keys, values)

    comparison = compare_attention(
        query,
        keys,
        values,
        cache.retrieve(query, budget=7),
    )

    torch.testing.assert_close(
        comparison.selected_output,
        comparison.full_output,
        rtol=1e-5,
        atol=1e-6,
    )
    assert comparison.relative_output_error == pytest.approx(0.0, abs=1e-6)
    assert comparison.selected_token_percentage == pytest.approx(100.0)


def test_reference_attention_reports_sparse_selection_percentage_and_error() -> None:
    keys = torch.tensor([[[[4.0], [3.0], [2.0], [1.0]]]])
    values = torch.tensor([[[[8.0], [4.0], [2.0], [1.0]]]])
    query = torch.ones(1, 1, 1)
    quest = QuestIndex(page_size=2)
    cache = KVCache(index=quest, storage=TensorStorage())
    cache.build(keys, values)

    sparse = compare_attention(query, keys, values, cache.retrieve(query, budget=2))
    full = compare_attention(query, keys, values, cache.retrieve(query, budget=4))

    assert sparse.selected_token_percentage == pytest.approx(50.0)
    assert sparse.relative_output_error > 0.0
    assert full.selected_token_percentage == pytest.approx(100.0)
    assert full.relative_output_error == pytest.approx(0.0, abs=1e-6)
    assert full.relative_output_error < sparse.relative_output_error


def test_masked_attention_ignores_invalid_keys_and_values() -> None:
    query = torch.tensor([[[1.0, -0.5]]])
    keys = torch.tensor([[[[2.0, 1.0], [1.0, 3.0], [float("nan"), float("inf")]]]])
    values = torch.tensor([[[[4.0, 2.0], [8.0, 6.0], [float("nan"), float("inf")]]]])
    valid_mask = torch.tensor([[[True, True, False]]])
    retrieved = RetrievedKV(keys=keys, values=values, valid_mask=valid_mask)

    output = selected_attention(
        query,
        retrieved.keys,
        retrieved.values,
        retrieved.valid_mask,
    )
    expected = selected_attention(query, keys[:, :, :2], values[:, :, :2])

    torch.testing.assert_close(output, expected)


def test_changing_invalid_values_does_not_change_masked_attention() -> None:
    query = torch.tensor([[[1.0, 1.0]]])
    keys = torch.tensor([[[[2.0, 0.0], [0.0, 2.0], [5.0, 5.0]]]])
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]]])
    valid_mask = torch.tensor([[[True, True, False]]])
    changed_values = values.clone()
    changed_values[0, 0, 2] = torch.tensor([float("nan"), float("inf")])

    original = selected_attention(query, keys, values, valid_mask)
    changed = selected_attention(query, keys, changed_values, valid_mask)

    torch.testing.assert_close(changed, original)


def test_selected_attention_rejects_an_all_invalid_row() -> None:
    with pytest.raises(ValueError, match="at least one valid token"):
        selected_attention(
            torch.ones(1, 1, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.zeros(1, 1, 2, dtype=torch.bool),
        )
