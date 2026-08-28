import pytest
import torch

from kvweave.indexes.quest import (
    QuestIndex,
    QuestMetadata,
    build_page_metadata,
    expand_pages_to_tokens,
    score_pages,
    token_budget_to_pages,
)


def loop_page_metadata(
    keys: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deliberately slow test-only oracle for page extrema."""
    batch_size, kv_heads, sequence_length, head_dim = keys.shape
    num_pages = (sequence_length + page_size - 1) // page_size
    minimum = torch.empty(batch_size, kv_heads, num_pages, head_dim)
    maximum = torch.empty_like(minimum)
    for batch_id in range(batch_size):
        for head_id in range(kv_heads):
            for page_id in range(num_pages):
                start = page_id * page_size
                stop = min(start + page_size, sequence_length)
                page = keys[batch_id, head_id, start:stop]
                for dimension in range(head_dim):
                    minimum[batch_id, head_id, page_id, dimension] = page[
                        :, dimension
                    ].min()
                    maximum[batch_id, head_id, page_id, dimension] = page[
                        :, dimension
                    ].max()
    return minimum, maximum


@pytest.mark.parametrize(
    ("shape", "page_size", "value_transform"),
    [
        ((1, 1, 3, 2), 8, "positive"),
        ((1, 1, 4, 3), 2, "negative"),
        ((1, 1, 5, 4), 2, "mixed"),
        ((2, 3, 7, 5), 3, "mixed"),
    ],
)
def test_metadata_matches_slow_loop_oracle(
    shape: tuple[int, int, int, int],
    page_size: int,
    value_transform: str,
) -> None:
    generator = torch.Generator().manual_seed(101 + sum(shape) + page_size)
    keys = torch.randn(*shape, generator=generator)
    if value_transform == "positive":
        keys = keys.abs() + 0.25
    elif value_transform == "negative":
        keys = -(keys.abs() + 0.25)

    metadata = build_page_metadata(keys, page_size)
    expected_minimum, expected_maximum = loop_page_metadata(keys, page_size)

    expected_pages = (shape[2] + page_size - 1) // page_size
    assert metadata.minimum.shape == (shape[0], shape[1], expected_pages, shape[3])
    assert metadata.maximum.shape == metadata.minimum.shape
    assert metadata.num_pages == expected_pages
    assert metadata.sequence_length == shape[2]
    torch.testing.assert_close(metadata.minimum, expected_minimum)
    torch.testing.assert_close(metadata.maximum, expected_maximum)


@pytest.mark.parametrize(
    ("sequence_length", "page_size", "expected_lengths"),
    [
        (3, 8, [3]),
        (4, 2, [2, 2]),
        (5, 2, [2, 2, 1]),
        (7, 3, [3, 3, 1]),
    ],
)
def test_page_count_and_valid_lengths(
    sequence_length: int,
    page_size: int,
    expected_lengths: list[int],
) -> None:
    metadata = build_page_metadata(
        torch.randn(2, 3, sequence_length, 4),
        page_size,
    )

    assert metadata.num_pages == len(expected_lengths)
    torch.testing.assert_close(
        metadata.page_lengths.cpu(),
        torch.tensor(expected_lengths),
    )


@pytest.mark.parametrize("page_size", [0, -1])
def test_invalid_page_size_value_is_rejected(page_size: int) -> None:
    with pytest.raises(ValueError, match="page_size"):
        QuestIndex(page_size)
    with pytest.raises(ValueError, match="page_size"):
        build_page_metadata(torch.randn(1, 1, 3, 2), page_size)


@pytest.mark.parametrize("page_size", [True, 1.5, "2"])
def test_invalid_page_size_type_is_rejected(page_size: object) -> None:
    with pytest.raises(TypeError, match="page_size"):
        QuestIndex(page_size)  # type: ignore[arg-type]


def test_page_scoring_uses_sign_aware_upper_bound() -> None:
    metadata = QuestMetadata(
        minimum=torch.tensor([[[[-2.0, -3.0], [1.0, -5.0]]]]),
        maximum=torch.tensor([[[[4.0, 7.0], [6.0, -1.0]]]]),
        page_size=2,
        sequence_length=4,
    )
    query = torch.tensor([[[3.0, -2.0]]])

    scores = score_pages(query, metadata)

    # Page 0: max(-6, 12) + max(6, -14) = 18.
    # Page 1: max(3, 18) + max(10, 2) = 28.
    torch.testing.assert_close(scores, torch.tensor([[[18.0, 28.0]]]))


@pytest.mark.parametrize(
    ("budget", "page_size", "num_pages", "expected"),
    [
        (1, 4, 5, 1),
        (4, 4, 5, 1),
        (5, 4, 5, 2),
        (19, 4, 5, 5),
        (20, 4, 5, 5),
        (1_000, 4, 5, 5),
    ],
)
def test_token_budget_rounds_up_and_caps_at_available_pages(
    budget: int,
    page_size: int,
    num_pages: int,
    expected: int,
) -> None:
    assert (
        token_budget_to_pages(
            budget,
            page_size=page_size,
            num_pages=num_pages,
        )
        == expected
    )


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_value_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError, match="token_budget"):
        token_budget_to_pages(budget, page_size=4, num_pages=2)


@pytest.mark.parametrize("budget", [True, 1.5, "2"])
def test_invalid_budget_type_is_rejected(budget: object) -> None:
    with pytest.raises(TypeError, match="token_budget"):
        token_budget_to_pages(
            budget,  # type: ignore[arg-type]
            page_size=4,
            num_pages=2,
        )


def test_partial_page_expansion_never_selects_out_of_range_tokens() -> None:
    metadata = build_page_metadata(torch.randn(1, 1, 5, 2), page_size=2)
    page_indices = torch.tensor([[[2, 0]]])

    selection = expand_pages_to_tokens(page_indices, metadata)

    torch.testing.assert_close(selection.indices, torch.tensor([[[4, 0, 1]]]))
    assert selection.valid_mask is not None
    assert torch.all(selection.valid_mask).item()
    torch.testing.assert_close(selection.valid_token_counts, torch.tensor([[3]]))
    assert torch.all(selection.indices < metadata.sequence_length).item()


def test_partial_page_can_make_selection_ragged_across_batches() -> None:
    metadata = build_page_metadata(torch.randn(2, 1, 5, 2), page_size=2)
    # Batch 0 selects the one-token tail; batch 1 selects two full pages.
    page_indices = torch.tensor([[[2, 0]], [[0, 1]]])

    selection = expand_pages_to_tokens(page_indices, metadata)

    assert selection.valid_mask is not None
    torch.testing.assert_close(selection.valid_token_counts, torch.tensor([[3], [4]]))
    torch.testing.assert_close(selection.indices[0, 0], torch.tensor([4, 0, 1, 0]))
    torch.testing.assert_close(
        selection.valid_mask[0, 0],
        torch.tensor([True, True, True, False]),
    )
    assert torch.all(
        selection.indices[selection.valid_mask] < metadata.sequence_length
    ).item()


def test_page_ranking_is_independent_per_batch_and_head() -> None:
    keys = torch.tensor(
        [
            [
                [[[9.0], [8.0], [1.0], [0.0]]],
                [[[1.0], [0.0], [9.0], [8.0]]],
            ],
            [
                [[[-9.0], [-8.0], [-1.0], [0.0]]],
                [[[-1.0], [0.0], [-9.0], [-8.0]]],
            ],
        ]
    ).squeeze(2)
    query = torch.tensor([[[1.0], [1.0]], [[-1.0], [-1.0]]])
    index = QuestIndex(page_size=2)
    index.build(keys)

    result = index.search_with_details(query, budget=1)

    torch.testing.assert_close(
        result.page_indices,
        torch.tensor([[[0], [1]], [[0], [1]]]),
    )
    torch.testing.assert_close(
        result.selection.indices,
        torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]]]),
    )


def test_partial_page_is_ranked_normally_not_force_included() -> None:
    keys = torch.tensor([[[[5.0], [4.0], [3.0], [2.0], [-100.0]]]])
    index = QuestIndex(page_size=2)
    index.build(keys)

    result = index.search_with_details(torch.ones(1, 1, 1), budget=2)

    torch.testing.assert_close(result.page_indices, torch.tensor([[[0]]]))
    torch.testing.assert_close(result.selection.indices, torch.tensor([[[0, 1]]]))


def test_ties_choose_lower_page_id_deterministically() -> None:
    index = QuestIndex(page_size=2)
    index.build(torch.zeros(1, 1, 7, 3))
    query = torch.ones(1, 1, 3)

    results = [index.search_with_details(query, budget=4) for _ in range(5)]

    for result in results:
        torch.testing.assert_close(result.page_indices, torch.tensor([[[0, 1]]]))
        torch.testing.assert_close(
            result.selection.indices,
            torch.tensor([[[0, 1, 2, 3]]]),
        )


def test_search_reports_requested_pages_and_actual_tokens_separately() -> None:
    keys = torch.tensor([[[[0.0], [0.0], [1.0], [1.0], [10.0]]]])
    index = QuestIndex(page_size=2)
    index.build(keys)

    result = index.search_with_details(torch.ones(1, 1, 1), budget=4)

    assert result.requested_token_budget == 4
    assert result.num_pages_to_select == 2
    torch.testing.assert_close(result.page_indices, torch.tensor([[[2, 1]]]))
    torch.testing.assert_close(result.actual_token_counts, torch.tensor([[3]]))
    torch.testing.assert_close(result.selection.indices, torch.tensor([[[4, 2, 3]]]))


def test_search_requires_build() -> None:
    with pytest.raises(RuntimeError, match="build must be called"):
        QuestIndex(page_size=2).search(torch.randn(1, 1, 3), budget=1)


def test_search_rejects_gqa_query_shape() -> None:
    index = QuestIndex(page_size=2)
    index.build(torch.randn(1, 2, 4, 3))

    with pytest.raises(ValueError, match="query must have shape"):
        index.search(torch.randn(1, 4, 3), budget=2)
