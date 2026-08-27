import math

import pytest
import torch

from benchmarks.phase3a import (
    TEXT_FIXTURES,
    TextFixture,
    aggregate_quest_bound_looseness,
    attention_entropy,
    build_deterministic_fixture,
    calculate_query_positions,
    canonicalize_selection_for_attention,
    effective_attention_support,
    normalized_attention_entropy,
    pq_score_approximation_metrics,
    quest_bound_quality,
    top_attention_mass,
)
from kvdb.core.types import Selection
from kvdb.indexes.quest import build_page_metadata


class IntegerTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [len(piece) for piece in text.split()]}


def test_phase3a_fixtures_cover_eight_unique_structures() -> None:
    assert len(TEXT_FIXTURES) == 8
    assert len({fixture.fixture_id for fixture in TEXT_FIXTURES}) == 8
    assert len({fixture.structure for fixture in TEXT_FIXTURES}) == 8
    assert all(fixture.text.strip() for fixture in TEXT_FIXTURES)


def test_deterministic_fixture_repeats_and_truncates_to_exact_length() -> None:
    fixture = TextFixture("tiny", "three word structure", "a bb ccc")

    first = build_deterministic_fixture(IntegerTokenizer(), fixture, 8)
    second = build_deterministic_fixture(IntegerTokenizer(), fixture, 8)

    assert first.base_token_count == 3
    assert first.repetitions == 3
    assert first.input_ids.tolist() == [[1, 2, 3, 1, 2, 3, 1, 2]]
    assert first.token_ids_sha256 == second.token_ids_sha256
    torch.testing.assert_close(first.input_ids, second.input_ids)


def test_query_positions_use_inclusive_fractional_causal_prefixes() -> None:
    positions = calculate_query_positions(512)

    assert [position.label for position in positions] == [
        "25_percent",
        "50_percent",
        "75_percent",
        "final",
    ]
    assert [position.token_index for position in positions] == [127, 255, 383, 511]
    assert [position.causal_token_count for position in positions] == [
        128,
        256,
        384,
        512,
    ]


def test_query_positions_round_nonintegral_prefixes_up_deterministically() -> None:
    positions = calculate_query_positions(7)

    assert [position.token_index for position in positions] == [1, 3, 5, 6]
    assert [position.causal_token_count for position in positions] == [2, 4, 6, 7]


def test_attention_selection_is_canonicalized_without_changing_candidates() -> None:
    selection = Selection(
        indices=torch.tensor([[[3, 0, 2, 0]]], dtype=torch.int64),
        scores=torch.tensor([[[0.9, 0.8, 0.7, 0.0]]]),
        valid_mask=torch.tensor([[[True, True, True, False]]]),
    )

    canonical = canonicalize_selection_for_attention(selection)

    assert canonical.indices.tolist() == [[[0, 2, 3, 0]]]
    assert canonical.scores is not None
    torch.testing.assert_close(
        canonical.scores,
        torch.tensor([[[0.8, 0.7, 0.9, 0.0]]]),
    )
    assert canonical.valid_mask is not None
    assert canonical.valid_mask.tolist() == [[[True, True, True, False]]]
    assert set(canonical.indices[canonical.valid_mask].tolist()) == {0, 2, 3}


def test_attention_entropy_top_mass_and_effective_support_have_explicit_units() -> None:
    weights = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
            ]
        ]
    )

    entropy = attention_entropy(weights)

    torch.testing.assert_close(entropy, torch.tensor([[0.0, math.log(4.0)]]))
    torch.testing.assert_close(
        normalized_attention_entropy(weights),
        torch.tensor([[0.0, 1.0]]),
    )
    torch.testing.assert_close(
        effective_attention_support(weights),
        torch.tensor([[1.0, 4.0]]),
    )
    torch.testing.assert_close(
        top_attention_mass(weights, 1),
        torch.tensor([[1.0, 0.25]]),
    )
    torch.testing.assert_close(
        top_attention_mass(weights, 16),
        torch.tensor([[1.0, 1.0]]),
    )


def test_quest_bound_quality_reports_exact_page_looseness() -> None:
    keys = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 2.0],
                    [1.0, 3.0],
                ]
            ]
        ]
    )
    query = torch.tensor([[[1.0, 1.0]]])
    metadata = build_page_metadata(keys, page_size=2)

    quality = quest_bound_quality(query, keys, metadata)

    torch.testing.assert_close(quality.upper_bound, torch.tensor([[[2.0, 5.0]]]))
    torch.testing.assert_close(
        quality.true_page_maximum,
        torch.tensor([[[1.0, 4.0]]]),
    )
    torch.testing.assert_close(quality.looseness, torch.tensor([[[1.0, 1.0]]]))


def test_quest_bound_looseness_aggregates_selected_and_nonselected_pages() -> None:
    looseness = torch.tensor([[[1.0, 3.0, 8.0]]])
    selected = torch.tensor([[[0, 2]]], dtype=torch.int64)

    aggregates = aggregate_quest_bound_looseness(looseness, selected)

    assert aggregates["quest_bound_looseness_all_mean"].item() == 4.0
    assert aggregates["quest_bound_looseness_all_max"].item() == 8.0
    assert aggregates["quest_bound_looseness_selected_mean"].item() == 4.5
    assert aggregates["quest_bound_looseness_selected_max"].item() == 8.0
    assert aggregates["quest_bound_looseness_nonselected_mean"].item() == 3.0
    assert aggregates["quest_bound_looseness_nonselected_max"].item() == 3.0


def test_pq_score_approximation_metrics_include_tie_aware_rank_and_top_errors() -> None:
    exact = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    approximate = torch.tensor([[[3.0, 3.0, 2.0, 0.0]]])

    metrics = pq_score_approximation_metrics(
        approximate,
        exact,
        high_attention_count=2,
    )

    torch.testing.assert_close(metrics["pq_score_mae"], torch.tensor([[0.5]]))
    torch.testing.assert_close(
        metrics["pq_score_rmse"],
        torch.tensor([[math.sqrt(0.5)]]),
    )
    assert metrics["pq_score_spearman_rank_correlation"].item() == pytest.approx(
        0.9486832980505138
    )
    assert metrics["pq_exact_top_token_signed_score_error"].item() == -1.0
    assert metrics["pq_exact_top_token_absolute_score_error"].item() == 1.0
    assert metrics["pq_exact_top_16_score_mae"].item() == 0.5


@pytest.mark.parametrize("sequence_length", [0, -1, True, 1.5])
def test_invalid_fixture_lengths_and_query_lengths_are_rejected(
    sequence_length: object,
) -> None:
    fixture = TextFixture("tiny", "tiny", "token")
    with pytest.raises((TypeError, ValueError)):
        build_deterministic_fixture(
            IntegerTokenizer(),
            fixture,
            sequence_length,  # type: ignore[arg-type]
        )
    with pytest.raises((TypeError, ValueError)):
        calculate_query_positions(sequence_length)  # type: ignore[arg-type]
