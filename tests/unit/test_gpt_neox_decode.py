import pytest
import torch

from kvweave import KVCache, PQIndex, QuestIndex, TensorStorage
from kvweave.core.types import Selection
from kvweave.integrations.transformers import (
    DecodeMode,
    DecodeStrategy,
    QuestMetadataUpdateMode,
    append_causal_kv,
    generation_divergence_metrics,
    logit_comparison_metrics,
    prepare_decode_selection,
    select_decode_input,
    update_decode_cache,
)


def make_kv(sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.arange(
        2 * sequence_length * 4,
        dtype=torch.float32,
    ).reshape(1, 2, sequence_length, 4)
    return keys, keys + 100.0


def test_causal_kv_append_preserves_existing_order_and_adds_newest_last() -> None:
    keys, values = make_kv(3)
    new_keys = torch.full((1, 2, 1, 4), 999.0)
    new_values = torch.full((1, 2, 1, 4), 1_999.0)

    appended_keys, appended_values = append_causal_kv(
        keys,
        values,
        new_keys,
        new_values,
    )

    torch.testing.assert_close(appended_keys[:, :, :3], keys)
    torch.testing.assert_close(appended_values[:, :, :3], values)
    torch.testing.assert_close(appended_keys[:, :, -1:], new_keys)
    torch.testing.assert_close(appended_values[:, :, -1:], new_values)


def test_decode_selection_forces_newest_and_sorts_in_causal_order() -> None:
    selection = Selection(
        indices=torch.tensor([[[3, 1, 0], [4, 2, 1]]], dtype=torch.int64),
        scores=torch.tensor([[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]]),
    )

    prepared = prepare_decode_selection(selection, newest_token_index=4)

    torch.testing.assert_close(
        prepared.indices,
        torch.tensor([[[1, 3, 4], [1, 2, 4]]], dtype=torch.int64),
    )
    assert prepared.valid_mask is None
    assert torch.all((prepared.indices == 4).any(dim=-1)).item()


def test_decode_selection_preserves_ragged_counts_and_masks_padding() -> None:
    selection = Selection(
        indices=torch.tensor([[[2, 0, 0], [3, 1, 0]]], dtype=torch.int64),
        valid_mask=torch.tensor([[[True, False, False], [True, True, False]]]),
    )

    prepared = prepare_decode_selection(selection, newest_token_index=4)

    torch.testing.assert_close(
        prepared.valid_token_counts,
        torch.tensor([[1, 2]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        prepared.indices,
        torch.tensor([[[4, 0, 0], [3, 4, 0]]], dtype=torch.int64),
    )
    assert prepared.valid_mask is not None
    assert torch.all(((prepared.indices == 4) & prepared.valid_mask).any(dim=-1)).item()


@pytest.mark.parametrize(
    "update_mode",
    [
        QuestMetadataUpdateMode.FULL_REBUILD,
        QuestMetadataUpdateMode.INCREMENTAL,
    ],
)
def test_quest_decode_update_extends_metadata_for_appended_length(
    update_mode: QuestMetadataUpdateMode,
) -> None:
    keys, values = make_kv(4)
    cache = KVCache(index=QuestIndex(page_size=2), storage=TensorStorage())
    cache.build(keys, values)
    new_keys = torch.full((1, 2, 1, 4), 10.0)
    new_values = torch.full((1, 2, 1, 4), 20.0)
    appended_keys, appended_values = append_causal_kv(
        keys,
        values,
        new_keys,
        new_values,
    )

    update_decode_cache(
        DecodeStrategy.QUEST,
        cache,
        keys=appended_keys,
        values=appended_values,
        new_keys=new_keys,
        quest_metadata_update_mode=update_mode,
    )

    assert cache.index.metadata.sequence_length == 5
    retrieved = cache.retrieve(torch.ones(1, 2, 4), budget=5)
    assert retrieved.keys.shape[2] == 5


def test_quest_decode_incremental_update_matches_full_rebuild_oracle() -> None:
    generator = torch.Generator().manual_seed(8_117)
    all_keys = torch.randn(2, 3, 13, 5, generator=generator)
    all_values = torch.randn(2, 3, 13, 5, generator=generator)
    oracle = KVCache(index=QuestIndex(page_size=4), storage=TensorStorage())
    incremental = KVCache(index=QuestIndex(page_size=4), storage=TensorStorage())
    oracle.build(all_keys[:, :, :3], all_values[:, :, :3])
    incremental.build(all_keys[:, :, :3], all_values[:, :, :3])

    for sequence_length in range(4, all_keys.shape[2] + 1):
        keys = all_keys[:, :, :sequence_length]
        values = all_values[:, :, :sequence_length]
        new_keys = all_keys[:, :, sequence_length - 1 : sequence_length]
        update_decode_cache(
            DecodeStrategy.QUEST,
            oracle,
            keys=keys,
            values=values,
            new_keys=new_keys,
            quest_metadata_update_mode=QuestMetadataUpdateMode.FULL_REBUILD,
        )
        update_decode_cache(
            DecodeStrategy.QUEST,
            incremental,
            keys=keys,
            values=values,
            new_keys=new_keys,
            quest_metadata_update_mode=QuestMetadataUpdateMode.INCREMENTAL,
        )

        assert torch.equal(
            incremental.index.metadata.minimum,
            oracle.index.metadata.minimum,
        )
        assert torch.equal(
            incremental.index.metadata.maximum,
            oracle.index.metadata.maximum,
        )
        query = torch.randn(2, 3, 5, generator=generator)
        budget = max(1, sequence_length // 2)
        oracle_selection = oracle.index.search(query, budget)
        incremental_selection = incremental.index.search(query, budget)
        assert torch.equal(incremental_selection.indices, oracle_selection.indices)
        assert torch.equal(incremental_selection.scores, oracle_selection.scores)
        assert (
            incremental_selection.valid_mask is None
            and oracle_selection.valid_mask is None
        ) or torch.equal(
            incremental_selection.valid_mask,
            oracle_selection.valid_mask,
        )
        oracle_kv = oracle.storage.fetch(oracle_selection)
        incremental_kv = incremental.storage.fetch(incremental_selection)
        assert torch.equal(incremental_kv.keys, oracle_kv.keys)
        assert torch.equal(incremental_kv.values, oracle_kv.values)


def test_pq_decode_update_freezes_codebooks_and_encodes_only_new_key() -> None:
    keys, values = make_kv(8)
    cache = KVCache(
        index=PQIndex(
            num_subspaces=2,
            num_centroids=2,
            max_iterations=2,
            seed=0,
        ),
        storage=TensorStorage(),
    )
    cache.build(keys, values)
    original_codebooks = cache.index.metadata.codebooks
    original_codes = cache.index.metadata.codes.clone()
    new_keys = torch.full((1, 2, 1, 4), 10.0)
    new_values = torch.full((1, 2, 1, 4), 20.0)
    appended_keys, appended_values = append_causal_kv(
        keys,
        values,
        new_keys,
        new_values,
    )

    update_decode_cache(
        DecodeStrategy.PQ,
        cache,
        keys=appended_keys,
        values=appended_values,
        new_keys=new_keys,
    )

    assert cache.index.metadata.codebooks is original_codebooks
    torch.testing.assert_close(cache.index.metadata.codes[:, :, :8], original_codes)
    assert cache.index.metadata.codes.shape == (1, 2, 9, 2)
    retrieved = cache.retrieve(torch.ones(1, 2, 4), budget=9)
    assert retrieved.keys.shape[2] == 9


def test_teacher_forced_and_free_running_bookkeeping_choose_expected_token() -> None:
    dense = torch.tensor([[7]], dtype=torch.int64)
    approximate = torch.tensor([[9]], dtype=torch.int64)

    assert torch.equal(
        select_decode_input(
            DecodeMode.TEACHER_FORCED,
            dense_token=dense,
            path_token=approximate,
        ),
        dense,
    )
    assert torch.equal(
        select_decode_input(
            DecodeMode.FREE_RUNNING,
            dense_token=dense,
            path_token=approximate,
        ),
        approximate,
    )


def test_generation_divergence_metrics_record_first_difference_and_reagreement() -> (
    None
):
    metrics = generation_divergence_metrics(
        [1, 2, 3, 4, 5],
        [1, 2, 9, 4, 8],
    )

    assert metrics["first_divergence_position"] == 2
    assert metrics["longest_common_prefix_tokens"] == 2
    assert metrics["token_agreement_rate"] == pytest.approx(0.6)
    assert metrics["reconverged_after_first_divergence"] is True
    assert metrics["differing_positions"] == [2, 4]
    assert metrics["cumulative_difference_count_by_position"] == [0, 0, 1, 1, 2]


def test_logit_metrics_report_rank_agreement_overlap_and_finite_kl() -> None:
    dense = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, -1.0]])
    approximate = torch.tensor([[3.0, 4.0, 2.0, 1.0, 0.0, -1.0]])

    metrics = logit_comparison_metrics(approximate, dense)

    assert metrics["dense_top_1_token"] == 0
    assert metrics["approximate_top_1_token"] == 1
    assert metrics["dense_top_1_rank_under_approximate_logits"] == 2
    assert metrics["top_1_agreement"] is False
    assert metrics["top_5_overlap_count"] == 5
    assert metrics["top_5_overlap_fraction"] == 1.0
    assert math_is_finite(metrics["kl_divergence_dense_to_approximate"])


def math_is_finite(value: object) -> bool:
    return isinstance(value, float) and value >= 0.0 and value < float("inf")
