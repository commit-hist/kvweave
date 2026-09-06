import pytest
import torch

from kvweave import BruteForceIndex, KVCache, QuestIndex, TensorStorage
from kvweave.metrics.reference import selected_attention


def test_cache_build_retrieve_interaction() -> None:
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.0, 2.0]]]])
    values = torch.tensor([[[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]]])
    cache = KVCache(index=BruteForceIndex(), storage=TensorStorage())
    cache.build(keys, values)

    retrieved = cache.retrieve(
        query=torch.tensor([[[1.0, 0.0]]]),
        budget=2,
    )

    assert retrieved.valid_mask is None
    torch.testing.assert_close(retrieved.keys, keys[:, :, [2, 0], :])
    torch.testing.assert_close(retrieved.values, values[:, :, [2, 0], :])


def test_cache_retrieve_before_build_fails() -> None:
    cache = KVCache(index=BruteForceIndex(), storage=TensorStorage())

    with pytest.raises(RuntimeError, match="build must be called"):
        cache.retrieve(torch.randn(1, 1, 2), budget=1)


def test_cache_build_rejects_mismatched_kv() -> None:
    cache = KVCache(index=BruteForceIndex(), storage=TensorStorage())

    with pytest.raises(ValueError, match="identical shapes"):
        cache.build(torch.randn(1, 1, 3, 2), torch.randn(1, 1, 4, 2))


def test_cache_accepts_uniform_quest_page_selection() -> None:
    keys = torch.tensor([[[[3.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]])
    values = keys + 10.0
    cache = KVCache(index=QuestIndex(page_size=2), storage=TensorStorage())
    cache.build(keys, values)

    retrieved = cache.retrieve(
        query=torch.tensor([[[1.0, 0.0]]]),
        budget=2,
    )

    assert retrieved.valid_mask is None
    torch.testing.assert_close(retrieved.keys, keys[:, :, [0, 1], :])
    torch.testing.assert_close(retrieved.values, values[:, :, [0, 1], :])


def test_ragged_quest_selection_survives_storage_cache_and_attention() -> None:
    sequence_length = 65
    page_size = 8
    keys = torch.zeros(1, 2, sequence_length, 2)
    keys[0, 0, 64, 0] = 10.0
    keys[0, 1, :page_size, 0] = 10.0
    keys[0, 1, 64, 0] = -10.0
    values = torch.arange(
        1 * 2 * sequence_length * 2,
        dtype=torch.float32,
    ).reshape(1, 2, sequence_length, 2)
    query = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    index = QuestIndex(page_size=page_size)
    storage = TensorStorage()
    cache = KVCache(index=index, storage=storage)
    cache.build(keys, values)

    selection = index.search(query, budget=page_size)
    fetched = storage.fetch(selection)
    retrieved = cache.retrieve(query, budget=page_size)

    assert selection.valid_mask is not None
    torch.testing.assert_close(selection.valid_token_counts, torch.tensor([[1, 8]]))
    assert fetched.valid_mask is not None
    assert retrieved.valid_mask is not None
    torch.testing.assert_close(retrieved.valid_mask, fetched.valid_mask)
    torch.testing.assert_close(retrieved.keys, fetched.keys)
    torch.testing.assert_close(retrieved.values, fetched.values)
    output = selected_attention(
        query,
        retrieved.keys,
        retrieved.values,
        retrieved.valid_mask,
    )

    torch.testing.assert_close(output[0, 0], values[0, 0, 64])
    torch.testing.assert_close(
        output[0, 1],
        values[0, 1, :page_size].mean(dim=0),
    )
