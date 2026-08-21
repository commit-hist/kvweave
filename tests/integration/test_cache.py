import pytest
import torch

from kvdb import BruteForceIndex, KVCache, TensorStorage


def test_cache_build_retrieve_interaction() -> None:
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.0, 2.0]]]]
    )
    values = torch.tensor(
        [[[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]]]
    )
    cache = KVCache(index=BruteForceIndex(), storage=TensorStorage())
    cache.build(keys, values)

    selected_keys, selected_values = cache.retrieve(
        query=torch.tensor([[[1.0, 0.0]]]),
        budget=2,
    )

    torch.testing.assert_close(selected_keys, keys[:, :, [2, 0], :])
    torch.testing.assert_close(selected_values, values[:, :, [2, 0], :])


def test_cache_retrieve_before_build_fails() -> None:
    cache = KVCache(index=BruteForceIndex(), storage=TensorStorage())

    with pytest.raises(RuntimeError, match="build must be called"):
        cache.retrieve(torch.randn(1, 1, 2), budget=1)


def test_cache_build_rejects_mismatched_kv() -> None:
    cache = KVCache(index=BruteForceIndex(), storage=TensorStorage())

    with pytest.raises(ValueError, match="identical shapes"):
        cache.build(torch.randn(1, 1, 3, 2), torch.randn(1, 1, 4, 2))
