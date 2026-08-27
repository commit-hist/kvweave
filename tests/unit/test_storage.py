import pytest
import torch

from kvdb.core.types import Selection
from kvdb.storage.tensor import TensorStorage


def test_fetch_gathers_keys_and_values_per_head() -> None:
    keys = torch.arange(1 * 2 * 4 * 2, dtype=torch.float32).reshape(1, 2, 4, 2)
    values = keys + 100.0
    selection = Selection(indices=torch.tensor([[[2, 0], [1, 3]]]))
    storage = TensorStorage()
    storage.put(keys, values)

    retrieved = storage.fetch(selection)

    expected_keys = torch.stack(
        [
            torch.stack([keys[0, 0, 2], keys[0, 0, 0]]),
            torch.stack([keys[0, 1, 1], keys[0, 1, 3]]),
        ]
    ).unsqueeze(0)
    assert retrieved.keys.shape == (1, 2, 2, 2)
    assert retrieved.valid_mask is None
    torch.testing.assert_close(retrieved.keys, expected_keys)
    torch.testing.assert_close(retrieved.values, expected_keys + 100.0)


def test_fetch_requires_put() -> None:
    selection = Selection(indices=torch.tensor([[[0]]]))

    with pytest.raises(RuntimeError, match="put must be called"):
        TensorStorage().fetch(selection)


def test_fetch_rejects_out_of_range_index() -> None:
    storage = TensorStorage()
    storage.put(torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2))
    selection = Selection(indices=torch.tensor([[[3]]]))

    with pytest.raises(IndexError, match="sequence length"):
        storage.fetch(selection)


def test_fetch_rejects_mismatched_batch_or_heads() -> None:
    storage = TensorStorage()
    storage.put(torch.randn(1, 2, 3, 2), torch.randn(1, 2, 3, 2))
    selection = Selection(indices=torch.tensor([[[0]]]))

    with pytest.raises(ValueError, match="batch and KV-head"):
        storage.fetch(selection)


def test_fetch_preserves_ragged_mask_and_ignores_invalid_placeholder_index() -> None:
    storage = TensorStorage()
    keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    values = keys + 100.0
    storage.put(keys, values)
    selection = Selection(
        indices=torch.tensor([[[2, 99]]]),
        valid_mask=torch.tensor([[[True, False]]]),
    )

    retrieved = storage.fetch(selection)

    assert retrieved.valid_mask is selection.valid_mask
    torch.testing.assert_close(retrieved.keys[0, 0, 0], keys[0, 0, 2])
    torch.testing.assert_close(retrieved.values[0, 0, 0], values[0, 0, 2])


def test_fetch_elides_an_all_valid_mask() -> None:
    storage = TensorStorage()
    keys = torch.randn(1, 1, 3, 2)
    values = torch.randn(1, 1, 3, 2)
    storage.put(keys, values)

    retrieved = storage.fetch(
        Selection(
            indices=torch.tensor([[[2, 0]]]),
            valid_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        )
    )

    assert retrieved.valid_mask is None
