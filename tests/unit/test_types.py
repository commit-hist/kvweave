import pytest
import torch

from kvdb.core.types import Selection, validate_kv_tensors


def test_selection_accepts_canonical_shape() -> None:
    indices = torch.tensor([[[2, 0], [1, 3]]], dtype=torch.int64)
    scores = torch.tensor([[[2.0, 1.0], [4.0, 3.0]]])

    selection = Selection(indices=indices, scores=scores)

    assert selection.indices.shape == (1, 2, 2)
    assert selection.scores is not None
    assert selection.scores.shape == selection.indices.shape


@pytest.mark.parametrize(
    ("indices", "error_type"),
    [
        (torch.tensor([[0, 1]], dtype=torch.int64), ValueError),
        (torch.tensor([[[0, 1]]], dtype=torch.int32), TypeError),
        (torch.tensor([[[-1, 1]]], dtype=torch.int64), ValueError),
        (torch.empty((1, 1, 0), dtype=torch.int64), ValueError),
    ],
)
def test_selection_rejects_invalid_indices(
    indices: torch.Tensor,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        Selection(indices=indices)


def test_selection_rejects_mismatched_scores() -> None:
    indices = torch.tensor([[[0, 1]]], dtype=torch.int64)

    with pytest.raises(ValueError, match="scores must match"):
        Selection(indices=indices, scores=torch.ones(1, 1, 1))


def test_kv_validation_accepts_canonical_layout() -> None:
    keys = torch.randn(2, 3, 5, 7)
    values = torch.randn(2, 3, 5, 7)

    validate_kv_tensors(keys, values)


@pytest.mark.parametrize(
    ("keys", "values", "error_type"),
    [
        (torch.randn(2, 3, 5), torch.randn(2, 3, 5), ValueError),
        (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 4, 4), ValueError),
        (torch.ones(1, 2, 3, 4, dtype=torch.int64), torch.ones(1, 2, 3, 4), TypeError),
        (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4, dtype=torch.float64), ValueError),
        (torch.empty(1, 2, 0, 4), torch.empty(1, 2, 0, 4), ValueError),
    ],
)
def test_kv_validation_rejects_invalid_tensors(
    keys: torch.Tensor,
    values: torch.Tensor,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        validate_kv_tensors(keys, values)
