import math

import pytest
import torch

from kvweave.core.types import Selection
from kvweave.integrations.transformers import relative_tensor_error
from kvweave.metrics import relative_l2_error
from kvweave.metrics.reference import selection_mask


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_relative_l2_preserves_native_and_float32_arithmetic(
    dtype: torch.dtype,
) -> None:
    exact = torch.tensor([1.25, -2.5, 0.75], dtype=dtype)
    approximate = torch.tensor([1.375, -2.75, 0.7], dtype=dtype)
    native = (
        torch.linalg.vector_norm(approximate - exact) / torch.linalg.vector_norm(exact)
    ).item()
    float32 = (
        torch.linalg.vector_norm(approximate.float() - exact.float())
        / torch.linalg.vector_norm(exact.float())
    ).item()
    assert relative_l2_error(approximate, exact) == native
    assert relative_l2_error(approximate, exact, dtype=torch.float32) == float32
    assert relative_tensor_error(approximate, exact) == float32


def test_native_precision_is_not_silently_narrowed() -> None:
    exact = torch.tensor([2e100], dtype=torch.float64)
    approximate = torch.tensor([1e100], dtype=torch.float64)
    assert relative_l2_error(approximate, exact) == 0.5
    assert math.isnan(relative_tensor_error(approximate, exact))


@pytest.mark.parametrize("dtype", [None, torch.float32])
def test_relative_l2_zero_and_nonfinite_cases(dtype: torch.dtype | None) -> None:
    zeros = torch.zeros(3)
    assert relative_l2_error(zeros, zeros, dtype=dtype) == 0.0
    assert relative_l2_error(torch.ones(3), zeros, dtype=dtype) == float("inf")
    assert math.isnan(
        relative_l2_error(torch.full((3,), float("nan")), torch.ones(3), dtype=dtype)
    )


def test_relative_l2_rejects_broadcasting() -> None:
    with pytest.raises(ValueError, match="same shape"):
        relative_l2_error(torch.ones(1, 3), torch.ones(3))


@pytest.mark.parametrize("padded", [False, True])
def test_selection_mask_preserves_explicit_masks(padded: bool) -> None:
    indices = torch.tensor([[[0, 1, 2]]])
    mask = torch.tensor([[[True, True, False]]]) if padded else None
    selection = Selection(indices=indices, valid_mask=mask)
    actual = selection_mask(selection)
    if padded:
        assert actual is mask
    else:
        assert torch.equal(actual, torch.ones_like(indices, dtype=torch.bool))
