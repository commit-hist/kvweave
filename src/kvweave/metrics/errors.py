"""Relative-error metrics with explicit arithmetic precision."""

import torch


def relative_l2_error(
    approximate: torch.Tensor,
    exact: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> float:
    """Return scalar L2 relative error over equally shaped tensors.

    ``dtype=None`` preserves input-dtype arithmetic; decode controls explicitly
    request float32. A zero reference has error zero for an exact zero match,
    otherwise infinity. This helper does not filter nonfinite observations.
    """
    if approximate.shape != exact.shape:
        raise ValueError("compared tensors must have the same shape")
    if dtype is not None:
        approximate = approximate.to(dtype=dtype)
        exact = exact.to(dtype=dtype)
    numerator = torch.linalg.vector_norm(approximate - exact)
    denominator = torch.linalg.vector_norm(exact)
    if denominator.item() == 0:
        return 0.0 if numerator.item() == 0 else float("inf")
    return float((numerator / denominator).item())
