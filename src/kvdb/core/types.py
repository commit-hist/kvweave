"""Canonical tensor validation and shared retrieval types."""

from dataclasses import dataclass

import torch


def validate_keys(keys: torch.Tensor) -> None:
    """Validate canonical keys with shape ``[B, Hkv, S, D]``."""
    if not isinstance(keys, torch.Tensor):
        raise TypeError("keys must be a torch.Tensor")
    if keys.ndim != 4:
        raise ValueError("keys must have shape [B, Hkv, S, D]")
    if any(size <= 0 for size in keys.shape):
        raise ValueError("keys dimensions B, Hkv, S, and D must be positive")
    if not torch.is_floating_point(keys):
        raise TypeError("keys must use a floating-point dtype")


def validate_kv_tensors(keys: torch.Tensor, values: torch.Tensor) -> None:
    """Validate matching canonical key/value tensors ``[B, Hkv, S, D]``."""
    validate_keys(keys)
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 4:
        raise ValueError("values must have shape [B, Hkv, S, D]")
    if values.shape != keys.shape:
        raise ValueError("keys and values must have identical shapes")
    if values.dtype != keys.dtype:
        raise ValueError("keys and values must have the same dtype")
    if values.device != keys.device:
        raise ValueError("keys and values must be on the same device")


def validate_query(query: torch.Tensor, keys: torch.Tensor) -> None:
    """Validate one decode query per batch item and KV head: ``[B, Hkv, D]``."""
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if query.ndim != 3:
        raise ValueError("query must have shape [B, Hkv, D]")
    expected_shape = (keys.shape[0], keys.shape[1], keys.shape[3])
    if query.shape != expected_shape:
        raise ValueError(
            f"query must have shape {expected_shape} to match the indexed keys"
        )
    if query.dtype != keys.dtype:
        raise ValueError("query and keys must have the same dtype")
    if query.device != keys.device:
        raise ValueError("query and keys must be on the same device")


def validate_budget(budget: int, sequence_length: int) -> None:
    """Validate an exact token budget for a sequence of known length."""
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise TypeError("budget must be an integer")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > sequence_length:
        raise ValueError("budget cannot exceed the indexed sequence length")


@dataclass(frozen=True)
class Selection:
    """Per-batch, per-head token selection.

    ``indices`` and optional ``scores`` have shape ``[B, Hkv, K]``. Indices
    address the sequence dimension of canonical KV tensors.
    """

    indices: torch.Tensor
    scores: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.indices, torch.Tensor):
            raise TypeError("selection indices must be a torch.Tensor")
        if self.indices.ndim != 3:
            raise ValueError("selection indices must have shape [B, Hkv, K]")
        if self.indices.dtype != torch.int64:
            raise TypeError("selection indices must use torch.int64")
        if any(size <= 0 for size in self.indices.shape):
            raise ValueError("selection dimensions B, Hkv, and K must be positive")
        if torch.any(self.indices < 0).item():
            raise ValueError("selection indices must be non-negative")

        if self.scores is None:
            return
        if not isinstance(self.scores, torch.Tensor):
            raise TypeError("selection scores must be a torch.Tensor or None")
        if self.scores.shape != self.indices.shape:
            raise ValueError("selection scores must match the indices shape")
        if not torch.is_floating_point(self.scores):
            raise TypeError("selection scores must use a floating-point dtype")
        if self.scores.device != self.indices.device:
            raise ValueError("selection scores and indices must be on the same device")
