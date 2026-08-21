"""Minimal in-memory tensor storage for reference experiments."""

import torch

from kvdb.core.types import Selection, validate_kv_tensors


class TensorStorage:
    """Store canonical KV tensors by reference on their existing device."""

    def __init__(self) -> None:
        self._keys: torch.Tensor | None = None
        self._values: torch.Tensor | None = None

    def put(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Store tensors without copying or moving them."""
        validate_kv_tensors(keys, values)
        self._keys = keys
        self._values = values

    def fetch(self, selection: Selection) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather selected tokens along the canonical sequence dimension."""
        if self._keys is None or self._values is None:
            raise RuntimeError("put must be called before fetch")

        indices = selection.indices
        expected_prefix = self._keys.shape[:2]
        if indices.shape[:2] != expected_prefix:
            raise ValueError(
                "selection batch and KV-head dimensions must match stored tensors"
            )
        if indices.device != self._keys.device:
            raise ValueError("selection indices and stored tensors must share a device")
        if torch.any(indices >= self._keys.shape[2]).item():
            raise IndexError("selection index exceeds the stored sequence length")

        gather_indices = indices.unsqueeze(-1).expand(
            *indices.shape,
            self._keys.shape[-1],
        )
        selected_keys = torch.gather(self._keys, dim=2, index=gather_indices)
        selected_values = torch.gather(self._values, dim=2, index=gather_indices)
        return selected_keys, selected_values
