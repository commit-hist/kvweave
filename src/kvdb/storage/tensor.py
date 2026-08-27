"""Minimal in-memory tensor storage for reference experiments."""

import torch

from kvdb.core.types import RetrievedKV, Selection, validate_kv_tensors
from kvdb.profiling import profile_component


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

    def fetch(self, selection: Selection) -> RetrievedKV:
        """Gather selected tokens and preserve rectangular validity."""
        if self._keys is None or self._values is None:
            raise RuntimeError("put must be called before fetch")

        with profile_component("storage.fetch.index_preparation"):
            indices = selection.indices
            expected_prefix = self._keys.shape[:2]
            if indices.shape[:2] != expected_prefix:
                raise ValueError(
                    "selection batch and KV-head dimensions must match stored tensors"
                )
            if indices.device != self._keys.device:
                raise ValueError(
                    "selection indices and stored tensors must share a device"
                )
            if selection.valid_mask is None:
                safe_indices = indices
            else:
                safe_indices = torch.where(
                    selection.valid_mask,
                    indices,
                    torch.zeros_like(indices),
                )
            if torch.any(safe_indices >= self._keys.shape[2]).item():
                raise IndexError("selection index exceeds the stored sequence length")

            gather_indices = safe_indices.unsqueeze(-1).expand(
                *safe_indices.shape,
                self._keys.shape[-1],
            )
        with profile_component("storage.fetch.key_gather"):
            selected_keys = torch.gather(self._keys, dim=2, index=gather_indices)
        with profile_component("storage.fetch.value_gather"):
            selected_values = torch.gather(self._values, dim=2, index=gather_indices)
        with profile_component("storage.fetch.result_construction"):
            valid_mask = selection.valid_mask
            if valid_mask is not None and torch.all(valid_mask).item():
                valid_mask = None
            retrieved = RetrievedKV(
                keys=selected_keys,
                values=selected_values,
                valid_mask=valid_mask,
            )
        return retrieved
