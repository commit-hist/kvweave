"""Minimal structural interfaces for retrieval and storage strategies."""

from typing import Protocol

import torch

from kvdb.core.types import RetrievedKV, Selection


class KVIndex(Protocol):
    """Choose which cached tokens should participate in attention."""

    def build(self, keys: torch.Tensor) -> None:
        """Build index state from canonical keys ``[B, Hkv, S, D]``."""
        ...

    def search(self, query: torch.Tensor, budget: int) -> Selection:
        """Select tokens for decode queries ``[B, Hkv, D]``."""
        ...


class KVStorage(Protocol):
    """Store and fetch canonical key/value tensors."""

    def put(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Store canonical tensors ``[B, Hkv, S, D]``."""
        ...

    def fetch(self, selection: Selection) -> RetrievedKV:
        """Fetch selected tensors and preserve rectangular validity."""
        ...
