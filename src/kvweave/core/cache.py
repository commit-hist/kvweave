"""High-level coordination between a KV index and storage."""

import torch

from kvweave.core.interfaces import KVIndex, KVStorage
from kvweave.core.types import RetrievedKV, validate_kv_tensors


class KVCache:
    """Coordinate index construction, selection, and KV fetching."""

    def __init__(self, index: KVIndex, storage: KVStorage) -> None:
        self.index = index
        self.storage = storage

    def build(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Build the index and store canonical KV tensors."""
        validate_kv_tensors(keys, values)
        self.index.build(keys)
        self.storage.put(keys, values)

    def retrieve(
        self,
        query: torch.Tensor,
        budget: int,
    ) -> RetrievedKV:
        """Select and fetch mask-preserving KV for one query per KV head."""
        selection = self.index.search(query, budget)
        return self.storage.fetch(selection)
