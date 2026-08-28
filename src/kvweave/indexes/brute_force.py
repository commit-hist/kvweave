"""Exact dot-product retrieval used as a correctness oracle."""

import torch

from kvweave.core.types import (
    Selection,
    validate_budget,
    validate_keys,
    validate_query,
)


class BruteForceIndex:
    """Retrieve exact Top-K keys independently for each batch item and KV head."""

    def __init__(self) -> None:
        self._keys: torch.Tensor | None = None

    def build(self, keys: torch.Tensor) -> None:
        """Retain canonical keys ``[B, Hkv, S, D]`` for exact search."""
        validate_keys(keys)
        self._keys = keys

    def search(self, query: torch.Tensor, budget: int) -> Selection:
        """Return exact raw-dot-product Top-K token indices and scores.

        Args:
            query: Decode queries with shape ``[B, Hkv, D]``.
            budget: Exact number of tokens ``K`` selected per batch/head.

        Returns:
            A selection whose indices and scores have shape ``[B, Hkv, K]``.
        """
        if self._keys is None:
            raise RuntimeError("build must be called before search")

        validate_query(query, self._keys)
        validate_budget(budget, self._keys.shape[2])

        with torch.no_grad():
            scores = torch.einsum("bhd,bhsd->bhs", query, self._keys)
            top_scores, top_indices = torch.topk(
                scores,
                k=budget,
                dim=-1,
                largest=True,
                sorted=True,
            )
        return Selection(indices=top_indices, scores=top_scores)
