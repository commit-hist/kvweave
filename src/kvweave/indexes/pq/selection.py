"""Experimental deterministic PQ candidate selection and full-ranking oracle.

These controls are strategy-internal and are not exported from ``kvweave``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum

import torch

from kvweave.core.types import validate_budget


class PQRankingMode(str, Enum):
    FULL_SORT = "full_stable_sort_oracle"
    PARTIAL = "deterministic_partial_selection"


# The Phase 5B matrix did not improve integrated 50% retrieval. Keep the
# accepted full-sort path as the default; partial selection remains opt-in.
_ranking_mode: ContextVar[PQRankingMode] = ContextVar(
    "pq_ranking_mode", default=PQRankingMode.FULL_SORT
)


@contextmanager
def pq_ranking_mode(mode: PQRankingMode) -> Iterator[None]:
    """Force a ranking implementation within an experimental test/replay."""
    if not isinstance(mode, PQRankingMode):
        raise TypeError("mode must be a PQRankingMode")
    token = _ranking_mode.set(mode)
    try:
        yield
    finally:
        _ranking_mode.reset(token)


def full_stable_ranking(scores: torch.Tensor) -> torch.Tensor:
    """Unchanged Phase 2–4 full ranking; ties retain ascending token IDs."""
    return torch.argsort(
        scores,
        dim=-1,
        descending=True,
        stable=True,
    )


def deterministic_partial_selection(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Return exactly ranked token IDs ``[B, H, K]`` from scores ``[B, H, S]``.

    Top-K supplies only the threshold. Membership is rebuilt in token-ID order,
    including the first required threshold ties, then only K scores are sorted
    stably. Neither topk's arbitrary tied indices nor altered scores are used.
    Full-budget and NaN inputs retain the complete stable-ranking behavior.
    """
    sequence_length = scores.shape[-1]
    validate_budget(budget, sequence_length)
    if budget == sequence_length or torch.isnan(scores).any():
        return full_stable_ranking(scores)[..., :budget]

    threshold = torch.topk(scores, budget, dim=-1, sorted=False).values.amin(
        dim=-1, keepdim=True
    )
    above = scores > threshold
    tied = scores == threshold
    remaining = budget - above.sum(dim=-1, keepdim=True)
    selected = above | (tied & (tied.cumsum(dim=-1) <= remaining))
    # nonzero emits row-major coordinates. Each row has exactly K candidates,
    # already ordered by token ID, which is the stable sort's secondary key.
    coordinates = selected.reshape(-1, sequence_length).nonzero()
    indices = coordinates[:, 1].reshape(*scores.shape[:-1], budget)
    candidate_scores = torch.gather(scores, dim=-1, index=indices)
    order = torch.argsort(candidate_scores, dim=-1, descending=True, stable=True)
    return torch.gather(indices, dim=-1, index=order)


def rank_pq_candidates(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Dispatch internally; callers retain the existing slice/gather policy."""
    if _ranking_mode.get() is PQRankingMode.FULL_SORT:
        return full_stable_ranking(scores)
    return deterministic_partial_selection(scores, budget)
