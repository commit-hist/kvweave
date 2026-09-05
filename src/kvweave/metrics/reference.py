"""Strategy-independent synthetic attention and retrieval correctness metrics."""

from dataclasses import dataclass
import math

import torch

from kvweave.core.types import (
    RetrievedKV,
    Selection,
    validate_kv_tensors,
    validate_query,
)
from kvweave.metrics.errors import relative_l2_error


def selection_mask(selection: Selection) -> torch.Tensor:
    """Return the explicit validity mask, or all-valid for an unpadded selection."""
    if selection.valid_mask is not None:
        return selection.valid_mask
    return torch.ones_like(selection.indices, dtype=torch.bool)


def candidate_recall(
    candidates: Selection,
    exact_topk: Selection,
) -> torch.Tensor:
    """Return exact Top-K containment recall for every ``[B, Hkv]``.

    The denominator is the number of valid tokens in ``exact_topk``. A token is
    recalled when it occurs anywhere in the valid page-expanded candidate set,
    even when Quest's actual candidate count exceeds the requested token budget.
    """
    if candidates.indices.shape[:2] != exact_topk.indices.shape[:2]:
        raise ValueError("candidate and exact selections must share B and Hkv")
    if candidates.indices.device != exact_topk.indices.device:
        raise ValueError("candidate and exact selections must share a device")

    candidate_mask = selection_mask(candidates)
    exact_mask = selection_mask(exact_topk)
    sentinel = torch.iinfo(torch.int64).max
    sortable_candidates = torch.where(
        candidate_mask,
        candidates.indices,
        sentinel,
    )
    sorted_candidates = sortable_candidates.sort(dim=-1).values
    candidate_positions = torch.searchsorted(
        sorted_candidates,
        exact_topk.indices,
    ).clamp(max=sorted_candidates.shape[-1] - 1)
    exact_token_found = (
        torch.gather(sorted_candidates, dim=-1, index=candidate_positions)
        == exact_topk.indices
    ) & exact_mask
    denominator = exact_mask.sum(dim=-1)
    if torch.any(denominator == 0).item():
        raise ValueError("exact selection must contain at least one valid token")
    return exact_token_found.sum(dim=-1).to(torch.float32) / denominator


def full_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute scaled dot-product attention over every KV token.

    Shapes are query ``[B, Hkv, D]`` and keys/values ``[B, Hkv, S, D]``.
    The output has shape ``[B, Hkv, D]``.
    """
    validate_kv_tensors(keys, values)
    validate_query(query, keys)
    scale = 1.0 / math.sqrt(keys.shape[-1])
    logits = torch.einsum("bhd,bhsd->bhs", query, keys) * scale
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("bhs,bhsd->bhd", weights, values)


def selected_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute attention over retrieved KV, excluding rectangular padding.

    Shapes are query ``[B, Hkv, D]``, keys/values ``[B, Hkv, K, D]``, and an
    optional validity mask ``[B, Hkv, K]``. A missing mask is the dense path in
    which every retrieved position is valid.
    """
    retrieved = RetrievedKV(keys=keys, values=values, valid_mask=valid_mask)
    validate_query(query, keys)
    scale = 1.0 / math.sqrt(keys.shape[-1])
    logits = torch.einsum("bhd,bhkd->bhk", query, retrieved.keys) * scale

    if retrieved.valid_mask is None:
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bhk,bhkd->bhd", weights, retrieved.values)

    logits = logits.masked_fill(~retrieved.valid_mask, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    valid_values = torch.where(
        retrieved.valid_mask.unsqueeze(-1),
        retrieved.values,
        torch.zeros_like(retrieved.values),
    )
    return torch.einsum("bhk,bhkd->bhd", weights, valid_values)


@dataclass(frozen=True)
class AttentionComparison:
    """Full and selected attention outputs with transparent summary metrics."""

    full_output: torch.Tensor
    selected_output: torch.Tensor
    relative_output_error: float
    selected_token_percentage: float


def compare_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    retrieved: RetrievedKV,
) -> AttentionComparison:
    """Compare full and selected synthetic attention without quality claims."""
    full_output = full_attention(query, keys, values)
    selected_output = selected_attention(
        query,
        retrieved.keys,
        retrieved.values,
        retrieved.valid_mask,
    )
    relative_error = relative_l2_error(selected_output, full_output)

    if retrieved.valid_mask is None:
        selected_tokens = (
            retrieved.keys.shape[0] * retrieved.keys.shape[1] * retrieved.keys.shape[2]
        )
    else:
        selected_tokens = retrieved.valid_mask.sum().item()
    total_tokens = keys.shape[0] * keys.shape[1] * keys.shape[2]
    return AttentionComparison(
        full_output=full_output,
        selected_output=selected_output,
        relative_output_error=relative_error,
        selected_token_percentage=100.0 * selected_tokens / total_tokens,
    )
