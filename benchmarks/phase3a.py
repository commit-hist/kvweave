"""Deterministic inputs and diagnostics for Phase 3A replication.

This module is benchmark support code, not part of KVDB's public package API.
It keeps locally authored fixtures and diagnostic definitions independently
testable without importing Transformers or downloading a model.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch

from kvdb.core.types import Selection
from kvdb.indexes.quest import QuestMetadata, score_pages


@dataclass(frozen=True)
class TextFixture:
    """One locally authored deterministic activation input."""

    fixture_id: str
    structure: str
    text: str


TEXT_FIXTURES = (
    TextFixture(
        fixture_id="repetitive_prose",
        structure=(
            "Short repetitive prose with recurring nouns, verbs, and clause order."
        ),
        text=(
            "The amber signal crossed the quiet station. The amber signal crossed "
            "the quiet station again. A patient observer recorded the amber signal, "
            "and the quiet station returned to its steady rhythm. "
        ),
    ),
    TextFixture(
        fixture_id="narrative_prose",
        structure=(
            "Chronological narrative with named characters, places, and changing events."
        ),
        text=(
            "Mara left the harbor before sunrise with a brass compass in her coat. "
            "At the ridge she met Ivo, who carried a torn map and news of a blocked "
            "bridge. They followed the river north, repaired a lantern at dusk, and "
            "reached the observatory after the first winter stars appeared. "
        ),
    ),
    TextFixture(
        fixture_id="technical_exposition",
        structure=(
            "Technical exposition with definitions, causal claims, and numeric terms."
        ),
        text=(
            "A cache index maps each query vector to a bounded candidate set. The "
            "storage layer then gathers full precision keys and values by token index. "
            "For a causal prefix of length S, attention normalizes scaled dot products "
            "over positions zero through t, where t is strictly less than S. "
        ),
    ),
    TextFixture(
        fixture_id="code_like",
        structure=(
            "Python-like functions with indentation, identifiers, branches, and literals."
        ),
        text=(
            "def select_pages(query, bounds, budget):\n"
            "    scores = bounds.upper(query)\n"
            "    page_count = max(1, budget // 16)\n"
            "    if page_count >= len(scores):\n"
            "        return list(range(len(scores)))\n"
            "    return stable_topk(scores, page_count)\n"
        ),
    ),
    TextFixture(
        fixture_id="list_table",
        structure=(
            "Repeated list and table-like rows with labels, delimiters, and quantities."
        ),
        text=(
            "Inventory:\n"
            "- alpha | sector A | count 03 | status ready\n"
            "- beta  | sector C | count 11 | status hold\n"
            "- gamma | sector B | count 07 | status ready\n"
            "Summary: ready=10; hold=11; sectors=A,B,C.\n"
        ),
    ),
    TextFixture(
        fixture_id="dialogue_qa",
        structure=(
            "Alternating dialogue and question-answer turns with explicit speaker roles."
        ),
        text=(
            "Analyst: Which page contains the highest exact token score?\n"
            "Engineer: Measure every token before judging the bound.\n"
            "Analyst: Does a larger budget recover the missing mass?\n"
            "Engineer: Test the causal prefix and report each head separately.\n"
        ),
    ),
    TextFixture(
        fixture_id="mixed_sentence_lengths",
        structure=(
            "Alternating very short statements and long multi-clause sentences."
        ),
        text=(
            "Stop. Observe. The first measurement looked ordinary, but after the query "
            "moved deeper into the sequence, several heads concentrated nearly all of "
            "their probability on a few distant positions that the earlier summary had "
            "treated as unremarkable. Continue. Record every detail before concluding. "
        ),
    ),
    TextFixture(
        fixture_id="symbolic_pattern",
        structure=(
            "Highly repetitive symbolic fields with cyclic markers and sparse changes."
        ),
        text=(
            "A0::A1::A2::A3 | 00110011 | [x][x][y][x] | "
            "B0::B1::B2::B3 | 11001100 | [y][x][x][x] | "
            "A0::A1::A2::A3 | 00110011 | [x][x][y][x] | "
        ),
    ),
)


@dataclass(frozen=True)
class TokenizedFixture:
    """Exact-length token IDs and deterministic construction metadata."""

    input_ids: torch.Tensor
    base_token_count: int
    repetitions: int
    token_ids_sha256: str


def build_deterministic_fixture(
    tokenizer: Any,
    fixture: TextFixture,
    sequence_length: int,
) -> TokenizedFixture:
    """Tokenize locally authored text, repeat it, and truncate to exact length."""
    if not isinstance(fixture, TextFixture):
        raise TypeError("fixture must be a TextFixture")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer")
    encoded = tokenizer(fixture.text, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer must return a mapping containing input_ids")
    base_ids = list(encoded["input_ids"])
    if not base_ids:
        raise RuntimeError(f"fixture {fixture.fixture_id!r} produced no tokenizer IDs")
    if any(not isinstance(token_id, int) for token_id in base_ids):
        raise TypeError("tokenizer input_ids must contain integers")
    repetitions = math.ceil(sequence_length / len(base_ids))
    token_ids = (base_ids * repetitions)[:sequence_length]
    input_ids = torch.tensor([token_ids], dtype=torch.int64)
    digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    return TokenizedFixture(
        input_ids=input_ids,
        base_token_count=len(base_ids),
        repetitions=repetitions,
        token_ids_sha256=digest,
    )


@dataclass(frozen=True)
class QueryPosition:
    """A named query position and its inclusive causal-prefix length."""

    label: str
    requested_fraction: float
    token_index: int
    causal_token_count: int


def calculate_query_positions(sequence_length: int) -> tuple[QueryPosition, ...]:
    """Map 25%/50%/75%/final to deterministic, valid zero-based indices.

    Fractional positions use ``ceil(sequence_length * fraction) - 1`` so the
    inclusive causal prefix is the smallest integer prefix covering the
    requested fraction. The final position is always ``sequence_length - 1``.
    """
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer")
    definitions = (
        ("25_percent", 0.25),
        ("50_percent", 0.50),
        ("75_percent", 0.75),
        ("final", 1.00),
    )
    return tuple(
        QueryPosition(
            label=label,
            requested_fraction=fraction,
            token_index=math.ceil(sequence_length * fraction) - 1,
            causal_token_count=math.ceil(sequence_length * fraction),
        )
        for label, fraction in definitions
    )


def canonicalize_selection_for_attention(selection: Selection) -> Selection:
    """Sort valid selected token IDs into causal order before attention.

    Candidate rankings remain untouched for retrieval diagnostics. Attention is
    mathematically permutation invariant, but float32 softmax/value reductions
    can otherwise cross the established full-budget equivalence tolerance when
    the same complete token set arrives in strategy-ranked order.
    """
    valid_mask = selection.valid_mask
    if valid_mask is None:
        order = torch.argsort(selection.indices, dim=-1, stable=True)
    else:
        sentinel = torch.full_like(selection.indices, torch.iinfo(torch.int64).max)
        sort_keys = torch.where(valid_mask, selection.indices, sentinel)
        order = torch.argsort(sort_keys, dim=-1, stable=True)
    indices = torch.gather(selection.indices, dim=-1, index=order)
    scores = (
        None
        if selection.scores is None
        else torch.gather(selection.scores, dim=-1, index=order)
    )
    canonical_mask = (
        None if valid_mask is None else torch.gather(valid_mask, dim=-1, index=order)
    )
    return Selection(indices=indices, scores=scores, valid_mask=canonical_mask)


def _validate_attention_weights(weights: torch.Tensor) -> None:
    if not isinstance(weights, torch.Tensor):
        raise TypeError("attention weights must be a torch.Tensor")
    if weights.ndim != 3 or any(size <= 0 for size in weights.shape):
        raise ValueError("attention weights must have shape [B, H, S]")
    if not torch.is_floating_point(weights):
        raise TypeError("attention weights must use a floating-point dtype")
    if not torch.isfinite(weights).all().item() or torch.any(weights < 0).item():
        raise ValueError("attention weights must be finite and non-negative")


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Return natural-log Shannon entropy in nats for every ``[B, H]``."""
    _validate_attention_weights(weights)
    positive_terms = torch.where(
        weights > 0,
        weights * torch.log(weights),
        torch.zeros_like(weights),
    )
    return -positive_terms.sum(dim=-1)


def normalized_attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Return entropy divided by ``ln(S)``; a one-token distribution is zero."""
    entropy = attention_entropy(weights)
    if weights.shape[-1] == 1:
        return torch.zeros_like(entropy)
    return entropy / math.log(weights.shape[-1])


def top_attention_mass(weights: torch.Tensor, count: int) -> torch.Tensor:
    """Return mass carried by the largest ``min(count, S)`` probabilities."""
    _validate_attention_weights(weights)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    return torch.topk(weights, k=min(count, weights.shape[-1]), dim=-1).values.sum(
        dim=-1
    )


def effective_attention_support(weights: torch.Tensor) -> torch.Tensor:
    """Return ``exp(H)`` in effective-token units using natural-log entropy."""
    return torch.exp(attention_entropy(weights))


def attention_distribution_metrics(
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute all Phase 3A exact-attention sparsity diagnostics."""
    entropy = attention_entropy(weights)
    normalized = (
        torch.zeros_like(entropy)
        if weights.shape[-1] == 1
        else entropy / math.log(weights.shape[-1])
    )
    return {
        "attention_entropy_nats": entropy,
        "normalized_attention_entropy": normalized,
        "top_1_attention_mass": top_attention_mass(weights, 1),
        "top_4_attention_mass": top_attention_mass(weights, 4),
        "top_16_attention_mass": top_attention_mass(weights, 16),
        "effective_attention_support_tokens": torch.exp(entropy),
    }


@dataclass(frozen=True)
class QuestBoundQuality:
    """Quest page upper bounds, exact page maxima, and their difference."""

    upper_bound: torch.Tensor
    true_page_maximum: torch.Tensor
    looseness: torch.Tensor


def quest_bound_quality(
    query: torch.Tensor,
    keys: torch.Tensor,
    metadata: QuestMetadata,
) -> QuestBoundQuality:
    """Compare Quest page scores with exact maximum raw QK score per page."""
    if keys.ndim != 4 or query.ndim != 3:
        raise ValueError("query and keys must have shape [B, H, D] and [B, H, S, D]")
    if query.shape != (keys.shape[0], keys.shape[1], keys.shape[3]):
        raise ValueError("query must match key batch, head, and feature dimensions")
    if metadata.sequence_length != keys.shape[2]:
        raise ValueError("Quest metadata sequence length must match keys")
    upper_bound = score_pages(query, metadata)
    exact_scores = torch.einsum("bhd,bhsd->bhs", query, keys)
    padded_length = metadata.num_pages * metadata.page_size
    if padded_length != keys.shape[2]:
        exact_scores = torch.nn.functional.pad(
            exact_scores,
            (0, padded_length - keys.shape[2]),
            value=float("-inf"),
        )
    true_page_maximum = exact_scores.reshape(
        keys.shape[0],
        keys.shape[1],
        metadata.num_pages,
        metadata.page_size,
    ).amax(dim=-1)
    return QuestBoundQuality(
        upper_bound=upper_bound,
        true_page_maximum=true_page_maximum,
        looseness=upper_bound - true_page_maximum,
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.sum(dim=-1)
    sums = torch.where(mask, values, torch.zeros_like(values)).sum(dim=-1)
    missing = torch.full_like(sums, float("nan"))
    return torch.where(counts > 0, sums / counts.clamp_min(1), missing)


def _masked_maximum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, float("-inf")))
    maxima = masked.amax(dim=-1)
    return torch.where(
        mask.any(dim=-1),
        maxima,
        torch.full_like(maxima, float("nan")),
    )


def aggregate_quest_bound_looseness(
    looseness: torch.Tensor,
    selected_page_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Aggregate page-bound looseness over selected, non-selected, and all pages."""
    if looseness.ndim != 3 or selected_page_indices.ndim != 3:
        raise ValueError("looseness and page indices must have shape [B, H, P/K]")
    if selected_page_indices.shape[:2] != looseness.shape[:2]:
        raise ValueError("page indices must match looseness batch/head dimensions")
    if selected_page_indices.dtype != torch.int64:
        raise TypeError("selected page indices must use torch.int64")
    if selected_page_indices.device != looseness.device:
        raise ValueError("page indices and looseness must share a device")
    if torch.any(
        (selected_page_indices < 0) | (selected_page_indices >= looseness.shape[-1])
    ).item():
        raise IndexError("selected page index is outside the page range")
    selected = torch.zeros_like(looseness, dtype=torch.bool)
    selected.scatter_(dim=-1, index=selected_page_indices, value=True)
    nonselected = ~selected
    return {
        "quest_bound_looseness_all_mean": looseness.mean(dim=-1),
        "quest_bound_looseness_all_max": looseness.amax(dim=-1),
        "quest_bound_looseness_selected_mean": _masked_mean(looseness, selected),
        "quest_bound_looseness_selected_max": _masked_maximum(looseness, selected),
        "quest_bound_looseness_nonselected_mean": _masked_mean(looseness, nonselected),
        "quest_bound_looseness_nonselected_max": _masked_maximum(
            looseness, nonselected
        ),
    }


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return zero-based average ranks for a one-dimensional tensor with ties."""
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    sorted_ranks = torch.empty_like(sorted_values, dtype=torch.float64)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and bool(
            sorted_values[end] == sorted_values[start]
        ):
            end += 1
        sorted_ranks[start:end] = (start + end - 1) / 2.0
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _pearson_1d(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if denominator.item() == 0:
        return left.new_tensor(float("nan"), dtype=torch.float64)
    return (left_centered * right_centered).sum() / denominator


def pq_score_approximation_metrics(
    approximate_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    *,
    high_attention_count: int = 16,
) -> dict[str, torch.Tensor]:
    """Compare approximate and exact raw QK token scores per ``[B, H]``.

    Rank correlation is tie-aware Spearman correlation. High-attention error is
    MAE on the exact Top-N raw scores; positive attention scaling and softmax
    preserve that ordering.
    """
    if approximate_scores.shape != exact_scores.shape or exact_scores.ndim != 3:
        raise ValueError("score tensors must share shape [B, H, S]")
    if not torch.is_floating_point(approximate_scores) or not torch.is_floating_point(
        exact_scores
    ):
        raise TypeError("score tensors must use floating-point dtypes")
    if approximate_scores.device != exact_scores.device:
        raise ValueError("score tensors must share a device")
    if (
        not isinstance(high_attention_count, int)
        or isinstance(high_attention_count, bool)
        or high_attention_count <= 0
    ):
        raise ValueError("high_attention_count must be a positive integer")
    if (
        not torch.isfinite(approximate_scores).all().item()
        or not torch.isfinite(exact_scores).all().item()
    ):
        raise ValueError("score tensors must be finite")

    errors = approximate_scores - exact_scores
    absolute_errors = errors.abs()
    exact_top_token = exact_scores.argmax(dim=-1, keepdim=True)
    top_token_signed_error = torch.gather(
        errors, dim=-1, index=exact_top_token
    ).squeeze(-1)
    top_count = min(high_attention_count, exact_scores.shape[-1])
    high_attention_indices = torch.topk(
        exact_scores,
        k=top_count,
        dim=-1,
    ).indices
    high_attention_errors = torch.gather(
        absolute_errors,
        dim=-1,
        index=high_attention_indices,
    )

    rank_correlations = torch.empty(
        exact_scores.shape[:2],
        dtype=torch.float64,
        device=exact_scores.device,
    )
    for batch_index in range(exact_scores.shape[0]):
        for head_index in range(exact_scores.shape[1]):
            approximate_ranks = _average_ranks(
                approximate_scores[batch_index, head_index]
            )
            exact_ranks = _average_ranks(exact_scores[batch_index, head_index])
            rank_correlations[batch_index, head_index] = _pearson_1d(
                approximate_ranks,
                exact_ranks,
            )

    return {
        "pq_score_mae": absolute_errors.mean(dim=-1),
        "pq_score_rmse": torch.sqrt((errors * errors).mean(dim=-1)),
        "pq_score_spearman_rank_correlation": rank_correlations,
        "pq_exact_top_token_signed_score_error": top_token_signed_error,
        "pq_exact_top_token_absolute_score_error": top_token_signed_error.abs(),
        "pq_exact_top_16_score_mae": high_attention_errors.mean(dim=-1),
    }
