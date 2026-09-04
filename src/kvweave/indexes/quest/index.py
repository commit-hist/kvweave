"""Readable PyTorch implementation of Quest-style page retrieval.

This module independently implements the page min/max estimator described in
the Quest paper. It contains no code copied from the upstream Quest repository.
"""

from dataclasses import dataclass

import torch

from kvweave.core.types import Selection, validate_keys, validate_query
from kvweave.profiling import profile_component


def _validate_page_size(page_size: int) -> None:
    if not isinstance(page_size, int) or isinstance(page_size, bool):
        raise TypeError("page_size must be an integer")
    if page_size <= 0:
        raise ValueError("page_size must be positive")


def _validate_quest_budget(token_budget: int) -> None:
    if not isinstance(token_budget, int) or isinstance(token_budget, bool):
        raise TypeError("token_budget must be an integer")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")


@dataclass(frozen=True)
class QuestMetadata:
    """Per-page key extrema for canonical keys ``[B, Hkv, S, D]``.

    ``minimum`` and ``maximum`` have shape ``[B, Hkv, P, D]``. The original
    sequence length and page size reconstruct every page's valid token range.
    """

    minimum: torch.Tensor
    maximum: torch.Tensor
    page_size: int
    sequence_length: int

    def __post_init__(self) -> None:
        _validate_page_size(self.page_size)
        if not isinstance(self.sequence_length, int) or isinstance(
            self.sequence_length, bool
        ):
            raise TypeError("sequence_length must be an integer")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not isinstance(self.minimum, torch.Tensor) or not isinstance(
            self.maximum, torch.Tensor
        ):
            raise TypeError("minimum and maximum must be torch.Tensor instances")
        if self.minimum.ndim != 4:
            raise ValueError("Quest metadata must have shape [B, Hkv, P, D]")
        if self.minimum.shape != self.maximum.shape:
            raise ValueError("minimum and maximum metadata shapes must match")
        if any(size <= 0 for size in self.minimum.shape):
            raise ValueError("Quest metadata dimensions must be positive")
        if self.minimum.dtype != self.maximum.dtype:
            raise ValueError("minimum and maximum metadata dtypes must match")
        if self.minimum.device != self.maximum.device:
            raise ValueError("minimum and maximum metadata devices must match")
        if not torch.is_floating_point(self.minimum):
            raise TypeError("Quest metadata must use a floating-point dtype")
        expected_pages = (self.sequence_length + self.page_size - 1) // self.page_size
        if self.minimum.shape[2] != expected_pages:
            raise ValueError(
                "Quest metadata page count does not match sequence_length and page_size"
            )

    @property
    def num_pages(self) -> int:
        """Return the number of indexed pages."""
        return self.minimum.shape[2]

    @property
    def page_lengths(self) -> torch.Tensor:
        """Return valid token counts for pages as an integer tensor ``[P]``."""
        page_starts = (
            torch.arange(
                self.num_pages,
                device=self.minimum.device,
                dtype=torch.int64,
            )
            * self.page_size
        )
        return (self.sequence_length - page_starts).clamp(
            min=0,
            max=self.page_size,
        )


@dataclass(frozen=True)
class QuestSearchResult:
    """A token selection plus page-level budget and ranking details."""

    selection: Selection
    page_indices: torch.Tensor
    page_scores: torch.Tensor
    requested_token_budget: int
    num_pages_to_select: int

    @property
    def actual_token_counts(self) -> torch.Tensor:
        """Return actual valid candidates selected for each ``[B, Hkv]``."""
        return self.selection.valid_token_counts


def build_page_metadata(keys: torch.Tensor, page_size: int) -> QuestMetadata:
    """Compute page extrema for keys shaped ``[B, Hkv, S, D]``.

    The final page is padded only for the reduction. Positive infinity cannot
    affect its minimum and negative infinity cannot affect its maximum.
    """
    validate_keys(keys)
    _validate_page_size(page_size)

    with profile_component("quest.metadata.page_reshape_padding"):
        batch_size, kv_heads, sequence_length, head_dim = keys.shape
        num_pages = (sequence_length + page_size - 1) // page_size
        padded_length = num_pages * page_size
        padding_length = padded_length - sequence_length

        if padding_length:
            padding_shape = (batch_size, kv_heads, padding_length, head_dim)
            minimum_input = torch.cat(
                [keys, keys.new_full(padding_shape, float("inf"))],
                dim=2,
            )
            maximum_input = torch.cat(
                [keys, keys.new_full(padding_shape, float("-inf"))],
                dim=2,
            )
        else:
            minimum_input = keys
            maximum_input = keys

        paged_shape = (batch_size, kv_heads, num_pages, page_size, head_dim)
        minimum_pages = minimum_input.reshape(paged_shape)
        maximum_pages = maximum_input.reshape(paged_shape)
    with profile_component("quest.metadata.page_minimum"):
        minimum = minimum_pages.amin(dim=3)
    with profile_component("quest.metadata.page_maximum"):
        maximum = maximum_pages.amax(dim=3)
    with profile_component("quest.metadata.object_construction"):
        metadata = QuestMetadata(
            minimum=minimum,
            maximum=maximum,
            page_size=page_size,
            sequence_length=sequence_length,
        )
    return metadata


def append_page_metadata(
    metadata: QuestMetadata,
    new_keys: torch.Tensor,
) -> QuestMetadata:
    """Extend Quest extrema by exactly one causal key token.

    ``new_keys`` has canonical shape ``[B, Hkv, 1, D]``. If the current final
    page is partial, only its extrema are recomputed from the old extrema and
    the new token. If the final page is full, one new metadata page is
    appended. The returned metadata owns replacement tensors so references to
    the previous metadata remain unchanged.
    """
    with profile_component("quest.metadata.incremental.identify_page"):
        if not isinstance(metadata, QuestMetadata):
            raise TypeError("metadata must be a QuestMetadata instance")
        validate_keys(new_keys)
        expected_shape = (
            metadata.minimum.shape[0],
            metadata.minimum.shape[1],
            1,
            metadata.minimum.shape[3],
        )
        if new_keys.shape != expected_shape:
            raise ValueError(
                f"new_keys must have shape {expected_shape} for one causal append"
            )
        if new_keys.dtype != metadata.minimum.dtype:
            raise ValueError("new_keys and metadata must have the same dtype")
        if new_keys.device != metadata.minimum.device:
            raise ValueError("new_keys and metadata must be on the same device")
        opens_new_page = metadata.sequence_length % metadata.page_size == 0

    if opens_new_page:
        with profile_component("quest.metadata.incremental.new_page_append"):
            minimum = torch.cat((metadata.minimum, new_keys), dim=2)
            maximum = torch.cat((metadata.maximum, new_keys), dim=2)
    else:
        new_token = new_keys[:, :, 0, :]
        with profile_component("quest.metadata.incremental.existing_page_minimum"):
            minimum = metadata.minimum.clone()
            minimum[:, :, -1, :] = torch.minimum(
                metadata.minimum[:, :, -1, :],
                new_token,
            )
        with profile_component("quest.metadata.incremental.existing_page_maximum"):
            maximum = metadata.maximum.clone()
            maximum[:, :, -1, :] = torch.maximum(
                metadata.maximum[:, :, -1, :],
                new_token,
            )

    with profile_component("quest.metadata.incremental.state_bookkeeping"):
        return QuestMetadata(
            minimum=minimum,
            maximum=maximum,
            page_size=metadata.page_size,
            sequence_length=metadata.sequence_length + 1,
        )


def score_pages(query: torch.Tensor, metadata: QuestMetadata) -> torch.Tensor:
    """Compute Quest upper-bound page scores with shape ``[B, Hkv, P]``.

    For every page dimension, the contribution is
    ``max(query * page_minimum, query * page_maximum)``. Contributions are
    summed over the final feature dimension ``D``.
    """
    validate_query(query, metadata.minimum)
    with profile_component("quest.search.query_expansion"):
        query_by_page = query.unsqueeze(2)
    with profile_component("quest.search.min_max_score"):
        minimum_contribution = query_by_page * metadata.minimum
        maximum_contribution = query_by_page * metadata.maximum
        maximum_contribution = torch.maximum(
            minimum_contribution,
            maximum_contribution,
        )
    with profile_component("quest.search.dimension_reduction"):
        scores = maximum_contribution.sum(dim=-1)
    return scores


def token_budget_to_pages(
    token_budget: int,
    *,
    page_size: int,
    num_pages: int,
) -> int:
    """Round a positive token budget up to pages and cap at available pages."""
    _validate_quest_budget(token_budget)
    _validate_page_size(page_size)
    if not isinstance(num_pages, int) or isinstance(num_pages, bool):
        raise TypeError("num_pages must be an integer")
    if num_pages <= 0:
        raise ValueError("num_pages must be positive")
    implied_pages = (token_budget + page_size - 1) // page_size
    return min(implied_pages, num_pages)


def expand_pages_to_tokens(
    page_indices: torch.Tensor,
    metadata: QuestMetadata,
    *,
    page_scores: torch.Tensor | None = None,
) -> Selection:
    """Expand ranked page IDs to valid token IDs.

    Page rank is preserved, and tokens within a page use increasing sequence
    order. If actual counts differ across batch/head because only some rankings
    contain the partial final page, zero-valued rectangular placeholders are
    marked false in ``Selection.valid_mask`` and are not selected tokens. When
    page scores are supplied, each valid token receives its page's score.
    """
    if not isinstance(page_indices, torch.Tensor):
        raise TypeError("page_indices must be a torch.Tensor")
    if page_indices.ndim != 3:
        raise ValueError("page_indices must have shape [B, Hkv, page_count]")
    if page_indices.dtype != torch.int64:
        raise TypeError("page_indices must use torch.int64")
    if page_indices.shape[:2] != metadata.minimum.shape[:2]:
        raise ValueError(
            "page_indices batch and KV-head dimensions must match metadata"
        )
    if page_indices.shape[-1] <= 0:
        raise ValueError("at least one page must be selected")
    if page_indices.device != metadata.minimum.device:
        raise ValueError("page_indices and metadata must be on the same device")
    if torch.any((page_indices < 0) | (page_indices >= metadata.num_pages)).item():
        raise IndexError("page index exceeds the indexed page range")

    if page_scores is not None:
        if not isinstance(page_scores, torch.Tensor):
            raise TypeError("page_scores must be a torch.Tensor or None")
        if page_scores.shape != page_indices.shape:
            raise ValueError("page_scores must match page_indices shape")
        if not torch.is_floating_point(page_scores):
            raise TypeError("page_scores must use a floating-point dtype")
        if page_scores.device != page_indices.device:
            raise ValueError("page_scores and page_indices must share a device")

    with profile_component("quest.expand.page_expansion"):
        offsets = torch.arange(
            metadata.page_size,
            dtype=torch.int64,
            device=page_indices.device,
        )
        expanded = page_indices.unsqueeze(-1) * metadata.page_size + offsets
    with profile_component("quest.expand.partial_page_handling"):
        valid_mask = expanded < metadata.sequence_length
        safe_indices = torch.where(valid_mask, expanded, torch.zeros_like(expanded))

    with profile_component("quest.expand.validity_mask_handling"):
        flattened_indices = safe_indices.flatten(start_dim=2)
        flattened_mask = valid_mask.flatten(start_dim=2)
        # Stable compaction keeps page-rank and in-page order while moving masked
        # tail-page slots behind every real token.
        compact_order = torch.argsort(
            flattened_mask.to(torch.int8),
            dim=-1,
            descending=True,
            stable=True,
        )
        compacted_indices = torch.gather(
            flattened_indices,
            dim=-1,
            index=compact_order,
        )
        compacted_mask = torch.gather(flattened_mask, dim=-1, index=compact_order)
        max_valid_tokens = int(compacted_mask.sum(dim=-1).max().item())
        compacted_indices = compacted_indices[..., :max_valid_tokens]
        compacted_mask = compacted_mask[..., :max_valid_tokens]

    token_scores: torch.Tensor | None = None
    if page_scores is not None:
        with profile_component("quest.expand.page_score_expansion"):
            expanded_scores = page_scores.unsqueeze(-1).expand_as(expanded)
            flattened_scores = expanded_scores.flatten(start_dim=2)
            token_scores = torch.gather(
                flattened_scores,
                dim=-1,
                index=compact_order,
            )
            token_scores = token_scores[..., :max_valid_tokens]
            token_scores = torch.where(
                compacted_mask,
                token_scores,
                torch.zeros_like(token_scores),
            )

    return Selection(
        indices=compacted_indices,
        scores=token_scores,
        valid_mask=compacted_mask,
    )


class QuestIndex:
    """Select pages with the Quest min/max upper-bound estimator.

    Equal page scores are resolved by ascending page ID. Selected pages are
    returned in descending score order, followed by tokens in ascending order
    within each page. This deterministic policy is local to KVWeave and is not a
    claim about upstream tie behavior.
    """

    def __init__(self, page_size: int) -> None:
        _validate_page_size(page_size)
        self.page_size = page_size
        self._metadata: QuestMetadata | None = None

    @property
    def metadata(self) -> QuestMetadata:
        """Return built metadata, or fail if ``build`` has not run."""
        if self._metadata is None:
            raise RuntimeError("build must be called before accessing metadata")
        return self._metadata

    def build(self, keys: torch.Tensor) -> None:
        """Build page min/max metadata from canonical keys ``[B, Hkv, S, D]``."""
        with torch.no_grad():
            self._metadata = build_page_metadata(keys, self.page_size)

    def append(self, new_keys: torch.Tensor) -> None:
        """Update page metadata for one appended causal key token."""
        with torch.no_grad():
            self._metadata = append_page_metadata(self.metadata, new_keys)

    def search(self, query: torch.Tensor, budget: int) -> Selection:
        """Return page-expanded token candidates for a requested token budget."""
        return self.search_with_details(query, budget).selection

    def search_with_details(
        self,
        query: torch.Tensor,
        budget: int,
    ) -> QuestSearchResult:
        """Return token candidates and explicit page/budget accounting."""
        metadata = self.metadata
        validate_query(query, metadata.minimum)
        pages_to_select = token_budget_to_pages(
            budget,
            page_size=metadata.page_size,
            num_pages=metadata.num_pages,
        )

        with torch.no_grad():
            all_page_scores = score_pages(query, metadata)
            with profile_component("quest.search.page_ranking"):
                ranked_page_indices = torch.argsort(
                    all_page_scores,
                    dim=-1,
                    descending=True,
                    stable=True,
                )
            with profile_component("quest.search.page_id_handling"):
                page_indices = ranked_page_indices[..., :pages_to_select]
                selected_page_scores = torch.gather(
                    all_page_scores,
                    dim=-1,
                    index=page_indices,
                )
            selection = expand_pages_to_tokens(
                page_indices,
                metadata,
                page_scores=selected_page_scores,
            )

        return QuestSearchResult(
            selection=selection,
            page_indices=page_indices,
            page_scores=selected_page_scores,
            requested_token_budget=budget,
            num_pages_to_select=pages_to_select,
        )
