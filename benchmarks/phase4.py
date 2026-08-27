"""Reusable analysis helpers for the fixed Phase 4 profiling experiment.

This module contains accounting and aggregation only.  It does not replace or
optimize any Quest, PQ, storage, or attention operation.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any

import torch

from kvdb.profiling import ComponentTiming, estimate_tensor_bytes, percentile


MODEL_COMPONENT_GROUPS: dict[str, tuple[str, ...]] = {
    "qkv_projection": ("model.qkv_projection",),
    "rope_query_key_preparation": ("model.rope_query_key_preparation",),
    "kv_cache_append": ("model.kv_cache_append",),
    "selected_attention": ("model.selected_attention",),
    "attention_output_projection": ("model.attention_output_projection",),
    "mlp": ("model.mlp",),
    "layer_norm_residual": ("model.layer_norm_residual",),
}

QUEST_COMPONENT_GROUPS: dict[str, tuple[str, ...]] = {
    "metadata_rebuild": (
        "quest.metadata.page_reshape_padding",
        "quest.metadata.page_minimum",
        "quest.metadata.page_maximum",
        "quest.metadata.object_construction",
    ),
    "query_page_scoring": (
        "quest.search.query_expansion",
        "quest.search.min_max_score",
        "quest.search.dimension_reduction",
    ),
    "page_ranking": (
        "quest.search.page_ranking",
        "quest.search.page_id_handling",
    ),
    "page_to_token_expansion": (
        "quest.expand.page_expansion",
        "quest.expand.partial_page_handling",
        "quest.expand.validity_mask_handling",
        "quest.expand.page_score_expansion",
    ),
    "newest_token_inclusion": ("policy.newest_token_inclusion",),
    "causal_reordering": ("policy.causal_reordering",),
    "storage_fetch_gather": (
        "storage.fetch.index_preparation",
        "storage.fetch.key_gather",
        "storage.fetch.value_gather",
        "storage.fetch.result_construction",
    ),
}

PQ_COMPONENT_GROUPS: dict[str, tuple[str, ...]] = {
    "frozen_append": (
        "pq.encode.subspace_split",
        "pq.encode.centroid_distance",
        "pq.encode.centroid_assignment",
        "pq.append.code_append",
    ),
    "lookup_table_construction": (
        "pq.search.query_split",
        "pq.search.query_centroid_dot_products",
    ),
    "approximate_score_reconstruction": (
        "pq.search.score_allocation",
        "pq.search.code_lookup",
        "pq.search.subspace_summation",
    ),
    "ranking_topk": (
        "pq.search.ranking",
        "pq.search.token_id_handling",
    ),
    "newest_token_inclusion": ("policy.newest_token_inclusion",),
    "causal_reordering": ("policy.causal_reordering",),
    "storage_fetch_gather": (
        "storage.fetch.index_preparation",
        "storage.fetch.key_gather",
        "storage.fetch.value_gather",
        "storage.fetch.result_construction",
    ),
}

QUEST_RETRIEVAL_CATEGORIES = (
    "metadata_rebuild",
    "query_page_scoring",
    "page_ranking",
    "page_to_token_expansion",
    "newest_token_inclusion",
    "causal_reordering",
    "storage_fetch_gather",
)
PQ_RETRIEVAL_CATEGORIES = (
    "frozen_append",
    "lookup_table_construction",
    "approximate_score_reconstruction",
    "ranking_topk",
    "newest_token_inclusion",
    "causal_reordering",
    "storage_fetch_gather",
)


def component_groups(strategy: str) -> dict[str, tuple[str, ...]]:
    """Return non-overlapping layer categories for one decode strategy."""
    groups = dict(MODEL_COMPONENT_GROUPS)
    if strategy == "quest":
        groups.update(QUEST_COMPONENT_GROUPS)
    elif strategy == "pq":
        groups.update(PQ_COMPONENT_GROUPS)
    elif strategy != "dense":
        raise ValueError(f"unsupported strategy {strategy!r}")
    return groups


def build_layer_step_records(
    component_records: Sequence[ComponentTiming],
    coarse_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join atomic timings to the runner's wall-clock layer observations."""
    atomic_by_key: dict[tuple[Any, ...], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for record in component_records:
        if record.context.get("phase") != "decode":
            continue
        key = (
            record.context.get("fixture_id"),
            record.context.get("strategy"),
            record.context.get("budget_fraction"),
            record.context.get("decode_step"),
            record.context.get("layer"),
        )
        atomic_by_key[key][record.component] += record.duration_ms

    results: list[dict[str, Any]] = []
    for coarse in coarse_records:
        key = (
            coarse["fixture_id"],
            coarse["strategy"],
            coarse["budget_fraction"],
            coarse["decode_step"],
            coarse["layer"],
        )
        atomic = dict(atomic_by_key.get(key, {}))
        groups = component_groups(str(coarse["strategy"]))
        categories = {
            category: sum(atomic.get(component, 0.0) for component in components)
            for category, components in groups.items()
        }
        measured_components_ms = sum(atomic.values())
        total_layer_time_ms = float(coarse["total_layer_time_ms"])
        categories["miscellaneous_layer_overhead"] = max(
            0.0,
            total_layer_time_ms - measured_components_ms,
        )
        results.append(
            {
                **coarse,
                "atomic_components_ms": atomic,
                "component_categories_ms": categories,
                "instrumented_component_sum_ms": measured_components_ms,
            }
        )
    return results


def build_step_records(
    layer_records: Sequence[Mapping[str, Any]],
    step_wall_times: Mapping[tuple[str, str, float, int], float],
) -> list[dict[str, Any]]:
    """Sum layer categories into complete 24-layer decode-step records."""
    grouped: dict[tuple[str, str, float, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in layer_records:
        key = (
            str(row["fixture_id"]),
            str(row["strategy"]),
            float(row["budget_fraction"]),
            int(row["decode_step"]),
        )
        grouped[key].append(row)

    results = []
    for key, layers in sorted(grouped.items()):
        categories: dict[str, float] = defaultdict(float)
        for layer in layers:
            for category, duration in layer["component_categories_ms"].items():
                categories[category] += float(duration)
        results.append(
            {
                "fixture_id": key[0],
                "strategy": key[1],
                "budget_fraction": key[2],
                "decode_step": key[3],
                "layer_count": len(layers),
                "component_categories_ms": dict(categories),
                "total_layer_time_ms": sum(
                    float(layer["total_layer_time_ms"]) for layer in layers
                ),
                "total_decode_step_time_ms": step_wall_times[key],
            }
        )
    return results


def summarize_step_components(
    step_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize per-step component totals across fixtures and positions."""
    grouped: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in step_records:
        for component, duration in row["component_categories_ms"].items():
            grouped[
                (str(row["strategy"]), float(row["budget_fraction"]), component)
            ].append(float(duration))
        grouped[
            (
                str(row["strategy"]),
                float(row["budget_fraction"]),
                "total_decode_step",
            )
        ].append(float(row["total_decode_step_time_ms"]))
        grouped[
            (
                str(row["strategy"]),
                float(row["budget_fraction"]),
                "total_layers",
            )
        ].append(float(row["total_layer_time_ms"]))

    results = []
    for (strategy, budget, component), values in sorted(grouped.items()):
        results.append(
            {
                "strategy": strategy,
                "budget_fraction": budget,
                "component": component,
                "call_count": len(values),
                "mean_ms_per_decode_step": statistics.fmean(values),
                "median_ms_per_decode_step": statistics.median(values),
                "p90_ms_per_decode_step": percentile(values, 0.90),
                "p95_ms_per_decode_step": percentile(values, 0.95),
            }
        )
    return results


def retrieval_overhead_summary(
    step_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report retrieval-only costs and per-component shares for Quest/PQ."""
    output = []
    for strategy, categories in (
        ("quest", QUEST_RETRIEVAL_CATEGORIES),
        ("pq", PQ_RETRIEVAL_CATEGORIES),
    ):
        budgets = sorted(
            {
                float(row["budget_fraction"])
                for row in step_records
                if row["strategy"] == strategy
            }
        )
        for budget in budgets:
            rows = [
                row
                for row in step_records
                if row["strategy"] == strategy
                and float(row["budget_fraction"]) == budget
            ]
            overheads = [
                sum(
                    float(row["component_categories_ms"].get(name, 0.0))
                    for name in categories
                )
                for row in rows
            ]
            for category in categories:
                durations = [
                    float(row["component_categories_ms"].get(category, 0.0))
                    for row in rows
                ]
                shares = [
                    0.0 if overhead == 0 else duration / overhead
                    for duration, overhead in zip(durations, overheads, strict=True)
                ]
                output.append(
                    {
                        "strategy": strategy,
                        "budget_fraction": budget,
                        "component": category,
                        "decode_step_count": len(rows),
                        "median_ms_per_decode_step": statistics.median(durations),
                        "p90_ms_per_decode_step": percentile(durations, 0.90),
                        "p95_ms_per_decode_step": percentile(durations, 0.95),
                        "median_share_of_retrieval_overhead": (
                            statistics.median(shares)
                        ),
                        "mean_share_of_retrieval_overhead": statistics.fmean(shares),
                    }
                )
            output.append(
                {
                    "strategy": strategy,
                    "budget_fraction": budget,
                    "component": "total_retrieval_overhead",
                    "decode_step_count": len(rows),
                    "median_ms_per_decode_step": statistics.median(overheads),
                    "p90_ms_per_decode_step": percentile(overheads, 0.90),
                    "p95_ms_per_decode_step": percentile(overheads, 0.95),
                    "median_share_of_retrieval_overhead": 1.0,
                    "mean_share_of_retrieval_overhead": 1.0,
                }
            )
    return output


def compare_budgets(
    step_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare matched 50% and 100% per-step component totals."""
    indexed = {
        (
            row["fixture_id"],
            row["strategy"],
            row["budget_fraction"],
            row["decode_step"],
        ): row
        for row in step_records
    }
    output = []
    for strategy in ("quest", "pq"):
        half_rows = [
            row
            for row in step_records
            if row["strategy"] == strategy and row["budget_fraction"] == 0.5
        ]
        categories = sorted(
            {
                component
                for row in half_rows
                for component in row["component_categories_ms"]
            }
        )
        for component in [*categories, "total_decode_step"]:
            half_values = []
            full_values = []
            for half in half_rows:
                full = indexed[
                    (
                        half["fixture_id"],
                        strategy,
                        1.0,
                        half["decode_step"],
                    )
                ]
                if component == "total_decode_step":
                    half_value = float(half["total_decode_step_time_ms"])
                    full_value = float(full["total_decode_step_time_ms"])
                else:
                    half_value = float(
                        half["component_categories_ms"].get(component, 0.0)
                    )
                    full_value = float(
                        full["component_categories_ms"].get(component, 0.0)
                    )
                half_values.append(half_value)
                full_values.append(full_value)
            half_median = statistics.median(half_values)
            full_median = statistics.median(full_values)
            output.append(
                {
                    "strategy": strategy,
                    "component": component,
                    "matched_step_count": len(half_values),
                    "budget_50_median_ms": half_median,
                    "budget_100_median_ms": full_median,
                    "absolute_change_ms": full_median - half_median,
                    "ratio_100_to_50": (
                        None if half_median == 0 else full_median / half_median
                    ),
                }
            )
    return output


def compare_dense_baseline(
    step_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Decompose matched approximate-minus-dense decode wall time."""
    indexed = {
        (
            row["fixture_id"],
            row["strategy"],
            row["budget_fraction"],
            row["decode_step"],
        ): row
        for row in step_records
    }
    model_categories = tuple(MODEL_COMPONENT_GROUPS)
    output = []
    for strategy, retrieval_categories in (
        ("quest", QUEST_RETRIEVAL_CATEGORIES),
        ("pq", PQ_RETRIEVAL_CATEGORIES),
    ):
        for budget in (0.5, 1.0):
            records: dict[str, list[float]] = defaultdict(list)
            approximate_rows = [
                row
                for row in step_records
                if row["strategy"] == strategy and row["budget_fraction"] == budget
            ]
            for approximate in approximate_rows:
                dense = indexed[
                    (
                        approximate["fixture_id"],
                        "dense",
                        1.0,
                        approximate["decode_step"],
                    )
                ]
                approximate_categories = approximate["component_categories_ms"]
                dense_categories = dense["component_categories_ms"]
                retrieval = sum(
                    float(approximate_categories.get(name, 0.0))
                    for name in retrieval_categories
                )
                attention_delta = float(
                    approximate_categories.get("selected_attention", 0.0)
                ) - float(dense_categories.get("selected_attention", 0.0))
                normal_delta = sum(
                    float(approximate_categories.get(name, 0.0))
                    - float(dense_categories.get(name, 0.0))
                    for name in model_categories
                    if name != "selected_attention"
                )
                normal_delta += float(
                    approximate_categories.get("miscellaneous_layer_overhead", 0.0)
                ) - float(dense_categories.get("miscellaneous_layer_overhead", 0.0))
                total_delta = float(approximate["total_decode_step_time_ms"]) - float(
                    dense["total_decode_step_time_ms"]
                )
                records["retrieval_overhead"].append(retrieval)
                records["selected_attention_delta"].append(attention_delta)
                records["comparable_model_compute_delta"].append(normal_delta)
                records["total_decode_delta"].append(total_delta)
                records["unattributed_step_delta"].append(
                    total_delta - retrieval - attention_delta - normal_delta
                )
            output.append(
                {
                    "strategy": strategy,
                    "budget_fraction": budget,
                    "matched_step_count": len(approximate_rows),
                    **{
                        f"median_{name}_ms": statistics.median(values)
                        for name, values in records.items()
                    },
                    "definition": (
                        "retrieval overhead is index update/search, policy, and "
                        "storage fetch; selected attention and comparable normal "
                        "model computation are reported separately"
                    ),
                }
            )
    return output


def rank_retrieval_bottlenecks(
    step_records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Rank measured retrieval components without choosing an implementation."""
    complexity = {
        "metadata_rebuild": "approximately O(H * S * D)",
        "query_page_scoring": "approximately O(H * ceil(S/P) * D)",
        "page_ranking": "approximately O(H * (S/P) log(S/P)) in this reference",
        "page_to_token_expansion": "approximately O(H * K)",
        "frozen_append": "O(H * C * D) assignment plus O(H * S * M) code append",
        "lookup_table_construction": "approximately O(H * C * D)",
        "approximate_score_reconstruction": "approximately O(H * S * M)",
        "ranking_topk": "approximately O(H * S log S) in this reference",
        "newest_token_inclusion": "approximately O(H * K) plus Python row loops",
        "causal_reordering": "approximately O(H * K log K) in this reference",
        "storage_fetch_gather": "approximately O(H * K * D)",
    }
    classification = {
        "metadata_rebuild": "reference-runtime policy overhead",
        "query_page_scoring": "algorithmic Quest work",
        "page_ranking": "algorithmic ranking expressed as reference full sort",
        "page_to_token_expansion": "reference representation/policy overhead",
        "frozen_append": "algorithmic assignment plus reference append/validation overhead",
        "lookup_table_construction": "algorithmic PQ work",
        "approximate_score_reconstruction": "algorithmic PQ work",
        "ranking_topk": "algorithmic selection expressed as reference stable full sort",
        "newest_token_inclusion": "integration-policy overhead",
        "causal_reordering": "integration-policy overhead",
        "storage_fetch_gather": "shared storage-boundary overhead",
    }
    semantic_risk = {
        "metadata_rebuild": "low if incremental extrema remain bit/selection equivalent",
        "query_page_scoring": "high unless page scores and tie behavior are preserved",
        "page_ranking": "medium; exact stable tie policy must be retained",
        "page_to_token_expansion": "low if IDs, order, and masks remain exact",
        "frozen_append": "low for storage/validation changes; high if assignments change",
        "lookup_table_construction": "high unless lookup values remain ranking-equivalent",
        "approximate_score_reconstruction": "high unless approximate scores/order remain equivalent",
        "ranking_topk": "medium; selected IDs and stable tie policy must remain exact",
        "newest_token_inclusion": "high unless replacement semantics remain exact",
        "causal_reordering": "high unless causal order and masks remain exact",
        "storage_fetch_gather": "low if fetched tensors and masks remain exact",
    }
    output = {}
    for strategy, categories in (
        ("quest", QUEST_RETRIEVAL_CATEGORIES),
        ("pq", PQ_RETRIEVAL_CATEGORIES),
    ):
        rows = [row for row in step_records if row["strategy"] == strategy]
        overheads = [
            sum(
                float(row["component_categories_ms"].get(category, 0.0))
                for category in categories
            )
            for row in rows
        ]
        ranked = []
        for category in categories:
            values = [
                float(row["component_categories_ms"].get(category, 0.0)) for row in rows
            ]
            shares = [
                0.0 if overhead == 0 else value / overhead
                for value, overhead in zip(values, overheads, strict=True)
            ]
            ranked.append(
                {
                    "component": category,
                    "budget_cells": [0.5, 1.0],
                    "decode_step_count": len(values),
                    "median_ms_per_decode_step": statistics.median(values),
                    "p95_ms_per_decode_step": percentile(values, 0.95),
                    "median_share_of_retrieval_overhead": statistics.median(shares),
                    "scaling": complexity[category],
                    "classification": classification[category],
                    "quality_semantic_risk": semantic_risk[category],
                }
            )
        ranked.sort(key=lambda row: row["median_ms_per_decode_step"], reverse=True)
        output[strategy] = ranked[:3]
    return output


def _allocation(
    name: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    *,
    frequency: str,
    lifetime: str,
    reusable_in_principle: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(shape),
        "dtype": str(dtype).removeprefix("torch."),
        "estimated_bytes": estimate_tensor_bytes(shape, dtype),
        "frequency": frequency,
        "lifetime": lifetime,
        "reusable_in_principle": reusable_in_principle,
        "notes": notes,
        "estimate_kind": "analytical logical tensor size; not allocator peak",
    }


def allocation_estimates(
    *,
    strategy: str,
    sequence_length: int,
    budget_fraction: float,
    batch_size: int,
    heads: int,
    head_dimension: int,
    selection_width: int,
    page_size: int = 64,
    pq_subspaces: int = 4,
    pq_centroids: int = 8,
) -> list[dict[str, Any]]:
    """Estimate major steady-state allocations for one layer/decode step."""
    previous_length = sequence_length - 1
    common = [
        _allocation(
            "causal_key_concat_output",
            (batch_size, heads, sequence_length, head_dimension),
            torch.float32,
            frequency="every layer/decode step",
            lifetime="persistent until next append",
            reusable_in_principle=False,
            notes=f"torch.cat replaces prior length {previous_length} key state",
        ),
        _allocation(
            "causal_value_concat_output",
            (batch_size, heads, sequence_length, head_dimension),
            torch.float32,
            frequency="every layer/decode step",
            lifetime="persistent until next append",
            reusable_in_principle=False,
            notes=f"torch.cat replaces prior length {previous_length} value state",
        ),
    ]
    if strategy == "dense":
        return common + [
            _allocation(
                "dense_attention_logits_or_weights",
                (batch_size, heads, sequence_length),
                torch.float32,
                frequency="twice every layer/decode step",
                lifetime="temporary",
                reusable_in_principle=True,
                notes="one logical tensor each for logits and softmax weights",
            )
        ]

    selections = [
        _allocation(
            "causally_ordered_selection_indices",
            (batch_size, heads, selection_width),
            torch.int64,
            frequency="multiple tensors every layer/decode step",
            lifetime="temporary through fetch",
            reusable_in_principle=True,
            notes="clones, sentinel/sort order, and gathered index outputs",
        ),
        _allocation(
            "selected_keys",
            (batch_size, heads, selection_width, head_dimension),
            torch.float32,
            frequency="every layer/decode step",
            lifetime="temporary through attention",
            reusable_in_principle=True,
            notes="TensorStorage torch.gather output",
        ),
        _allocation(
            "selected_values",
            (batch_size, heads, selection_width, head_dimension),
            torch.float32,
            frequency="every layer/decode step",
            lifetime="temporary through attention",
            reusable_in_principle=True,
            notes="TensorStorage torch.gather output",
        ),
        _allocation(
            "selected_attention_logits_or_weights",
            (batch_size, heads, selection_width),
            torch.float32,
            frequency="twice every layer/decode step",
            lifetime="temporary",
            reusable_in_principle=True,
            notes="one logical tensor each for logits and softmax weights",
        ),
    ]
    if strategy == "quest":
        pages = math.ceil(sequence_length / page_size)
        selected_pages = min(
            pages,
            math.ceil(math.ceil(sequence_length * budget_fraction) / page_size),
        )
        padding = pages * page_size - sequence_length
        allocations = list(common)
        if padding:
            allocations.extend(
                [
                    _allocation(
                        "quest_minimum_padding",
                        (batch_size, heads, padding, head_dimension),
                        torch.float32,
                        frequency="every layer/decode step",
                        lifetime="temporary during metadata rebuild",
                        reusable_in_principle=True,
                        notes="positive-infinity tail padding",
                    ),
                    _allocation(
                        "quest_maximum_padding",
                        (batch_size, heads, padding, head_dimension),
                        torch.float32,
                        frequency="every layer/decode step",
                        lifetime="temporary during metadata rebuild",
                        reusable_in_principle=True,
                        notes="negative-infinity tail padding",
                    ),
                    _allocation(
                        "quest_padded_minimum_input",
                        (batch_size, heads, pages * page_size, head_dimension),
                        torch.float32,
                        frequency="every layer/decode step",
                        lifetime="temporary during minimum reduction",
                        reusable_in_principle=True,
                        notes="torch.cat result; reshape is a view",
                    ),
                    _allocation(
                        "quest_padded_maximum_input",
                        (batch_size, heads, pages * page_size, head_dimension),
                        torch.float32,
                        frequency="every layer/decode step",
                        lifetime="temporary during maximum reduction",
                        reusable_in_principle=True,
                        notes="torch.cat result; reshape is a view",
                    ),
                ]
            )
        allocations.extend(
            [
                _allocation(
                    "quest_page_minimum",
                    (batch_size, heads, pages, head_dimension),
                    torch.float32,
                    frequency="every layer/decode step",
                    lifetime="persistent until next rebuild",
                    reusable_in_principle=True,
                    notes="metadata tensor",
                ),
                _allocation(
                    "quest_page_maximum",
                    (batch_size, heads, pages, head_dimension),
                    torch.float32,
                    frequency="every layer/decode step",
                    lifetime="persistent until next rebuild",
                    reusable_in_principle=True,
                    notes="metadata tensor",
                ),
                _allocation(
                    "quest_score_dimension_intermediate",
                    (batch_size, heads, pages, head_dimension),
                    torch.float32,
                    frequency="three tensors every layer/decode step",
                    lifetime="temporary during page scoring",
                    reusable_in_principle=True,
                    notes="minimum product, maximum product, elementwise maximum",
                ),
                _allocation(
                    "quest_page_scores",
                    (batch_size, heads, pages),
                    torch.float32,
                    frequency="every layer/decode step",
                    lifetime="temporary through page ranking",
                    reusable_in_principle=True,
                    notes="dimension-reduction output",
                ),
                _allocation(
                    "quest_expanded_page_token_ids",
                    (batch_size, heads, selected_pages, page_size),
                    torch.int64,
                    frequency="several tensors every layer/decode step",
                    lifetime="temporary through mask compaction",
                    reusable_in_principle=True,
                    notes="expanded IDs, safe IDs, and int64 compaction order",
                ),
                _allocation(
                    "quest_expanded_validity_mask",
                    (batch_size, heads, selected_pages, page_size),
                    torch.bool,
                    frequency="several tensors every layer/decode step",
                    lifetime="temporary through attention",
                    reusable_in_principle=True,
                    notes="partial-page and compacted validity masks",
                ),
            ]
        )
        return allocations + selections

    if strategy == "pq":
        subspace_dimension = head_dimension // pq_subspaces
        return common + [
            _allocation(
                "pq_append_centroid_differences",
                (
                    batch_size,
                    heads,
                    1,
                    pq_subspaces,
                    pq_centroids,
                    subspace_dimension,
                ),
                torch.float32,
                frequency="every layer/decode step",
                lifetime="temporary during frozen append",
                reusable_in_principle=True,
                notes="broadcast subtraction for one appended key",
            ),
            _allocation(
                "pq_append_centroid_distances",
                (batch_size, heads, 1, pq_subspaces, pq_centroids),
                torch.float32,
                frequency="every layer/decode step",
                lifetime="temporary during frozen append",
                reusable_in_principle=True,
                notes="squared-distance reduction output",
            ),
            _allocation(
                "pq_appended_code_tensor",
                (batch_size, heads, sequence_length, pq_subspaces),
                torch.int64,
                frequency="every layer/decode step",
                lifetime="persistent until next append",
                reusable_in_principle=False,
                notes="torch.cat copies every prior int64 code plus new codes",
            ),
            _allocation(
                "pq_query_lookup_table",
                (batch_size, heads, pq_subspaces, pq_centroids),
                torch.float32,
                frequency="every layer/decode step",
                lifetime="temporary through approximate scoring",
                reusable_in_principle=True,
                notes="query-to-centroid dot products",
            ),
            _allocation(
                "pq_approximate_scores",
                (batch_size, heads, sequence_length),
                torch.float32,
                frequency="every layer/decode step",
                lifetime="temporary through ranking",
                reusable_in_principle=True,
                notes="zero initialization followed by subspace accumulation",
            ),
            _allocation(
                "pq_subspace_lookup_result",
                (batch_size, heads, sequence_length),
                torch.float32,
                frequency=f"{pq_subspaces} times every layer/decode step",
                lifetime="temporary during score reconstruction",
                reusable_in_principle=True,
                notes="one gathered score vector per subspace",
            ),
            _allocation(
                "pq_ranked_token_ids",
                (batch_size, heads, sequence_length),
                torch.int64,
                frequency="every layer/decode step",
                lifetime="temporary through candidate slicing",
                reusable_in_principle=True,
                notes="stable full argsort output; implementation intentionally unchanged",
            ),
            *selections,
        ]
    raise ValueError(f"unsupported strategy {strategy!r}")


def initialization_allocation_estimates(
    *,
    strategy: str,
    sequence_length: int,
    batch_size: int,
    heads: int,
    head_dimension: int,
    page_size: int = 64,
    pq_subspaces: int = 4,
    pq_centroids: int = 8,
) -> list[dict[str, Any]]:
    """Estimate major one-time index-construction allocations per layer."""
    if strategy == "quest":
        pages = math.ceil(sequence_length / page_size)
        padding = pages * page_size - sequence_length
        estimates = []
        if padding:
            estimates.extend(
                [
                    _allocation(
                        "quest_initial_minimum_padding",
                        (batch_size, heads, padding, head_dimension),
                        torch.float32,
                        frequency="once per layer initialization",
                        lifetime="temporary",
                        reusable_in_principle=True,
                        notes="positive-infinity tail padding",
                    ),
                    _allocation(
                        "quest_initial_maximum_padding",
                        (batch_size, heads, padding, head_dimension),
                        torch.float32,
                        frequency="once per layer initialization",
                        lifetime="temporary",
                        reusable_in_principle=True,
                        notes="negative-infinity tail padding",
                    ),
                ]
            )
        estimates.extend(
            [
                _allocation(
                    "quest_initial_page_minimum",
                    (batch_size, heads, pages, head_dimension),
                    torch.float32,
                    frequency="once per layer initialization",
                    lifetime="persistent until first decode rebuild",
                    reusable_in_principle=True,
                    notes="initial minimum metadata",
                ),
                _allocation(
                    "quest_initial_page_maximum",
                    (batch_size, heads, pages, head_dimension),
                    torch.float32,
                    frequency="once per layer initialization",
                    lifetime="persistent until first decode rebuild",
                    reusable_in_principle=True,
                    notes="initial maximum metadata",
                ),
            ]
        )
        return estimates
    if strategy != "pq":
        raise ValueError(f"unsupported strategy {strategy!r}")
    subspace_dimension = head_dimension // pq_subspaces
    return [
        _allocation(
            "pq_codebooks",
            (
                batch_size,
                heads,
                pq_subspaces,
                pq_centroids,
                subspace_dimension,
            ),
            torch.float32,
            frequency="once per layer initialization",
            lifetime="persistent",
            reusable_in_principle=False,
            notes="trained centroid storage",
        ),
        _allocation(
            "pq_kmeans_group_differences",
            (sequence_length, pq_centroids, subspace_dimension),
            torch.float32,
            frequency=f"up to 8 iterations for each of {heads * pq_subspaces} groups",
            lifetime="temporary per head/subspace group",
            reusable_in_principle=True,
            notes="representative peak group-local broadcast subtraction",
        ),
        _allocation(
            "pq_prefill_encoding_differences",
            (
                batch_size,
                heads,
                sequence_length,
                pq_subspaces,
                pq_centroids,
                subspace_dimension,
            ),
            torch.float32,
            frequency="once per layer initialization",
            lifetime="temporary during prefill encoding",
            reusable_in_principle=True,
            notes="full prefill broadcast subtraction",
        ),
        _allocation(
            "pq_prefill_encoding_distances",
            (batch_size, heads, sequence_length, pq_subspaces, pq_centroids),
            torch.float32,
            frequency="once per layer initialization",
            lifetime="temporary during prefill encoding",
            reusable_in_principle=True,
            notes="full prefill squared-distance reductions",
        ),
        _allocation(
            "pq_initial_codes",
            (batch_size, heads, sequence_length, pq_subspaces),
            torch.int64,
            frequency="once per layer initialization",
            lifetime="persistent and extended during decode",
            reusable_in_principle=False,
            notes="reference code storage is int64, not bit packed",
        ),
    ]


def _traffic(
    operation: str,
    *,
    read_bytes: int,
    write_bytes: int,
    scaling: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "estimated_read_bytes": read_bytes,
        "estimated_write_bytes": write_bytes,
        "estimated_total_bytes": read_bytes + write_bytes,
        "scaling": scaling,
        "notes": notes,
        "estimate_kind": (
            "analytical logical tensor traffic; excludes caches, allocator, "
            "kernel internals, and validation reads"
        ),
    }


def tensor_traffic_estimates(
    *,
    strategy: str,
    sequence_length: int,
    batch_size: int,
    heads: int,
    head_dimension: int,
    selection_width: int,
    page_size: int = 64,
    pq_subspaces: int = 4,
    pq_centroids: int = 8,
) -> list[dict[str, Any]]:
    """Estimate major logical reads/writes for one layer/decode step."""
    dense_key_bytes = estimate_tensor_bytes(
        (batch_size, heads, sequence_length, head_dimension),
        torch.float32,
    )
    selected_kv_bytes = 2 * estimate_tensor_bytes(
        (batch_size, heads, selection_width, head_dimension),
        torch.float32,
    )
    common = [
        _traffic(
            "causal_kv_append",
            read_bytes=2 * dense_key_bytes,
            write_bytes=2 * dense_key_bytes,
            scaling="O(H * S * D)",
            notes="two torch.cat operations for full-precision K and V",
        )
    ]
    if strategy == "dense":
        return common + [
            _traffic(
                "dense_attention_kv_consumption",
                read_bytes=2 * dense_key_bytes,
                write_bytes=estimate_tensor_bytes(
                    (batch_size, heads, sequence_length), torch.float32
                ),
                scaling="O(H * S * D)",
                notes="K/V reads plus one logical score tensor write",
            )
        ]
    selection_index_bytes = estimate_tensor_bytes(
        (batch_size, heads, selection_width),
        torch.int64,
    )
    retrieval = [
        _traffic(
            "selected_index_reads",
            read_bytes=2 * selection_index_bytes,
            write_bytes=0,
            scaling="O(H * K)",
            notes="logical index consumption for K and V gathers",
        ),
        _traffic(
            "full_precision_kv_gather",
            read_bytes=selected_kv_bytes,
            write_bytes=selected_kv_bytes,
            scaling="O(H * K * D)",
            notes="selected K/V source reads and gathered destination writes",
        ),
        _traffic(
            "selected_attention_kv_consumption",
            read_bytes=selected_kv_bytes,
            write_bytes=estimate_tensor_bytes(
                (batch_size, heads, selection_width), torch.float32
            ),
            scaling="O(H * K * D)",
            notes="selected K/V reads plus one logical score tensor write",
        ),
    ]
    if strategy == "quest":
        pages = math.ceil(sequence_length / page_size)
        metadata_bytes = estimate_tensor_bytes(
            (batch_size, heads, pages, head_dimension), torch.float32
        )
        return common + [
            _traffic(
                "quest_full_key_metadata_rebuild",
                read_bytes=2 * dense_key_bytes,
                write_bytes=2 * metadata_bytes,
                scaling="O(H * S * D)",
                notes="separate minimum and maximum reductions",
            ),
            _traffic(
                "quest_page_scoring",
                read_bytes=2 * metadata_bytes,
                write_bytes=3 * metadata_bytes
                + estimate_tensor_bytes((batch_size, heads, pages), torch.float32),
                scaling="O(H * ceil(S/P) * D)",
                notes="metadata reads and three dimension-sized intermediates",
            ),
            *retrieval,
        ]
    if strategy == "pq":
        codebook_bytes = estimate_tensor_bytes(
            (
                batch_size,
                heads,
                pq_subspaces,
                pq_centroids,
                head_dimension // pq_subspaces,
            ),
            torch.float32,
        )
        old_code_bytes = estimate_tensor_bytes(
            (batch_size, heads, sequence_length - 1, pq_subspaces),
            torch.int64,
        )
        new_code_bytes = estimate_tensor_bytes(
            (batch_size, heads, sequence_length, pq_subspaces),
            torch.int64,
        )
        score_bytes = estimate_tensor_bytes(
            (batch_size, heads, sequence_length), torch.float32
        )
        return common + [
            _traffic(
                "pq_frozen_key_assignment",
                read_bytes=estimate_tensor_bytes(
                    (batch_size, heads, 1, head_dimension), torch.float32
                )
                + codebook_bytes,
                write_bytes=estimate_tensor_bytes(
                    (batch_size, heads, 1, pq_subspaces), torch.int64
                ),
                scaling="O(H * M * C * D/M) = O(H * C * D)",
                notes="new-key and codebook logical traffic; excludes temporaries",
            ),
            _traffic(
                "pq_code_append",
                read_bytes=old_code_bytes
                + estimate_tensor_bytes(
                    (batch_size, heads, 1, pq_subspaces), torch.int64
                ),
                write_bytes=new_code_bytes,
                scaling="O(H * S * M)",
                notes="reference torch.cat copies all prior int64 codes",
            ),
            _traffic(
                "pq_lookup_table_construction",
                read_bytes=codebook_bytes
                + estimate_tensor_bytes(
                    (batch_size, heads, head_dimension), torch.float32
                ),
                write_bytes=estimate_tensor_bytes(
                    (batch_size, heads, pq_subspaces, pq_centroids),
                    torch.float32,
                ),
                scaling="O(H * M * C * D/M) = O(H * C * D)",
                notes="query and codebook reads plus lookup-table write",
            ),
            _traffic(
                "pq_approximate_score_reconstruction",
                read_bytes=new_code_bytes + pq_subspaces * score_bytes,
                write_bytes=(pq_subspaces + 1) * score_bytes,
                scaling="O(H * S * M)",
                notes="int64 code reads, gathered subspace scores, and accumulation",
            ),
            *retrieval,
        ]
    raise ValueError(f"unsupported strategy {strategy!r}")
