#!/usr/bin/env python3
"""Validate and profile exact incremental Quest metadata maintenance."""

import argparse
import cProfile
from dataclasses import asdict
import math
from pathlib import Path
import pstats
import statistics
import sys
import time
from typing import Any, Callable

import torch

from benchmarks.artifacts import write_json
from benchmarks.phase4 import load_phase4_baseline
from benchmarks.statistics import latency_distribution as distribution
from benchmarks.support import git_commit, git_is_dirty, machine_metadata
from benchmarks.phase3a import TEXT_FIXTURES, build_deterministic_fixture
from benchmarks.decode import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TRANSFORMERS_REVISION,
    DEFAULT_TRANSFORMERS_VERSION,
    assert_full_budget_step,
    build_dense_trace,
    validate_hugging_face_generation,
)
from kvweave import QuestIndex
from kvweave.indexes.quest import (
    QuestMetadata,
    append_page_metadata,
    build_page_metadata,
    score_pages,
)
from kvweave.integrations.transformers import (
    DecodeStrategy,
    DensePrefillSnapshot,
    GPTNeoXDecodeRunner,
    GPTNeoXDecodeState,
    GPTNeoXDecodeStep,
    QuestMetadataUpdateMode,
    attention_mass_captured,
    logit_comparison_metrics,
    per_head_relative_error,
    relative_tensor_error,
    validate_gpt_neox_config,
)
from kvweave.profiling import (
    ComponentProfiler,
    ComponentTiming,
    estimate_tensor_bytes,
    profile_context,
)


DEFAULT_OUTPUT = Path("benchmarks/results/pythia-410m-phase5a-quest-incremental.json")
DEFAULT_PHASE4_ARTIFACT = Path("benchmarks/results/pythia-410m-phase4-profile.json")
DEFAULT_PROFILE_DIRECTORY = Path("benchmarks/results/profile/pythia-410m-phase5a")
FIXTURE_IDS = ("technical_exposition", "code_like")
PROMPT_LENGTH = 1_024
GENERATED_TOKEN_POSITIONS = 32
DECODE_STEPS = GENERATED_TOKEN_POSITIONS - 1
BUDGET_FRACTIONS = (0.5, 1.0)
QUEST_PAGE_SIZE = 64
SEED = 0
FULL_BUDGET_RTOL = 1e-4
FULL_BUDGET_ATOL = 1e-5
SCALING_LENGTHS = (512, 2_048, 8_192, 32_768)
SCALING_WARMUPS = 5
SCALING_REPETITIONS = 25

MODE_LABELS = {
    QuestMetadataUpdateMode.FULL_REBUILD: "full_rebuild_oracle",
    QuestMetadataUpdateMode.INCREMENTAL: "incremental",
}
FULL_METADATA_COMPONENTS = (
    "quest.metadata.page_reshape_padding",
    "quest.metadata.page_minimum",
    "quest.metadata.page_maximum",
    "quest.metadata.object_construction",
)
INCREMENTAL_METADATA_COMPONENTS = (
    "quest.metadata.incremental.identify_page",
    "quest.metadata.incremental.existing_page_minimum",
    "quest.metadata.incremental.existing_page_maximum",
    "quest.metadata.incremental.new_page_append",
    "quest.metadata.incremental.state_bookkeeping",
)
RETRIEVAL_COMPONENT_GROUPS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase4-artifact",
        type=Path,
        default=DEFAULT_PHASE4_ARTIFACT,
    )
    parser.add_argument(
        "--profile-directory",
        type=Path,
        default=DEFAULT_PROFILE_DIRECTORY,
    )
    return parser.parse_args()


def _assert_optional_tensor_equal(
    incremental: torch.Tensor | None,
    oracle: torch.Tensor | None,
    *,
    name: str,
) -> None:
    if (incremental is None) != (oracle is None):
        raise AssertionError(f"Quest {name} presence changed")
    if (
        incremental is not None
        and oracle is not None
        and not torch.equal(
            incremental,
            oracle,
        )
    ):
        raise AssertionError(f"Quest {name} changed")


def assert_decode_steps_bit_exact(
    incremental: GPTNeoXDecodeStep,
    oracle: GPTNeoXDecodeStep,
) -> None:
    """Require exact observable decode equivalence, excluding wall times."""
    for name in ("input_token", "next_token_logits", "next_token"):
        if not torch.equal(getattr(incremental, name), getattr(oracle, name)):
            raise AssertionError(f"incremental Quest changed {name}")
    for incremental_layer, oracle_layer in zip(
        incremental.layers,
        oracle.layers,
        strict=True,
    ):
        for name in (
            "query",
            "attention_output",
            "attention_weights",
            "residual_output",
            "selected_token_counts",
        ):
            if not torch.equal(
                getattr(incremental_layer, name),
                getattr(oracle_layer, name),
            ):
                raise AssertionError(f"incremental Quest changed layer {name}")
        if incremental_layer.selection is None or oracle_layer.selection is None:
            raise AssertionError("Quest decode omitted a selection")
        for name in ("indices", "scores", "valid_mask"):
            _assert_optional_tensor_equal(
                getattr(incremental_layer.selection, name),
                getattr(oracle_layer.selection, name),
                name=f"selection {name}",
            )


def _quest_index(state: GPTNeoXDecodeState, layer: int) -> QuestIndex:
    cache = state.layers[layer].cache
    if cache is None or not isinstance(cache.index, QuestIndex):
        raise AssertionError("Quest decode state has no QuestIndex")
    return cache.index


def assert_state_and_search_exact(
    incremental_state: GPTNeoXDecodeState,
    oracle_state: GPTNeoXDecodeState,
    incremental_step: GPTNeoXDecodeStep,
    oracle_step: GPTNeoXDecodeStep,
) -> int:
    """Compare metadata, raw page search, expanded selection, and masks."""
    comparisons = 0
    for incremental_layer, oracle_layer in zip(
        incremental_step.layers,
        oracle_step.layers,
        strict=True,
    ):
        incremental_index = _quest_index(
            incremental_state,
            incremental_layer.layer_index,
        )
        oracle_index = _quest_index(oracle_state, oracle_layer.layer_index)
        incremental_metadata = incremental_index.metadata
        oracle_metadata = oracle_index.metadata
        if not torch.equal(
            incremental_metadata.minimum,
            oracle_metadata.minimum,
        ):
            raise AssertionError("incremental Quest minimum metadata changed")
        if not torch.equal(
            incremental_metadata.maximum,
            oracle_metadata.maximum,
        ):
            raise AssertionError("incremental Quest maximum metadata changed")
        if (
            incremental_metadata.sequence_length != oracle_metadata.sequence_length
            or incremental_metadata.num_pages != oracle_metadata.num_pages
            or incremental_metadata.minimum.shape != oracle_metadata.minimum.shape
        ):
            raise AssertionError("incremental Quest metadata shape/state changed")

        query = incremental_layer.query
        if not torch.equal(query, oracle_layer.query):
            raise AssertionError("Quest paths produced different queries")
        incremental_scores = score_pages(query, incremental_metadata)
        oracle_scores = score_pages(query, oracle_metadata)
        if not torch.equal(incremental_scores, oracle_scores):
            raise AssertionError("incremental Quest page scores changed")
        budget = max(
            1,
            math.ceil(
                incremental_layer.sequence_length * incremental_state.budget_fraction
            ),
        )
        incremental_result = incremental_index.search_with_details(query, budget)
        oracle_result = oracle_index.search_with_details(query, budget)
        for name in ("page_indices", "page_scores"):
            if not torch.equal(
                getattr(incremental_result, name),
                getattr(oracle_result, name),
            ):
                raise AssertionError(f"incremental Quest {name} changed")
        for name in ("indices", "scores", "valid_mask"):
            _assert_optional_tensor_equal(
                getattr(incremental_result.selection, name),
                getattr(oracle_result.selection, name),
                name=f"raw selection {name}",
            )
        if not torch.equal(
            incremental_result.actual_token_counts,
            oracle_result.actual_token_counts,
        ):
            raise AssertionError("incremental Quest candidate counts changed")
        comparisons += 1
    return comparisons


def initialize_quest_state(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    *,
    budget_fraction: float,
    update_mode: QuestMetadataUpdateMode,
) -> GPTNeoXDecodeState:
    return runner.initialize_state(
        snapshot,
        strategy=DecodeStrategy.QUEST,
        budget_fraction=budget_fraction,
        quest_page_size=QUEST_PAGE_SIZE,
        quest_metadata_update_mode=update_mode,
    )


def run_quest_trace(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
    *,
    budget_fraction: float,
    update_mode: QuestMetadataUpdateMode,
    fixture_id: str,
    recorder: ComponentProfiler | None = None,
) -> tuple[GPTNeoXDecodeState, list[GPTNeoXDecodeStep]]:
    state = initialize_quest_state(
        runner,
        snapshot,
        budget_fraction=budget_fraction,
        update_mode=update_mode,
    )
    steps = []
    for decode_step in range(1, GENERATED_TOKEN_POSITIONS):
        input_token = torch.tensor(
            [[dense_tokens[decode_step - 1]]],
            dtype=torch.int64,
            device=snapshot.input_ids.device,
        )
        if recorder is None:
            step = runner.step(state, input_token)
        else:
            with (
                recorder.activate(),
                profile_context(
                    phase="decode",
                    fixture_id=fixture_id,
                    strategy="quest",
                    budget_fraction=budget_fraction,
                    metadata_update_mode=MODE_LABELS[update_mode],
                    decode_step=decode_step,
                ),
            ):
                step = runner.step(state, input_token)
        steps.append(step)
    return state, steps


def run_lockstep_correctness(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
    dense_steps: list[GPTNeoXDecodeStep],
    *,
    budget_fraction: float,
) -> dict[str, Any]:
    oracle_state = initialize_quest_state(
        runner,
        snapshot,
        budget_fraction=budget_fraction,
        update_mode=QuestMetadataUpdateMode.FULL_REBUILD,
    )
    incremental_state = initialize_quest_state(
        runner,
        snapshot,
        budget_fraction=budget_fraction,
        update_mode=QuestMetadataUpdateMode.INCREMENTAL,
    )
    layer_search_comparisons = 0
    for decode_step, dense_step in enumerate(dense_steps, start=1):
        input_token = torch.tensor(
            [[dense_tokens[decode_step - 1]]],
            dtype=torch.int64,
            device=snapshot.input_ids.device,
        )
        oracle_step = runner.step(oracle_state, input_token)
        incremental_step = runner.step(incremental_state, input_token)
        assert_decode_steps_bit_exact(incremental_step, oracle_step)
        layer_search_comparisons += assert_state_and_search_exact(
            incremental_state,
            oracle_state,
            incremental_step,
            oracle_step,
        )
        if budget_fraction == 1.0:
            assert_full_budget_step(
                incremental_step,
                dense_step,
                rtol=FULL_BUDGET_RTOL,
                atol=FULL_BUDGET_ATOL,
            )
    return {
        "passed": True,
        "decode_steps_compared": len(dense_steps),
        "layer_steps_compared": len(dense_steps) * len(dense_steps[0].layers),
        "metadata_and_raw_search_comparisons": layer_search_comparisons,
        "metadata_minimum_bit_exact": True,
        "metadata_maximum_bit_exact": True,
        "page_scores_bit_exact": True,
        "page_ids_bit_exact": True,
        "selection_ids_scores_masks_bit_exact": True,
        "attention_outputs_and_weights_bit_exact": True,
        "residual_streams_bit_exact": True,
        "logits_bit_exact": True,
        "full_budget_dense_control_passed": budget_fraction == 1.0,
    }


def _layer_total_time_ms(step: GPTNeoXDecodeStep, layer: int) -> float:
    observation = step.layers[layer]
    return (
        observation.index_update_time_ms
        + observation.retrieval_time_ms
        + observation.storage_fetch_time_ms
        + observation.selected_attention_time_ms
        + observation.remaining_layer_time_ms
    )


def build_step_records(
    records: list[ComponentTiming],
    measured: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    atomic: dict[tuple[Any, ...], dict[str, float]] = {}
    for record in records:
        key = (
            record.context["fixture_id"],
            record.context["budget_fraction"],
            record.context["metadata_update_mode"],
            record.context["decode_step"],
            record.context["layer"],
        )
        values = atomic.setdefault(key, {})
        values[record.component] = (
            values.get(record.component, 0.0) + record.duration_ms
        )

    step_records = []
    for run in measured:
        update_mode = run["metadata_update_mode"]
        metadata_components = (
            FULL_METADATA_COMPONENTS
            if update_mode == MODE_LABELS[QuestMetadataUpdateMode.FULL_REBUILD]
            else INCREMENTAL_METADATA_COMPONENTS
        )
        for decode_step, step_timing in enumerate(run["step_timings"], start=1):
            categories: dict[str, float] = {
                "metadata_maintenance": 0.0,
                **{name: 0.0 for name in RETRIEVAL_COMPONENT_GROUPS},
            }
            for layer in range(step_timing["layer_count"]):
                key = (
                    run["fixture_id"],
                    run["budget_fraction"],
                    update_mode,
                    decode_step,
                    layer,
                )
                layer_atomic = atomic[key]
                categories["metadata_maintenance"] += sum(
                    layer_atomic.get(component, 0.0)
                    for component in metadata_components
                )
                for name, components in RETRIEVAL_COMPONENT_GROUPS.items():
                    categories[name] += sum(
                        layer_atomic.get(component, 0.0) for component in components
                    )
            total_retrieval = sum(categories.values())
            step_records.append(
                {
                    "fixture_id": run["fixture_id"],
                    "budget_fraction": run["budget_fraction"],
                    "metadata_update_mode": update_mode,
                    "decode_step": decode_step,
                    "component_categories_ms": categories,
                    "total_retrieval_overhead_ms": total_retrieval,
                    "total_layers_ms": step_timing["total_layers_ms"],
                    "total_decode_step_ms": step_timing["total_decode_step_ms"],
                }
            )
    return step_records


def summarize_step_records(step_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for update_mode in MODE_LABELS.values():
        for budget in BUDGET_FRACTIONS:
            rows = [
                row
                for row in step_records
                if row["metadata_update_mode"] == update_mode
                and row["budget_fraction"] == budget
            ]
            components = list(rows[0]["component_categories_ms"])
            components.extend(
                ["total_retrieval_overhead", "total_layers", "total_decode_step"]
            )
            for component in components:
                if component in rows[0]["component_categories_ms"]:
                    values = [row["component_categories_ms"][component] for row in rows]
                else:
                    values = [row[f"{component}_ms"] for row in rows]
                output.append(
                    {
                        "metadata_update_mode": update_mode,
                        "budget_fraction": budget,
                        "component": component,
                        **{
                            f"{name}_ms": value
                            for name, value in distribution(values).items()
                        },
                    }
                )
    return output


def paired_before_after(step_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            row["fixture_id"],
            row["budget_fraction"],
            row["metadata_update_mode"],
            row["decode_step"],
        ): row
        for row in step_records
    }
    output = []
    for budget in BUDGET_FRACTIONS:
        for component in (
            "metadata_maintenance",
            "total_retrieval_overhead",
            "total_decode_step",
        ):
            old_values = []
            new_values = []
            savings = []
            for fixture_id in FIXTURE_IDS:
                for decode_step in range(1, GENERATED_TOKEN_POSITIONS):
                    old = indexed[
                        (
                            fixture_id,
                            budget,
                            MODE_LABELS[QuestMetadataUpdateMode.FULL_REBUILD],
                            decode_step,
                        )
                    ]
                    new = indexed[
                        (
                            fixture_id,
                            budget,
                            MODE_LABELS[QuestMetadataUpdateMode.INCREMENTAL],
                            decode_step,
                        )
                    ]
                    if component == "metadata_maintenance":
                        old_value = old["component_categories_ms"][component]
                        new_value = new["component_categories_ms"][component]
                    else:
                        old_value = old[f"{component}_ms"]
                        new_value = new[f"{component}_ms"]
                    old_values.append(old_value)
                    new_values.append(new_value)
                    savings.append(old_value - new_value)
            old_summary = distribution(old_values)
            new_summary = distribution(new_values)
            old_median = float(old_summary["median"])
            new_median = float(new_summary["median"])
            output.append(
                {
                    "budget_fraction": budget,
                    "component": component,
                    "old_full_rebuild_ms": old_summary,
                    "new_incremental_ms": new_summary,
                    "paired_savings_ms": distribution(savings),
                    "difference_of_medians_ms": old_median - new_median,
                    "median_time_reduction_fraction": (
                        (old_median - new_median) / old_median
                    ),
                    "old_to_new_median_ratio": old_median / new_median,
                }
            )
    return output


def incremental_component_summary(
    records: list[ComponentTiming],
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if record.context.get("metadata_update_mode") == "incremental"
        and record.component in INCREMENTAL_METADATA_COMPONENTS
    ]
    atomic = []
    for component in INCREMENTAL_METADATA_COMPONENTS:
        values = [
            record.duration_ms for record in selected if record.component == component
        ]
        atomic.append(
            {
                "component": component,
                "milliseconds_per_layer_call": (
                    None if not values else distribution(values)
                ),
            }
        )

    existing_groups: dict[tuple[Any, ...], float] = {}
    new_page_groups: dict[tuple[Any, ...], float] = {}
    for record in selected:
        key = (
            record.context["fixture_id"],
            record.context["budget_fraction"],
            record.context["decode_step"],
            record.context["layer"],
        )
        target = (
            new_page_groups
            if record.component == "quest.metadata.incremental.new_page_append"
            else existing_groups
        )
        target[key] = target.get(key, 0.0) + record.duration_ms
    new_keys = set(new_page_groups)
    existing_values = [
        value for key, value in existing_groups.items() if key not in new_keys
    ]
    new_values = [
        new_page_groups[key] + existing_groups.get(key, 0.0) for key in new_keys
    ]
    return {
        "atomic_components": atomic,
        "existing_partial_page_ms_per_layer": distribution(existing_values),
        "new_page_ms_per_layer": distribution(new_values),
        "classification": (
            "existing-page timing includes safe full-metadata clone/replacement; "
            "new-page timing includes contiguous tensor concatenation"
        ),
    }


def quality_summary(
    measured: list[dict[str, Any]],
    dense_by_fixture: dict[str, list[GPTNeoXDecodeStep]],
) -> list[dict[str, Any]]:
    output = []
    for run in measured:
        rows = []
        for approximate, dense in zip(
            run["steps"],
            dense_by_fixture[run["fixture_id"]],
            strict=True,
        ):
            masses = []
            attention_errors = []
            residual_errors = []
            for approximate_layer, dense_layer in zip(
                approximate.layers,
                dense.layers,
                strict=True,
            ):
                if approximate_layer.selection is None:
                    raise AssertionError("Quest path omitted selection")
                masses.extend(
                    attention_mass_captured(
                        dense_layer.attention_weights,
                        approximate_layer.selection,
                    )[0]
                    .float()
                    .tolist()
                )
                attention_errors.extend(
                    per_head_relative_error(
                        approximate_layer.attention_output,
                        dense_layer.attention_output,
                    )[0]
                    .float()
                    .tolist()
                )
                residual_errors.append(
                    relative_tensor_error(
                        approximate_layer.residual_output,
                        dense_layer.residual_output,
                    )
                )
            rows.append(
                {
                    **logit_comparison_metrics(
                        approximate.next_token_logits,
                        dense.next_token_logits,
                    ),
                    "attention_mass": statistics.fmean(masses),
                    "attention_error": statistics.fmean(attention_errors),
                    "residual_error": statistics.fmean(residual_errors),
                }
            )
        output.append(
            {
                "fixture_id": run["fixture_id"],
                "metadata_update_mode": run["metadata_update_mode"],
                "budget_fraction": run["budget_fraction"],
                "decode_step_count": len(rows),
                "top_1_agreement_rate": statistics.fmean(
                    float(row["top_1_agreement"]) for row in rows
                ),
                "mean_top_5_overlap_fraction": statistics.fmean(
                    row["top_5_overlap_fraction"] for row in rows
                ),
                "mean_logit_relative_error": statistics.fmean(
                    row["logit_relative_error"] for row in rows
                ),
                "mean_kl_divergence": statistics.fmean(
                    row["kl_divergence_dense_to_approximate"] for row in rows
                ),
                "mean_attention_mass_captured": statistics.fmean(
                    row["attention_mass"] for row in rows
                ),
                "mean_attention_output_relative_error": statistics.fmean(
                    row["attention_error"] for row in rows
                ),
                "mean_residual_stream_relative_error": statistics.fmean(
                    row["residual_error"] for row in rows
                ),
            }
        )
    return output


def pooled_quality_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool the equal-length two-fixture quality summaries by path and budget."""
    output = []
    metric_names = (
        "top_1_agreement_rate",
        "mean_top_5_overlap_fraction",
        "mean_logit_relative_error",
        "mean_kl_divergence",
        "mean_attention_mass_captured",
        "mean_attention_output_relative_error",
        "mean_residual_stream_relative_error",
    )
    for update_mode in MODE_LABELS.values():
        for budget in BUDGET_FRACTIONS:
            selected = [
                row
                for row in rows
                if row["metadata_update_mode"] == update_mode
                and row["budget_fraction"] == budget
            ]
            output.append(
                {
                    "metadata_update_mode": update_mode,
                    "budget_fraction": budget,
                    "fixture_ids": [row["fixture_id"] for row in selected],
                    "decode_step_count": sum(
                        row["decode_step_count"] for row in selected
                    ),
                    **{
                        name: statistics.fmean(row[name] for row in selected)
                        for name in metric_names
                    },
                }
            )
    return output


def _measure_operation(
    operation: Callable[[], QuestMetadata],
    *,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, float | int], float]:
    checksum = 0.0
    for _ in range(warmups):
        result = operation()
        checksum += float(result.minimum[0, 0, -1, 0].item())
    values = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = operation()
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
        checksum += float(result.maximum[0, 0, -1, 0].item())
    return distribution(values), checksum


def synthetic_scaling_benchmark() -> list[dict[str, Any]]:
    results = []
    generator = torch.Generator().manual_seed(SEED)
    for sequence_length in SCALING_LENGTHS:
        keys = torch.randn(
            1,
            16,
            sequence_length + 2,
            64,
            generator=generator,
            dtype=torch.float32,
        )
        full_input = keys[:, :, : sequence_length + 1]
        new_page_base = build_page_metadata(
            keys[:, :, :sequence_length],
            QUEST_PAGE_SIZE,
        )
        existing_base = build_page_metadata(full_input, QUEST_PAGE_SIZE)
        with torch.no_grad():
            full, full_checksum = _measure_operation(
                lambda: build_page_metadata(full_input, QUEST_PAGE_SIZE),
                warmups=SCALING_WARMUPS,
                repetitions=SCALING_REPETITIONS,
            )
            new_page, new_page_checksum = _measure_operation(
                lambda: append_page_metadata(
                    new_page_base,
                    keys[:, :, sequence_length : sequence_length + 1],
                ),
                warmups=SCALING_WARMUPS,
                repetitions=SCALING_REPETITIONS,
            )
            existing_page, existing_checksum = _measure_operation(
                lambda: append_page_metadata(
                    existing_base,
                    keys[:, :, sequence_length + 1 : sequence_length + 2],
                ),
                warmups=SCALING_WARMUPS,
                repetitions=SCALING_REPETITIONS,
            )
        results.append(
            {
                "sequence_length_before_append": sequence_length,
                "batch_size": 1,
                "kv_heads": 16,
                "head_dimension": 64,
                "page_size": QUEST_PAGE_SIZE,
                "full_rebuild_for_length_s_plus_1_ms": full,
                "incremental_new_page_ms": new_page,
                "incremental_existing_page_ms": existing_page,
                "full_to_new_page_median_ratio": (
                    float(full["median"]) / float(new_page["median"])
                ),
                "full_to_existing_page_median_ratio": (
                    float(full["median"]) / float(existing_page["median"])
                ),
                "checksum": full_checksum + new_page_checksum + existing_checksum,
            }
        )
    return results


def metadata_allocation_and_traffic() -> dict[str, Any]:
    batch_size = 1
    heads = 16
    head_dimension = 64
    sequence_length = PROMPT_LENGTH + DECODE_STEPS
    pages = math.ceil(sequence_length / QUEST_PAGE_SIZE)
    padded_length = pages * QUEST_PAGE_SIZE
    padding = padded_length - sequence_length
    scalar_bytes = torch.empty((), dtype=torch.float32).element_size()
    key_bytes = estimate_tensor_bytes(
        (batch_size, heads, sequence_length, head_dimension),
        torch.float32,
    )
    metadata_bytes = estimate_tensor_bytes(
        (batch_size, heads, pages, head_dimension),
        torch.float32,
    )
    token_bytes = estimate_tensor_bytes(
        (batch_size, heads, 1, head_dimension),
        torch.float32,
    )
    padding_bytes = padding * batch_size * heads * head_dimension * scalar_bytes
    padded_key_bytes = estimate_tensor_bytes(
        (batch_size, heads, padded_length, head_dimension),
        torch.float32,
    )
    old_total_traffic = 2 * key_bytes + 2 * metadata_bytes
    incremental_read = 2 * metadata_bytes + 4 * token_bytes
    incremental_write = 2 * metadata_bytes + 2 * token_bytes
    return {
        "representative_existing_page_append": {
            "sequence_length_after_append": sequence_length,
            "pages": pages,
            "persistent_metadata_bytes_both_extrema": 2 * metadata_bytes,
            "persistent_metadata_change_bytes": 0,
            "old_full_rebuild_temporary_allocations": {
                "two_padding_tensors_bytes": 2 * padding_bytes,
                "two_padded_key_inputs_bytes": 2 * padded_key_bytes,
                "replacement_metadata_outputs_bytes": 2 * metadata_bytes,
            },
            "new_incremental_temporary_and_replacement_allocations": {
                "replacement_metadata_outputs_bytes": 2 * metadata_bytes,
                "tail_extrema_results_bytes": 2 * token_bytes,
                "padded_key_inputs_bytes": 0,
            },
            "old_logical_tensor_traffic_bytes": {
                "read": 2 * key_bytes,
                "write": 2 * metadata_bytes,
                "total": old_total_traffic,
            },
            "new_logical_tensor_traffic_bytes": {
                "read": incremental_read,
                "write": incremental_write,
                "total": incremental_read + incremental_write,
            },
            "logical_traffic_reduction_fraction": (
                1.0 - (incremental_read + incremental_write) / old_total_traffic
            ),
            "qualification": (
                "analytical logical tensor bytes; the safe ownership model clones "
                "contiguous metadata and does not claim cache/allocator counters"
            ),
        },
        "new_page_append": {
            "behavior": (
                "two torch.cat operations read old min/max plus the new token and "
                "write replacement min/max tensors with one additional page"
            ),
            "persistent_metadata_growth_bytes": 2 * token_bytes,
            "padded_key_inputs_bytes": 0,
        },
    }


def _advance_to_final_incremental_step(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
) -> tuple[GPTNeoXDecodeState, torch.Tensor]:
    state = initialize_quest_state(
        runner,
        snapshot,
        budget_fraction=0.5,
        update_mode=QuestMetadataUpdateMode.INCREMENTAL,
    )
    for decode_step in range(1, DECODE_STEPS):
        runner.step(
            state,
            torch.tensor(
                [[dense_tokens[decode_step - 1]]],
                dtype=torch.int64,
                device=snapshot.input_ids.device,
            ),
        )
    return state, torch.tensor(
        [[dense_tokens[DECODE_STEPS - 1]]],
        dtype=torch.int64,
        device=snapshot.input_ids.device,
    )


def python_profile(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
) -> dict[str, Any]:
    state, input_token = _advance_to_final_incremental_step(
        runner,
        snapshot,
        dense_tokens,
    )
    profiler = cProfile.Profile()
    profiler.enable()
    runner.step(state, input_token)
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), values in stats.stats.items():
        primitive, total, self_time, cumulative, _ = values
        if filename.startswith("~"):
            continue
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}({function})",
                "primitive_call_count": primitive,
                "total_call_count": total,
                "self_time_ms": self_time * 1_000.0,
                "cumulative_time_ms": cumulative * 1_000.0,
            }
        )
    return {
        "profile_total_time_ms": stats.total_tt * 1_000.0,
        "top_python_functions_by_self_time": sorted(
            rows,
            key=lambda row: row["self_time_ms"],
            reverse=True,
        )[:30],
        "notes": "separate final-step replay; diagnostic and excluded from timings",
    }


def operator_profile(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
    profile_directory: Path,
) -> dict[str, Any]:
    state, input_token = _advance_to_final_incremental_step(
        runner,
        snapshot,
        dense_tokens,
    )
    recorder = ComponentProfiler(emit_operator_ranges=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as torch_profile:
        with (
            recorder.activate(),
            profile_context(
                phase="decode",
                fixture_id="technical_exposition",
                strategy="quest",
                budget_fraction=0.5,
                metadata_update_mode="incremental",
                decode_step=DECODE_STEPS,
            ),
        ):
            runner.step(state, input_token)
    profile_directory.mkdir(parents=True, exist_ok=True)
    trace_path = profile_directory / "quest-incremental-budget-50-step-31.json"
    torch_profile.export_chrome_trace(str(trace_path))
    events = [
        event
        for event in torch_profile.key_averages(group_by_input_shape=True)
        if event.key.startswith("aten::")
    ]
    events.sort(key=lambda event: event.self_cpu_time_total, reverse=True)
    return {
        "trace_path": str(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "top_operators_by_self_cpu_time": [
            {
                "operator": event.key,
                "call_count": event.count,
                "self_cpu_time_ms": event.self_cpu_time_total / 1_000.0,
                "total_cpu_time_ms": event.cpu_time_total / 1_000.0,
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "total_cpu_memory_bytes": event.cpu_memory_usage,
                "input_shapes": str(event.input_shapes),
            }
            for event in events[:30]
        ],
        "notes": "separate final-step replay; trace is gitignored and not committed",
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    except ImportError as error:
        raise RuntimeError(
            "install the model-experiment dependency for Phase 5A"
        ) from error
    if __version__ != DEFAULT_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {DEFAULT_TRANSFORMERS_VERSION}, found {__version__}"
        )
    phase4 = load_phase4_baseline(args.phase4_artifact)

    torch.manual_seed(SEED)
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    architecture = validate_gpt_neox_config(model.config)
    resolved_revision = getattr(model.config, "_commit_hash", DEFAULT_MODEL_REVISION)
    if resolved_revision != DEFAULT_MODEL_REVISION:
        raise RuntimeError("resolved model revision did not match the exact pin")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Phase 5A requires eager attention")

    fixtures = {fixture.fixture_id: fixture for fixture in TEXT_FIXTURES}
    runner = GPTNeoXDecodeRunner(model)
    recorder = ComponentProfiler()
    measured: list[dict[str, Any]] = []
    correctness = []
    dense_hf = []
    quality_summaries = []
    representative_snapshot: DensePrefillSnapshot | None = None
    representative_tokens: list[int] | None = None

    for fixture_id in FIXTURE_IDS:
        tokenized = build_deterministic_fixture(
            tokenizer,
            fixtures[fixture_id],
            PROMPT_LENGTH,
        )
        snapshot = runner.dense_prefill(tokenized.input_ids.to(device))
        warmup_tokens, _, warmup_dense_steps = build_dense_trace(
            runner,
            snapshot,
            generated_tokens=GENERATED_TOKEN_POSITIONS,
        )
        dense_tokens, dense_logits, dense_steps = build_dense_trace(
            runner,
            snapshot,
            generated_tokens=GENERATED_TOKEN_POSITIONS,
        )
        if dense_tokens != warmup_tokens:
            raise AssertionError("dense warmup changed generated tokens")
        for measured_step, warmup_step in zip(
            dense_steps,
            warmup_dense_steps,
            strict=True,
        ):
            if not torch.equal(
                measured_step.next_token_logits,
                warmup_step.next_token_logits,
            ):
                raise AssertionError("dense warmup changed logits")
        dense_hf.append(
            {
                "fixture_id": fixture_id,
                **validate_hugging_face_generation(
                    model,
                    snapshot.input_ids,
                    generated_tokens=GENERATED_TOKEN_POSITIONS,
                    custom_tokens=dense_tokens,
                    custom_logits=dense_logits,
                ),
            }
        )
        # Keep the accepted Phase 4 oracle order (50%, then 100%) before the
        # incremental cells. Correctness-only lockstep replays run afterward so
        # they cannot heat or pressure memory before primary timing.
        for update_mode in MODE_LABELS:
            for budget in BUDGET_FRACTIONS:
                _, warmup_steps = run_quest_trace(
                    runner,
                    snapshot,
                    dense_tokens,
                    budget_fraction=budget,
                    update_mode=update_mode,
                    fixture_id=fixture_id,
                )
                _, measured_steps = run_quest_trace(
                    runner,
                    snapshot,
                    dense_tokens,
                    budget_fraction=budget,
                    update_mode=update_mode,
                    fixture_id=fixture_id,
                    recorder=recorder,
                )
                for measured_step, warmup_step in zip(
                    measured_steps,
                    warmup_steps,
                    strict=True,
                ):
                    assert_decode_steps_bit_exact(measured_step, warmup_step)
                quality_summaries.extend(
                    quality_summary(
                        [
                            {
                                "fixture_id": fixture_id,
                                "budget_fraction": budget,
                                "metadata_update_mode": MODE_LABELS[update_mode],
                                "steps": measured_steps,
                            }
                        ],
                        {fixture_id: dense_steps},
                    )
                )
                measured.append(
                    {
                        "fixture_id": fixture_id,
                        "budget_fraction": budget,
                        "metadata_update_mode": MODE_LABELS[update_mode],
                        "step_timings": [
                            {
                                "layer_count": len(step.layers),
                                "total_layers_ms": sum(
                                    _layer_total_time_ms(step, layer)
                                    for layer in range(len(step.layers))
                                ),
                                "total_decode_step_ms": step.total_time_ms,
                            }
                            for step in measured_steps
                        ],
                    }
                )
        for budget in BUDGET_FRACTIONS:
            correctness.append(
                {
                    "fixture_id": fixture_id,
                    "budget_fraction": budget,
                    **run_lockstep_correctness(
                        runner,
                        snapshot,
                        dense_tokens,
                        dense_steps,
                        budget_fraction=budget,
                    ),
                }
            )
        print(f"{fixture_id}: correctness and measured paths passed", flush=True)
        if fixture_id == "technical_exposition":
            representative_snapshot = snapshot
            representative_tokens = dense_tokens

    if representative_snapshot is None or representative_tokens is None:
        raise AssertionError("representative Phase 5A fixture was not captured")
    step_records = build_step_records(recorder.records, measured)
    current_environment = {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": resolved_revision,
        "transformers_version": __version__,
        "transformers_source_revision": DEFAULT_TRANSFORMERS_REVISION,
        "transformers_attention_implementation": "eager",
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "dtype": "float32",
        "device": str(device),
        "git_commit": git_commit(),
        "git_dirty_before_result_write": git_is_dirty(),
        **machine_metadata(device),
    }
    comparison_fields = (
        "hardware",
        "cpu_brand",
        "physical_memory_bytes",
        "platform",
        "machine",
        "python_implementation",
        "torch_num_threads",
        "torch_num_interop_threads",
        "torch_version",
        "transformers_version",
        "model_revision",
    )
    environment_differences = {
        field: {
            "phase4": phase4["provenance"].get(field),
            "phase5a": current_environment.get(field),
        }
        for field in comparison_fields
        if phase4["provenance"].get(field) != current_environment.get(field)
    }
    component_summary = summarize_step_records(step_records)
    comparisons = paired_before_after(step_records)
    incremental_rows = [
        row for row in step_records if row["metadata_update_mode"] == "incremental"
    ]
    bottlenecks = []
    for component in (
        "metadata_maintenance",
        *RETRIEVAL_COMPONENT_GROUPS,
    ):
        values = [row["component_categories_ms"][component] for row in incremental_rows]
        bottlenecks.append(
            {
                "component": component,
                "milliseconds_per_decode_step": distribution(values),
            }
        )
    bottlenecks.sort(
        key=lambda row: row["milliseconds_per_decode_step"]["median"],
        reverse=True,
    )

    print("running synthetic scaling benchmark", flush=True)
    scaling = synthetic_scaling_benchmark()
    print("running diagnostic operator and Python profiles", flush=True)
    operator = operator_profile(
        runner,
        representative_snapshot,
        representative_tokens,
        args.profile_directory,
    )
    python = python_profile(
        runner,
        representative_snapshot,
        representative_tokens,
    )
    return {
        "schema_version": 1,
        "phase": "Phase 5A exact incremental Quest metadata maintenance",
        "status": "complete",
        "provenance": current_environment,
        "architecture": asdict(architecture),
        "phase4_frozen_baseline": phase4,
        "environment_differences_from_phase4": environment_differences,
        "protocol": {
            "prompt_length": PROMPT_LENGTH,
            "fixture_ids": list(FIXTURE_IDS),
            "generation_mode": "teacher_forced_only",
            "generated_token_positions": GENERATED_TOKEN_POSITIONS,
            "explicit_decode_steps": DECODE_STEPS,
            "budget_fractions": list(BUDGET_FRACTIONS),
            "quest_configuration": f"p{QUEST_PAGE_SIZE}",
            "seed": SEED,
            "warmup": "one complete uninstrumented 31-step replay per measured path",
            "measurement": (
                "one measured 31-step replay per fixture/budget/update mode; "
                "31 decode-step and 744 layer-step observations per fixture cell; "
                "summaries pool 62 and 1488 observations across two fixtures"
            ),
            "timing_clock": "time.perf_counter_ns named scopes plus runner wall time",
            "initial_prefill_metadata": "unchanged full build_page_metadata",
            "oracle": "unchanged full build_page_metadata after every append",
            "incremental_ownership": (
                "replacement QuestMetadata and replacement min/max tensors; prior "
                "metadata objects and tensors are not mutated"
            ),
            "shared_interface_changes": {
                "KVIndex": False,
                "Selection": False,
                "KVStorage": False,
                "RetrievedKV": False,
                "KVCache": False,
            },
            "public_root_api_changes": False,
        },
        "correctness": {
            "lockstep_full_rebuild_vs_incremental": correctness,
            "dense_vs_hugging_face": dense_hf,
            "full_budget_tolerance": {
                "rtol": FULL_BUDGET_RTOL,
                "atol": FULL_BUDGET_ATOL,
            },
            "quality_summary_by_fixture": quality_summaries,
            "quality_summary": pooled_quality_summary(quality_summaries),
        },
        "timing": {
            "step_component_summary": component_summary,
            "paired_before_after": comparisons,
            "incremental_metadata_breakdown": incremental_component_summary(
                recorder.records
            ),
            "post_optimization_retrieval_bottlenecks": bottlenecks,
        },
        "allocation_and_tensor_traffic": metadata_allocation_and_traffic(),
        "synthetic_metadata_scaling": {
            "sequence_lengths": list(SCALING_LENGTHS),
            "warmups": SCALING_WARMUPS,
            "repetitions": SCALING_REPETITIONS,
            "results": scaling,
            "qualification": (
                "synthetic eager-CPU reference scaling evidence, not a production "
                "performance claim"
            ),
        },
        "operator_profile": operator,
        "python_profile": python,
        "raw_component_timing_records": [
            record.as_dict() for record in recorder.records
        ],
        "limitations": [
            "one pinned 410M standard-MHA model and one CPU machine",
            "two deterministic fixtures at one prompt length",
            "31 teacher-forced decode steps per fixture and no free-running path",
            "safe replacement ownership still copies contiguous metadata tensors",
            "new-page metadata growth still uses torch.cat",
            "allocation and tensor-traffic figures are analytical, not hardware counters",
            "operator and cProfile replays are diagnostic and excluded from timings",
            "reference eager PyTorch results are not production speedup claims",
        ],
    }


def main() -> None:
    args = parse_args()
    artifact = run_experiment(args)
    write_json(args.output, artifact, overwrite=True, sort_keys=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
