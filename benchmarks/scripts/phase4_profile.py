#!/usr/bin/env python3
"""Profile the frozen Pythia-410M Phase 4 teacher-forced decode matrix."""

import argparse
import cProfile
from dataclasses import asdict
import hashlib
from pathlib import Path
import pstats
import statistics
import sys
import time
from typing import Any

import torch

from benchmarks.artifacts import write_json
from benchmarks.support import git_commit, git_is_dirty, machine_metadata
from benchmarks.phase3a import TEXT_FIXTURES, build_deterministic_fixture
from benchmarks.phase4 import (
    allocation_estimates,
    build_layer_step_records,
    build_step_records,
    compare_budgets,
    compare_dense_baseline,
    initialization_allocation_estimates,
    rank_retrieval_bottlenecks,
    retrieval_overhead_summary,
    summarize_step_components,
    tensor_traffic_estimates,
)
from benchmarks.decode import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TRANSFORMERS_REVISION,
    DEFAULT_TRANSFORMERS_VERSION,
    assert_full_budget_step,
    build_dense_trace,
    validate_hugging_face_generation,
)
from kvweave.integrations.transformers import (
    DecodeStrategy,
    DensePrefillSnapshot,
    GPTNeoXDecodeRunner,
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
    aggregate_component_timings,
    percentile,
    profile_component,
    profile_context,
)


DEFAULT_OUTPUT = Path("benchmarks/results/pythia-410m-phase4-profile.json")
DEFAULT_PROFILE_DIRECTORY = Path("benchmarks/results/profile/pythia-410m-phase4")
FIXTURE_IDS = ("technical_exposition", "code_like")
PROMPT_LENGTH = 1_024
GENERATED_TOKEN_POSITIONS = 32
APPROXIMATE_DECODE_STEPS = GENERATED_TOKEN_POSITIONS - 1
BUDGET_FRACTIONS = (0.5, 1.0)
QUEST_PAGE_SIZE = 64
PQ_SUBSPACES = 4
PQ_CENTROIDS = 8
PQ_ITERATIONS = 8
SEED = 0
FULL_BUDGET_RTOL = 1e-4
FULL_BUDGET_ATOL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profile-directory",
        type=Path,
        default=DEFAULT_PROFILE_DIRECTORY,
    )
    return parser.parse_args()


def _layer_total_time_ms(step: GPTNeoXDecodeStep, layer_index: int) -> float:
    layer = step.layers[layer_index]
    return (
        layer.index_update_time_ms
        + layer.retrieval_time_ms
        + layer.storage_fetch_time_ms
        + layer.selected_attention_time_ms
        + layer.remaining_layer_time_ms
    )


def selection_digest(steps: list[GPTNeoXDecodeStep]) -> str | None:
    digest = hashlib.sha256()
    selected = False
    for step in steps:
        for layer in step.layers:
            if layer.selection is None:
                continue
            selected = True
            digest.update(layer.selection.indices.detach().cpu().contiguous().numpy())
            if layer.selection.scores is not None:
                digest.update(
                    layer.selection.scores.detach().cpu().contiguous().numpy()
                )
            if layer.selection.valid_mask is not None:
                digest.update(
                    layer.selection.valid_mask.detach().cpu().contiguous().numpy()
                )
    return digest.hexdigest() if selected else None


def assert_profiled_semantics_unchanged(
    profiled_steps: list[GPTNeoXDecodeStep],
    plain_steps: list[GPTNeoXDecodeStep],
) -> dict[str, Any]:
    """Require bit-exact outputs and selections from profiled/plain replays."""
    if len(profiled_steps) != len(plain_steps):
        raise AssertionError("profiled and plain paths have different step counts")
    selection_rows = 0
    for profiled, plain in zip(profiled_steps, plain_steps, strict=True):
        if not torch.equal(profiled.input_token, plain.input_token):
            raise AssertionError("profiling changed a decode input token")
        if not torch.equal(profiled.next_token_logits, plain.next_token_logits):
            raise AssertionError("profiling changed next-token logits")
        if not torch.equal(profiled.next_token, plain.next_token):
            raise AssertionError("profiling changed a greedy token")
        for profiled_layer, plain_layer in zip(
            profiled.layers,
            plain.layers,
            strict=True,
        ):
            for name in ("query", "attention_output", "residual_output"):
                if not torch.equal(
                    getattr(profiled_layer, name),
                    getattr(plain_layer, name),
                ):
                    raise AssertionError(f"profiling changed layer {name}")
            if (profiled_layer.selection is None) != (plain_layer.selection is None):
                raise AssertionError("profiling changed selection presence")
            if profiled_layer.selection is not None:
                selection_rows += (
                    profiled_layer.selection.indices.shape[0]
                    * (profiled_layer.selection.indices.shape[1])
                )
                for name in ("indices", "scores", "valid_mask"):
                    profiled_value = getattr(profiled_layer.selection, name)
                    plain_value = getattr(plain_layer.selection, name)
                    if (profiled_value is None) != (plain_value is None):
                        raise AssertionError(f"profiling changed selection {name}")
                    if profiled_value is not None and not torch.equal(
                        profiled_value,
                        plain_value,
                    ):
                        raise AssertionError(f"profiling changed selection {name}")
    profiled_digest = selection_digest(profiled_steps)
    plain_digest = selection_digest(plain_steps)
    if profiled_digest != plain_digest:
        raise AssertionError("profiled and plain selection digests differ")
    return {
        "passed": True,
        "decode_steps_compared": len(profiled_steps),
        "layer_steps_compared": sum(len(step.layers) for step in profiled_steps),
        "selection_rows_compared": selection_rows,
        "selection_sha256": profiled_digest,
        "logits_bit_exact": True,
        "queries_bit_exact": True,
        "attention_outputs_bit_exact": True,
        "residual_streams_bit_exact": True,
        "selection_ids_scores_masks_bit_exact": True,
    }


def run_profiled_dense_trace(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    recorder: ComponentProfiler,
    *,
    fixture_id: str,
) -> tuple[list[int], list[torch.Tensor], list[GPTNeoXDecodeStep]]:
    state = runner.initialize_state(snapshot, strategy=DecodeStrategy.DENSE)
    generated = [int(snapshot.next_token_logits.argmax(dim=-1).item())]
    logits = [snapshot.next_token_logits]
    steps: list[GPTNeoXDecodeStep] = []
    for decode_step in range(1, GENERATED_TOKEN_POSITIONS):
        input_token = torch.tensor(
            [[generated[-1]]],
            dtype=torch.int64,
            device=snapshot.input_ids.device,
        )
        with (
            recorder.activate(),
            profile_context(
                fixture_id=fixture_id,
                strategy=DecodeStrategy.DENSE.value,
                budget_fraction=1.0,
                decode_step=decode_step,
            ),
        ):
            step = runner.step(state, input_token)
        steps.append(step)
        logits.append(step.next_token_logits)
        generated.append(int(step.next_token.item()))
    return generated, logits, steps


def run_teacher_forced_trace(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
    *,
    strategy: DecodeStrategy,
    budget_fraction: float,
    recorder: ComponentProfiler | None,
    fixture_id: str,
    record_initialization: bool,
) -> tuple[Any, list[GPTNeoXDecodeStep]]:
    initialization_context = profile_context(
        fixture_id=fixture_id,
        strategy=strategy.value,
        budget_fraction=budget_fraction,
    )
    if recorder is not None and record_initialization:
        with recorder.activate(), initialization_context:
            state = runner.initialize_state(
                snapshot,
                strategy=strategy,
                budget_fraction=budget_fraction,
                quest_page_size=QUEST_PAGE_SIZE,
                pq_num_subspaces=PQ_SUBSPACES,
                pq_num_centroids=PQ_CENTROIDS,
                pq_max_iterations=PQ_ITERATIONS,
                seed=SEED,
                quest_metadata_update_mode=QuestMetadataUpdateMode.FULL_REBUILD,
            )
    else:
        state = runner.initialize_state(
            snapshot,
            strategy=strategy,
            budget_fraction=budget_fraction,
            quest_page_size=QUEST_PAGE_SIZE,
            pq_num_subspaces=PQ_SUBSPACES,
            pq_num_centroids=PQ_CENTROIDS,
            pq_max_iterations=PQ_ITERATIONS,
            seed=SEED,
            quest_metadata_update_mode=QuestMetadataUpdateMode.FULL_REBUILD,
        )

    steps: list[GPTNeoXDecodeStep] = []
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
                    fixture_id=fixture_id,
                    strategy=strategy.value,
                    budget_fraction=budget_fraction,
                    decode_step=decode_step,
                ),
            ):
                step = runner.step(state, input_token)
        steps.append(step)
    return state, steps


def add_coarse_records(
    *,
    fixture_id: str,
    strategy: str,
    budget_fraction: float,
    steps: list[GPTNeoXDecodeStep],
    coarse_records: list[dict[str, Any]],
    step_wall_times: dict[tuple[str, str, float, int], float],
) -> None:
    for decode_step, step in enumerate(steps, start=1):
        step_wall_times[(fixture_id, strategy, budget_fraction, decode_step)] = (
            step.total_time_ms
        )
        for layer in step.layers:
            selection_width = (
                layer.sequence_length
                if layer.selection is None
                else layer.selection.indices.shape[-1]
            )
            coarse_records.append(
                {
                    "fixture_id": fixture_id,
                    "strategy": strategy,
                    "budget_fraction": budget_fraction,
                    "decode_step": decode_step,
                    "layer": layer.layer_index,
                    "sequence_length": layer.sequence_length,
                    "selection_width": selection_width,
                    "mean_selected_tokens_per_head": float(
                        layer.selected_token_counts.float().mean().item()
                    ),
                    "total_layer_time_ms": _layer_total_time_ms(
                        step,
                        layer.layer_index,
                    ),
                    "coarse_index_update_time_ms": layer.index_update_time_ms,
                    "coarse_retrieval_time_ms": layer.retrieval_time_ms,
                    "coarse_storage_fetch_time_ms": layer.storage_fetch_time_ms,
                    "coarse_attention_time_ms": layer.selected_attention_time_ms,
                    "coarse_remaining_layer_time_ms": (layer.remaining_layer_time_ms),
                }
            )


def quality_records(
    *,
    fixture_id: str,
    strategy: str,
    budget_fraction: float,
    approximate_steps: list[GPTNeoXDecodeStep],
    dense_steps: list[GPTNeoXDecodeStep],
) -> list[dict[str, Any]]:
    rows = []
    for decode_step, (approximate, dense) in enumerate(
        zip(approximate_steps, dense_steps, strict=True),
        start=1,
    ):
        layer_mass = []
        layer_error = []
        residual_errors = []
        for approximate_layer, dense_layer in zip(
            approximate.layers,
            dense.layers,
            strict=True,
        ):
            if approximate_layer.selection is None:
                raise AssertionError("approximate path omitted its selection")
            layer_mass.extend(
                attention_mass_captured(
                    dense_layer.attention_weights,
                    approximate_layer.selection,
                )[0]
                .float()
                .tolist()
            )
            layer_error.extend(
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
                "fixture_id": fixture_id,
                "strategy": strategy,
                "budget_fraction": budget_fraction,
                "decode_step": decode_step,
                **logit_comparison_metrics(
                    approximate.next_token_logits,
                    dense.next_token_logits,
                ),
                "mean_attention_mass_captured": statistics.fmean(layer_mass),
                "mean_attention_output_relative_error": statistics.fmean(layer_error),
                "mean_residual_stream_relative_error": statistics.fmean(
                    residual_errors
                ),
            }
        )
    return rows


def summarize_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for strategy in ("quest", "pq"):
        for budget in BUDGET_FRACTIONS:
            selected = [
                row
                for row in rows
                if row["strategy"] == strategy and row["budget_fraction"] == budget
            ]
            output.append(
                {
                    "strategy": strategy,
                    "budget_fraction": budget,
                    "decode_step_count": len(selected),
                    "top_1_agreement_rate": statistics.fmean(
                        float(row["top_1_agreement"]) for row in selected
                    ),
                    "mean_top_5_overlap_fraction": statistics.fmean(
                        row["top_5_overlap_fraction"] for row in selected
                    ),
                    "mean_logit_relative_error": statistics.fmean(
                        row["logit_relative_error"] for row in selected
                    ),
                    "mean_kl_divergence": statistics.fmean(
                        row["kl_divergence_dense_to_approximate"] for row in selected
                    ),
                    "mean_attention_mass_captured": statistics.fmean(
                        row["mean_attention_mass_captured"] for row in selected
                    ),
                    "mean_attention_output_relative_error": statistics.fmean(
                        row["mean_attention_output_relative_error"] for row in selected
                    ),
                    "mean_residual_stream_relative_error": statistics.fmean(
                        row["mean_residual_stream_relative_error"] for row in selected
                    ),
                }
            )
    return output


def analyze_initialization(
    records: list[ComponentTiming],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    atomic: dict[tuple[str, str, int], dict[str, float]] = {}
    grouped: dict[tuple[str, str, int], dict[str, float]] = {}
    for record in records:
        if record.context.get("phase") != "initialization":
            continue
        key = (
            str(record.context["fixture_id"]),
            str(record.context["strategy"]),
            int(record.context["layer"]),
        )
        atomic.setdefault(key, {})
        atomic[key][record.component] = (
            atomic[key].get(record.component, 0.0) + record.duration_ms
        )

    layer_rows = []
    for run in runs:
        for layer, total in enumerate(run["initial_index_build_time_ms_by_layer"]):
            key = (run["fixture_id"], run["strategy"], layer)
            values = atomic[key]
            if run["strategy"] == "quest":
                categories = {
                    "page_reshape_padding": values.get(
                        "quest.metadata.page_reshape_padding", 0.0
                    ),
                    "page_minimum": values.get("quest.metadata.page_minimum", 0.0),
                    "page_maximum": values.get("quest.metadata.page_maximum", 0.0),
                    "metadata_object_construction": values.get(
                        "quest.metadata.object_construction", 0.0
                    ),
                }
            else:
                categories = {
                    "codebook_input_and_allocation": values.get(
                        "pq.init.subspace_split", 0.0
                    )
                    + values.get("pq.init.codebook_allocation", 0.0),
                    "kmeans_training": values.get("pq.init.kmeans_training", 0.0),
                    "prefill_encoding": sum(
                        duration
                        for component, duration in values.items()
                        if component.startswith("pq.encode.")
                    ),
                    "initial_code_storage": values.get(
                        "pq.init.initial_code_storage", 0.0
                    ),
                }
            categories["miscellaneous_initialization"] = max(
                0.0,
                float(total) - sum(categories.values()),
            )
            grouped[key] = categories
            layer_rows.append(
                {
                    "fixture_id": key[0],
                    "strategy": key[1],
                    "layer": key[2],
                    "total_initialization_ms": total,
                    "components_ms": categories,
                }
            )

    summaries = []
    for strategy in ("quest", "pq"):
        strategy_rows = [row for row in layer_rows if row["strategy"] == strategy]
        component_names = sorted(
            {name for row in strategy_rows for name in row["components_ms"]}
        )
        for component in component_names:
            values = [row["components_ms"][component] for row in strategy_rows]
            summaries.append(
                {
                    "strategy": strategy,
                    "component": component,
                    "layer_call_count": len(values),
                    "median_ms_per_layer": statistics.median(values),
                    "p90_ms_per_layer": percentile(values, 0.90),
                    "p95_ms_per_layer": percentile(values, 0.95),
                    "total_ms_across_measured_layers": sum(values),
                }
            )
    total_summary = []
    for strategy in ("quest", "pq"):
        totals = [
            run["total_initialization_ms"]
            for run in runs
            if run["strategy"] == strategy
        ]
        per_layer = [
            value
            for run in runs
            if run["strategy"] == strategy
            for value in run["initial_index_build_time_ms_by_layer"]
        ]
        total_summary.append(
            {
                "strategy": strategy,
                "fixture_run_count": len(totals),
                "median_total_ms_all_layers": statistics.median(totals),
                "minimum_total_ms_all_layers": min(totals),
                "maximum_total_ms_all_layers": max(totals),
                "median_ms_per_layer": statistics.median(per_layer),
                "p90_ms_per_layer": percentile(per_layer, 0.90),
                "p95_ms_per_layer": percentile(per_layer, 0.95),
            }
        )
    return {
        "runs": runs,
        "total_summary": total_summary,
        "layer_records": layer_rows,
        "component_summary": summaries,
        "atomic_component_summary": aggregate_component_timings(
            [
                record
                for record in records
                if record.context.get("phase") == "initialization"
            ],
            group_fields=("strategy", "layer"),
        ),
    }


def measure_scope_overhead(iterations: int = 10_000) -> dict[str, Any]:
    recorder = ComponentProfiler()
    durations = []
    with recorder.activate(), profile_context(phase="calibration"):
        for _ in range(iterations):
            start = time.perf_counter_ns()
            with profile_component("calibration.empty_scope"):
                pass
            durations.append((time.perf_counter_ns() - start) / 1_000.0)
    return {
        "iterations": iterations,
        "median_total_scope_overhead_microseconds": statistics.median(durations),
        "p95_total_scope_overhead_microseconds": percentile(durations, 0.95),
        "median_recorded_empty_body_microseconds": 1_000
        * statistics.median(record.duration_ms for record in recorder.records),
        "notes": (
            "calibration includes context lookup, two perf-counter reads, and "
            "record append; it is not subtracted from measurements"
        ),
    }


def _advance_to_representative_step(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
    *,
    strategy: DecodeStrategy,
    budget_fraction: float,
) -> tuple[Any, torch.Tensor]:
    state = runner.initialize_state(
        snapshot,
        strategy=strategy,
        budget_fraction=budget_fraction,
        quest_page_size=QUEST_PAGE_SIZE,
        pq_num_subspaces=PQ_SUBSPACES,
        pq_num_centroids=PQ_CENTROIDS,
        pq_max_iterations=PQ_ITERATIONS,
        seed=SEED,
        quest_metadata_update_mode=QuestMetadataUpdateMode.FULL_REBUILD,
    )
    for decode_step in range(1, APPROXIMATE_DECODE_STEPS):
        input_token = torch.tensor(
            [[dense_tokens[decode_step - 1]]],
            dtype=torch.int64,
            device=snapshot.input_ids.device,
        )
        runner.step(state, input_token)
    final_input = torch.tensor(
        [[dense_tokens[APPROXIMATE_DECODE_STEPS - 1]]],
        dtype=torch.int64,
        device=snapshot.input_ids.device,
    )
    return state, final_input


def _operator_rows(
    events: Any, *, sort_field: str, limit: int = 25
) -> list[dict[str, Any]]:
    selected = [event for event in events if event.key.startswith("aten::")]
    selected.sort(key=lambda event: getattr(event, sort_field), reverse=True)
    return [
        {
            "operator": event.key,
            "call_count": event.count,
            "self_cpu_time_ms": event.self_cpu_time_total / 1_000.0,
            "total_cpu_time_ms": event.cpu_time_total / 1_000.0,
            "self_cpu_memory_bytes": event.self_cpu_memory_usage,
            "total_cpu_memory_bytes": event.cpu_memory_usage,
            "input_shapes": str(event.input_shapes),
        }
        for event in selected[:limit]
    ]


def operator_profile(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
    *,
    strategy: DecodeStrategy,
    budget_fraction: float,
    profile_directory: Path,
) -> dict[str, Any]:
    state, input_token = _advance_to_representative_step(
        runner,
        snapshot,
        dense_tokens,
        strategy=strategy,
        budget_fraction=budget_fraction,
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
                fixture_id="technical_exposition",
                strategy=strategy.value,
                budget_fraction=budget_fraction,
                decode_step=APPROXIMATE_DECODE_STEPS,
                profile_kind="operator",
            ),
        ):
            runner.step(state, input_token)
    profile_directory.mkdir(parents=True, exist_ok=True)
    trace_path = profile_directory / (
        f"{strategy.value}-budget-{int(budget_fraction * 100)}-step-31.json"
    )
    torch_profile.export_chrome_trace(str(trace_path))
    events = torch_profile.key_averages(group_by_input_shape=True)
    return {
        "strategy": strategy.value,
        "budget_fraction": budget_fraction,
        "fixture_id": "technical_exposition",
        "decode_step": APPROXIMATE_DECODE_STEPS,
        "sequence_length": PROMPT_LENGTH + APPROXIMATE_DECODE_STEPS,
        "trace_path": str(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "top_operators_by_self_cpu_time": _operator_rows(
            events,
            sort_field="self_cpu_time_total",
        ),
        "top_operators_by_total_cpu_time": _operator_rows(
            events,
            sort_field="cpu_time_total",
        ),
        "top_operators_by_self_cpu_memory": _operator_rows(
            events,
            sort_field="self_cpu_memory_usage",
        ),
        "notes": (
            "separate replay of the final steady-state decode step; profiler "
            "overhead is excluded from primary wall-time distributions"
        ),
    }


def python_profile(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
    *,
    strategy: DecodeStrategy,
    budget_fraction: float,
) -> dict[str, Any]:
    state, input_token = _advance_to_representative_step(
        runner,
        snapshot,
        dense_tokens,
        strategy=strategy,
        budget_fraction=budget_fraction,
    )
    profiler = cProfile.Profile()
    profiler.enable()
    runner.step(state, input_token)
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), (
        primitive,
        total,
        self_time,
        cumulative,
        _,
    ) in stats.stats.items():
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}({function})",
                "primitive_call_count": primitive,
                "total_call_count": total,
                "self_time_ms": self_time * 1_000.0,
                "cumulative_time_ms": cumulative * 1_000.0,
                "is_python_source": not filename.startswith("~"),
            }
        )
    by_cumulative = sorted(
        rows,
        key=lambda row: row["cumulative_time_ms"],
        reverse=True,
    )[:30]
    by_self = sorted(rows, key=lambda row: row["self_time_ms"], reverse=True)[:30]
    python_rows = [row for row in rows if row["is_python_source"]]
    python_rows.sort(key=lambda row: row["self_time_ms"], reverse=True)
    return {
        "strategy": strategy.value,
        "budget_fraction": budget_fraction,
        "fixture_id": "technical_exposition",
        "decode_step": APPROXIMATE_DECODE_STEPS,
        "sequence_length": PROMPT_LENGTH + APPROXIMATE_DECODE_STEPS,
        "profile_total_time_ms": stats.total_tt * 1_000.0,
        "top_functions_by_cumulative_time": by_cumulative,
        "top_functions_by_self_time": by_self,
        "top_python_source_functions_by_self_time": python_rows[:30],
        "notes": (
            "separate replay; cProfile attributes PyTorch dispatch to builtin "
            "calls and cannot cleanly partition asynchronous/native internals"
        ),
    }


def accounting_artifacts(
    coarse_records: list[dict[str, Any]],
    *,
    architecture: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    representative = [
        row
        for row in coarse_records
        if row["fixture_id"] == "technical_exposition"
        and row["decode_step"] == APPROXIMATE_DECODE_STEPS
        and row["layer"] == architecture.num_hidden_layers - 1
    ]
    allocations = []
    traffic = []
    for row in representative:
        context = {
            "strategy": row["strategy"],
            "budget_fraction": row["budget_fraction"],
            "sequence_length": row["sequence_length"],
            "representative_fixture": row["fixture_id"],
            "representative_decode_step": row["decode_step"],
            "representative_layer": row["layer"],
        }
        for estimate in allocation_estimates(
            strategy=row["strategy"],
            sequence_length=row["sequence_length"],
            budget_fraction=row["budget_fraction"],
            batch_size=1,
            heads=architecture.num_attention_heads,
            head_dimension=architecture.head_dimension,
            selection_width=row["selection_width"],
            page_size=QUEST_PAGE_SIZE,
            pq_subspaces=PQ_SUBSPACES,
            pq_centroids=PQ_CENTROIDS,
        ):
            allocations.append({**context, **estimate})
        for estimate in tensor_traffic_estimates(
            strategy=row["strategy"],
            sequence_length=row["sequence_length"],
            batch_size=1,
            heads=architecture.num_attention_heads,
            head_dimension=architecture.head_dimension,
            selection_width=row["selection_width"],
            page_size=QUEST_PAGE_SIZE,
            pq_subspaces=PQ_SUBSPACES,
            pq_centroids=PQ_CENTROIDS,
        ):
            traffic.append({**context, **estimate})
    initialization_allocations = []
    for strategy in ("quest", "pq"):
        for estimate in initialization_allocation_estimates(
            strategy=strategy,
            sequence_length=PROMPT_LENGTH,
            batch_size=1,
            heads=architecture.num_attention_heads,
            head_dimension=architecture.head_dimension,
            page_size=QUEST_PAGE_SIZE,
            pq_subspaces=PQ_SUBSPACES,
            pq_centroids=PQ_CENTROIDS,
        ):
            initialization_allocations.append(
                {
                    "strategy": strategy,
                    "sequence_length": PROMPT_LENGTH,
                    **estimate,
                }
            )
    return allocations, initialization_allocations, traffic


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    except ImportError as error:
        raise RuntimeError(
            "install the optional model experiment dependency before Phase 4"
        ) from error
    if __version__ != DEFAULT_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {DEFAULT_TRANSFORMERS_VERSION}, found {__version__}"
        )

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
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    if attention_implementation != "eager":
        raise RuntimeError("Phase 4 requires eager attention")

    fixtures = {fixture.fixture_id: fixture for fixture in TEXT_FIXTURES}
    if any(fixture_id not in fixtures for fixture_id in FIXTURE_IDS):
        raise RuntimeError("frozen Phase 4 fixtures are unavailable")
    runner = GPTNeoXDecodeRunner(model)
    component_recorder = ComponentProfiler()
    coarse_records: list[dict[str, Any]] = []
    step_wall_times: dict[tuple[str, str, float, int], float] = {}
    initialization_runs = []
    instrumentation_checks = []
    dense_hf_checks = []
    quality_rows: list[dict[str, Any]] = []
    dense_case_rows = []
    representative_input_ids: torch.Tensor | None = None
    representative_dense_tokens: list[int] | None = None

    for fixture_index, fixture_id in enumerate(FIXTURE_IDS, start=1):
        fixture = fixtures[fixture_id]
        tokenized = build_deterministic_fixture(
            tokenizer,
            fixture,
            PROMPT_LENGTH,
        )
        input_ids = tokenized.input_ids.to(device)
        snapshot = runner.dense_prefill(input_ids)

        # One complete uninstrumented replay warms every layer/operator and is
        # also the semantic control for the instrumented dense measurement.
        plain_dense_tokens, _, plain_dense_steps = build_dense_trace(
            runner,
            snapshot,
            generated_tokens=GENERATED_TOKEN_POSITIONS,
        )
        dense_tokens, dense_logits, dense_steps = run_profiled_dense_trace(
            runner,
            snapshot,
            component_recorder,
            fixture_id=fixture_id,
        )
        dense_instrumentation = assert_profiled_semantics_unchanged(
            dense_steps,
            plain_dense_steps,
        )
        dense_instrumentation.update(
            {
                "fixture_id": fixture_id,
                "strategy": "dense",
                "budget_fraction": 1.0,
                "warmup_decode_runs": 1,
                "warmup_decode_steps": APPROXIMATE_DECODE_STEPS,
            }
        )
        instrumentation_checks.append(dense_instrumentation)
        if dense_tokens != plain_dense_tokens:
            raise AssertionError("profiled dense tokens changed after warmup replay")
        hf_check = validate_hugging_face_generation(
            model,
            input_ids,
            generated_tokens=GENERATED_TOKEN_POSITIONS,
            custom_tokens=dense_tokens,
            custom_logits=dense_logits,
        )
        hf_check["fixture_id"] = fixture_id
        dense_hf_checks.append(hf_check)
        add_coarse_records(
            fixture_id=fixture_id,
            strategy="dense",
            budget_fraction=1.0,
            steps=dense_steps,
            coarse_records=coarse_records,
            step_wall_times=step_wall_times,
        )
        dense_case_rows.append(
            {
                "fixture_id": fixture_id,
                "prompt_length": PROMPT_LENGTH,
                "token_ids_sha256": tokenized.token_ids_sha256,
                "prefill_time_ms": snapshot.prefill_time_ms,
                "generated_token_ids": dense_tokens,
                "median_dense_decode_step_ms": statistics.median(
                    step.total_time_ms for step in dense_steps
                ),
            }
        )
        print(
            f"fixture {fixture_index}/{len(FIXTURE_IDS)} {fixture_id}: "
            "dense warmup/profile/HF gates passed",
            flush=True,
        )

        for strategy in (DecodeStrategy.QUEST, DecodeStrategy.PQ):
            for budget_fraction in BUDGET_FRACTIONS:
                # Warmup is a full 31-step uninstrumented teacher-forced trace.
                _, plain_steps = run_teacher_forced_trace(
                    runner,
                    snapshot,
                    dense_tokens,
                    strategy=strategy,
                    budget_fraction=budget_fraction,
                    recorder=None,
                    fixture_id=fixture_id,
                    record_initialization=False,
                )
                state, profiled_steps = run_teacher_forced_trace(
                    runner,
                    snapshot,
                    dense_tokens,
                    strategy=strategy,
                    budget_fraction=budget_fraction,
                    recorder=component_recorder,
                    fixture_id=fixture_id,
                    record_initialization=budget_fraction == 0.5,
                )
                check = assert_profiled_semantics_unchanged(
                    profiled_steps,
                    plain_steps,
                )
                check.update(
                    {
                        "fixture_id": fixture_id,
                        "strategy": strategy.value,
                        "budget_fraction": budget_fraction,
                        "warmup_decode_runs": 1,
                        "warmup_decode_steps": APPROXIMATE_DECODE_STEPS,
                    }
                )
                instrumentation_checks.append(check)
                if budget_fraction == 1.0:
                    for approximate, dense in zip(
                        profiled_steps,
                        dense_steps,
                        strict=True,
                    ):
                        assert_full_budget_step(
                            approximate,
                            dense,
                            rtol=FULL_BUDGET_RTOL,
                            atol=FULL_BUDGET_ATOL,
                        )
                if budget_fraction == 0.5:
                    initialization_runs.append(
                        {
                            "fixture_id": fixture_id,
                            "strategy": strategy.value,
                            "configuration": (
                                f"p{QUEST_PAGE_SIZE}"
                                if strategy is DecodeStrategy.QUEST
                                else f"M{PQ_SUBSPACES}/C{PQ_CENTROIDS}"
                            ),
                            "total_initialization_ms": sum(
                                layer.initial_index_build_time_ms
                                for layer in state.layers
                            ),
                            "initial_index_build_time_ms_by_layer": [
                                layer.initial_index_build_time_ms
                                for layer in state.layers
                            ],
                        }
                    )
                add_coarse_records(
                    fixture_id=fixture_id,
                    strategy=strategy.value,
                    budget_fraction=budget_fraction,
                    steps=profiled_steps,
                    coarse_records=coarse_records,
                    step_wall_times=step_wall_times,
                )
                quality_rows.extend(
                    quality_records(
                        fixture_id=fixture_id,
                        strategy=strategy.value,
                        budget_fraction=budget_fraction,
                        approximate_steps=profiled_steps,
                        dense_steps=dense_steps,
                    )
                )
                print(
                    f"  {strategy.value} budget={budget_fraction:g}: "
                    "warmup/profile/semantic gates passed",
                    flush=True,
                )

        if fixture_id == "technical_exposition":
            representative_input_ids = input_ids.detach().clone()
            representative_dense_tokens = list(dense_tokens)

    layer_step_records = build_layer_step_records(
        component_recorder.records,
        coarse_records,
    )
    step_records = build_step_records(layer_step_records, step_wall_times)
    allocations, initialization_allocations, traffic = accounting_artifacts(
        coarse_records,
        architecture=architecture,
    )
    if representative_input_ids is None or representative_dense_tokens is None:
        raise AssertionError("representative fixture was not captured")
    representative_snapshot = runner.dense_prefill(representative_input_ids)
    operator_profiles = []
    python_profiles = []
    for strategy, budget in (
        (DecodeStrategy.DENSE, 1.0),
        (DecodeStrategy.QUEST, 0.5),
        (DecodeStrategy.PQ, 0.5),
    ):
        print(f"operator profile: {strategy.value} budget={budget:g}", flush=True)
        operator_profiles.append(
            operator_profile(
                runner,
                representative_snapshot,
                representative_dense_tokens,
                strategy=strategy,
                budget_fraction=budget,
                profile_directory=args.profile_directory,
            )
        )
        python_profiles.append(
            python_profile(
                runner,
                representative_snapshot,
                representative_dense_tokens,
                strategy=strategy,
                budget_fraction=budget,
            )
        )

    raw_timing_records = [
        record.as_dict()
        for record in component_recorder.records
        if record.context.get("phase") in ("decode", "initialization")
    ]
    artifact = {
        "schema_version": 1,
        "phase": "Phase 4 profiling",
        "status": "complete",
        "provenance": {
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": resolved_revision,
            "transformers_version": __version__,
            "transformers_source_revision": DEFAULT_TRANSFORMERS_REVISION,
            "transformers_attention_implementation": attention_implementation,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "dtype": "float32",
            "device": str(device),
            "git_commit": git_commit(),
            "git_dirty_before_result_write": git_is_dirty(),
            **machine_metadata(device),
        },
        "architecture": asdict(architecture),
        "protocol": {
            "prompt_length": PROMPT_LENGTH,
            "fixture_ids": list(FIXTURE_IDS),
            "generation_mode": "teacher_forced_only",
            "generated_token_positions": GENERATED_TOKEN_POSITIONS,
            "approximate_decode_steps": APPROXIMATE_DECODE_STEPS,
            "first_generated_token_source": "dense prefill final-position logits",
            "strategies": ["dense", "quest", "pq"],
            "budget_fractions": list(BUDGET_FRACTIONS),
            "quest_configuration": f"p{QUEST_PAGE_SIZE}",
            "pq_configuration": f"M{PQ_SUBSPACES}/C{PQ_CENTROIDS}",
            "pq_iterations": PQ_ITERATIONS,
            "seed": SEED,
            "execution": "CPU float32 eager/reference",
            "warmup": (
                "one complete uninstrumented 31-step replay before each measured "
                "path; this also serves as the instrumentation semantic control"
            ),
            "measurement": (
                "one measured 31-step replay per fixture/strategy/budget, yielding "
                "62 decode-step and 1488 layer-step observations per cell"
            ),
            "timing_clock": "time.perf_counter_ns wall clock around existing ops",
            "synchronization": (
                "no CPU synchronization call; CUDA/MPS branches exist in the "
                "unchanged reference timer but are inactive"
            ),
            "operator_profile": (
                "separate PyTorch-profiler replay of technical_exposition final "
                "steady-state step for dense, Quest 50%, and PQ 50%"
            ),
            "python_profile": (
                "separate cProfile replay of the same representative steps"
            ),
            "initialization_accounting": (
                "measured once per fixture/strategy before the 50% path; excluded "
                "from steady-state decode distributions"
            ),
            "shared_interface_changes": {
                "KVIndex": False,
                "Selection": False,
                "KVStorage": False,
                "RetrievedKV": False,
                "KVCache": False,
            },
            "algorithm_expression_changes": False,
            "instrumentation_refactor": (
                "named scopes and intermediate variable binding only; stable "
                "argsort, tensor shapes, ranking, policy, fetch, and attention "
                "expressions are unchanged"
            ),
            "scope_overhead_calibration": measure_scope_overhead(),
        },
        "correctness": {
            "instrumented_vs_uninstrumented": instrumentation_checks,
            "dense_vs_hugging_face": dense_hf_checks,
            "full_budget_controls": {
                "quest": "passed every fixture/step/layer",
                "pq": "passed every fixture/step/layer",
                "rtol": FULL_BUDGET_RTOL,
                "atol": FULL_BUDGET_ATOL,
            },
            "partial_quality_summary": summarize_quality(quality_rows),
        },
        "dense_cases": dense_case_rows,
        "initialization": analyze_initialization(
            component_recorder.records,
            initialization_runs,
        ),
        "steady_state": {
            "raw_component_timing_records": raw_timing_records,
            "atomic_component_summary": aggregate_component_timings(
                [
                    record
                    for record in component_recorder.records
                    if record.context.get("phase") == "decode"
                ],
                group_fields=("strategy", "budget_fraction", "component"),
            ),
            "layer_step_records": layer_step_records,
            "step_records": step_records,
            "step_component_summary": summarize_step_components(step_records),
            "retrieval_overhead_summary": retrieval_overhead_summary(step_records),
            "budget_comparison": compare_budgets(step_records),
            "dense_baseline_comparison": compare_dense_baseline(step_records),
        },
        "allocation_accounting": {
            "steady_state_representative_final_step": allocations,
            "initialization": initialization_allocations,
            "profiler_memory_operator_summary": [
                {
                    "strategy": profile["strategy"],
                    "budget_fraction": profile["budget_fraction"],
                    "operators": profile["top_operators_by_self_cpu_memory"],
                }
                for profile in operator_profiles
            ],
        },
        "tensor_traffic_accounting": traffic,
        "operator_profiles": operator_profiles,
        "python_profiles": python_profiles,
        "bottleneck_analysis": {
            "ranked_top_three": rank_retrieval_bottlenecks(step_records),
            "selected_targets": {
                "quest": {
                    "target": "metadata_rebuild",
                    "selection_rule": (
                        "largest repeated median retrieval component across the "
                        "frozen 50% and 100% cells"
                    ),
                    "next_experiment": (
                        "incrementally update only the newest Quest page metadata, "
                        "then require exact selection/logit/full-budget equivalence "
                        "and compare against this Phase 4 artifact"
                    ),
                },
                "pq": {
                    "target": "ranking_topk",
                    "selection_rule": (
                        "largest repeated median retrieval component across the "
                        "frozen 50% and 100% cells"
                    ),
                    "next_experiment": (
                        "evaluate a deterministic partial-selection implementation "
                        "against the current stable full argsort, requiring exact "
                        "selected IDs/tie behavior and all quality controls"
                    ),
                },
            },
            "backend_options": {
                "quest_metadata_rebuild": [
                    "portable eager PyTorch incremental update as the semantic oracle",
                    "portable C++ or Rust SIMD only if the incremental oracle remains costly",
                    "MLX or Metal for a later device-resident Apple path",
                    "Triton or CUDA only for a later GPU profile",
                ],
                "pq_ranking_topk": [
                    "PyTorch partial selection with explicit deterministic tie repair",
                    "portable C++ or Rust deterministic selection kernel",
                    "MLX or Metal selection for a later Apple device-resident path",
                    "Triton or CUDA selection only for a later GPU profile",
                ],
                "qualification": (
                    "ranked by fit for the measured operation shapes, not selected "
                    "for implementation in this phase"
                ),
            },
        },
        "complexity": {
            "quest_metadata_rebuild": "approximately O(H * S * D)",
            "quest_query_page_scoring": "approximately O(H * ceil(S/P) * D)",
            "quest_page_ranking": (
                "reference stable full argsort approximately O(H * (S/P) log(S/P))"
            ),
            "quest_page_to_token_expansion": "approximately O(H * K)",
            "pq_frozen_assignment": "approximately O(H * M * C * (D/M))",
            "pq_code_append": "reference torch.cat approximately O(H * S * M)",
            "pq_lookup_table_construction": ("approximately O(H * M * C * (D/M))"),
            "pq_approximate_score_reconstruction": "approximately O(H * S * M)",
            "pq_ranking": "reference stable full argsort approximately O(H * S log S)",
            "storage_fetch_and_selected_attention": "approximately O(H * K * D)",
            "qualification": (
                "algorithmic tensor dimensions only; Python dispatch, allocation, "
                "validation, and PyTorch CPU threading add implementation overhead"
            ),
        },
        "limitations": [
            "one pinned 410M standard-MHA model on one CPU machine",
            "two deterministic repeated local fixtures at one 1024-token prompt length",
            "31 measured teacher-forced decode steps per fixture and no free-running path",
            "reference Python/PyTorch execution only; no optimized backend evaluated",
            "wall-time scopes perturb small operations; calibrated overhead is reported",
            "analytical allocation and traffic estimates are not allocator or cache counters",
            "PyTorch operator and cProfile replays are diagnostic and excluded from primary timings",
            "no speedup claim and no Phase 5 implementation is included",
        ],
    }
    return artifact


def main() -> None:
    args = parse_args()
    artifact = run_experiment(args)
    write_json(args.output, artifact, overwrite=True, sort_keys=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
