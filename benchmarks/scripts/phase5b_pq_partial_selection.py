#!/usr/bin/env python3
"""Exact PQ ranking replay and matched Phase 4 CPU profiling for Phase 5B."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import statistics
import time
from typing import Any
from unittest.mock import patch

import torch

from benchmarks.artifacts import (
    load_json,
    require_schema_version,
    _sha256 as sha256_file,
    write_report,
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
from benchmarks.phase3a import TEXT_FIXTURES, build_deterministic_fixture
from benchmarks.phase4 import (
    build_layer_step_records,
    build_step_records,
    PQ_RETRIEVAL_CATEGORIES,
)
from benchmarks.pq_selection import (
    TieStatistics,
    assert_tensor_exact,
    compare_methods,
    scaling_benchmark,
    compare_score_batches,
    selection_accounting,
    selection_profile,
    update_digest,
)
from benchmarks.report_statistics import latency_distribution as distribution
from benchmarks.scripts.phase4_profile import (
    add_coarse_records,
    assert_profiled_semantics_unchanged,
    measure_scope_overhead,
    operator_profile,
    python_profile,
    quality_records,
    run_teacher_forced_trace,
)
from benchmarks.support import git_commit, git_is_dirty, machine_metadata
from kvweave import PQIndex, TensorStorage
from kvweave.indexes.pq import index as pq_module
from kvweave.indexes.pq.selection import PQRankingMode, pq_ranking_mode
from kvweave.integrations.transformers import (
    DecodeStrategy,
    GPTNeoXDecodeRunner,
    validate_gpt_neox_config,
)
from kvweave.integrations.transformers import gpt_neox_decode as decode_module
from kvweave.profiling import ComponentProfiler


FIXTURES = ("technical_exposition", "code_like")
BUDGETS = (0.5, 1.0)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/pythia-410m-phase5b-pq-partial-selection.json"
)


class SelectionCapture:
    """Capture actual intermediates only during an untimed correctness replay."""

    def __init__(self) -> None:
        self.records: list[dict[str, torch.Tensor]] = []
        self.stack = ExitStack()

    def __enter__(self) -> "SelectionCapture":
        original_scores = pq_module.score_pq_codes
        original_search = PQIndex.search
        original_policy = decode_module.prepare_decode_selection
        original_fetch = TensorStorage.fetch
        records = self.records

        def score(query: Any, metadata: Any) -> torch.Tensor:
            result = original_scores(query, metadata)
            records.append({"reconstructed_scores": result})
            return result

        def search(index: Any, query: Any, budget: int) -> Any:
            result = original_search(index, query, budget)
            if result.valid_mask is not None or result.indices.shape[-1] != budget:
                raise AssertionError("PQ search budget/mask contract changed")
            records[-1].update(
                ranked_ids=result.indices,
                ranked_scores=result.scores,
                candidate_counts=result.valid_token_counts,
            )
            return result

        class PolicyTorch:
            def __getattr__(self, name: str) -> Any:
                return getattr(torch, name)

            def argsort(self, sort_keys: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                # The unchanged policy sorts its actual newest-adjusted IDs here.
                records[-1]["newest_adjusted_ids"] = sort_keys.clone()
                return torch.argsort(sort_keys, **kwargs)

        def policy(selection: Any, *, newest_token_index: int) -> Any:
            with patch.object(decode_module, "torch", PolicyTorch()):
                result = original_policy(
                    selection, newest_token_index=newest_token_index
                )
            records[-1].update(causal_ids=result.indices, causal_scores=result.scores)
            return result

        def fetch(storage: Any, selection: Any) -> Any:
            result = original_fetch(storage, selection)
            records[-1].update(fetched_keys=result.keys, fetched_values=result.values)
            return result

        self.stack.enter_context(patch.object(pq_module, "score_pq_codes", score))
        self.stack.enter_context(patch.object(PQIndex, "search", search))
        self.stack.enter_context(
            patch.object(decode_module, "prepare_decode_selection", policy)
        )
        self.stack.enter_context(patch.object(TensorStorage, "fetch", fetch))
        return self

    def __exit__(self, *args: Any) -> None:
        self.stack.__exit__(*args)


def initialize(runner: Any, snapshot: Any, budget: float) -> Any:
    return runner.initialize_state(
        snapshot,
        strategy=DecodeStrategy.PQ,
        budget_fraction=budget,
        pq_num_subspaces=4,
        pq_num_centroids=8,
        pq_max_iterations=8,
        seed=0,
    )


def correctness_replay(
    runner: Any,
    snapshot: Any,
    dense_tokens: list[int],
    dense_steps: list[Any],
    *,
    fixture_id: str,
    budget: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor | None]:
    states = {mode: initialize(runner, snapshot, budget) for mode in PQRankingMode}
    frozen = [
        layer.cache.index.metadata.codebooks.clone()
        for layer in states[PQRankingMode.FULL_SORT].layers
    ]
    digests = {mode: hashlib.sha256() for mode in PQRankingMode}
    ties = TieStatistics()
    quality = []
    representative_scores = None
    for position in range(1, 32):
        token = torch.tensor([[dense_tokens[position - 1]]], dtype=torch.int64)
        steps = {}
        captured = {}
        for mode in PQRankingMode:
            with pq_ranking_mode(mode), SelectionCapture() as capture:
                step = runner.step(states[mode], token)
            if len(capture.records) != len(frozen):
                raise AssertionError("capture must cover every decode layer")
            steps[mode] = step
            captured[mode] = capture.records
        oracle = steps[PQRankingMode.FULL_SORT]
        partial = steps[PQRankingMode.PARTIAL]
        assert_profiled_semantics_unchanged([partial], [oracle])
        assert_tensor_exact(
            partial.next_token_logits, oracle.next_token_logits, "logits"
        )
        if budget == 1.0:
            assert_full_budget_step(
                partial, dense_steps[position - 1], rtol=1e-4, atol=1e-5
            )
        for layer, (old, new) in enumerate(
            zip(
                captured[PQRankingMode.FULL_SORT],
                captured[PQRankingMode.PARTIAL],
                strict=True,
            )
        ):
            if old.keys() != new.keys():
                raise AssertionError("captured stages differ")
            for name in old:
                assert_tensor_exact(new[name], old[name], name)
                for mode, values in (
                    (PQRankingMode.FULL_SORT, old),
                    (PQRankingMode.PARTIAL, new),
                ):
                    update_digest(digests[mode], values[name])
            assert_tensor_exact(
                partial.layers[layer].attention_weights,
                oracle.layers[layer].attention_weights,
                "attention weights",
            )
            for mode in PQRankingMode:
                metadata = states[mode].layers[layer].cache.index.metadata
                assert_tensor_exact(
                    metadata.codebooks, frozen[layer], "frozen codebook"
                )
                for value in (
                    steps[mode].layers[layer].attention_output,
                    steps[mode].layers[layer].residual_output,
                ):
                    update_digest(digests[mode], value)
            ties.add(old["reconstructed_scores"], old["ranked_ids"].shape[-1])
            if position == 31 and layer == 12:
                representative_scores = old["reconstructed_scores"].clone()
        for mode in PQRankingMode:
            update_digest(digests[mode], steps[mode].next_token_logits)
        row = quality_records(
            fixture_id=fixture_id,
            strategy="pq",
            budget_fraction=budget,
            approximate_steps=[partial],
            dense_steps=[dense_steps[position - 1]],
        )[0]
        row["decode_step"] = position
        quality.append(row)
    hashes = {mode.value: digest.hexdigest() for mode, digest in digests.items()}
    if len(set(hashes.values())) != 1:
        raise AssertionError("oracle/partial correctness hashes differ")
    return (
        {
            "fixture_id": fixture_id,
            "budget_fraction": budget,
            "passed": True,
            "decode_steps": 31,
            "layer_steps": 31 * len(frozen),
            "selection_rows": 31
            * len(frozen)
            * snapshot.layers[0].keys.shape[0]
            * snapshot.layers[0].keys.shape[1],
            "hashes": hashes,
            "stages_bit_exact": list(old)
            + ["attention_weights", "attention_output", "residual_stream", "logits"],
            "frozen_codebooks_exact": True,
            "full_budget_dense_control_passed": budget == 1.0,
            "tie_statistics": ties.report(),
        },
        quality,
        representative_scores,
    )


def load_baselines(
    phase4: Path, phase3b: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    historical = load_json(phase4)
    source = load_json(phase3b)
    for payload, phase in ((historical, "phase4"), (source, "phase3b")):
        require_schema_version(payload, supported=(1,))
        if payload["status"] != "complete":
            raise ValueError(f"{phase} baseline is not complete")
        if payload["provenance"]["model_revision"] != DEFAULT_MODEL_REVISION:
            raise ValueError(f"{phase} model pin differs")
    protocol = historical["protocol"]
    for field, expected in {
        "prompt_length": 1024,
        "fixture_ids": list(FIXTURES),
        "approximate_decode_steps": 31,
        "budget_fractions": list(BUDGETS),
        "pq_configuration": "M4/C8",
        "pq_iterations": 8,
        "seed": 0,
    }.items():
        if protocol[field] != expected:
            raise ValueError(f"Phase 4 protocol mismatch: {field}")
    runs = [
        run
        for run in source["runs"]
        if run["strategy"] == "pq"
        and run["mode"] == "teacher_forced"
        and run["prompt_length"] == 1024
        and run["fixture_id"] in FIXTURES
        and run["budget_fraction"] in BUDGETS
    ]
    cells = {(run["fixture_id"], run["budget_fraction"]) for run in runs}
    expected_cells = {(fixture, budget) for fixture in FIXTURES for budget in BUDGETS}
    if (
        len(runs) != 4
        or cells != expected_cells
        or any(len(run["steps"]) != 31 for run in runs)
    ):
        raise ValueError("Phase 3B comparison cells are incomplete")
    return historical, runs


def compare_historical_quality(
    quality: list[dict[str, Any]], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    comparisons = []
    metric_names = (
        "top_1_agreement",
        "top_5_overlap_fraction",
        "logit_relative_error",
        "kl_divergence_dense_to_approximate",
        "logit_cosine_similarity",
    )
    for row in quality:
        run = next(
            run
            for run in runs
            if run["fixture_id"] == row["fixture_id"]
            and run["budget_fraction"] == row["budget_fraction"]
        )
        old = run["steps"][row["decode_step"] - 1]
        for name in metric_names:
            if row[name] != old[name]:
                raise AssertionError(
                    f"Phase 3B quality changed: {row['fixture_id']} step {row['decode_step']} {name}: {row[name]} != {old[name]}"
                )
        # Phase 4 pools individual heads using fmean; reconstruct this from Phase 3B raw heads.
        mass = statistics.fmean(
            value
            for layer in old["layers"]
            for value in layer["attention_mass_captured_by_head"]
        )
        error = statistics.fmean(
            value
            for layer in old["layers"]
            for value in layer["attention_output_relative_error_by_head"]
        )
        residual = statistics.fmean(
            layer["residual_stream_relative_error"] for layer in old["layers"]
        )
        for name, expected in (
            ("mean_attention_mass_captured", mass),
            ("mean_attention_output_relative_error", error),
            ("mean_residual_stream_relative_error", residual),
        ):
            if row[name] != expected:
                raise AssertionError(f"Phase 3B quality changed: {name}")
        comparisons.append(
            {
                "fixture_id": row["fixture_id"],
                "budget_fraction": row["budget_fraction"],
                "decode_step": row["decode_step"],
                "all_metrics_exact": True,
            }
        )
    return {
        "passed": True,
        "steps_compared": len(comparisons),
        "comparisons": comparisons,
    }


def summarize_timing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for budget in BUDGETS:
        for metric in (
            "ranking_ms",
            "ranking_and_id_handling_ms",
            "retrieval_overhead_ms",
            "coarse_retrieval_overhead_ms",
            "total_decode_step_time_ms",
        ):
            timings = {
                mode.value: distribution(
                    [
                        row[metric]
                        for row in rows
                        if row["mode"] == mode.value
                        and row["budget_fraction"] == budget
                    ]
                )
                for mode in PQRankingMode
            }
            old = float(timings[PQRankingMode.FULL_SORT.value]["median"])
            new = float(timings[PQRankingMode.PARTIAL.value]["median"])
            summary.append(
                {
                    "budget_fraction": budget,
                    "metric": metric,
                    "timing_ms": timings,
                    "median_ms_saved": old - new,
                    "median_reduction_percent": 100 * (old - new) / old,
                    "median_speed_ratio": old / new,
                }
            )
    return summary


def load_pinned_runner() -> tuple[Any, Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

    if __version__ != DEFAULT_TRANSFORMERS_VERSION:
        raise RuntimeError("Transformers version differs from pinned decode setup")
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.to("cpu").eval()
    architecture = validate_gpt_neox_config(model.config)
    if model.config._commit_hash != DEFAULT_MODEL_REVISION:
        raise RuntimeError("Resolved model revision differs from pin")
    runner = GPTNeoXDecodeRunner(model)
    return runner, tokenizer, architecture


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    historical, phase3b_runs = load_baselines(
        args.phase4_artifact, args.phase3b_artifact
    )
    if transformers.__version__ != DEFAULT_TRANSFORMERS_VERSION:
        raise RuntimeError("Transformers version differs from pinned decode setup")
    torch.manual_seed(0)
    provenance = {
        "git_commit": git_commit(),
        "git_dirty_before_result_write": git_is_dirty(),
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "transformers_version": transformers.__version__,
        "transformers_source_revision": DEFAULT_TRANSFORMERS_REVISION,
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        **machine_metadata(torch.device("cpu")),
    }
    differences = {
        name: {"phase4": value, "phase5b": provenance.get(name)}
        for name, value in historical["provenance"].items()
        if name in provenance
        and not name.startswith("git_")
        and value != provenance[name]
    }
    runner, tokenizer, architecture = load_pinned_runner()
    model = runner.model
    fixtures = {fixture.fixture_id: fixture for fixture in TEXT_FIXTURES}
    cases = {}
    correctness = []
    quality = []
    representatives = {}
    hf_checks = []
    # All correctness/quality gates precede the primary timing matrix.
    for fixture_id in FIXTURES:
        tokens = build_deterministic_fixture(tokenizer, fixtures[fixture_id], 1024)
        snapshot = runner.dense_prefill(tokens.input_ids)
        dense_tokens, dense_logits, dense_steps = build_dense_trace(
            runner, snapshot, generated_tokens=32
        )
        hf_checks.append(
            {
                "fixture_id": fixture_id,
                **validate_hugging_face_generation(
                    model,
                    tokens.input_ids,
                    generated_tokens=32,
                    custom_tokens=dense_tokens,
                    custom_logits=dense_logits,
                ),
            }
        )
        cases[fixture_id] = (snapshot, dense_tokens, dense_steps)
        for budget in BUDGETS:
            check, metrics, scores = correctness_replay(
                runner,
                snapshot,
                dense_tokens,
                dense_steps,
                fixture_id=fixture_id,
                budget=budget,
            )
            correctness.append(check)
            quality.extend(metrics)
            if scores is None:
                raise AssertionError("pinned model omitted representative layer 12")
            representatives[(fixture_id, budget)] = scores
            print(
                f"exact decode passed fixture={fixture_id} budget={budget}", flush=True
            )
    regression = compare_historical_quality(quality, phase3b_runs)
    print(
        "Phase 3B quality regression passed exactly; starting primary timing matrix",
        flush=True,
    )
    step_rows = []
    layer_rows = []
    raw_records = []
    instrumentation_checks = []
    for fixture_index, fixture_id in enumerate(FIXTURES):
        snapshot, dense_tokens, dense_steps = cases[fixture_id]
        for budget in BUDGETS:
            modes = list(PQRankingMode)
            if fixture_index % 2:
                modes.reverse()
            for mode in modes:
                recorder = ComponentProfiler()
                with pq_ranking_mode(mode):
                    _, warmup = run_teacher_forced_trace(
                        runner,
                        snapshot,
                        dense_tokens,
                        strategy=DecodeStrategy.PQ,
                        budget_fraction=budget,
                        recorder=None,
                        fixture_id=fixture_id,
                        record_initialization=False,
                    )
                    _, measured = run_teacher_forced_trace(
                        runner,
                        snapshot,
                        dense_tokens,
                        strategy=DecodeStrategy.PQ,
                        budget_fraction=budget,
                        recorder=recorder,
                        fixture_id=fixture_id,
                        record_initialization=False,
                    )
                instrumentation_checks.append(
                    {
                        "fixture_id": fixture_id,
                        "budget_fraction": budget,
                        "mode": mode.value,
                        **assert_profiled_semantics_unchanged(measured, warmup),
                    }
                )
                coarse = []
                walls = {}
                add_coarse_records(
                    fixture_id=fixture_id,
                    strategy="pq",
                    budget_fraction=budget,
                    steps=measured,
                    coarse_records=coarse,
                    step_wall_times=walls,
                )
                layers = build_layer_step_records(recorder.records, coarse)
                steps = build_step_records(layers, walls)
                for row in steps:
                    layer_group = [
                        layer
                        for layer in layers
                        if layer["decode_step"] == row["decode_step"]
                    ]
                    row.update(
                        mode=mode.value,
                        ranking_ms=sum(
                            layer["atomic_components_ms"].get("pq.search.ranking", 0.0)
                            for layer in layer_group
                        ),
                        ranking_and_id_handling_ms=row["component_categories_ms"][
                            "ranking_topk"
                        ],
                        retrieval_overhead_ms=sum(
                            row["component_categories_ms"].get(name, 0.0)
                            for name in PQ_RETRIEVAL_CATEGORIES
                        ),
                        coarse_retrieval_overhead_ms=sum(
                            layer["coarse_index_update_time_ms"]
                            + layer["coarse_retrieval_time_ms"]
                            + layer["coarse_storage_fetch_time_ms"]
                            for layer in layer_group
                        ),
                    )
                    step_rows.append(row)
                layer_rows.extend({**row, "mode": mode.value} for row in layers)
                raw_records.extend(
                    {**asdict(row), "mode": mode.value} for row in recorder.records
                )
                print(
                    f"profile complete fixture={fixture_id} budget={budget} mode={mode.value}",
                    flush=True,
                )
    # Isolated diagnostics follow the matrix and cannot contaminate its timings.
    scaling = scaling_benchmark(repetitions=args.scaling_repetitions)
    representative_timings = []
    for (fixture_id, source_budget), scores in representatives.items():
        for fraction in (0.125, 0.25, 0.5, 1.0):
            budget = math.ceil(scores.shape[-1] * fraction)
            representative_timings.append(
                {
                    "fixture_id": fixture_id,
                    "source_decode_budget": source_budget,
                    "budget_fraction": fraction,
                    **compare_methods(scores, budget, repetitions=101),
                }
            )
    profiles = []
    snapshot, dense_tokens, _ = cases["technical_exposition"]
    for budget in BUDGETS:
        scores = representatives[("technical_exposition", budget)]
        profiles.append(
            {
                "budget_fraction": budget,
                "selection_profiles": selection_profile(
                    scores, math.ceil(scores.shape[-1] * budget)
                ),
            }
        )
        for mode in PQRankingMode:
            with pq_ranking_mode(mode):
                profiles.append(
                    {
                        "mode": mode.value,
                        "budget_fraction": budget,
                        "decode_operator_profile": operator_profile(
                            runner,
                            snapshot,
                            dense_tokens,
                            strategy=DecodeStrategy.PQ,
                            budget_fraction=budget,
                            profile_directory=args.profile_directory / mode.value,
                        ),
                        "decode_python_profile": python_profile(
                            runner,
                            snapshot,
                            dense_tokens,
                            strategy=DecodeStrategy.PQ,
                            budget_fraction=budget,
                        ),
                    }
                )
    quality_summary = [
        {
            "budget_fraction": budget,
            **{
                metric: statistics.fmean(
                    float(row[metric])
                    for row in quality
                    if row["budget_fraction"] == budget
                )
                for metric in (
                    "top_1_agreement",
                    "top_5_overlap_fraction",
                    "logit_relative_error",
                    "kl_divergence_dense_to_approximate",
                    "mean_attention_mass_captured",
                    "mean_attention_output_relative_error",
                    "mean_residual_stream_relative_error",
                )
            },
        }
        for budget in BUDGETS
    ]
    return {
        "schema_version": 1,
        "phase": "5B",
        "status": "complete",
        "provenance": provenance,
        "architecture": asdict(architecture),
        "environment_differences_from_phase4": differences,
        "protocol": {
            "fixture_ids": list(FIXTURES),
            "prompt_length": 1024,
            "mode": "teacher_forced",
            "generated_token_positions": 32,
            "retrieval_steps": 31,
            "pq_configuration": "M4/C8",
            "pq_iterations": 8,
            "seed": 0,
            "codebooks": "frozen prefill-trained",
            "budget_fractions": list(BUDGETS),
            "warmup": "one complete uninstrumented 31-step replay before each measured path",
            "measurement": "one 31-step replay per fixture/mode/budget; mode order reversed for second fixture",
            "clock_and_scopes": "unchanged Phase 4 perf_counter_ns ComponentProfiler scopes; initialization excluded",
            "scope_overhead": measure_scope_overhead(),
            "correctness_before_primary_timings": True,
        },
        "baselines": {
            "phase4_sha256": sha256_file(args.phase4_artifact),
            "phase3b_sha256": sha256_file(args.phase3b_artifact),
            "phase4_provenance": historical["provenance"],
            "phase4_component_summary": historical["steady_state"][
                "step_component_summary"
            ],
            "phase4_retrieval_summary": historical["steady_state"][
                "retrieval_overhead_summary"
            ],
        },
        "correctness": {
            "oracle_vs_partial": correctness,
            "dense_vs_hugging_face": hf_checks,
            "instrumentation": instrumentation_checks,
            "phase3b_quality_regression": regression,
        },
        "quality": {"summary": quality_summary, "rows": quality},
        "steady_state": {
            "summary": summarize_timing(step_rows),
            "step_records": step_rows,
            "layer_records": layer_rows,
            "raw_component_records": raw_records,
        },
        "scaling": scaling,
        "real_decode_score_budget_sensitivity": representative_timings,
        "profiles": profiles,
        "allocation_and_traffic": [
            selection_accounting(
                batch=1, heads=16, length=1055, budget=math.ceil(1055 * fraction)
            )
            for fraction in (0.125, 0.25, 0.5, 1.0)
        ],
        "limitations": [
            "CPU float32 reference; no production runtime or kernel speed claim",
            "single machine; matched timings do not eliminate thermal/scheduling variation",
            "native selection/sort workspaces and hardware memory traffic are not inferred from tensor payloads",
            "full budget deliberately retains globally ranked IDs",
            "no free-running replay in this teacher-forced matrix",
        ],
        "shared_abstractions_changed": False,
        "root_public_api_changed": False,
    }


def run_prototypes() -> dict[str, Any]:
    """Compare all candidates on actual PQ scores across layers/positions."""
    torch.manual_seed(0)
    runner, tokenizer, _ = load_pinned_runner()
    fixtures = {fixture.fixture_id: fixture for fixture in TEXT_FIXTURES}
    samples = []
    original_scores = pq_module.score_pq_codes
    for fixture_id in FIXTURES:
        snapshot = runner.dense_prefill(
            build_deterministic_fixture(tokenizer, fixtures[fixture_id], 1024).input_ids
        )
        tokens, _, _ = build_dense_trace(runner, snapshot, generated_tokens=32)
        state = initialize(runner, snapshot, 0.5)
        for position in range(1, 32):
            captured = []

            def capture_scores(query: Any, metadata: Any) -> torch.Tensor:
                result = original_scores(query, metadata)
                if position in (1, 16, 31):
                    captured.append(result.clone())
                return result

            with (
                pq_ranking_mode(PQRankingMode.FULL_SORT),
                patch.object(pq_module, "score_pq_codes", capture_scores),
            ):
                runner.step(
                    state, torch.tensor([[tokens[position - 1]]], dtype=torch.int64)
                )
            if captured:
                samples.append((fixture_id, position, captured))
        print(f"prototype scores collected fixture={fixture_id}", flush=True)
    rows = []
    for fixture_id, position, scores in samples:
        result = compare_score_batches(scores, budget_fraction=0.5, repetitions=31)
        rows.append({"fixture_id": fixture_id, "decode_step": position, **result})
        print(
            f"prototype comparison fixture={fixture_id} step={position} "
            + str(
                {name: values["median"] for name, values in result["timing_ms"].items()}
            ),
            flush=True,
        )
    return {
        "schema_version": 1,
        "phase": "5B-prototypes",
        "status": "complete",
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "transformers_version": DEFAULT_TRANSFORMERS_VERSION,
        "torch_version": torch.__version__,
        **machine_metadata(torch.device("cpu")),
        "protocol": "teacher-forced frozen M4/C8, seed0, 8 iterations, 1024 prompt, 50%; actual scores from all 24 layers at steps1/16/31; ranking-only hot replay, not integrated decode latency",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--phase4-artifact",
        type=Path,
        default=Path("benchmarks/results/pythia-410m-phase4-profile.json"),
    )
    parser.add_argument(
        "--phase3b-artifact",
        type=Path,
        default=Path("benchmarks/results/pythia-410m-phase3b-decode.json"),
    )
    parser.add_argument(
        "--profile-directory",
        type=Path,
        default=Path("benchmarks/results/profile/pythia-410m-phase5b"),
    )
    parser.add_argument("--scaling-repetitions", type=int, default=51)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--scaling-only", action="store_true")
    modes.add_argument("--prototypes-only", action="store_true")
    args = parser.parse_args()
    if args.scaling_repetitions < 1:
        parser.error("scaling repetitions must be positive")
    if args.output is None:
        args.output = (
            Path("benchmarks/results/phase5b-pq-prototypes.json")
            if args.prototypes_only
            else Path("benchmarks/results/phase5b-pq-ranking-scaling.json")
            if args.scaling_only
            else DEFAULT_OUTPUT
        )
    start = time.perf_counter()
    if args.prototypes_only:
        result = run_prototypes()
    elif args.scaling_only:
        result = {
            "schema_version": 1,
            "phase": "5B-scaling",
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "torch_version": torch.__version__,
            **machine_metadata(torch.device("cpu")),
            "scaling": scaling_benchmark(repetitions=args.scaling_repetitions),
        }
    else:
        result = run_experiment(args)
    write_report(args.output, result, overwrite=False)
    print(
        f"output={args.output} elapsed_seconds={time.perf_counter() - start:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
