#!/usr/bin/env python3
"""Replicate Phase 3A across local text structures and query positions."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Iterable

import torch

from benchmarks.phase3a import (
    QueryPosition,
    TextFixture,
    aggregate_quest_bound_looseness,
    attention_distribution_metrics,
    build_deterministic_fixture,
    calculate_query_positions,
    canonicalize_selection_for_attention,
    pq_score_approximation_metrics,
    quest_bound_quality,
)
from benchmarks.policy_feasibility import (
    FIXTURE_SPLITS,
    validate_fixture_lock,
    validate_tokenized_fixture_lock,
)
from kvdb import BruteForceIndex, PQIndex, QuestIndex, TensorStorage
from kvdb.core.types import Selection
from kvdb.indexes.pq import reconstruct_keys, score_pq_codes
from kvdb.indexes.quest.reference import candidate_recall
from kvdb.integrations.transformers import (
    GPTNeoXLayerActivations,
    attention_mass_captured,
    capture_gpt_neox_activations,
    causal_slice,
    per_head_relative_error,
    project_head_outputs,
    reference_attention,
    reference_causal_attention,
    validate_gpt_neox_config,
    validate_layer_indices,
)


DEFAULT_MODEL_ID = "EleutherAI/pythia-410m"
DEFAULT_MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
DEFAULT_TRANSFORMERS_VERSION = "5.15.1"
DEFAULT_OUTPUT = Path("benchmarks/results/pythia-410m-phase3a-replication.json")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
COMMON_QUALITY_METRICS = (
    "candidate_recall",
    "attention_mass_captured",
    "relative_attention_output_error",
)
ATTENTION_METRICS = (
    "attention_entropy_nats",
    "normalized_attention_entropy",
    "top_1_attention_mass",
    "top_4_attention_mass",
    "top_16_attention_mass",
    "effective_attention_support_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--fixture-split",
        choices=sorted(FIXTURE_SPLITS),
        default="development",
    )
    parser.add_argument(
        "--fixture-ids",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[512, 2_048],
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 12, 23])
    parser.add_argument("--page-sizes", nargs="+", type=int, default=[16, 64])
    parser.add_argument(
        "--budget-fractions",
        nargs="+",
        type=float,
        default=[0.125, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--pq-configs",
        nargs="+",
        default=["2x4", "4x8"],
        metavar="SUBSPACESxCENTROIDS",
    )
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--reconstruction-rtol", type=float, default=1e-4)
    parser.add_argument("--reconstruction-atol", type=float, default=1e-5)
    parser.add_argument("--full-budget-max-relative-error", type=float, default=1e-3)
    parser.add_argument("--full-budget-max-absolute-error", type=float, default=5e-4)
    parser.add_argument("--full-budget-attention-mass-atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_pq_configurations(values: list[str]) -> list[tuple[int, int]]:
    configurations: list[tuple[int, int]] = []
    for value in values:
        parts = value.lower().split("x", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid PQ configuration {value!r}; expected MxC")
        try:
            num_subspaces, num_centroids = (int(part) for part in parts)
        except ValueError as error:
            raise ValueError(
                f"invalid PQ configuration {value!r}; expected integer MxC"
            ) from error
        if num_subspaces <= 0 or num_centroids <= 0:
            raise ValueError("PQ subspaces and centroids must be positive")
        configurations.append((num_subspaces, num_centroids))
    if len(set(configurations)) != len(configurations):
        raise ValueError("PQ configurations must be unique")
    return configurations


def selected_fixtures(
    split: str,
    fixture_ids: list[str] | None,
) -> tuple[TextFixture, ...]:
    fixtures = FIXTURE_SPLITS[split]
    available = {fixture.fixture_id: fixture for fixture in fixtures}
    if fixture_ids is None:
        fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if not fixture_ids:
        raise ValueError("at least one fixture ID is required")
    if len(set(fixture_ids)) != len(fixture_ids):
        raise ValueError("fixture IDs must be unique")
    unknown = sorted(set(fixture_ids) - set(available))
    if unknown:
        raise ValueError(f"unknown fixture IDs: {', '.join(unknown)}")
    return tuple(available[fixture_id] for fixture_id in fixture_ids)


def validate_args(
    args: argparse.Namespace,
) -> tuple[tuple[TextFixture, ...], list[tuple[int, int]]]:
    if not args.model_id or not args.model_revision:
        raise ValueError("model ID and revision must be non-empty")
    if any(length <= 0 for length in args.sequence_lengths):
        raise ValueError("sequence lengths must be positive")
    if len(set(args.sequence_lengths)) != len(args.sequence_lengths):
        raise ValueError("sequence lengths must be unique")
    if any(page_size <= 0 for page_size in args.page_sizes):
        raise ValueError("page sizes must be positive")
    if len(set(args.page_sizes)) != len(args.page_sizes):
        raise ValueError("page sizes must be unique")
    if any(fraction <= 0.0 or fraction > 1.0 for fraction in args.budget_fractions):
        raise ValueError("budget fractions must be in (0, 1]")
    if len(set(args.budget_fractions)) != len(args.budget_fractions):
        raise ValueError("budget fractions must be unique")
    if args.kmeans_iterations <= 0:
        raise ValueError("kmeans iterations must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    tolerances = (
        args.reconstruction_rtol,
        args.reconstruction_atol,
        args.full_budget_max_relative_error,
        args.full_budget_max_absolute_error,
        args.full_budget_attention_mass_atol,
    )
    if any(tolerance < 0 for tolerance in tolerances):
        raise ValueError("floating-point tolerances must be non-negative")
    validate_fixture_lock(args.fixture_split)
    return selected_fixtures(
        args.fixture_split,
        args.fixture_ids,
    ), parse_pq_configurations(args.pq_configs)


def git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def tensor_relative_error(
    approximate: torch.Tensor,
    exact: torch.Tensor,
) -> float:
    numerator = torch.linalg.vector_norm(approximate - exact)
    denominator = torch.linalg.vector_norm(exact)
    if denominator.item() == 0:
        return 0.0 if numerator.item() == 0 else float("inf")
    return (numerator / denominator).item()


def pq_reconstruction_error_by_head(
    reconstructed: torch.Tensor,
    exact: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(reconstructed - exact, dim=(-2, -1))
    denominator = torch.linalg.vector_norm(exact, dim=(-2, -1))
    infinite = torch.full_like(numerator, float("inf"))
    return torch.where(
        denominator == 0,
        torch.where(numerator == 0, torch.zeros_like(numerator), infinite),
        numerator / denominator,
    )


def selection_full_coverage_by_head(
    selection: Selection,
    sequence_length: int,
) -> torch.Tensor:
    expected = torch.arange(
        sequence_length,
        device=selection.indices.device,
        dtype=torch.int64,
    )
    valid_mask = selection.valid_mask
    if valid_mask is None:
        valid_mask = torch.ones_like(selection.indices, dtype=torch.bool)
    coverage = torch.zeros(selection.indices.shape[:2], dtype=torch.bool)
    for batch_index in range(selection.indices.shape[0]):
        for head_index in range(selection.indices.shape[1]):
            actual = selection.indices[batch_index, head_index][
                valid_mask[batch_index, head_index]
            ]
            coverage[batch_index, head_index] = actual.shape[
                0
            ] == sequence_length and torch.equal(actual.sort().values, expected)
    return coverage


def _optional_head_value(
    metrics: dict[str, torch.Tensor] | None,
    metric: str,
    head_index: int,
) -> float | None:
    if metrics is None:
        return None
    value = float(metrics[metric][0, head_index].item())
    return value if math.isfinite(value) else None


def requested_token_budget(causal_token_count: int, fraction: float) -> int:
    return max(1, min(causal_token_count, round(causal_token_count * fraction)))


def record_selection(
    *,
    records: list[dict[str, Any]],
    full_budget_invariants: list[dict[str, Any]],
    fixture: TextFixture,
    layer_index: int,
    sequence_length: int,
    query_position: QueryPosition,
    budget_fraction: float,
    requested_budget: int,
    strategy: str,
    configuration: str,
    selection: Selection,
    exact_topk: Selection,
    query: torch.Tensor,
    keys: torch.Tensor,
    storage: TensorStorage,
    full_output: torch.Tensor,
    full_weights: torch.Tensor,
    attention_scale: float,
    attention_metrics: dict[str, torch.Tensor],
    index_build_seconds: float,
    retrieval_seconds: float,
    quest_metrics: dict[str, torch.Tensor] | None = None,
    pq_reconstruction_errors: torch.Tensor | None = None,
    pq_score_metrics: dict[str, torch.Tensor] | None = None,
    full_budget_max_relative_error: float,
    full_budget_max_absolute_error: float,
    full_budget_attention_mass_atol: float,
) -> None:
    attention_selection = canonicalize_selection_for_attention(selection)
    retrieved = storage.fetch(attention_selection)
    selected_attention = reference_attention(
        query,
        retrieved.keys,
        retrieved.values,
        valid_mask=retrieved.valid_mask,
        scale=attention_scale,
    )
    recalls = candidate_recall(selection, exact_topk)
    captured_mass = attention_mass_captured(full_weights, selection)
    output_errors = per_head_relative_error(selected_attention.output, full_output)
    absolute_output_errors = (
        (selected_attention.output - full_output).abs().amax(dim=-1)
    )
    actual_counts = selection.valid_token_counts

    is_full_budget = requested_budget == query_position.causal_token_count
    coverage = None
    if is_full_budget:
        coverage = selection_full_coverage_by_head(
            selection,
            query_position.causal_token_count,
        )

    for head_index in range(keys.shape[1]):
        invariant_passed: bool | None = None
        covers_all: bool | None = None
        mass_matches: bool | None = None
        attention_matches: bool | None = None
        if is_full_budget:
            assert coverage is not None
            covers_all = bool(coverage[0, head_index].item())
            mass_matches = (
                abs(float(captured_mass[0, head_index].item()) - 1.0)
                <= full_budget_attention_mass_atol
            )
            attention_matches = (
                float(output_errors[0, head_index].item())
                <= full_budget_max_relative_error
                and float(absolute_output_errors[0, head_index].item())
                <= full_budget_max_absolute_error
            )
            invariant_passed = covers_all and mass_matches and attention_matches
            invariant = {
                "text_fixture_id": fixture.fixture_id,
                "sequence_length": sequence_length,
                "query_position_label": query_position.label,
                "query_position": query_position.token_index,
                "causal_token_count": query_position.causal_token_count,
                "layer": layer_index,
                "head": head_index,
                "strategy": strategy,
                "configuration": configuration,
                "covers_every_causal_token": covers_all,
                "attention_mass_is_one": mass_matches,
                "selected_attention_matches_full": attention_matches,
                "attention_mass_captured": float(captured_mass[0, head_index].item()),
                "head_relative_output_error": float(
                    output_errors[0, head_index].item()
                ),
                "maximum_absolute_output_error": float(
                    absolute_output_errors[0, head_index].item()
                ),
            }
            full_budget_invariants.append(invariant)
            if not invariant_passed:
                raise AssertionError(f"full-budget invariant failed: {invariant}")

        record: dict[str, Any] = {
            "text_fixture_id": fixture.fixture_id,
            "text_structure": fixture.structure,
            "sequence_length": sequence_length,
            "query_position_label": query_position.label,
            "query_position_fraction": query_position.requested_fraction,
            "query_position": query_position.token_index,
            "causal_token_count": query_position.causal_token_count,
            "layer": layer_index,
            "head": head_index,
            "budget_fraction": budget_fraction,
            "requested_token_budget": requested_budget,
            "actual_candidate_count": int(actual_counts[0, head_index].item()),
            "strategy": strategy,
            "configuration": configuration,
            "candidate_recall": float(recalls[0, head_index].item()),
            "attention_mass_captured": float(captured_mass[0, head_index].item()),
            "relative_attention_output_error": float(
                output_errors[0, head_index].item()
            ),
            "index_build_seconds": index_build_seconds,
            "retrieval_seconds": retrieval_seconds,
            "pq_reconstruction_relative_error": (
                None
                if pq_reconstruction_errors is None
                else float(pq_reconstruction_errors[0, head_index].item())
            ),
            "full_budget_invariant_passed": invariant_passed,
            "full_budget_covers_every_causal_token": covers_all,
            "full_budget_attention_mass_is_one": mass_matches,
            "full_budget_attention_matches_reference": attention_matches,
        }
        for metric in ATTENTION_METRICS:
            record[metric] = float(attention_metrics[metric][0, head_index].item())
        for metric in (
            "quest_bound_looseness_all_mean",
            "quest_bound_looseness_all_max",
            "quest_bound_looseness_selected_mean",
            "quest_bound_looseness_selected_max",
            "quest_bound_looseness_nonselected_mean",
            "quest_bound_looseness_nonselected_max",
        ):
            record[metric] = _optional_head_value(
                quest_metrics,
                metric,
                head_index,
            )
        for metric in (
            "pq_score_mae",
            "pq_score_rmse",
            "pq_score_spearman_rank_correlation",
            "pq_exact_top_token_signed_score_error",
            "pq_exact_top_token_absolute_score_error",
            "pq_exact_top_16_score_mae",
        ):
            record[metric] = _optional_head_value(
                pq_score_metrics,
                metric,
                head_index,
            )
        records.append(record)


def reconstruct_and_validate_attention(
    activations: GPTNeoXLayerActivations,
    *,
    fixture_id: str,
    sequence_length: int,
    query_positions: tuple[QueryPosition, ...],
    attention_scale: float,
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    """Apply the accepted full causal reconstruction at every query position."""
    if sequence_length != activations.key.shape[2]:
        raise ValueError("reconstruction requires a matching-length model forward")
    causal_output = reference_causal_attention(
        activations.query,
        activations.key,
        activations.value,
        scale=attention_scale,
    )
    results: list[dict[str, Any]] = []
    for query_position in query_positions:
        projected = project_head_outputs(
            causal_output[:, :, query_position.token_index, :],
            activations.dense_weight,
            activations.dense_bias,
        )
        model_output = activations.projected_attention_output[
            :, query_position.token_index
        ]
        passed = torch.allclose(projected, model_output, rtol=rtol, atol=atol)
        result = {
            "text_fixture_id": fixture_id,
            "layer": activations.layer_index,
            "sequence_length": sequence_length,
            "query_position_label": query_position.label,
            "query_position": query_position.token_index,
            "causal_token_count": query_position.causal_token_count,
            "passed": passed,
            "relative_output_error": tensor_relative_error(projected, model_output),
            "maximum_absolute_error": (projected - model_output).abs().max().item(),
            "rtol": rtol,
            "atol": atol,
        }
        if not passed:
            raise AssertionError(f"full attention reconstruction failed: {result}")
        results.append(result)
    return results


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_distribution(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "p10": percentile(finite, 0.10),
        "p25": percentile(finite, 0.25),
        "p75": percentile(finite, 0.75),
        "p90": percentile(finite, 0.90),
        "max": max(finite),
    }


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
    metrics: tuple[str, ...] = COMMON_QUALITY_METRICS,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in group_fields)].append(record)
    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        aggregate = dict(zip(group_fields, key, strict=True))
        aggregate["sample_count"] = len(group)
        for metric in metrics:
            aggregate[metric] = metric_distribution(
                float(record[metric])
                for record in group
                if record.get(metric) is not None
            )
        aggregates.append(aggregate)
    return aggregates


def pearson_correlation(
    records: list[dict[str, Any]],
    left: str,
    right: str,
) -> tuple[int, float | None]:
    pairs = [
        (float(record[left]), float(record[right]))
        for record in records
        if record.get(left) is not None
        and record.get(right) is not None
        and math.isfinite(float(record[left]))
        and math.isfinite(float(record[right]))
    ]
    if len(pairs) < 2:
        return len(pairs), None
    left_values, right_values = zip(*pairs, strict=True)
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_values)
        * sum((value - right_mean) ** 2 for value in right_values)
    )
    return len(pairs), None if denominator == 0 else numerator / denominator


def _correlation_scopes(
    records: list[dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], list[dict[str, Any]]]]:
    yield {"scope": "pooled"}, records
    for fields, scope in (
        (("layer",), "per_layer"),
        (("budget_fraction",), "per_budget"),
        (("layer", "budget_fraction"), "per_layer_and_budget"),
    ):
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[tuple(record[field] for field in fields)].append(record)
        for key, group in sorted(groups.items()):
            labels = {"scope": scope}
            labels.update(dict(zip(fields, key, strict=True)))
            yield labels, group


def calculate_correlations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return descriptive Pearson correlations on partial-budget records."""
    partial = [record for record in records if record["budget_fraction"] < 1.0]
    populations: list[
        tuple[str, list[dict[str, Any]], tuple[tuple[str, str], ...]]
    ] = []
    common_pairs = (
        ("candidate_recall", "relative_attention_output_error"),
        ("attention_mass_captured", "relative_attention_output_error"),
        ("candidate_recall", "attention_mass_captured"),
        ("attention_entropy_nats", "candidate_recall"),
        ("attention_entropy_nats", "attention_mass_captured"),
        ("attention_entropy_nats", "relative_attention_output_error"),
        ("effective_attention_support_tokens", "candidate_recall"),
        ("effective_attention_support_tokens", "attention_mass_captured"),
        ("effective_attention_support_tokens", "relative_attention_output_error"),
    )
    approximate = [
        record for record in partial if record["strategy"] in {"quest", "pq"}
    ]
    populations.append(("all_approximate", approximate, common_pairs))
    for strategy in ("quest", "pq"):
        strategy_records = [
            record for record in partial if record["strategy"] == strategy
        ]
        populations.append((strategy, strategy_records, common_pairs))
        for configuration in sorted(
            {record["configuration"] for record in strategy_records}
        ):
            populations.append(
                (
                    f"{strategy}:{configuration}",
                    [
                        record
                        for record in strategy_records
                        if record["configuration"] == configuration
                    ],
                    common_pairs,
                )
            )

    quest_pairs = tuple(
        (bound_metric, quality_metric)
        for bound_metric in (
            "quest_bound_looseness_all_mean",
            "quest_bound_looseness_selected_mean",
            "quest_bound_looseness_selected_max",
        )
        for quality_metric in COMMON_QUALITY_METRICS
    )
    populations.append(
        (
            "quest_bound_diagnostics",
            [record for record in partial if record["strategy"] == "quest"],
            quest_pairs,
        )
    )
    pq_pairs = tuple(
        (diagnostic, quality_metric)
        for diagnostic in (
            "pq_score_mae",
            "pq_score_rmse",
            "pq_score_spearman_rank_correlation",
            "pq_exact_top_token_absolute_score_error",
            "pq_exact_top_16_score_mae",
            "pq_reconstruction_relative_error",
        )
        for quality_metric in COMMON_QUALITY_METRICS
    )
    populations.append(
        (
            "pq_score_diagnostics",
            [record for record in partial if record["strategy"] == "pq"],
            pq_pairs,
        )
    )

    results: list[dict[str, Any]] = []
    for population, population_records, pairs in populations:
        for labels, scoped_records in _correlation_scopes(population_records):
            for left, right in pairs:
                sample_count, correlation = pearson_correlation(
                    scoped_records,
                    left,
                    right,
                )
                row = {
                    "population": population,
                    **labels,
                    "left_metric": left,
                    "right_metric": right,
                    "sample_count": sample_count,
                    "pearson_correlation": correlation,
                    "interpretation": "descriptive association, not causal inference",
                }
                results.append(row)
    return results


def entropy_tercile_thresholds(
    attention_diagnostics: list[dict[str, Any]],
) -> dict[str, float]:
    entropies = [
        float(row["normalized_attention_entropy"]) for row in attention_diagnostics
    ]
    return {
        "low_to_middle": percentile(entropies, 1.0 / 3.0),
        "middle_to_high": percentile(entropies, 2.0 / 3.0),
    }


def entropy_stratum(value: float, thresholds: dict[str, float]) -> str:
    if value <= thresholds["low_to_middle"]:
        return "low_entropy_tercile"
    if value <= thresholds["middle_to_high"]:
        return "middle_entropy_tercile"
    return "high_entropy_tercile"


def _pairwise_outcome(
    left: float,
    right: float,
    *,
    left_label: str,
    right_label: str,
    higher_is_better: bool,
) -> str:
    difference = left - right
    if abs(difference) < 1e-7:
        return "tie"
    left_is_better = (difference > 0) == higher_is_better
    return left_label if left_is_better else right_label


def _paired_configuration_outcomes(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    left_configuration: str,
    right_configuration: str,
    left_label: str,
    right_label: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    partial = [
        record
        for record in records
        if record["strategy"] == strategy and record["budget_fraction"] < 1.0
    ]
    key_fields = (
        "text_fixture_id",
        "sequence_length",
        "query_position_label",
        "layer",
        "head",
        "budget_fraction",
    )
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in partial:
        groups[tuple(record[field] for field in key_fields)][
            record["configuration"]
        ] = record

    outcome_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {"all": []}
    equal_actual_counts = 0
    unequal_actual_counts = 0
    for configurations in groups.values():
        if (
            left_configuration not in configurations
            or right_configuration not in configurations
        ):
            continue
        left = configurations[left_configuration]
        right = configurations[right_configuration]
        if left["actual_candidate_count"] == right["actual_candidate_count"]:
            equal_actual_counts += 1
        else:
            unequal_actual_counts += 1
        outcome_groups["all"].append((left, right))
        stratum = entropy_stratum(
            float(left["normalized_attention_entropy"]),
            thresholds,
        )
        outcome_groups.setdefault(stratum, []).append((left, right))

    summaries: list[dict[str, Any]] = []
    for stratum, pairs in outcome_groups.items():
        summary: dict[str, Any] = {
            "entropy_stratum": stratum,
            "comparison_count": len(pairs),
            "outcomes": {},
        }
        for metric, higher_is_better in (
            ("candidate_recall", True),
            ("attention_mass_captured", True),
            ("relative_attention_output_error", False),
        ):
            counts = {left_label: 0, right_label: 0, "tie": 0}
            for left, right in pairs:
                outcome = _pairwise_outcome(
                    float(left[metric]),
                    float(right[metric]),
                    left_label=left_label,
                    right_label=right_label,
                    higher_is_better=higher_is_better,
                )
                counts[outcome] += 1
            summary["outcomes"][metric] = counts
        summaries.append(summary)
    return {
        "scope": "equal requested partial budgets",
        "left_configuration": left_configuration,
        "right_configuration": right_configuration,
        "equal_actual_candidate_count_comparisons": equal_actual_counts,
        "unequal_actual_candidate_count_comparisons": unequal_actual_counts,
        "by_entropy_stratum": summaries,
    }


def quest_page_size_comparison(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    return _paired_configuration_outcomes(
        records,
        strategy="quest",
        left_configuration="page_size=16",
        right_configuration="page_size=64",
        left_label="p16_better",
        right_label="p64_better",
        thresholds=thresholds,
    )


def pq_capacity_comparison(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    kmeans_iterations: int,
) -> dict[str, Any]:
    left_configuration = f"M=2,C=4,iterations={kmeans_iterations}"
    right_configuration = f"M=4,C=8,iterations={kmeans_iterations}"
    result = _paired_configuration_outcomes(
        records,
        strategy="pq",
        left_configuration=left_configuration,
        right_configuration=right_configuration,
        left_label="m2_c4_better",
        right_label="m4_c8_better",
        thresholds=thresholds,
    )

    reconstruction_source = [
        record
        for record in records
        if record["strategy"] == "pq" and record["budget_fraction"] == 0.125
    ]
    key_fields = (
        "text_fixture_id",
        "sequence_length",
        "query_position_label",
        "layer",
        "head",
    )
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in reconstruction_source:
        groups[tuple(record[field] for field in key_fields)][
            record["configuration"]
        ] = record
    strata: dict[str, dict[str, int]] = defaultdict(
        lambda: {"m2_c4_lower_error": 0, "m4_c8_lower_error": 0, "tie": 0}
    )
    for configurations in groups.values():
        if (
            left_configuration not in configurations
            or right_configuration not in configurations
        ):
            continue
        left = configurations[left_configuration]
        right = configurations[right_configuration]
        labels = (
            "all",
            f"layer={left['layer']}",
            f"query_position={left['query_position_label']}",
            entropy_stratum(float(left["normalized_attention_entropy"]), thresholds),
        )
        outcome = _pairwise_outcome(
            float(left["pq_reconstruction_relative_error"]),
            float(right["pq_reconstruction_relative_error"]),
            left_label="m2_c4_lower_error",
            right_label="m4_c8_lower_error",
            higher_is_better=False,
        )
        for label in labels:
            strata[label][outcome] += 1
    result["reconstruction_error_outcomes"] = [
        {"stratum": label, **counts} for label, counts in sorted(strata.items())
    ]
    return result


def _threshold_summary(
    records: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...],
    comparison: str,
) -> dict[str, Any]:
    values = [float(record["attention_mass_captured"]) for record in records]
    counts: dict[str, int] = {}
    for threshold in thresholds:
        if comparison == "at_least":
            count = sum(value >= threshold for value in values)
            label = f"at_least_{threshold:.2f}"
        elif comparison == "less_than":
            count = sum(value < threshold for value in values)
            label = f"less_than_{threshold:.2f}"
        else:
            raise ValueError("unsupported threshold comparison")
        counts[label] = count
    return {
        "sample_count": len(values),
        "counts": counts,
        "fractions": {
            label: (count / len(values) if values else None)
            for label, count in counts.items()
        },
    }


def _threshold_breakdown(
    records: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    thresholds: tuple[float, ...],
    comparison: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in fields)].append(record)
    rows = []
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        row = dict(zip(fields, key, strict=True))
        row.update(
            _threshold_summary(
                group,
                thresholds=thresholds,
                comparison=comparison,
            )
        )
        rows.append(row)
    return rows


def layer_23_sparse_replication(records: list[dict[str, Any]]) -> dict[str, Any]:
    source = [
        record
        for record in records
        if record["layer"] == 23 and record["budget_fraction"] == 0.125
    ]
    exact = [record for record in source if record["strategy"] == "brute_force"]
    approximate = [record for record in source if record["strategy"] in {"quest", "pq"}]
    dimensions = (
        ("text_fixture_id",),
        ("query_position_label",),
        ("head",),
    )
    exact_result = {
        "overall": _threshold_summary(
            exact,
            thresholds=(0.90, 0.95, 0.99),
            comparison="at_least",
        )
    }
    approximate_result = {
        "overall": _threshold_summary(
            approximate,
            thresholds=(0.50, 0.75, 0.90),
            comparison="less_than",
        )
    }
    for fields in dimensions:
        label = f"by_{fields[0]}"
        exact_result[label] = _threshold_breakdown(
            exact,
            fields=fields,
            thresholds=(0.90, 0.95, 0.99),
            comparison="at_least",
        )
        approximate_result[label] = _threshold_breakdown(
            approximate,
            fields=fields,
            thresholds=(0.50, 0.75, 0.90),
            comparison="less_than",
        )
    approximate_result["by_configuration"] = _threshold_breakdown(
        approximate,
        fields=("strategy", "configuration"),
        thresholds=(0.50, 0.75, 0.90),
        comparison="less_than",
    )
    approximate_result["by_text_query_head"] = _threshold_breakdown(
        approximate,
        fields=(
            "strategy",
            "configuration",
            "text_fixture_id",
            "query_position_label",
            "head",
        ),
        thresholds=(0.50, 0.75, 0.90),
        comparison="less_than",
    )
    return {
        "scope": "layer 23 at 12.5% of each valid causal prefix",
        "exact_topk": exact_result,
        "approximate": approximate_result,
    }


def retrospective_configuration_oracle(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure heterogeneity without defining or implementing a routing policy."""
    approximate = [
        record
        for record in records
        if record["strategy"] in {"quest", "pq"} and record["budget_fraction"] < 1.0
    ]
    key_fields = (
        "text_fixture_id",
        "sequence_length",
        "query_position_label",
        "layer",
        "head",
        "budget_fraction",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in approximate:
        groups[tuple(record[field] for field in key_fields)].append(record)

    oracle_rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        if len(group) != 4:
            raise AssertionError("retrospective oracle requires four configurations")
        row = dict(zip(key_fields, key, strict=True))
        row["attention_entropy_stratum"] = group[0]["attention_entropy_stratum"]
        for metric, operation in (
            ("candidate_recall", max),
            ("attention_mass_captured", max),
            ("relative_attention_output_error", min),
        ):
            oracle_value = operation(float(record[metric]) for record in group)
            row[f"oracle_{metric}"] = oracle_value
            row[f"winning_{metric}_configurations"] = sorted(
                record["configuration"]
                for record in group
                if abs(float(record[metric]) - oracle_value) < 1e-7
            )
        row["configurations"] = {
            record["configuration"]: {
                metric: float(record[metric]) for metric in COMMON_QUALITY_METRICS
            }
            for record in group
        }
        oracle_rows.append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        configurations = sorted(next(iter(group))["configurations"])
        summary: dict[str, Any] = {"sample_count": len(group), "metrics": {}}
        for metric, higher_is_better in (
            ("candidate_recall", True),
            ("attention_mass_captured", True),
            ("relative_attention_output_error", False),
        ):
            fixed_means = {
                configuration: statistics.fmean(
                    row["configurations"][configuration][metric] for row in group
                )
                for configuration in configurations
            }
            best_fixed_configuration = (
                max(fixed_means, key=fixed_means.get)  # type: ignore[arg-type]
                if higher_is_better
                else min(fixed_means, key=fixed_means.get)  # type: ignore[arg-type]
            )
            best_fixed_mean = fixed_means[best_fixed_configuration]
            oracle_mean = statistics.fmean(
                float(row[f"oracle_{metric}"]) for row in group
            )
            gap = (
                oracle_mean - best_fixed_mean
                if higher_is_better
                else best_fixed_mean - oracle_mean
            )
            win_counts = {
                configuration: sum(
                    configuration in row[f"winning_{metric}_configurations"]
                    for row in group
                )
                for configuration in configurations
            }
            summary["metrics"][metric] = {
                "fixed_configuration_means": fixed_means,
                "best_fixed_configuration": best_fixed_configuration,
                "best_fixed_mean": best_fixed_mean,
                "retrospective_oracle_mean": oracle_mean,
                "retrospective_oracle_absolute_improvement": gap,
                "configuration_win_counts_including_ties": win_counts,
            }
        return summary

    by_budget = []
    for budget_fraction in sorted({row["budget_fraction"] for row in oracle_rows}):
        group = [
            row for row in oracle_rows if row["budget_fraction"] == budget_fraction
        ]
        by_budget.append({"budget_fraction": budget_fraction, **summarize(group)})

    at_smallest_budget = [row for row in oracle_rows if row["budget_fraction"] == 0.125]
    stratified: dict[str, list[dict[str, Any]]] = {}
    for field in ("layer", "attention_entropy_stratum"):
        field_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in at_smallest_budget:
            field_groups[row[field]].append(row)
        stratified[field] = [
            {field: value, **summarize(group)}
            for value, group in sorted(
                field_groups.items(), key=lambda item: str(item[0])
            )
        ]
    return {
        "scope": "partial budgets over the four already-tested approximate configurations",
        "interpretation": (
            "retrospective per-head/query upper bound only; no routing policy, feature, "
            "runtime cost, or causal claim"
        ),
        "by_budget": by_budget,
        "at_12_5_percent_by_layer": stratified["layer"],
        "at_12_5_percent_by_entropy_stratum": stratified["attention_entropy_stratum"],
    }


def analyze_results(
    records: list[dict[str, Any]],
    attention_diagnostics: list[dict[str, Any]],
    *,
    kmeans_iterations: int,
) -> dict[str, Any]:
    thresholds = entropy_tercile_thresholds(attention_diagnostics)
    records_with_strata = []
    for record in records:
        enriched = dict(record)
        enriched["attention_entropy_stratum"] = entropy_stratum(
            float(record["normalized_attention_entropy"]),
            thresholds,
        )
        records_with_strata.append(enriched)
    for index, record in enumerate(records_with_strata):
        records[index]["attention_entropy_stratum"] = record[
            "attention_entropy_stratum"
        ]

    partial_approximate = [
        record
        for record in records
        if record["strategy"] in {"quest", "pq"} and record["budget_fraction"] < 1.0
    ]
    return {
        "entropy_stratification": {
            "method": "empirical terciles of normalized entropy over unique heads/queries",
            "thresholds": thresholds,
        },
        "attention_distributions": {
            "pooled": aggregate_records(
                attention_diagnostics,
                group_fields=(),
                metrics=ATTENTION_METRICS,
            ),
            "by_layer": aggregate_records(
                attention_diagnostics,
                group_fields=("layer",),
                metrics=ATTENTION_METRICS,
            ),
            "by_text": aggregate_records(
                attention_diagnostics,
                group_fields=("text_fixture_id",),
                metrics=ATTENTION_METRICS,
            ),
            "by_query_position": aggregate_records(
                attention_diagnostics,
                group_fields=("query_position_label",),
                metrics=ATTENTION_METRICS,
            ),
        },
        "retrieval_aggregates": {
            "by_strategy_configuration_budget": aggregate_records(
                records,
                group_fields=("strategy", "configuration", "budget_fraction"),
            ),
            "by_strategy_configuration_budget_layer": aggregate_records(
                records,
                group_fields=(
                    "strategy",
                    "configuration",
                    "budget_fraction",
                    "layer",
                ),
            ),
            "by_strategy_configuration_budget_text": aggregate_records(
                records,
                group_fields=(
                    "strategy",
                    "configuration",
                    "budget_fraction",
                    "text_fixture_id",
                ),
            ),
            "by_strategy_configuration_budget_query_position": aggregate_records(
                records,
                group_fields=(
                    "strategy",
                    "configuration",
                    "budget_fraction",
                    "query_position_label",
                ),
            ),
            "partial_approximate_by_entropy": aggregate_records(
                partial_approximate,
                group_fields=(
                    "strategy",
                    "configuration",
                    "budget_fraction",
                    "attention_entropy_stratum",
                ),
            ),
        },
        "layer_23_sparse_replication": layer_23_sparse_replication(records),
        "quest_page_size_comparison": quest_page_size_comparison(
            records,
            thresholds,
        ),
        "pq_capacity_comparison": pq_capacity_comparison(
            records,
            thresholds,
            kmeans_iterations=kmeans_iterations,
        ),
        "retrospective_configuration_oracle": retrospective_configuration_oracle(
            records
        ),
        "correlations": calculate_correlations(records),
    }


def _append_attention_diagnostics(
    rows: list[dict[str, Any]],
    *,
    fixture: TextFixture,
    sequence_length: int,
    query_position: QueryPosition,
    layer_index: int,
    metrics: dict[str, torch.Tensor],
) -> None:
    for head_index in range(next(iter(metrics.values())).shape[1]):
        row: dict[str, Any] = {
            "text_fixture_id": fixture.fixture_id,
            "text_structure": fixture.structure,
            "sequence_length": sequence_length,
            "query_position_label": query_position.label,
            "query_position_fraction": query_position.requested_fraction,
            "query_position": query_position.token_index,
            "causal_token_count": query_position.causal_token_count,
            "layer": layer_index,
            "head": head_index,
        }
        for metric in ATTENTION_METRICS:
            row[metric] = float(metrics[metric][0, head_index].item())
        rows.append(row)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    fixtures, pq_configurations = validate_args(args)
    try:
        from transformers import AutoModel, AutoTokenizer, __version__
    except ImportError as error:
        raise RuntimeError(
            "install the optional model experiment dependency: "
            "pip install -e '.[model-experiment]'"
        ) from error
    if __version__ != DEFAULT_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {DEFAULT_TRANSFORMERS_VERSION}, found {__version__}"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_dtype = DTYPES[args.model_dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
    )
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=model_dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    architecture = validate_gpt_neox_config(model.config)
    layers = validate_layer_indices(
        args.layers,
        num_hidden_layers=architecture.num_hidden_layers,
    )
    maximum_length = max(args.sequence_lengths)
    if maximum_length > architecture.max_position_embeddings:
        raise ValueError("requested sequence exceeds the model context limit")
    if any(
        architecture.head_dimension % num_subspaces != 0
        for num_subspaces, _ in pq_configurations
    ):
        raise ValueError("head dimension must be divisible by every PQ subspace count")
    minimum_causal_length = min(
        position.causal_token_count
        for sequence_length in args.sequence_lengths
        for position in calculate_query_positions(sequence_length)
    )
    if any(
        num_centroids > minimum_causal_length for _, num_centroids in pq_configurations
    ):
        raise ValueError("PQ centroids cannot exceed a tested causal prefix length")

    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision is None:
        resolved_revision = args.model_revision
    if len(args.model_revision) == 40 and resolved_revision != args.model_revision:
        raise RuntimeError(
            f"resolved model revision {resolved_revision} did not match the pin"
        )
    attention_implementation = getattr(
        model.config,
        "_attn_implementation",
        "unknown",
    )
    if attention_implementation != "eager":
        raise RuntimeError("the reference experiment requires eager model attention")

    records: list[dict[str, Any]] = []
    attention_diagnostics: list[dict[str, Any]] = []
    full_budget_invariants: list[dict[str, Any]] = []
    reconstruction: list[dict[str, Any]] = []
    input_metadata: list[dict[str, Any]] = []
    sorted_lengths = sorted(args.sequence_lengths)
    sorted_budgets = sorted(args.budget_fractions)

    for fixture_index, fixture in enumerate(fixtures, start=1):
        for sequence_length in sorted_lengths:
            print(
                f"capture fixture={fixture.fixture_id} length={sequence_length} "
                f"({fixture_index}/{len(fixtures)})",
                flush=True,
            )
            tokenized = build_deterministic_fixture(
                tokenizer,
                fixture,
                sequence_length,
            )
            validate_tokenized_fixture_lock(
                args.fixture_split,
                fixture,
                sequence_length,
                tokenized,
            )
            input_metadata.append(
                {
                    "text_fixture_id": fixture.fixture_id,
                    "sequence_length": sequence_length,
                    "base_token_count": tokenized.base_token_count,
                    "repetitions_before_truncation": tokenized.repetitions,
                    "resulting_token_count": tokenized.input_ids.shape[1],
                    "token_ids_sha256": tokenized.token_ids_sha256,
                }
            )
            sequence_input_ids = tokenized.input_ids.to(device)
            capture = capture_gpt_neox_activations(
                model,
                sequence_input_ids,
                layer_indices=layers,
                attention_mask=torch.ones_like(sequence_input_ids, device=device),
                capture_device="cpu",
                capture_dtype=torch.float32,
            )
            query_positions = calculate_query_positions(sequence_length)

            for layer_index in layers:
                activations = capture.layers[layer_index]
                reconstruction.extend(
                    reconstruct_and_validate_attention(
                        activations,
                        fixture_id=fixture.fixture_id,
                        sequence_length=sequence_length,
                        query_positions=query_positions,
                        attention_scale=architecture.attention_scale,
                        rtol=args.reconstruction_rtol,
                        atol=args.reconstruction_atol,
                    )
                )
                for query_position in query_positions:
                    sliced = causal_slice(activations, query_position.token_index)
                    full = reference_attention(
                        sliced.query,
                        sliced.keys,
                        sliced.values,
                        scale=architecture.attention_scale,
                    )
                    attention_metrics = attention_distribution_metrics(full.weights)
                    _append_attention_diagnostics(
                        attention_diagnostics,
                        fixture=fixture,
                        sequence_length=sequence_length,
                        query_position=query_position,
                        layer_index=layer_index,
                        metrics=attention_metrics,
                    )
                    storage = TensorStorage()
                    storage.put(sliced.keys, sliced.values)
                    exact_scores = torch.einsum(
                        "bhd,bhsd->bhs",
                        sliced.query,
                        sliced.keys,
                    )

                    start = time.perf_counter()
                    brute_force = BruteForceIndex()
                    brute_force.build(sliced.keys)
                    brute_build_seconds = time.perf_counter() - start
                    exact_by_budget: dict[int, Selection] = {}
                    for budget_fraction in sorted_budgets:
                        requested_budget = requested_token_budget(
                            query_position.causal_token_count,
                            budget_fraction,
                        )
                        exact_start = time.perf_counter()
                        exact_topk = brute_force.search(
                            sliced.query,
                            requested_budget,
                        )
                        exact_retrieval_seconds = time.perf_counter() - exact_start
                        exact_by_budget[requested_budget] = exact_topk
                        record_selection(
                            records=records,
                            full_budget_invariants=full_budget_invariants,
                            fixture=fixture,
                            layer_index=layer_index,
                            sequence_length=sequence_length,
                            query_position=query_position,
                            budget_fraction=budget_fraction,
                            requested_budget=requested_budget,
                            strategy="brute_force",
                            configuration="exact_raw_dot_product",
                            selection=exact_topk,
                            exact_topk=exact_topk,
                            query=sliced.query,
                            keys=sliced.keys,
                            storage=storage,
                            full_output=full.output,
                            full_weights=full.weights,
                            attention_scale=architecture.attention_scale,
                            attention_metrics=attention_metrics,
                            index_build_seconds=brute_build_seconds,
                            retrieval_seconds=exact_retrieval_seconds,
                            full_budget_max_relative_error=(
                                args.full_budget_max_relative_error
                            ),
                            full_budget_max_absolute_error=(
                                args.full_budget_max_absolute_error
                            ),
                            full_budget_attention_mass_atol=(
                                args.full_budget_attention_mass_atol
                            ),
                        )

                    for page_size in args.page_sizes:
                        start = time.perf_counter()
                        quest = QuestIndex(page_size=page_size)
                        quest.build(sliced.keys)
                        build_seconds = time.perf_counter() - start
                        bound_quality = quest_bound_quality(
                            sliced.query,
                            sliced.keys,
                            quest.metadata,
                        )
                        if bound_quality.looseness.min().item() < -1e-5:
                            raise AssertionError(
                                "Quest upper bound fell below true page max"
                            )
                        for budget_fraction in sorted_budgets:
                            requested_budget = requested_token_budget(
                                query_position.causal_token_count,
                                budget_fraction,
                            )
                            start = time.perf_counter()
                            search_result = quest.search_with_details(
                                sliced.query,
                                requested_budget,
                            )
                            retrieval_seconds = time.perf_counter() - start
                            quest_metrics = aggregate_quest_bound_looseness(
                                bound_quality.looseness,
                                search_result.page_indices,
                            )
                            record_selection(
                                records=records,
                                full_budget_invariants=full_budget_invariants,
                                fixture=fixture,
                                layer_index=layer_index,
                                sequence_length=sequence_length,
                                query_position=query_position,
                                budget_fraction=budget_fraction,
                                requested_budget=requested_budget,
                                strategy="quest",
                                configuration=f"page_size={page_size}",
                                selection=search_result.selection,
                                exact_topk=exact_by_budget[requested_budget],
                                query=sliced.query,
                                keys=sliced.keys,
                                storage=storage,
                                full_output=full.output,
                                full_weights=full.weights,
                                attention_scale=architecture.attention_scale,
                                attention_metrics=attention_metrics,
                                index_build_seconds=build_seconds,
                                retrieval_seconds=retrieval_seconds,
                                quest_metrics=quest_metrics,
                                full_budget_max_relative_error=(
                                    args.full_budget_max_relative_error
                                ),
                                full_budget_max_absolute_error=(
                                    args.full_budget_max_absolute_error
                                ),
                                full_budget_attention_mass_atol=(
                                    args.full_budget_attention_mass_atol
                                ),
                            )

                    for num_subspaces, num_centroids in pq_configurations:
                        start = time.perf_counter()
                        pq = PQIndex(
                            num_subspaces=num_subspaces,
                            num_centroids=num_centroids,
                            max_iterations=args.kmeans_iterations,
                            seed=args.seed,
                        )
                        pq.build(sliced.keys)
                        build_seconds = time.perf_counter() - start
                        reconstruction_errors = pq_reconstruction_error_by_head(
                            reconstruct_keys(pq.metadata),
                            sliced.keys,
                        )
                        approximate_scores = score_pq_codes(
                            sliced.query,
                            pq.metadata,
                        )
                        score_metrics = pq_score_approximation_metrics(
                            approximate_scores,
                            exact_scores,
                            high_attention_count=16,
                        )
                        configuration = (
                            f"M={num_subspaces},C={num_centroids},"
                            f"iterations={args.kmeans_iterations}"
                        )
                        for budget_fraction in sorted_budgets:
                            requested_budget = requested_token_budget(
                                query_position.causal_token_count,
                                budget_fraction,
                            )
                            start = time.perf_counter()
                            selection = pq.search(sliced.query, requested_budget)
                            retrieval_seconds = time.perf_counter() - start
                            record_selection(
                                records=records,
                                full_budget_invariants=full_budget_invariants,
                                fixture=fixture,
                                layer_index=layer_index,
                                sequence_length=sequence_length,
                                query_position=query_position,
                                budget_fraction=budget_fraction,
                                requested_budget=requested_budget,
                                strategy="pq",
                                configuration=configuration,
                                selection=selection,
                                exact_topk=exact_by_budget[requested_budget],
                                query=sliced.query,
                                keys=sliced.keys,
                                storage=storage,
                                full_output=full.output,
                                full_weights=full.weights,
                                attention_scale=architecture.attention_scale,
                                attention_metrics=attention_metrics,
                                index_build_seconds=build_seconds,
                                retrieval_seconds=retrieval_seconds,
                                pq_reconstruction_errors=reconstruction_errors,
                                pq_score_metrics=score_metrics,
                                full_budget_max_relative_error=(
                                    args.full_budget_max_relative_error
                                ),
                                full_budget_max_absolute_error=(
                                    args.full_budget_max_absolute_error
                                ),
                                full_budget_attention_mass_atol=(
                                    args.full_budget_attention_mass_atol
                                ),
                            )
            del capture

    configuration_count = 1 + len(args.page_sizes) + len(pq_configurations)
    expected_attention_rows = (
        len(fixtures)
        * len(args.sequence_lengths)
        * 4
        * len(layers)
        * architecture.num_attention_heads
    )
    expected_record_count = (
        expected_attention_rows * len(args.budget_fractions) * configuration_count
    )
    expected_invariant_count = expected_attention_rows * configuration_count
    if len(attention_diagnostics) != expected_attention_rows:
        raise AssertionError("attention diagnostic matrix is incomplete")
    if len(records) != expected_record_count:
        raise AssertionError("retrieval record matrix is incomplete")
    if len(full_budget_invariants) != expected_invariant_count:
        raise AssertionError("full-budget invariant matrix is incomplete")

    analysis = analyze_results(
        records,
        attention_diagnostics,
        kmeans_iterations=args.kmeans_iterations,
    )
    result = {
        "schema_version": 2,
        "benchmark": "pythia_410m_phase3a_structural_replication",
        "scope": (
            "single-query internal activation retrieval only; no generation, "
            "perplexity, downstream quality, optimized latency, or speed claim"
        ),
        "provenance": {
            "model_id": args.model_id,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": resolved_revision,
            "transformers_version": __version__,
            "transformers_source_revision": (
                "550d7b3834670483a4df436541272c055dc364bf"
            ),
            "transformers_attention_implementation": attention_implementation,
            "torch_version": torch.__version__,
            "model_dtype": args.model_dtype,
            "capture_and_reference_dtype": "float32",
            "device": str(device),
            "hardware": platform.platform(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "seed": args.seed,
        },
        "architecture": asdict(architecture),
        "input": {
            "fixture_split": args.fixture_split,
            "method": (
                "each locally authored fixture is tokenized independently, repeated, "
                "and deterministically truncated to the exact requested length"
            ),
            "external_dataset": None,
            "fixtures": [
                {
                    "text_fixture_id": fixture.fixture_id,
                    "structure": fixture.structure,
                    "text": fixture.text,
                }
                for fixture in fixtures
            ],
            "tokenizations": input_metadata,
        },
        "configuration": {
            "sequence_lengths": sorted_lengths,
            "query_positions": [
                asdict(position)
                for length in sorted_lengths
                for position in calculate_query_positions(length)
            ],
            "layers": list(layers),
            "heads": list(range(architecture.num_attention_heads)),
            "budget_fractions": sorted_budgets,
            "quest_page_sizes": args.page_sizes,
            "pq_configurations": [
                {
                    "num_subspaces": num_subspaces,
                    "num_centroids": num_centroids,
                    "max_iterations": args.kmeans_iterations,
                    "seed": args.seed,
                }
                for num_subspaces, num_centroids in pq_configurations
            ],
            "retrieval_objective": "unscaled raw query-key dot product",
            "selected_attention_ordering": (
                "valid selected token IDs sorted into ascending causal order before "
                "storage fetch; original strategy ranking retained for diagnostics"
            ),
            "attention_scale": architecture.attention_scale,
            "reconstruction_tolerance": {
                "rtol": args.reconstruction_rtol,
                "atol": args.reconstruction_atol,
            },
            "full_budget_permutation_tolerance": {
                "maximum_head_relative_output_error": (
                    args.full_budget_max_relative_error
                ),
                "maximum_absolute_output_error": (args.full_budget_max_absolute_error),
                "attention_mass_absolute_tolerance": (
                    args.full_budget_attention_mass_atol
                ),
            },
        },
        "metric_definitions": {
            "attention_entropy_nats": (
                "Shannon entropy -sum(p * ln(p)) of exact full-attention weights; "
                "natural-log units (nats)"
            ),
            "normalized_attention_entropy": "entropy / ln(valid causal token count)",
            "effective_attention_support_tokens": (
                "exp(attention entropy in nats), measured in effective tokens"
            ),
            "top_n_attention_mass": (
                "sum of the N largest exact full-attention probabilities"
            ),
            "quest_bound_looseness": (
                "Quest upper-bound page score minus the maximum exact raw q dot k "
                "token score in that page"
            ),
            "pq_score_errors": (
                "differences between PQ approximate and exact unscaled raw q dot k "
                "token scores; rank correlation is tie-aware Spearman"
            ),
            "candidate_recall": (
                "fraction of exact raw-dot-product Top-K token IDs present in the "
                "strategy selection"
            ),
            "attention_mass_captured": (
                "sum of exact full-attention probability on selected token IDs"
            ),
            "relative_attention_output_error": (
                "per-head vector norm(selected output - full output) / norm(full output)"
            ),
        },
        "matrix_counts": {
            "attention_head_query_rows": len(attention_diagnostics),
            "retrieval_records": len(records),
            "full_budget_head_invariants": len(full_budget_invariants),
            "expected_attention_head_query_rows": expected_attention_rows,
            "expected_retrieval_records": expected_record_count,
            "expected_full_budget_head_invariants": expected_invariant_count,
        },
        "attention_reconstruction": reconstruction,
        "all_attention_reconstructions_passed": all(
            row["passed"] for row in reconstruction
        ),
        "attention_diagnostics": attention_diagnostics,
        "full_budget_invariants": full_budget_invariants,
        "all_full_budget_invariants_passed": all(
            row["covers_every_causal_token"]
            and row["attention_mass_is_one"]
            and row["selected_attention_matches_full"]
            for row in full_budget_invariants
        ),
        "analysis": analysis,
        "shared_architecture_changes_required": {
            "KVIndex": False,
            "Selection": False,
            "KVStorage": False,
            "RetrievedKV": False,
            "KVCache": False,
        },
        "records": records,
    }
    return result


def print_summary(result: dict[str, Any]) -> None:
    provenance = result["provenance"]
    counts = result["matrix_counts"]
    print(f"benchmark={result['benchmark']}")
    print(f"model={provenance['model_id']}")
    print(f"model_revision={provenance['resolved_model_revision']}")
    print(f"transformers={provenance['transformers_version']}")
    print(f"device={provenance['device']}")
    print(
        "attention_reconstruction="
        f"{sum(row['passed'] for row in result['attention_reconstruction'])}/"
        f"{len(result['attention_reconstruction'])} passed"
    )
    print(
        "full_budget_invariants="
        f"{result['all_full_budget_invariants_passed']} "
        f"count={counts['full_budget_head_invariants']}"
    )
    print(f"retrieval_records={counts['retrieval_records']}")
    layer_23 = result["analysis"]["layer_23_sparse_replication"]
    print(f"layer_23_exact_topk_12.5_percent={layer_23['exact_topk']['overall']}")
    print(f"layer_23_approximate_12.5_percent={layer_23['approximate']['overall']}")


def main() -> None:
    args = parse_args()
    if args.fixture_split == "held_out" and args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite locked held-out artifact: {args.output}"
        )
    result = run_experiment(args)
    print_summary(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # json.dump writes incrementally instead of materializing a second copy of
    # the large record matrix as one in-memory string.
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
