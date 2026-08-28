#!/usr/bin/env python3
"""Opt-in real GPT-NeoX activation retrieval reference experiment."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import torch

from kvweave import BruteForceIndex, PQIndex, QuestIndex, TensorStorage
from kvweave.core.types import Selection
from kvweave.indexes.pq import reconstruct_keys
from kvweave.indexes.quest.reference import candidate_recall
from kvweave.integrations.transformers import (
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
DETERMINISTIC_TEXT = (
    # Frozen before the rename; retain these bytes for comparable model inputs.
    "KVDB studies whether key value cache retrieval can be separated from "
    "storage while preserving the attention behavior of a transformer. "
    "This repeated local corpus creates deterministic real model activations "
    "without introducing an external evaluation dataset."
)
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[256, 512, 1_024, 2_048],
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
    parser.add_argument("--output", type=Path)
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


def validate_args(args: argparse.Namespace) -> list[tuple[int, int]]:
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
    )
    if any(tolerance < 0 for tolerance in tolerances):
        raise ValueError("floating-point tolerances must be non-negative")
    return parse_pq_configurations(args.pq_configs)


def git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def deterministic_input_ids(tokenizer: Any, sequence_length: int) -> torch.Tensor:
    """Repeat one locally authored tokenized text to an exact token length."""
    base_ids = tokenizer(
        DETERMINISTIC_TEXT,
        add_special_tokens=False,
    )["input_ids"]
    if not base_ids:
        raise RuntimeError("the deterministic text produced no tokenizer IDs")
    repetitions = math.ceil(sequence_length / len(base_ids))
    token_ids = (base_ids * repetitions)[:sequence_length]
    return torch.tensor([token_ids], dtype=torch.int64)


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


def selection_covers_every_token(selection: Selection, sequence_length: int) -> bool:
    expected = torch.arange(
        sequence_length,
        device=selection.indices.device,
        dtype=torch.int64,
    )
    valid_mask = selection.valid_mask
    if valid_mask is None:
        valid_mask = torch.ones_like(selection.indices, dtype=torch.bool)
    for batch_id in range(selection.indices.shape[0]):
        for head_id in range(selection.indices.shape[1]):
            valid_indices = selection.indices[batch_id, head_id][
                valid_mask[batch_id, head_id]
            ]
            if valid_indices.shape[0] != sequence_length:
                return False
            if not torch.equal(valid_indices.sort().values, expected):
                return False
    return True


def record_selection(
    *,
    records: list[dict[str, Any]],
    model_id: str,
    model_revision: str,
    layer_index: int,
    sequence_length: int,
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
    index_build_seconds: float,
    retrieval_seconds: float,
    pq_errors: torch.Tensor | None = None,
    full_budget_invariants: list[dict[str, Any]],
    full_budget_max_relative_error: float,
    full_budget_max_absolute_error: float,
) -> None:
    retrieved = storage.fetch(selection)
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
    actual_counts = selection.valid_token_counts

    if requested_budget == sequence_length:
        covers_all = selection_covers_every_token(selection, sequence_length)
        maximum_relative_error = output_errors.max().item()
        maximum_absolute_error = (
            (selected_attention.output - full_output).abs().max().item()
        )
        attention_matches = (
            maximum_relative_error <= full_budget_max_relative_error
            and maximum_absolute_error <= full_budget_max_absolute_error
        )
        invariant = {
            "strategy": strategy,
            "configuration": configuration,
            "layer": layer_index,
            "sequence_length": sequence_length,
            "covers_every_causal_token": covers_all,
            "selected_attention_matches_full": attention_matches,
            "maximum_head_relative_output_error": maximum_relative_error,
            "maximum_absolute_output_error": maximum_absolute_error,
            "maximum_allowed_head_relative_output_error": (
                full_budget_max_relative_error
            ),
            "maximum_allowed_absolute_output_error": (full_budget_max_absolute_error),
        }
        full_budget_invariants.append(invariant)
        if not covers_all or not attention_matches:
            raise AssertionError(f"full-budget invariant failed: {invariant}")

    for head_index in range(keys.shape[1]):
        records.append(
            {
                "model_id": model_id,
                "model_revision": model_revision,
                "layer": layer_index,
                "head": head_index,
                "sequence_length": sequence_length,
                "query_position": sequence_length - 1,
                "budget_fraction": budget_fraction,
                "requested_token_budget": requested_budget,
                "actual_candidate_count": int(actual_counts[0, head_index].item()),
                "strategy": strategy,
                "configuration": configuration,
                "candidate_recall": recalls[0, head_index].item(),
                "attention_mass_captured": captured_mass[0, head_index].item(),
                "relative_attention_output_error": output_errors[0, head_index].item(),
                "pq_reconstruction_relative_error": (
                    None if pq_errors is None else pq_errors[0, head_index].item()
                ),
                "index_build_seconds": index_build_seconds,
                "retrieval_seconds": retrieval_seconds,
            }
        )


def metric_distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    include_layer: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key: tuple[Any, ...] = (
            record["strategy"],
            record["configuration"],
            record["budget_fraction"],
        )
        if include_layer:
            key += (record["layer"],)
        groups[key].append(record)

    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "strategy": key[0],
            "configuration": key[1],
            "budget_fraction": key[2],
            "sample_count": len(group),
        }
        if include_layer:
            aggregate["layer"] = key[3]
        for metric in (
            "candidate_recall",
            "attention_mass_captured",
            "relative_attention_output_error",
        ):
            aggregate[metric] = metric_distribution(
                [float(record[metric]) for record in group]
            )
        aggregates.append(aggregate)
    return aggregates


def pearson_correlation(
    records: list[dict[str, Any]],
    left: str,
    right: str,
) -> float | None:
    pairs = [
        (float(record[left]), float(record[right]))
        for record in records
        if record[left] is not None and record[right] is not None
    ]
    if len(pairs) < 2:
        return None
    left_values, right_values = zip(*pairs, strict=True)
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    left_sum_squares = sum((left_value - left_mean) ** 2 for left_value in left_values)
    right_sum_squares = sum(
        (right_value - right_mean) ** 2 for right_value in right_values
    )
    denominator = math.sqrt(left_sum_squares * right_sum_squares)
    return None if denominator == 0 else numerator / denominator


def calculate_correlations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    correlations: list[dict[str, Any]] = []
    budget_fractions = sorted({record["budget_fraction"] for record in records})
    for budget_fraction in budget_fractions:
        approximate = [
            record
            for record in records
            if record["budget_fraction"] == budget_fraction
            and record["strategy"] in {"quest", "pq"}
        ]
        pq_records = [record for record in approximate if record["strategy"] == "pq"]
        correlations.append(
            {
                "budget_fraction": budget_fraction,
                "approximate_sample_count": len(approximate),
                "candidate_recall_vs_output_error": pearson_correlation(
                    approximate,
                    "candidate_recall",
                    "relative_attention_output_error",
                ),
                "attention_mass_vs_output_error": pearson_correlation(
                    approximate,
                    "attention_mass_captured",
                    "relative_attention_output_error",
                ),
                "pq_sample_count": len(pq_records),
                "pq_reconstruction_error_vs_candidate_recall": pearson_correlation(
                    pq_records,
                    "pq_reconstruction_relative_error",
                    "candidate_recall",
                ),
                "pq_reconstruction_error_vs_attention_mass": pearson_correlation(
                    pq_records,
                    "pq_reconstruction_relative_error",
                    "attention_mass_captured",
                ),
                "pq_reconstruction_error_vs_output_error": pearson_correlation(
                    pq_records,
                    "pq_reconstruction_relative_error",
                    "relative_attention_output_error",
                ),
            }
        )
    return correlations


def monotonic_budget_checks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    metric_directions = {
        "candidate_recall": "nondecreasing",
        "attention_mass_captured": "nondecreasing",
        "relative_attention_output_error": "nonincreasing",
    }
    for strategy in ("quest", "pq"):
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["strategy"] == strategy:
                groups[
                    (
                        record["configuration"],
                        record["sequence_length"],
                        record["layer"],
                        record["head"],
                    )
                ].append(record)
        for metric, direction in metric_directions.items():
            passing_groups = 0
            for group in groups.values():
                values = [
                    record[metric]
                    for record in sorted(
                        group,
                        key=lambda record: record["budget_fraction"],
                    )
                ]
                if direction == "nondecreasing":
                    passed = all(
                        right >= left - 1e-6
                        for left, right in zip(values, values[1:], strict=False)
                    )
                else:
                    passed = all(
                        right <= left + 1e-6
                        for left, right in zip(values, values[1:], strict=False)
                    )
                passing_groups += int(passed)
            checks.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "expected_direction": direction,
                    "passing_groups": passing_groups,
                    "total_groups": len(groups),
                    "passing_fraction": passing_groups / len(groups),
                }
            )
    return checks


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


def quest_page_size_pairwise(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["strategy"] == "quest" and record["budget_fraction"] < 1.0:
            groups[
                (
                    record["sequence_length"],
                    record["layer"],
                    record["head"],
                    record["budget_fraction"],
                )
            ].append(record)
    outcomes: dict[str, dict[str, int]] = {}
    equal_count_comparisons = 0
    skipped_unequal_count = 0
    for metric, higher_is_better in (
        ("candidate_recall", True),
        ("attention_mass_captured", True),
        ("relative_attention_output_error", False),
    ):
        metric_outcomes = {"smaller_page_better": 0, "larger_page_better": 0, "tie": 0}
        for group in groups.values():
            if len(group) != 2:
                continue
            smaller, larger = sorted(group, key=lambda record: record["configuration"])
            smaller_page = int(smaller["configuration"].split("=")[1])
            larger_page = int(larger["configuration"].split("=")[1])
            if smaller_page > larger_page:
                smaller, larger = larger, smaller
            if smaller["actual_candidate_count"] != larger["actual_candidate_count"]:
                if metric == "candidate_recall":
                    skipped_unequal_count += 1
                continue
            if metric == "candidate_recall":
                equal_count_comparisons += 1
            outcome = _pairwise_outcome(
                smaller[metric],
                larger[metric],
                left_label="smaller_page_better",
                right_label="larger_page_better",
                higher_is_better=higher_is_better,
            )
            metric_outcomes[outcome] += 1
        outcomes[metric] = metric_outcomes
    return {
        "scope": "partial budgets with equal actual candidate counts",
        "equal_count_comparisons": equal_count_comparisons,
        "skipped_unequal_count_comparisons": skipped_unequal_count,
        "outcomes": outcomes,
    }


def pq_quality_pairwise(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["strategy"] == "pq" and record["budget_fraction"] < 1.0:
            groups[
                (
                    record["sequence_length"],
                    record["layer"],
                    record["head"],
                    record["budget_fraction"],
                )
            ].append(record)
    outcomes: dict[str, dict[str, int]] = {}
    quality_comparisons = 0
    for metric, higher_is_better in (
        ("candidate_recall", True),
        ("attention_mass_captured", True),
        ("relative_attention_output_error", False),
    ):
        metric_outcomes = {
            "higher_quality_better": 0,
            "higher_quality_worse": 0,
            "tie": 0,
        }
        for group in groups.values():
            if len(group) < 2:
                continue
            ranked = sorted(
                group,
                key=lambda record: record["pq_reconstruction_relative_error"],
            )
            higher_quality = ranked[0]
            lower_quality = ranked[-1]
            if (
                higher_quality["pq_reconstruction_relative_error"]
                == lower_quality["pq_reconstruction_relative_error"]
            ):
                continue
            if metric == "candidate_recall":
                quality_comparisons += 1
            outcome = _pairwise_outcome(
                higher_quality[metric],
                lower_quality[metric],
                left_label="higher_quality_better",
                right_label="higher_quality_worse",
                higher_is_better=higher_is_better,
            )
            metric_outcomes[outcome] += 1
        outcomes[metric] = metric_outcomes
    return {
        "scope": "best versus worst PQ reconstruction at fixed context/layer/head/budget",
        "quality_comparisons": quality_comparisons,
        "outcomes": outcomes,
    }


def reconstruct_and_validate_attention(
    activations: GPTNeoXLayerActivations,
    *,
    sequence_length: int,
    attention_scale: float,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if sequence_length != activations.key.shape[2]:
        raise ValueError("reconstruction requires a matching-length model forward")
    causal_output = reference_causal_attention(
        activations.query,
        activations.key,
        activations.value,
        scale=attention_scale,
    )
    projected = project_head_outputs(
        causal_output[:, :, sequence_length - 1, :],
        activations.dense_weight,
        activations.dense_bias,
    )
    model_output = activations.projected_attention_output[:, sequence_length - 1]
    passed = torch.allclose(projected, model_output, rtol=rtol, atol=atol)
    result = {
        "layer": activations.layer_index,
        "sequence_length": sequence_length,
        "query_position": sequence_length - 1,
        "passed": passed,
        "relative_output_error": tensor_relative_error(projected, model_output),
        "maximum_absolute_error": (projected - model_output).abs().max().item(),
        "rtol": rtol,
        "atol": atol,
    }
    if not passed:
        raise AssertionError(f"full attention reconstruction failed: {result}")
    return result


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    pq_configurations = validate_args(args)
    try:
        from transformers import AutoModel, AutoTokenizer, __version__
    except ImportError as error:
        raise RuntimeError(
            "install the optional model experiment dependency: "
            "pip install -e '.[model-experiment]'"
        ) from error

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
    if any(
        num_centroids > sequence_length
        for _, num_centroids in pq_configurations
        for sequence_length in args.sequence_lengths
    ):
        raise ValueError("PQ centroids cannot exceed a tested sequence length")

    input_ids = deterministic_input_ids(tokenizer, maximum_length)
    input_hash = hashlib.sha256(
        json.dumps(input_ids.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    captures = {}
    for sequence_length in sorted(args.sequence_lengths):
        sequence_input_ids = input_ids[:, :sequence_length].to(device)
        captures[sequence_length] = capture_gpt_neox_activations(
            model,
            sequence_input_ids,
            layer_indices=layers,
            attention_mask=torch.ones_like(sequence_input_ids, device=device),
            capture_device="cpu",
            capture_dtype=torch.float32,
        )
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

    reconstruction = []
    for sequence_length in sorted(args.sequence_lengths):
        for layer_index in layers:
            reconstruction.append(
                reconstruct_and_validate_attention(
                    captures[sequence_length].layers[layer_index],
                    sequence_length=sequence_length,
                    attention_scale=architecture.attention_scale,
                    rtol=args.reconstruction_rtol,
                    atol=args.reconstruction_atol,
                )
            )

    records: list[dict[str, Any]] = []
    full_budget_invariants: list[dict[str, Any]] = []
    for sequence_length in sorted(args.sequence_lengths):
        for layer_index in layers:
            sliced = causal_slice(
                captures[sequence_length].layers[layer_index],
                sequence_length - 1,
            )
            full = reference_attention(
                sliced.query,
                sliced.keys,
                sliced.values,
                scale=architecture.attention_scale,
            )
            storage = TensorStorage()
            storage.put(sliced.keys, sliced.values)

            start = time.perf_counter()
            brute_force = BruteForceIndex()
            brute_force.build(sliced.keys)
            brute_build_seconds = time.perf_counter() - start

            for budget_fraction in sorted(args.budget_fractions):
                requested_budget = max(
                    1,
                    min(
                        sequence_length,
                        round(sequence_length * budget_fraction),
                    ),
                )
                exact_start = time.perf_counter()
                exact_topk = brute_force.search(sliced.query, requested_budget)
                exact_retrieval_seconds = time.perf_counter() - exact_start
                record_selection(
                    records=records,
                    model_id=args.model_id,
                    model_revision=resolved_revision,
                    layer_index=layer_index,
                    sequence_length=sequence_length,
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
                    index_build_seconds=brute_build_seconds,
                    retrieval_seconds=exact_retrieval_seconds,
                    full_budget_invariants=full_budget_invariants,
                    full_budget_max_relative_error=(
                        args.full_budget_max_relative_error
                    ),
                    full_budget_max_absolute_error=(
                        args.full_budget_max_absolute_error
                    ),
                )

            for page_size in args.page_sizes:
                start = time.perf_counter()
                quest = QuestIndex(page_size=page_size)
                quest.build(sliced.keys)
                build_seconds = time.perf_counter() - start
                for budget_fraction in sorted(args.budget_fractions):
                    requested_budget = max(
                        1,
                        min(
                            sequence_length,
                            round(sequence_length * budget_fraction),
                        ),
                    )
                    exact_topk = brute_force.search(sliced.query, requested_budget)
                    start = time.perf_counter()
                    selection = quest.search(sliced.query, requested_budget)
                    retrieval_seconds = time.perf_counter() - start
                    record_selection(
                        records=records,
                        model_id=args.model_id,
                        model_revision=resolved_revision,
                        layer_index=layer_index,
                        sequence_length=sequence_length,
                        budget_fraction=budget_fraction,
                        requested_budget=requested_budget,
                        strategy="quest",
                        configuration=f"page_size={page_size}",
                        selection=selection,
                        exact_topk=exact_topk,
                        query=sliced.query,
                        keys=sliced.keys,
                        storage=storage,
                        full_output=full.output,
                        full_weights=full.weights,
                        attention_scale=architecture.attention_scale,
                        index_build_seconds=build_seconds,
                        retrieval_seconds=retrieval_seconds,
                        full_budget_invariants=full_budget_invariants,
                        full_budget_max_relative_error=(
                            args.full_budget_max_relative_error
                        ),
                        full_budget_max_absolute_error=(
                            args.full_budget_max_absolute_error
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
                pq_errors = pq_reconstruction_error_by_head(
                    reconstruct_keys(pq.metadata),
                    sliced.keys,
                )
                for budget_fraction in sorted(args.budget_fractions):
                    requested_budget = max(
                        1,
                        min(
                            sequence_length,
                            round(sequence_length * budget_fraction),
                        ),
                    )
                    exact_topk = brute_force.search(sliced.query, requested_budget)
                    start = time.perf_counter()
                    selection = pq.search(sliced.query, requested_budget)
                    retrieval_seconds = time.perf_counter() - start
                    record_selection(
                        records=records,
                        model_id=args.model_id,
                        model_revision=resolved_revision,
                        layer_index=layer_index,
                        sequence_length=sequence_length,
                        budget_fraction=budget_fraction,
                        requested_budget=requested_budget,
                        strategy="pq",
                        configuration=(
                            f"subspaces={num_subspaces},centroids={num_centroids},"
                            f"iterations={args.kmeans_iterations}"
                        ),
                        selection=selection,
                        exact_topk=exact_topk,
                        query=sliced.query,
                        keys=sliced.keys,
                        storage=storage,
                        full_output=full.output,
                        full_weights=full.weights,
                        attention_scale=architecture.attention_scale,
                        index_build_seconds=build_seconds,
                        retrieval_seconds=retrieval_seconds,
                        pq_errors=pq_errors,
                        full_budget_invariants=full_budget_invariants,
                        full_budget_max_relative_error=(
                            args.full_budget_max_relative_error
                        ),
                        full_budget_max_absolute_error=(
                            args.full_budget_max_absolute_error
                        ),
                    )

    result = {
        "schema_version": 1,
        "benchmark": "real_model_activation_reference",
        "scope": (
            "internal activation retrieval only; no generation, perplexity, "
            "downstream quality, or speed claim"
        ),
        "provenance": {
            "model_id": args.model_id,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": resolved_revision,
            "transformers_version": __version__,
            "transformers_attention_implementation": attention_implementation,
            "torch_version": torch.__version__,
            "model_dtype": args.model_dtype,
            "capture_and_reference_dtype": "float32",
            "device": str(device),
            "hardware": platform.platform(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
        },
        "architecture": asdict(captures[maximum_length].architecture),
        "input": {
            "method": "locally authored text tokenized once and repeated",
            "external_dataset": None,
            "maximum_sequence_length": maximum_length,
            "token_ids_sha256": input_hash,
        },
        "configuration": {
            "sequence_lengths": sorted(args.sequence_lengths),
            "layers": list(layers),
            "heads": list(range(architecture.num_attention_heads)),
            "budget_fractions": sorted(args.budget_fractions),
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
            "query_position_policy": "last token in each causal prefix",
            "retrieval_objective": "unscaled raw query-key dot product",
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
            },
        },
        "attention_reconstruction": reconstruction,
        "full_budget_invariants": full_budget_invariants,
        "all_full_budget_invariants_passed": all(
            invariant["covers_every_causal_token"]
            and invariant["selected_attention_matches_full"]
            for invariant in full_budget_invariants
        ),
        "aggregates_by_strategy_budget": aggregate_records(
            records,
            include_layer=False,
        ),
        "aggregates_by_strategy_budget_layer": aggregate_records(
            records,
            include_layer=True,
        ),
        "correlations_by_budget": calculate_correlations(records),
        "monotonic_budget_checks": monotonic_budget_checks(records),
        "quest_page_size_pairwise": quest_page_size_pairwise(records),
        "pq_quality_pairwise": pq_quality_pairwise(records),
        "records": records,
    }
    return result


def print_summary(result: dict[str, Any]) -> None:
    provenance = result["provenance"]
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
        f"count={len(result['full_budget_invariants'])}"
    )
    for row in result["aggregates_by_strategy_budget"]:
        recall = row["candidate_recall"]
        mass = row["attention_mass_captured"]
        error = row["relative_attention_output_error"]
        print(
            f"strategy={row['strategy']} config={row['configuration']} "
            f"budget={row['budget_fraction']:.3f} n={row['sample_count']} "
            f"recall_mean={recall['mean']:.6f} recall_min={recall['min']:.6f} "
            f"mass_mean={mass['mean']:.6f} mass_min={mass['min']:.6f} "
            f"error_mean={error['mean']:.6f} error_max={error['max']:.6f}"
        )


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
