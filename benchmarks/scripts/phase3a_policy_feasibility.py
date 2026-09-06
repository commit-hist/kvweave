#!/usr/bin/env python3
"""Freeze and evaluate the experimental Phase 3A policy-feasibility model."""

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
from pathlib import Path
import statistics
import time
from typing import Any

from benchmarks.artifacts import load_json, write_new_json, require_schema_version
from benchmarks.support import git_commit, git_is_dirty
from benchmarks.policy_feasibility import (
    CANDIDATE_CONFIGURATIONS,
    DEVELOPMENT_FIXTURES,
    DEVELOPMENT_TOKEN_ID_SHA256,
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FORBIDDEN_INFERENCE_FEATURES,
    HELD_OUT_FIXTURES,
    LOCKED_FIXTURE_TEXT_SHA256,
    LOCKED_TOKEN_ID_SHA256,
    OBSERVATION_KEY_FIELDS,
    PARTIAL_BUDGETS,
    LogisticModel,
    assemble_policy_examples,
    bootstrap_fixture_mean_interval,
    distribution,
    error_regret,
    fit_mass_lookup,
    index_memory_bytes,
    maintained_key_metadata_bytes,
    mass_regret,
    measure_prediction_latency,
    oracle_gap_recovery,
    predict_budget_specific_logistic,
    predict_mass_lookup,
    train_development_logistic_models,
)


DEFAULT_DEVELOPMENT_OUTCOMES = Path(
    "benchmarks/results/pythia-410m-phase3a-replication.json"
)
DEFAULT_DEVELOPMENT_FEATURES = Path(
    "benchmarks/results/pythia-410m-phase3a-policy-development-features.json"
)
DEFAULT_FROZEN_MODEL = Path("benchmarks/results/pythia-410m-phase3a-policy-frozen.json")
DEFAULT_HELD_OUT_OUTCOMES = Path(
    "benchmarks/results/pythia-410m-phase3a-policy-held-out-outcomes.json"
)
DEFAULT_HELD_OUT_FEATURES = Path(
    "benchmarks/results/pythia-410m-phase3a-policy-held-out-features.json"
)
DEFAULT_EVALUATION = Path(
    "benchmarks/results/pythia-410m-phase3a-policy-evaluation.json"
)
MODEL_ID = "EleutherAI/pythia-410m"
MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
TRANSFORMERS_VERSION = "5.15.1"
TRAINING_SEED = 0
LOGISTIC_LEARNING_RATE = 0.05
LOGISTIC_EPOCHS = 250
BOOTSTRAP_SAMPLES = 2_000
NEAR_ZERO_ORACLE_GAP = 1e-8

# Frozen before held-out evaluation. These thresholds classify evidence rather
# than tune any retrieval or predictor behavior.
DECISION_CRITERIA = {
    "material_attention_mass": 0.01,
    "nearly_all_oracle_gap_recovered": 0.75,
    "budgets_required": 2,
    "query_adaptive_gain_requires_positive_cluster_ci": True,
    "maximum_query_overhead_fraction_of_retrieval": 0.10,
    "tiny_failure_mass_regret": 0.01,
    "large_failure_mass_regret": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze",
        help="train/select only on development fixtures and serialize the lock",
    )
    freeze.add_argument(
        "--development-outcomes",
        type=Path,
        default=DEFAULT_DEVELOPMENT_OUTCOMES,
    )
    freeze.add_argument(
        "--development-features",
        type=Path,
        default=DEFAULT_DEVELOPMENT_FEATURES,
    )
    freeze.add_argument("--output", type=Path, default=DEFAULT_FROZEN_MODEL)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="apply a frozen development model once to locked held-out outcomes",
    )
    evaluate.add_argument("--frozen-model", type=Path, default=DEFAULT_FROZEN_MODEL)
    evaluate.add_argument(
        "--held-out-outcomes",
        type=Path,
        default=DEFAULT_HELD_OUT_OUTCOMES,
    )
    evaluate.add_argument(
        "--held-out-features",
        type=Path,
        default=DEFAULT_HELD_OUT_FEATURES,
    )
    evaluate.add_argument("--output", type=Path, default=DEFAULT_EVALUATION)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_ids(fixtures: Sequence[Any]) -> set[str]:
    return {str(fixture.fixture_id) for fixture in fixtures}


def validate_feature_artifact(payload: Mapping[str, Any], split: str) -> None:
    require_schema_version(payload, supported=(1,))
    if payload.get("artifact") != "pythia_410m_phase3a_pre_retrieval_features":
        raise ValueError("unexpected feature artifact type")
    if payload.get("fixture_split") != split:
        raise ValueError("feature artifact split does not match the requested stage")
    provenance = payload["provenance"]
    if (
        provenance["model_id"] != MODEL_ID
        or provenance["resolved_model_revision"] != MODEL_REVISION
        or provenance["transformers_version"] != TRANSFORMERS_VERSION
        or provenance["transformers_attention_implementation"] != "eager"
    ):
        raise ValueError("feature artifact does not use the frozen model setup")
    expected_fixtures = (
        _fixture_ids(DEVELOPMENT_FIXTURES)
        if split == "development"
        else _fixture_ids(HELD_OUT_FIXTURES)
    )
    actual_fixtures = {
        str(row["text_fixture_id"]) for row in payload["fixture_manifest"]
    }
    if actual_fixtures != expected_fixtures:
        raise ValueError("feature artifact fixture membership is not locked")
    if payload["configuration"] != {
        "sequence_lengths": [512, 2048],
        "query_positions": ["25_percent", "50_percent", "75_percent", "final"],
        "layers": [0, 12, 23],
        "heads": list(range(16)),
    }:
        raise ValueError("feature artifact experiment matrix is not frozen")


def validate_outcome_artifact(payload: Mapping[str, Any], split: str) -> None:
    require_schema_version(payload, supported=(1, 2))
    if payload.get("benchmark") != "pythia_410m_phase3a_structural_replication":
        raise ValueError("unexpected outcome artifact type")
    provenance = payload["provenance"]
    if (
        provenance["model_id"] != MODEL_ID
        or provenance["resolved_model_revision"] != MODEL_REVISION
        or provenance["transformers_version"] != TRANSFORMERS_VERSION
        or provenance["transformers_attention_implementation"] != "eager"
        or int(provenance["seed"]) != TRAINING_SEED
    ):
        raise ValueError("outcome artifact does not use the frozen model setup")
    configuration = payload["configuration"]
    if (
        configuration["sequence_lengths"] != [512, 2048]
        or configuration["layers"] != [0, 12, 23]
        or configuration["heads"] != list(range(16))
        or configuration["budget_fractions"] != [0.125, 0.25, 0.5, 1.0]
        or configuration["quest_page_sizes"] != [16, 64]
        or configuration["pq_configurations"]
        != [
            {
                "max_iterations": 8,
                "num_centroids": 4,
                "num_subspaces": 2,
                "seed": 0,
            },
            {
                "max_iterations": 8,
                "num_centroids": 8,
                "num_subspaces": 4,
                "seed": 0,
            },
        ]
    ):
        raise ValueError(
            "outcome artifact candidate/matrix configuration is not frozen"
        )
    expected_fixtures = (
        _fixture_ids(DEVELOPMENT_FIXTURES)
        if split == "development"
        else _fixture_ids(HELD_OUT_FIXTURES)
    )
    actual_fixtures = {
        str(row["text_fixture_id"]) for row in payload["input"]["fixtures"]
    }
    if actual_fixtures != expected_fixtures:
        raise ValueError("outcome fixture membership is not locked")
    recorded_split = payload["input"].get("fixture_split")
    if split == "held_out" and recorded_split != "held_out":
        raise ValueError("held-out outcome artifact lacks the split lock")
    if split == "development" and recorded_split not in {None, "development"}:
        raise ValueError("development outcome artifact has the wrong split")


def validate_tokenizations(
    outcome_payload: Mapping[str, Any],
    feature_payload: Mapping[str, Any],
    split: str,
) -> None:
    expected = (
        DEVELOPMENT_TOKEN_ID_SHA256
        if split == "development"
        else LOCKED_TOKEN_ID_SHA256
    )
    outcome_rows = outcome_payload["input"]["tokenizations"]
    feature_rows = feature_payload["tokenizations"]
    for source_name, rows in (("outcomes", outcome_rows), ("features", feature_rows)):
        actual = {
            (str(row["text_fixture_id"]), int(row["sequence_length"])): str(
                row["token_ids_sha256"]
            )
            for row in rows
        }
        locked = {
            (fixture_id, length): digest
            for fixture_id, lengths in expected.items()
            for length, digest in lengths.items()
        }
        if actual != locked:
            raise ValueError(
                f"{source_name} token hashes do not match the {split} lock"
            )


def _lookup_models(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "global_fixed": {
            "group_fields": [],
            "rows": fit_mass_lookup(examples, group_fields=()),
        },
        "layer_fixed": {
            "group_fields": ["layer"],
            "rows": fit_mass_lookup(examples, group_fields=("layer",)),
        },
        "layer_head_fixed": {
            "group_fields": ["layer", "head"],
            "rows": fit_mass_lookup(examples, group_fields=("layer", "head")),
        },
    }


def _development_difficult_heads(
    examples: Sequence[Mapping[str, Any]],
    global_lookup: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source = [
        example
        for example in examples
        if int(example["layer"]) == 23 and float(example["budget_fraction"]) == 0.125
    ]
    fixed = predict_mass_lookup(global_lookup, source, group_fields=())
    by_head: dict[int, list[float]] = defaultdict(list)
    for example, configuration in zip(source, fixed, strict=True):
        oracle = str(example["mass_oracle_configuration"])
        by_head[int(example["head"])].append(
            float(
                example["outcomes"][oracle]["attention_mass_captured"]
                - example["outcomes"][configuration]["attention_mass_captured"]
            )
        )
    rows = [
        {
            "head": head,
            "development_mean_global_mass_regret_at_layer23_12_5_percent": (
                statistics.fmean(values)
            ),
        }
        for head, values in by_head.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            -float(row["development_mean_global_mass_regret_at_layer23_12_5_percent"]),
            int(row["head"]),
        ),
    )[:3]


def freeze_development(args: argparse.Namespace) -> dict[str, Any]:
    outcomes = load_json(args.development_outcomes)
    features = load_json(args.development_features)
    validate_outcome_artifact(outcomes, "development")
    validate_feature_artifact(features, "development")
    validate_tokenizations(outcomes, features, "development")
    examples = assemble_policy_examples(features["feature_rows"], outcomes["records"])
    lookup_models = _lookup_models(examples)
    logistic_models, cross_validation = train_development_logistic_models(
        examples,
        learning_rate=LOGISTIC_LEARNING_RATE,
        epochs=LOGISTIC_EPOCHS,
        seed=TRAINING_SEED,
    )
    difficult_heads = _development_difficult_heads(
        examples,
        lookup_models["global_fixed"]["rows"],
    )
    normalized_entropy_threshold = float(
        outcomes["analysis"]["entropy_stratification"]["thresholds"]["low_to_middle"]
    )
    return {
        "schema_version": 1,
        "artifact": "pythia_410m_phase3a_policy_development_freeze",
        "research_question": (
            "Can cheap pre-retrieval features beat the development-selected best "
            "fixed strategy on completely held-out structural inputs after cost?"
        ),
        "provenance": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "training_seed": TRAINING_SEED,
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "development_outcome_artifact": str(args.development_outcomes),
            "development_outcome_sha256": sha256_file(args.development_outcomes),
            "development_feature_artifact": str(args.development_features),
            "development_feature_sha256": sha256_file(args.development_features),
        },
        "development_fixture_ids": sorted(_fixture_ids(DEVELOPMENT_FIXTURES)),
        "held_out_fixture_ids_forbidden_during_training": sorted(
            _fixture_ids(HELD_OUT_FIXTURES)
        ),
        "development_token_id_sha256": DEVELOPMENT_TOKEN_ID_SHA256,
        "held_out_fixture_text_sha256": LOCKED_FIXTURE_TEXT_SHA256,
        "held_out_token_id_sha256": LOCKED_TOKEN_ID_SHA256,
        "candidate_order_and_tie_breaking": list(CANDIDATE_CONFIGURATIONS),
        "partial_budgets": list(PARTIAL_BUDGETS),
        "target": (
            "candidate with maximum post-hoc attention mass; exact ties use frozen "
            "candidate order"
        ),
        "error_oracle": (
            "candidate with minimum post-hoc per-head relative output error; analysis only"
        ),
        "feature_names": list(FEATURE_NAMES),
        "feature_definitions": [
            asdict(definition) for definition in FEATURE_DEFINITIONS
        ],
        "forbidden_inference_features": sorted(FORBIDDEN_INFERENCE_FEATURES),
        "predictor": {
            "family": "budget-specific multinomial logistic regression",
            "dependency": "pure PyTorch; no additional ML framework",
            "learning_rate": LOGISTIC_LEARNING_RATE,
            "epochs": LOGISTIC_EPOCHS,
            "l2_selection": "leave-one-development-fixture-out attention mass",
            "models": [model.to_json() for model in logistic_models],
            "development_cross_validation": cross_validation,
        },
        "baseline_models": lookup_models,
        "development_selected_difficult_heads": difficult_heads,
        "development_low_normalized_entropy_threshold": normalized_entropy_threshold,
        "decision_criteria_frozen_before_held_out_evaluation": DECISION_CRITERIA,
        "development_example_count": len(examples),
        "shared_architecture_changes_required": {
            "KVIndex": False,
            "Selection": False,
            "KVStorage": False,
            "RetrievedKV": False,
            "KVCache": False,
        },
    }


def _predict_all(
    frozen: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for name in ("global_fixed", "layer_fixed", "layer_head_fixed"):
        model = frozen["baseline_models"][name]
        group_fields = tuple(str(field) for field in model["group_fields"])
        predictions[name] = predict_mass_lookup(
            model["rows"],
            examples,
            group_fields=group_fields,
        )
    logistic_models = [
        LogisticModel.from_json(payload) for payload in frozen["predictor"]["models"]
    ]
    predictions["learned_cheap_features"] = predict_budget_specific_logistic(
        logistic_models,
        examples,
    )
    predictions["retrospective_mass_oracle"] = [
        str(example["mass_oracle_configuration"]) for example in examples
    ]
    predictions["retrospective_error_oracle"] = [
        str(example["error_oracle_configuration"]) for example in examples
    ]
    return predictions


def _prediction_rows(
    examples: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        oracle_configuration = str(example["mass_oracle_configuration"])
        error_oracle_configuration = str(example["error_oracle_configuration"])
        oracle_mass = float(
            example["outcomes"][oracle_configuration]["attention_mass_captured"]
        )
        oracle_error = float(
            example["outcomes"][error_oracle_configuration][
                "relative_attention_output_error"
            ]
        )
        fixed_configuration = predictions["global_fixed"][example_index]
        fixed_mass = float(
            example["outcomes"][fixed_configuration]["attention_mass_captured"]
        )
        row: dict[str, Any] = {
            **{field: example[field] for field in OBSERVATION_KEY_FIELDS},
            "query_position": example["query_position"],
            "budget_fraction": example["budget_fraction"],
            "mass_oracle_configuration": oracle_configuration,
            "mass_oracle_attention_mass": oracle_mass,
            "error_oracle_configuration": error_oracle_configuration,
            "error_oracle_output_error": oracle_error,
            "normalized_attention_entropy_post_hoc": example["post_hoc_diagnostics"][
                "normalized_attention_entropy"
            ],
            "candidate_outcomes_post_hoc": example["outcomes"],
            "predictors": {},
        }
        for name, predictor_configurations in predictions.items():
            configuration = predictor_configurations[example_index]
            outcome = example["outcomes"][configuration]
            predicted_mass = float(outcome["attention_mass_captured"])
            predicted_error = float(outcome["relative_attention_output_error"])
            row["predictors"][name] = {
                "configuration": configuration,
                "attention_mass": predicted_mass,
                "output_error": predicted_error,
                "mass_regret": mass_regret(oracle_mass, predicted_mass),
                "error_regret": error_regret(predicted_error, oracle_error),
                "oracle_configuration_correct": configuration == oracle_configuration,
                "oracle_gap_recovered": oracle_gap_recovery(
                    predicted_mass,
                    fixed_mass,
                    oracle_mass,
                    epsilon=NEAR_ZERO_ORACLE_GAP,
                ),
            }
        rows.append(row)
    return rows


def _flat_policy_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    predictor: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in prediction_rows:
        metrics = source["predictors"][predictor]
        rows.append(
            {
                **{
                    field: source[field]
                    for field in (
                        *OBSERVATION_KEY_FIELDS,
                        "query_position",
                        "budget_fraction",
                    )
                },
                **metrics,
            }
        )
    return rows


def _summarize_group(
    group: Sequence[Mapping[str, Any]],
    *,
    fixed_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": len(group),
        "attention_mass": distribution(row["attention_mass"] for row in group),
        "output_error": distribution(row["output_error"] for row in group),
        "mass_regret": distribution(row["mass_regret"] for row in group),
        "error_regret": distribution(row["error_regret"] for row in group),
        "oracle_gap_recovered_per_observation": distribution(
            row["oracle_gap_recovered"] for row in group
        ),
        "near_zero_oracle_gap_count": sum(
            row["oracle_gap_recovered"] is None for row in group
        ),
        "oracle_configuration_accuracy": statistics.fmean(
            float(row["oracle_configuration_correct"]) for row in group
        ),
    }
    if fixed_rows is not None:
        if len(fixed_rows) != len(group):
            raise ValueError("fixed/predictor rows must align")
        mean_predicted = statistics.fmean(float(row["attention_mass"]) for row in group)
        mean_fixed = statistics.fmean(
            float(row["attention_mass"]) for row in fixed_rows
        )
        mean_oracle = statistics.fmean(
            float(row["attention_mass"] + row["mass_regret"]) for row in group
        )
        result["oracle_gap_recovered_from_group_means"] = oracle_gap_recovery(
            mean_predicted,
            mean_fixed,
            mean_oracle,
            epsilon=NEAR_ZERO_ORACLE_GAP,
        )
    return result


def _stratified_summary(
    prediction_rows: Sequence[Mapping[str, Any]],
    predictor: str,
) -> dict[str, Any]:
    flat = _flat_policy_rows(prediction_rows, predictor)
    fixed = _flat_policy_rows(prediction_rows, "global_fixed")
    result: dict[str, Any] = {}
    for field in (
        "budget_fraction",
        "layer",
        "head",
        "query_position_label",
        "sequence_length",
        "text_fixture_id",
    ):
        groups: dict[Any, list[int]] = defaultdict(list)
        for index, row in enumerate(flat):
            groups[row[field]].append(index)
        result[f"by_{field}"] = [
            {
                field: value,
                **_summarize_group(
                    [flat[index] for index in indices],
                    fixed_rows=[fixed[index] for index in indices],
                ),
            }
            for value, indices in sorted(groups.items(), key=lambda item: str(item[0]))
        ]
    return result


def _budget_confidence_intervals(
    prediction_rows: Sequence[Mapping[str, Any]],
    predictor: str,
) -> list[dict[str, Any]]:
    flat = _flat_policy_rows(prediction_rows, predictor)
    rows: list[dict[str, Any]] = []
    for budget_index, budget in enumerate(PARTIAL_BUDGETS):
        group = [row for row in flat if float(row["budget_fraction"]) == budget]
        result: dict[str, Any] = {"budget_fraction": budget}
        for metric_index, metric in enumerate(
            ("attention_mass", "output_error", "mass_regret", "error_regret")
        ):
            result[metric] = bootstrap_fixture_mean_interval(
                group,
                metric,
                seed=TRAINING_SEED + budget_index * 10 + metric_index,
                samples=BOOTSTRAP_SAMPLES,
            )
        rows.append(result)
    return rows


def _paired_query_gain_intervals(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    learned = _flat_policy_rows(prediction_rows, "learned_cheap_features")
    identity = _flat_policy_rows(prediction_rows, "layer_head_fixed")
    gain_rows = [
        {
            **{field: learned_row[field] for field in OBSERVATION_KEY_FIELDS},
            "budget_fraction": learned_row["budget_fraction"],
            "attention_mass_gain_over_layer_head": (
                float(learned_row["attention_mass"])
                - float(identity_row["attention_mass"])
            ),
        }
        for learned_row, identity_row in zip(learned, identity, strict=True)
    ]
    results = []
    for budget_index, budget in enumerate(PARTIAL_BUDGETS):
        group = [row for row in gain_rows if float(row["budget_fraction"]) == budget]
        results.append(
            {
                "budget_fraction": budget,
                "attention_mass_gain_over_layer_head": distribution(
                    row["attention_mass_gain_over_layer_head"] for row in group
                ),
                "fixture_cluster_bootstrap_95": bootstrap_fixture_mean_interval(
                    group,
                    "attention_mass_gain_over_layer_head",
                    seed=TRAINING_SEED + 100 + budget_index,
                    samples=BOOTSTRAP_SAMPLES,
                ),
            }
        )
    return results


def _deduplicate_batch_costs(
    feature_payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    fields = (
        "text_fixture_id",
        "sequence_length",
        "query_position_label",
        "layer",
    )
    for row in feature_payload["feature_rows"]:
        key = tuple(row[field] for field in fields)
        rows.setdefault(key, row)
    return list(rows.values())


def _cost_analysis(
    frozen: Mapping[str, Any],
    features: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    batch_costs = _deduplicate_batch_costs(features)
    models = [LogisticModel.from_json(row) for row in frozen["predictor"]["models"]]
    prediction_costs = []
    for model in models:
        model_rows = [
            example["features"]
            for example in examples
            if float(example["budget_fraction"]) == model.budget_fraction
        ]
        prediction_costs.append(
            {
                "budget_fraction": model.budget_fraction,
                "seconds_per_head_observation": measure_prediction_latency(
                    model,
                    model_rows,
                    repetitions=20,
                ),
            }
        )
    retrieval_calls: dict[tuple[Any, ...], float] = {}
    for record in outcomes["records"]:
        if record["strategy"] not in {"quest", "pq"}:
            continue
        key = (
            *(record[field] for field in OBSERVATION_KEY_FIELDS if field != "head"),
            record["strategy"],
            record["configuration"],
            record["budget_fraction"],
        )
        retrieval_calls.setdefault(key, float(record["retrieval_seconds"]))
    feature_batch_median = statistics.median(
        float(row["feature_extraction_seconds_batch"]) for row in batch_costs
    )
    prediction_per_head_medians = [
        float(row["seconds_per_head_observation"]["median"]) for row in prediction_costs
    ]
    predicted_batch_median = statistics.median(prediction_per_head_medians) * 16
    retrieval_median = statistics.median(retrieval_calls.values())
    query_overhead = feature_batch_median + predicted_batch_median
    return {
        "feature_extraction_seconds_batch_all_16_heads": distribution(
            row["feature_extraction_seconds_batch"] for row in batch_costs
        ),
        "offline_prefix_metadata_construction_seconds_batch": distribution(
            row["metadata_construction_seconds_batch"] for row in batch_costs
        ),
        "prefix_metadata_note": (
            "measured construction scans the prefix for experiment convenience; in "
            "deployment D+4 float statistics and a count are maintainable in O(H*D) "
            "per appended token and are available before retrieval"
        ),
        "prediction_latency": prediction_costs,
        "persistent_feature_metadata": {
            "bytes_per_head": maintained_key_metadata_bytes(head_dimension=64),
            "bytes_per_layer_16_heads": maintained_key_metadata_bytes(
                head_dimension=64,
                num_heads=16,
            ),
            "bytes_full_24_layer_model": maintained_key_metadata_bytes(
                head_dimension=64,
                num_heads=16 * 24,
            ),
        },
        "approximate_candidate_retrieval_seconds_batch": distribution(
            retrieval_calls.values()
        ),
        "estimated_feature_plus_prediction_median_seconds_batch": query_overhead,
        "estimated_query_overhead_fraction_of_median_candidate_retrieval": (
            query_overhead / retrieval_median if retrieval_median > 0 else None
        ),
        "strategy_switching": {
            "implemented": False,
            "estimated_dispatch_cost": (
                "included in prediction timing only; no index build, migration, or "
                "lifecycle operation is performed per query"
            ),
            "all_four_choice_requires_both_quest_and_pq_resident": True,
        },
    }


def _index_coexistence() -> dict[str, Any]:
    return {
        "excludes_shared_full_precision_kv": True,
        "deployment_models": {
            "model_a": "all four frozen indexes are built and resident",
            "model_b": (
                "a selected subset is resident; Quest-only and PQ-only pairs are "
                "reported alongside individual index costs"
            ),
        },
        "reference_code_note": (
            "actual PQ code tensors use int64; logical packed estimates use 2-bit "
            "M2/C4 and 3-bit M4/C8 codes"
        ),
        "by_sequence_length": [
            {
                "sequence_length": sequence_length,
                "per_layer_16_heads": index_memory_bytes(
                    sequence_length=sequence_length,
                    head_dimension=64,
                    num_heads=16,
                    num_layers=1,
                ),
                "full_24_layer_model": index_memory_bytes(
                    sequence_length=sequence_length,
                    head_dimension=64,
                    num_heads=16,
                    num_layers=24,
                ),
            }
            for sequence_length in (512, 2048)
        ],
    }


def _failure_analysis(
    prediction_rows: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    learned = _flat_policy_rows(prediction_rows, "learned_cheap_features")
    difficult_heads = {
        int(row["head"]) for row in frozen["development_selected_difficult_heads"]
    }
    tiny = float(DECISION_CRITERIA["tiny_failure_mass_regret"])
    large = float(DECISION_CRITERIA["large_failure_mass_regret"])
    wrong = [row for row in learned if not row["oracle_configuration_correct"]]

    def clusters(field: str) -> list[dict[str, Any]]:
        grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for row in learned:
            grouped[row[field]].append(row)
        return [
            {
                field: value,
                "mass_regret": distribution(row["mass_regret"] for row in group),
                "large_mass_regret_count": sum(
                    float(row["mass_regret"]) >= large for row in group
                ),
                "sample_count": len(group),
            }
            for value, group in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]

    low_entropy_threshold = float(
        frozen["development_low_normalized_entropy_threshold"]
    )
    late_layer_source = [
        row
        for row in prediction_rows
        if int(row["layer"]) == 23
        and float(row["normalized_attention_entropy_post_hoc"]) <= low_entropy_threshold
    ]
    late_layer_flat = []
    for row in late_layer_source:
        metrics = row["predictors"]["learned_cheap_features"]
        late_layer_flat.append({**row, **metrics})
    return {
        "wrong_strategy_but_mass_regret_at_most_0_01": sum(
            float(row["mass_regret"]) <= tiny for row in wrong
        ),
        "wrong_strategy_count": len(wrong),
        "large_attention_mass_loss_at_least_0_10": sum(
            float(row["mass_regret"]) >= large for row in learned
        ),
        "by_head": clusters("head"),
        "by_fixture": clusters("text_fixture_id"),
        "by_query_position": clusters("query_position_label"),
        "layer_23": {
            "all": _summarize_group(
                [row for row in learned if int(row["layer"]) == 23]
            ),
            "by_budget": [
                {
                    "budget_fraction": budget,
                    **_summarize_group(
                        [
                            row
                            for row in learned
                            if int(row["layer"]) == 23
                            and float(row["budget_fraction"]) == budget
                        ]
                    ),
                }
                for budget in PARTIAL_BUDGETS
            ],
            "development_selected_difficult_heads": {
                "head_ids": sorted(difficult_heads),
                "summary": _summarize_group(
                    [
                        row
                        for row in learned
                        if int(row["layer"]) == 23
                        and int(row["head"]) in difficult_heads
                    ]
                ),
                "by_budget": [
                    {
                        "budget_fraction": budget,
                        **_summarize_group(
                            [
                                row
                                for row in learned
                                if int(row["layer"]) == 23
                                and int(row["head"]) in difficult_heads
                                and float(row["budget_fraction"]) == budget
                            ]
                        ),
                    }
                    for budget in PARTIAL_BUDGETS
                ],
            },
            "post_hoc_low_entropy": {
                "development_frozen_normalized_entropy_threshold": (
                    low_entropy_threshold
                ),
                "summary": _summarize_group(late_layer_flat),
                "by_budget": [
                    {
                        "budget_fraction": budget,
                        **_summarize_group(
                            [
                                row
                                for row in late_layer_flat
                                if float(row["budget_fraction"]) == budget
                            ]
                        ),
                    }
                    for budget in PARTIAL_BUDGETS
                ],
                "note": "exact entropy is post-prediction analysis only",
            },
        },
    }


def _classify_evidence(
    prediction_rows: Sequence[Mapping[str, Any]],
    query_gain: Sequence[Mapping[str, Any]],
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    global_rows = _flat_policy_rows(prediction_rows, "global_fixed")
    layer_head_rows = _flat_policy_rows(prediction_rows, "layer_head_fixed")
    material = float(DECISION_CRITERIA["material_attention_mass"])
    fixed_gaps: list[float] = []
    layer_head_recoveries: list[float | None] = []
    for budget in PARTIAL_BUDGETS:
        fixed_group = [
            row for row in global_rows if float(row["budget_fraction"]) == budget
        ]
        identity_group = [
            row for row in layer_head_rows if float(row["budget_fraction"]) == budget
        ]
        oracle_mean = statistics.fmean(
            float(row["attention_mass"] + row["mass_regret"]) for row in fixed_group
        )
        fixed_mean = statistics.fmean(
            float(row["attention_mass"]) for row in fixed_group
        )
        identity_mean = statistics.fmean(
            float(row["attention_mass"]) for row in identity_group
        )
        fixed_gaps.append(oracle_mean - fixed_mean)
        layer_head_recoveries.append(
            oracle_gap_recovery(
                identity_mean,
                fixed_mean,
                oracle_mean,
                epsilon=NEAR_ZERO_ORACLE_GAP,
            )
        )
    budgets_required = int(DECISION_CRITERIA["budgets_required"])
    material_budget_count = sum(gap >= material for gap in fixed_gaps)
    query_positive_count = sum(
        float(row["attention_mass_gain_over_layer_head"]["mean"]) >= material
        and float(row["fixture_cluster_bootstrap_95"]["lower_95"]) > 0
        for row in query_gain
    )
    overhead = cost["estimated_query_overhead_fraction_of_median_candidate_retrieval"]
    overhead_small = overhead is not None and float(overhead) <= float(
        DECISION_CRITERIA["maximum_query_overhead_fraction_of_retrieval"]
    )
    static_count = sum(
        recovery is not None
        and recovery >= float(DECISION_CRITERIA["nearly_all_oracle_gap_recovered"])
        for recovery in layer_head_recoveries
    )
    if material_budget_count < budgets_required:
        evidence = "A"
        label = "NO ADAPTIVE POLICY JUSTIFIED"
    elif query_positive_count >= budgets_required and overhead_small:
        evidence = "C"
        label = "QUERY-ADAPTIVE POLICY WORTH FURTHER STUDY"
    elif static_count >= budgets_required:
        evidence = "B"
        label = "STATIC PER-HEAD POLICY JUSTIFIED"
    else:
        evidence = "D"
        label = "ORACLE GAP NOT PREDICTABLE WITH CHEAP FEATURES"
    return {
        "classification": evidence,
        "label": label,
        "fixed_to_oracle_mass_gaps_by_budget": dict(
            zip((str(value) for value in PARTIAL_BUDGETS), fixed_gaps, strict=True)
        ),
        "layer_head_gap_recovery_by_budget": dict(
            zip(
                (str(value) for value in PARTIAL_BUDGETS),
                layer_head_recoveries,
                strict=True,
            )
        ),
        "query_adaptive_budgets_meeting_gain_and_ci": query_positive_count,
        "query_overhead_within_frozen_limit": overhead_small,
        "criteria": DECISION_CRITERIA,
        "adaptive_planner_justified_now": False,
        "interpretation": (
            "This feasibility result cannot authorize a public planner API or "
            "production-performance claim."
        ),
    }


def evaluate_held_out(args: argparse.Namespace) -> dict[str, Any]:
    frozen = load_json(args.frozen_model)
    require_schema_version(frozen, supported=(1,))
    outcomes = load_json(args.held_out_outcomes)
    features = load_json(args.held_out_features)
    if frozen.get("artifact") != "pythia_410m_phase3a_policy_development_freeze":
        raise ValueError("unexpected frozen model artifact")
    validate_outcome_artifact(outcomes, "held_out")
    validate_feature_artifact(features, "held_out")
    validate_tokenizations(outcomes, features, "held_out")
    if frozen["held_out_fixture_text_sha256"] != LOCKED_FIXTURE_TEXT_SHA256:
        raise ValueError("frozen model held-out content lock differs from source")
    if frozen["held_out_token_id_sha256"] != {
        key: {str(length): digest for length, digest in value.items()}
        for key, value in LOCKED_TOKEN_ID_SHA256.items()
    }:
        # JSON stringifies integer mapping keys.
        raise ValueError("frozen model held-out token lock differs from source")
    examples = assemble_policy_examples(features["feature_rows"], outcomes["records"])
    predictions = _predict_all(frozen, examples)
    prediction_rows = _prediction_rows(examples, predictions)
    summaries = {
        predictor: {
            "stratified": _stratified_summary(prediction_rows, predictor),
            "fixture_cluster_bootstrap_95_by_budget": _budget_confidence_intervals(
                prediction_rows,
                predictor,
            ),
        }
        for predictor in predictions
    }
    query_gain = _paired_query_gain_intervals(prediction_rows)
    cost = _cost_analysis(frozen, features, outcomes, examples)
    decision = _classify_evidence(prediction_rows, query_gain, cost)
    return {
        "schema_version": 1,
        "artifact": "pythia_410m_phase3a_policy_held_out_evaluation",
        "scope": (
            "single-query internal-activation policy feasibility; no decode, "
            "generation, perplexity, downstream quality, or production speed claim"
        ),
        "provenance": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "training_seed": TRAINING_SEED,
            "frozen_model_artifact": str(args.frozen_model),
            "frozen_model_sha256": sha256_file(args.frozen_model),
            "held_out_outcome_artifact": str(args.held_out_outcomes),
            "held_out_outcome_sha256": sha256_file(args.held_out_outcomes),
            "held_out_feature_artifact": str(args.held_out_features),
            "held_out_feature_sha256": sha256_file(args.held_out_features),
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
        },
        "held_out_fixture_text_sha256": LOCKED_FIXTURE_TEXT_SHA256,
        "held_out_token_id_sha256": LOCKED_TOKEN_ID_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "oracle_gap_recovery_reporting": {
            "primary": (
                "ratio of group mean predicted-minus-fixed mass to group mean "
                "oracle-minus-fixed mass"
            ),
            "per_observation_near_zero_rule": (
                f"denominators with absolute value <= {NEAR_ZERO_ORACLE_GAP} are null"
            ),
            "warning": (
                "per-observation ratios can remain numerically unstable just above "
                "the frozen cutoff; group-mean recovery is the reported comparison"
            ),
        },
        "leakage_audit": {
            "forbidden_features": sorted(FORBIDDEN_INFERENCE_FEATURES),
            "passed": True,
            "outcome_join_timing": (
                "predictions are computed only from separately collected feature rows; "
                "attention/retrieval outcomes are joined afterward for evaluation"
            ),
        },
        "predictor_summaries": summaries,
        "identity_only_versus_query_dependent": query_gain,
        "cost_analysis": cost,
        "index_coexistence_analysis": _index_coexistence(),
        "failure_analysis": _failure_analysis(prediction_rows, frozen),
        "evidence_decision": decision,
        "matrix_counts": {
            "held_out_observations_per_budget": len(examples) // len(PARTIAL_BUDGETS),
            "held_out_prediction_rows": len(prediction_rows),
            "expected_prediction_rows": 8 * 2 * 4 * 3 * 16 * 3,
        },
        "shared_architecture_changes_required": {
            "KVIndex": False,
            "Selection": False,
            "KVStorage": False,
            "RetrievedKV": False,
            "KVCache": False,
        },
        "held_out_predictions_and_post_hoc_outcomes": prediction_rows,
    }


def print_freeze_summary(result: Mapping[str, Any]) -> None:
    print(f"development_examples={result['development_example_count']}")
    for row in result["baseline_models"]["global_fixed"]["rows"]:
        print(
            f"budget={row['budget_fraction']} global_fixed={row['configuration']} "
            f"development_mass={row['development_mean_attention_mass']:.6f}"
        )
    print(
        "difficult_heads="
        + ",".join(
            str(row["head"]) for row in result["development_selected_difficult_heads"]
        )
    )


def print_evaluation_summary(result: Mapping[str, Any]) -> None:
    print(f"decision={result['evidence_decision']['classification']}")
    print(f"label={result['evidence_decision']['label']}")
    for row in result["identity_only_versus_query_dependent"]:
        interval = row["fixture_cluster_bootstrap_95"]
        print(
            f"budget={row['budget_fraction']} learned_minus_layer_head_mass="
            f"{interval['mean']:.6f} ci95=[{interval['lower_95']:.6f},"
            f"{interval['upper_95']:.6f}]"
        )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.command == "freeze":
        result = freeze_development(args)
        print_freeze_summary(result)
    else:
        result = evaluate_held_out(args)
        print_evaluation_summary(result)
    write_new_json(args.output, result)
    print(f"elapsed_seconds={time.perf_counter() - start:.3f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
