from copy import deepcopy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from types import SimpleNamespace
import hashlib

import pytest
import torch

from benchmarks.artifacts import load_json, write_new_json
from benchmarks.scripts import phase3b_decode
from benchmarks.decode import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TRANSFORMERS_VERSION,
)
from benchmarks.phase4 import load_phase4_baseline
from benchmarks.scripts.phase3a_replication import aggregate_records
from benchmarks.scripts.real_model_reference import (
    aggregate_records as aggregate_real_model,
)
from benchmarks.scripts.phase3a_policy_feasibility import (
    validate_feature_artifact,
    validate_outcome_artifact,
)


def test_replication_aggregation_preserves_groups_counts_and_missing_metrics() -> None:
    rows = [
        {"layer": 1, "metric": 3.0},
        {"layer": 0, "metric": 2.0},
        {"layer": 1, "metric": 1.0},
        {"layer": 1, "metric": None},
        {"layer": 1, "metric": float("inf")},
    ]
    report = aggregate_records(rows, group_fields=("layer",), metrics=("metric",))
    assert [row["layer"] for row in report] == [0, 1]
    assert [row["sample_count"] for row in report] == [1, 4]
    assert report[1]["metric"] == pytest.approx(
        {
            "count": 2,
            "mean": 2.0,
            "median": 2.0,
            "min": 1.0,
            "p10": 1.2,
            "p25": 1.5,
            "p75": 2.5,
            "p90": 2.8,
            "max": 3.0,
        }
    )


def test_real_model_aggregation_preserves_earlier_report_fields() -> None:
    rows = [
        {
            "strategy": "pq",
            "configuration": "M2/C4",
            "budget_fraction": 0.5,
            "layer": layer,
            "candidate_recall": recall,
            "attention_mass_captured": 0.75,
            "relative_attention_output_error": 0.25,
        }
        for layer, recall in [(0, 0.25), (1, 0.75)]
    ]
    pooled = aggregate_real_model(rows, include_layer=False)
    assert len(pooled) == 1 and pooled[0]["sample_count"] == 2
    assert pooled[0]["candidate_recall"] == {
        "mean": 0.5,
        "median": 0.5,
        "min": 0.25,
        "max": 0.75,
    }
    assert len(aggregate_real_model(rows, include_layer=True)) == 2


def phase4_artifact() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "Phase 4 profiling",
        "status": "complete",
        "provenance": {
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
            "transformers_version": DEFAULT_TRANSFORMERS_VERSION,
            "transformers_attention_implementation": "eager",
            "dtype": "float32",
            "device": "cpu",
            "hardware": "recorded historical hardware",
        },
        "protocol": {
            "prompt_length": 1024,
            "fixture_ids": ["technical_exposition", "code_like"],
            "generation_mode": "teacher_forced_only",
            "generated_token_positions": 32,
            "approximate_decode_steps": 31,
            "budget_fractions": [0.5, 1.0],
            "quest_configuration": "p64",
            "seed": 0,
        },
        "steady_state": {
            "retrieval_overhead_summary": [
                {"strategy": "quest", "component": "metadata_rebuild"},
                {"strategy": "quest", "component": "total_retrieval_overhead"},
                {"strategy": "pq", "component": "total_retrieval_overhead"},
            ],
            "step_component_summary": [
                {"strategy": "quest", "component": "total_decode_step"},
                {"strategy": "dense", "component": "total_decode_step"},
            ],
        },
        "correctness": {
            "partial_quality_summary": [{"strategy": "quest", "budget_fraction": 0.5}]
        },
    }


def test_baseline_reader_preserves_selected_rows_and_historical_environment(
    tmp_path: Path,
) -> None:
    artifact = phase4_artifact()
    path = tmp_path / "baseline.json"
    write_new_json(path, artifact)
    baseline = load_phase4_baseline(path)
    assert baseline == {
        "artifact_path": str(path),
        "provenance": artifact["provenance"],
        "quest_retrieval_summary": artifact["steady_state"][
            "retrieval_overhead_summary"
        ][:2],
        "quest_decode_summary": artifact["steady_state"]["step_component_summary"][:1],
        "quest_partial_quality_summary": artifact["correctness"][
            "partial_quality_summary"
        ],
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "schema_version", 2),
        (None, "status", "incomplete"),
        ("provenance", "model_revision", "another-revision"),
        ("provenance", "dtype", "float16"),
        ("provenance", "device", "cuda"),
        ("protocol", "prompt_length", 512),
        ("protocol", "quest_configuration", "p16"),
        ("protocol", "generation_mode", "free_running"),
        ("protocol", "budget_fractions", [1.0]),
        ("steady_state", "retrieval_overhead_summary", []),
        ("correctness", "partial_quality_summary", []),
    ],
)
def test_baseline_reader_rejects_incompatible_or_incomplete_evidence(
    tmp_path: Path,
    section: str | None,
    field: str,
    value: object,
) -> None:
    artifact = deepcopy(phase4_artifact())
    (artifact if section is None else artifact[section])[field] = value
    path = tmp_path / "baseline.json"
    write_new_json(path, artifact)
    with pytest.raises(ValueError):
        load_phase4_baseline(path)


@pytest.mark.parametrize(
    "validator", [validate_feature_artifact, validate_outcome_artifact]
)
def test_policy_readers_reject_future_schema_before_consuming_records(
    validator: Callable[[Mapping[str, Any], str], None],
) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        validator({"schema_version": 99}, "development")


def test_decode_report_records_a_reproducible_loadable_tensor_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        output=tmp_path / "report.json",
        dense_tensors_output=tmp_path / "dense.pt",
    )
    tensors = {"logits": torch.arange(6).reshape(2, 3)}
    monkeypatch.setattr(phase3b_decode, "parse_args", lambda: args)
    monkeypatch.setattr(
        phase3b_decode,
        "run_experiment",
        lambda args: ({"schema_version": 1}, tensors),
    )
    phase3b_decode.main()
    original = args.dense_tensors_output.read_bytes()
    artifact = load_json(args.output)
    assert (
        artifact["dense_tensor_artifact"]["sha256"]
        == hashlib.sha256(original).hexdigest()
    )
    assert artifact["dense_tensor_artifact"]["path"] == str(args.dense_tensors_output)
    restored = torch.load(args.dense_tensors_output, weights_only=True)
    assert torch.equal(restored["logits"], tensors["logits"])
    phase3b_decode.main()
    assert args.dense_tensors_output.read_bytes() == original
