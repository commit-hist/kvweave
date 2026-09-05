from copy import deepcopy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from types import SimpleNamespace
import hashlib
import json
import errno
import os

import pytest
import torch

from benchmarks.artifacts import ArtifactPublicationError, load_json, write_new_json
from benchmarks.scripts import phase3b_decode
from benchmarks.scripts import (
    phase4_profile,
    phase5a_quest_incremental,
    real_model_reference,
)
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
    artifact = load_json(args.output)
    sidecar = Path(artifact["dense_tensor_artifact"]["path"])
    original = sidecar.read_bytes()
    assert (
        artifact["dense_tensor_artifact"]["sha256"]
        == hashlib.sha256(original).hexdigest()
    )
    assert sidecar.name == f"dense.{hashlib.sha256(original).hexdigest()}.pt"
    assert not args.dense_tensors_output.exists()
    restored = torch.load(sidecar, weights_only=True)
    assert torch.equal(restored["logits"], tensors["logits"])
    phase3b_decode.main()
    assert sidecar.read_bytes() == original
    assert len(list(tmp_path.glob("*.pt"))) == 1


@pytest.mark.parametrize("failure", ["nonfinite", "serialization", "publication"])
def test_decode_rerun_keeps_previous_sidecar_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    args = SimpleNamespace(
        output=tmp_path / "report.json", dense_tensors_output=tmp_path / "dense.pt"
    )
    monkeypatch.setattr(phase3b_decode, "parse_args", lambda: args)
    monkeypatch.setattr(
        phase3b_decode,
        "run_experiment",
        lambda args: ({"schema_version": 1}, {"logits": torch.zeros(2)}),
    )
    phase3b_decode.main()
    previous_bytes = args.output.read_bytes()
    previous = load_json(args.output)
    previous_sidecar = Path(previous["dense_tensor_artifact"]["path"])
    metric = object() if failure == "serialization" else float("inf")
    monkeypatch.setattr(
        phase3b_decode,
        "run_experiment",
        lambda args: (
            {"schema_version": 1, "metric": metric},
            {"logits": torch.ones(2)},
        ),
    )
    if failure == "publication":
        replace = os.replace

        def fail_report(source: Path, destination: Path) -> None:
            if destination == args.output:
                raise OSError(errno.EIO, "report publication failed")
            replace(source, destination)

        monkeypatch.setattr(os, "replace", fail_report)
    if failure == "nonfinite":
        phase3b_decode.main()
        current = json.loads(args.output.read_text())
        assert current["metric"] is None
        assert current["nonfinite_metrics"] == [
            {"pointer": "/metric", "value": "Infinity"}
        ]
        assert current["dense_tensor_artifact"]["path"] != str(previous_sidecar)
        current_sidecar = Path(current["dense_tensor_artifact"]["path"])
        assert (
            hashlib.sha256(current_sidecar.read_bytes()).hexdigest()
            == current["dense_tensor_artifact"]["sha256"]
        )
    else:
        with pytest.raises(
            TypeError if failure == "serialization" else ArtifactPublicationError
        ):
            phase3b_decode.main()
        assert args.output.read_bytes() == previous_bytes
    assert (
        hashlib.sha256(previous_sidecar.read_bytes()).hexdigest()
        == previous["dense_tensor_artifact"]["sha256"]
    )


def test_decode_preserves_legacy_sidecar_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        output=tmp_path / "report.json", dense_tensors_output=tmp_path / "dense.pt"
    )
    args.dense_tensors_output.write_bytes(b"legacy sidecar")
    monkeypatch.setattr(phase3b_decode, "parse_args", lambda: args)
    monkeypatch.setattr(
        phase3b_decode,
        "run_experiment",
        lambda args: ({"schema_version": 1}, {"logits": torch.zeros(2)}),
    )
    phase3b_decode.main()
    assert args.dense_tensors_output.read_bytes() == b"legacy sidecar"


@pytest.mark.parametrize(
    "script", [real_model_reference, phase4_profile, phase5a_quest_incremental]
)
def test_experiment_entrypoints_publish_nonfinite_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: Any
) -> None:
    args = SimpleNamespace(output=tmp_path / "report.json")
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(
        script,
        "run_experiment",
        lambda args: {"schema_version": 1, "metric": float("nan")},
    )
    if script is real_model_reference:
        monkeypatch.setattr(script, "print_summary", lambda report: None)
    script.main()
    report = json.loads(args.output.read_text())
    assert report["metric"] is None
    assert report["nonfinite_metrics"] == [{"pointer": "/metric", "value": "NaN"}]
