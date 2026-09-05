import json
import errno
import os
from pathlib import Path
import stat

import pytest

from benchmarks import artifacts


def test_atomic_json_preserves_old_evidence_on_serialization_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    artifacts.write_json(path, {"accepted": 1}, overwrite=False)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        artifacts.write_json(
            path, {"part": [1, 2], "bad": float("inf")}, overwrite=True
        )
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    artifacts.write_json(path, {"accepted": 2}, overwrite=True)
    assert artifacts.load_json(path) == {"accepted": 2}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object()])
def test_failed_new_artifact_does_not_block_retry(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "frozen.json"
    with pytest.raises((ValueError, TypeError)):
        artifacts.write_new_json(path, {"nested": {"metric": value}})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
    artifacts.write_new_json(path, {"metric": 1.0})
    with pytest.raises(FileExistsError):
        artifacts.write_new_json(path, {"metric": 2.0})
    assert artifacts.load_json(path) == {"metric": 1.0}


def test_exclusive_publication_cannot_replace_a_concurrent_writer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frozen.json"
    with pytest.raises(FileExistsError):
        with artifacts.atomic_output(path, overwrite=False) as temporary:
            temporary.write_text('{"writer": 1}')
            path.write_text('{"writer": 2}')
    assert artifacts.load_json(path) == {"writer": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_interrupted_write_preserves_destination_and_cleans_temporary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    artifacts.write_new_json(path, {"accepted": True})
    with pytest.raises(KeyboardInterrupt):
        with artifacts.atomic_output(path, overwrite=True) as temporary:
            temporary.write_text('{"partial":')
            raise KeyboardInterrupt
    assert artifacts.load_json(path) == {"accepted": True}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "payload", ['{"metric": Infinity}', '{"metric": NaN}', '{"metric": 1e999}', "[]"]
)
def test_reader_rejects_nonfinite_or_nonobject_reports(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises((ValueError, TypeError)):
        artifacts.load_json(path)


def test_streamed_json_preserves_finite_serialization_and_key_order(
    tmp_path: Path,
) -> None:
    payload = {"z": [1, 2.5, None], "a": {"flag": True}}
    path = tmp_path / "result.json"
    artifacts.write_json(path, payload, overwrite=False, sort_keys=False)
    assert path.read_text() == json.dumps(payload, indent=2) + "\n"


@pytest.mark.parametrize("version", [None, True, "1", 0, 3])
def test_schema_version_must_be_explicit_and_supported(version: object) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        artifacts.require_schema_version({"schema_version": version}, supported=(1, 2))


@pytest.mark.parametrize("overwrite", [False, True])
def test_new_artifact_permissions_follow_normal_file_creation(
    tmp_path: Path, overwrite: bool
) -> None:
    control = tmp_path / "control.json"
    control.write_text("{}")
    path = tmp_path / "report.json"
    artifacts.write_json(path, {"ok": True}, overwrite=overwrite)
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)


@pytest.mark.parametrize("mode", [0o200, 0o600, 0o640, 0o664])
def test_overwrite_preserves_existing_permission_bits(
    tmp_path: Path, mode: int
) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}")
    path.chmod(mode)
    artifacts.write_json(path, {"ok": True}, overwrite=True)
    assert stat.S_IMODE(path.stat().st_mode) == mode


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
def test_symlink_destinations_are_rejected(
    tmp_path: Path, overwrite: bool, dangling: bool
) -> None:
    target = tmp_path / "target.json"
    if not dangling:
        target.write_text('{"original": true}')
    path = tmp_path / "report.json"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        artifacts.write_json(path, {"new": True}, overwrite=overwrite)
    assert path.is_symlink()
    assert target.exists() is not dangling
    if not dangling:
        assert artifacts.load_json(target) == {"original": True}


def test_symlink_created_during_write_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}")
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="symlink"):
        with artifacts.atomic_output(path, overwrite=True) as temporary:
            temporary.write_text('{"new": true}')
            path.symlink_to(target)
    assert path.is_symlink()
    assert target.read_text() == "{}"


@pytest.mark.parametrize("overwrite", [False, True])
def test_publication_failure_retains_completed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool
) -> None:
    path = tmp_path / "report.json"
    if overwrite:
        path.write_text('{"old": true}')

    def unsupported(*args: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "publication unavailable")

    monkeypatch.setattr(os, "replace" if overwrite else "link", unsupported)
    with pytest.raises(
        artifacts.ArtifactPublicationError, match="retained at"
    ) as caught:
        artifacts.write_json(path, {"completed": True}, overwrite=overwrite)
    assert artifacts.load_json(caught.value.temporary) == {"completed": True}
    assert str(caught.value.temporary) in str(caught.value)
    if overwrite:
        assert artifacts.load_json(path) == {"old": True}
    else:
        assert not path.exists()


def test_report_nonfinite_metrics_are_explicit_and_not_accepted_as_evidence(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "a/b~c": [float("inf"), float("-inf"), float("nan"), None, 2.0],
    }
    path = tmp_path / "report.json"
    artifacts.write_report(path, payload, overwrite=True)
    report = json.loads(path.read_text())
    assert report["a/b~c"] == [None, None, None, None, 2.0]
    assert report["nonfinite_metrics"] == [
        {"pointer": "/a~1b~0c/0", "value": "Infinity"},
        {"pointer": "/a~1b~0c/1", "value": "-Infinity"},
        {"pointer": "/a~1b~0c/2", "value": "NaN"},
    ]
    assert payload["a/b~c"][0] == float("inf")
    with pytest.raises(ValueError, match="not valid evidence"):
        artifacts.load_json(path)


def test_finite_report_bytes_remain_unchanged(tmp_path: Path) -> None:
    payload = {"z": [1, 2.5, None], "a": {"flag": True}}
    path = tmp_path / "report.json"
    artifacts.write_report(path, payload, overwrite=True, sort_keys=False)
    assert path.read_text() == json.dumps(payload, indent=2) + "\n"


def test_report_annotation_cannot_be_overwritten() -> None:
    with pytest.raises(ValueError, match="reserved"):
        artifacts.prepare_report({"nonfinite_metrics": []})


def test_content_addressed_output_reuses_matching_bytes_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dense.pt"

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"complete tensors")

    destination, digest = artifacts.write_content_addressed(path, write)
    assert destination.name == f"dense.{digest}.pt"
    assert artifacts.write_content_addressed(path, write) == (destination, digest)
    assert not path.exists()
    destination.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        artifacts.write_content_addressed(path, write)
    assert destination.read_bytes() == b"corrupt"
