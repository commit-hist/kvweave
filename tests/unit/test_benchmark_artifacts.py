import json
from pathlib import Path

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
