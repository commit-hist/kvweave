"""Strict JSON and atomic publication for local benchmark artifacts."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


@contextmanager
def atomic_output(path: Path, *, overwrite: bool) -> Iterator[Path]:
    """Publish a completed sibling file; leave existing evidence intact on error.

    A hard link makes no-overwrite publication exclusive even if another writer
    finishes after our initial existence check. Atomicity is per file, not a
    transaction spanning a report and all of its sidecars.
    """
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb") as completed:
            os.fsync(completed.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
    sort_keys: bool = True,
) -> None:
    """Stream finite JSON to a temporary file before publishing it.

    Undefined/nonfinite metrics must be handled explicitly by the experiment.
    They are never silently converted to zero, null, or nonstandard JSON tokens.
    """
    with atomic_output(path, overwrite=overwrite) as temporary:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=sort_keys, allow_nan=False)
            output.write("\n")


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish frozen/evaluation evidence without replacing an existing file."""
    write_json(path, payload, overwrite=False)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is not allowed: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"JSON number exceeds finite float range: {value}")
    return number


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(
            input_file,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def require_schema_version(
    payload: Mapping[str, Any],
    *,
    supported: Sequence[int],
) -> None:
    version = payload.get("schema_version")
    if type(version) is not int or version not in supported:
        raise ValueError(
            f"unsupported artifact schema_version {version!r}; expected {tuple(supported)}"
        )
