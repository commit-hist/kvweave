"""Strict JSON and atomic publication for local benchmark artifacts."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any


class ArtifactPublicationError(OSError):
    """A completed artifact could not be published and remains recoverable."""

    def __init__(self, path: Path, temporary: Path, error: OSError) -> None:
        self.temporary = temporary
        super().__init__(
            error.errno,
            f"could not publish {path}; completed artifact retained at {temporary}: {error}",
        )


def _destination_mode(path: Path, *, overwrite: bool) -> int | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"refusing symlink artifact destination: {path}")
    if not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact destination is not a regular file: {path}")
    return stat.S_IMODE(metadata.st_mode)


@contextmanager
def _temporary_output(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Let the kernel apply the umask, without reading/changing process-global
    # state. Exclusive creation prevents following an existing temporary link.
    while True:
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            continue
    os.close(descriptor)
    retain = False
    try:
        yield temporary
    except ArtifactPublicationError as error:
        retain = error.temporary == temporary
        raise
    finally:
        if not retain:
            temporary.unlink(missing_ok=True)


def _publish(temporary: Path, path: Path, *, overwrite: bool) -> None:
    try:
        mode = _destination_mode(path, overwrite=overwrite)
        with temporary.open("rb") as completed:
            if mode is not None:
                os.fchmod(completed.fileno(), mode)
            os.fsync(completed.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    except FileExistsError:
        # A competing frozen writer won. Never replace its evidence.
        raise
    except OSError as error:
        raise ArtifactPublicationError(path, temporary, error) from error


@contextmanager
def atomic_output(path: Path, *, overwrite: bool) -> Iterator[Path]:
    """Publish complete files, preserving permission bits on replacement.

    New files use 0666 filtered by the umask. Symlink destinations are rejected.
    Exclusive publication requires hard links; publication I/O failures retain
    the completed temporary file and report its recovery path. Atomicity is per
    file, not a transaction or a power-loss durability guarantee.
    """
    _destination_mode(path, overwrite=overwrite)
    with _temporary_output(path) as temporary:
        yield temporary
        _publish(temporary, path, overwrite=overwrite)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_content_addressed(
    path: Path, write: Callable[[Path], None]
) -> tuple[Path, str]:
    """Publish/reuse an immutable sibling named ``stem.<sha256>.suffix``.

    The requested base path is never replaced, including a legacy sidecar still
    referenced by an older report. A report can safely reference the returned
    path only after this function succeeds.
    """
    _destination_mode(path, overwrite=True)
    with _temporary_output(path) as temporary:
        write(temporary)
        digest = _sha256(temporary)
        destination = path.with_name(f"{path.stem}.{digest}{path.suffix}")
        try:
            _publish(temporary, destination, overwrite=False)
        except FileExistsError:
            _destination_mode(destination, overwrite=True)
            if _sha256(destination) != digest:
                raise ValueError(
                    f"content-addressed artifact hash mismatch: {destination}"
                )
        return destination, digest


def prepare_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Represent nonfinite metrics explicitly without changing finite values.

    Exceptional values become null, with JSON-pointer locations and original
    NaN/Infinity/-Infinity values recorded in a reserved top-level annotation.
    Such reports are diagnostic output, not admissible downstream evidence.
    """
    if "nonfinite_metrics" in payload:
        raise ValueError("nonfinite_metrics is reserved for report serialization")
    nonfinite: list[dict[str, str]] = []

    def convert(value: Any, pointer: str) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            label = (
                "NaN"
                if math.isnan(value)
                else ("Infinity" if value > 0 else "-Infinity")
            )
            nonfinite.append({"pointer": pointer, "value": label})
            return None
        if isinstance(value, dict):
            return {
                key: convert(
                    item, f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}"
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                convert(item, f"{pointer}/{index}") for index, item in enumerate(value)
            ]
        return value

    report = convert(dict(payload), "")
    if nonfinite:
        report["nonfinite_metrics"] = nonfinite
    # Validate all types before callers publish any associated sidecars, without
    # allocating another full serialized copy of a potentially large report.
    for _ in json.JSONEncoder(allow_nan=False).iterencode(report):
        pass
    return report


def write_report(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
    sort_keys: bool = True,
) -> None:
    write_json(path, prepare_report(payload), overwrite=overwrite, sort_keys=sort_keys)


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
    if "nonfinite_metrics" in payload:
        raise ValueError(
            f"report contains nonfinite metrics and is not valid evidence: {path}"
        )
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
