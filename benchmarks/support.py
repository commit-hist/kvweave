"""Shared environment metadata and unchanged synthetic timing operations."""

from collections.abc import Callable
import platform
import statistics
import subprocess
import time
from typing import Any, TypeVar

import torch

from kvweave.metrics import relative_l2_error

T = TypeVar("T")

# Preserve benchmark imports while keeping arithmetic in the neutral library.
relative_error = relative_l2_error


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def median_latency_ms(
    operation: Callable[[], T],
    *,
    warmups: int,
    repetitions: int,
    device: torch.device,
) -> tuple[float, T]:
    result: T
    for _ in range(warmups):
        result = operation()
    synchronize(device)

    latencies_ms: list[float] = []
    for _ in range(repetitions):
        synchronize(device)
        start = time.perf_counter()
        result = operation()
        synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1_000)
    return statistics.median(latencies_ms), result


def hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"{platform.machine()} Apple MPS"
    return platform.processor() or platform.machine() or "unknown"


def git_value(*arguments: str) -> str | None:
    """Read Git metadata, distinguishing unavailable output from an empty string."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_commit() -> str:
    """Keep the historical report string for unavailable commit provenance."""
    commit = git_value("rev-parse", "HEAD")
    return "unknown" if commit is None else commit


def git_is_dirty() -> bool | None:
    """Distinguish a dirty checkout from unavailable Git metadata."""
    status = git_value("status", "--porcelain")
    return None if status is None else bool(status)


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def machine_metadata(device: torch.device) -> dict[str, Any]:
    memory = _sysctl("hw.memsize")
    return {
        "hardware": hardware_name(device),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "physical_memory_bytes": None if memory is None else int(memory),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
