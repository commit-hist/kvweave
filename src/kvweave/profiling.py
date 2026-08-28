"""Lightweight, opt-in profiling support for reference experiments.

The recorder is intentionally internal instrumentation rather than a public
KVWeave interface.  Normal execution has no active recorder; benchmark programs
can activate one and attach fixture, strategy, decode-step, and layer context
without threading profiler arguments through the storage/index contracts.
"""

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import math
import statistics
import time
from typing import Any

import torch


ProfileValue = str | int | float | bool | None


@dataclass(frozen=True)
class ComponentTiming:
    """One wall-clock observation for a named profiled component."""

    component: str
    duration_ms: float
    context: Mapping[str, ProfileValue]

    def as_dict(self) -> dict[str, ProfileValue]:
        """Return a flat JSON-serializable representation."""
        return {
            **self.context,
            "component": self.component,
            "duration_ms": self.duration_ms,
        }


class ComponentProfiler:
    """Collect nested component wall times for one benchmark run."""

    def __init__(self, *, emit_operator_ranges: bool = False) -> None:
        self.emit_operator_ranges = emit_operator_ranges
        self.records: list[ComponentTiming] = []

    @contextmanager
    def activate(self) -> Iterator["ComponentProfiler"]:
        """Make this recorder visible to instrumented reference operations."""
        token = _ACTIVE_PROFILER.set(self)
        try:
            yield self
        finally:
            _ACTIVE_PROFILER.reset(token)

    def record(
        self,
        component: str,
        duration_ms: float,
        context: Mapping[str, ProfileValue],
    ) -> None:
        if not component:
            raise ValueError("component name must not be empty")
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("duration_ms must be finite and non-negative")
        self.records.append(
            ComponentTiming(
                component=component,
                duration_ms=duration_ms,
                context=dict(context),
            )
        )


_ACTIVE_PROFILER: ContextVar[ComponentProfiler | None] = ContextVar(
    "kvweave_active_component_profiler",
    default=None,
)
_PROFILE_CONTEXT: ContextVar[Mapping[str, ProfileValue]] = ContextVar(
    "kvweave_component_profile_context",
    default={},
)


@contextmanager
def profile_context(**values: ProfileValue) -> Iterator[None]:
    """Add fields to every component recorded within the context."""
    invalid = [name for name in values if not name]
    if invalid:
        raise ValueError("profile context field names must not be empty")
    merged = {**_PROFILE_CONTEXT.get(), **values}
    token = _PROFILE_CONTEXT.set(merged)
    try:
        yield
    finally:
        _PROFILE_CONTEXT.reset(token)


@contextmanager
def profile_component(component: str) -> Iterator[None]:
    """Time an existing operation when an opt-in recorder is active."""
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
        return

    operator_range: Any = (
        torch.profiler.record_function(f"kvweave::{component}")
        if profiler.emit_operator_ranges
        else nullcontext()
    )
    with operator_range:
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            profiler.record(component, duration_ms, _PROFILE_CONTEXT.get())


def estimate_tensor_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    """Return analytical logical bytes for a tensor shape and dtype."""
    if any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
        for dimension in shape
    ):
        raise ValueError("tensor dimensions must be non-negative integers")
    elements = math.prod(shape)
    return elements * torch.empty((), dtype=dtype).element_size()


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for non-empty observations."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_component_timings(
    records: Sequence[ComponentTiming],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, ProfileValue]]:
    """Summarize component observations by caller-selected context fields."""
    grouped: dict[tuple[ProfileValue, ...], list[float]] = defaultdict(list)
    for record in records:
        key = tuple(record.context.get(field) for field in group_fields) + (
            record.component,
        )
        grouped[key].append(record.duration_ms)

    summaries: list[dict[str, ProfileValue]] = []
    for key, durations in sorted(grouped.items(), key=lambda item: str(item[0])):
        context_values = key[:-1]
        component = key[-1]
        summaries.append(
            {
                **dict(zip(group_fields, context_values, strict=True)),
                "component": component,
                "call_count": len(durations),
                "total_ms": sum(durations),
                "mean_ms": statistics.fmean(durations),
                "median_ms": statistics.median(durations),
                "p90_ms": percentile(durations, 0.90),
                "p95_ms": percentile(durations, 0.95),
            }
        )
    return summaries
