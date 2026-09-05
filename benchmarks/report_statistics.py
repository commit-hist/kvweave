"""Report statistics with explicit historical filtering and output contracts."""

from collections.abc import Iterable
import math
import statistics
from typing import Any

from kvweave.profiling import percentile as percentile


def metric_distribution(
    values: Iterable[float], *, include_p95: bool = False
) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "p10": percentile(finite, 0.10),
        "p25": percentile(finite, 0.25),
        "p75": percentile(finite, 0.75),
        "p90": percentile(finite, 0.90),
        **({"p95": percentile(finite, 0.95)} if include_p95 else {}),
        "max": max(finite),
    }


def policy_distribution(values: Iterable[float | None]) -> dict[str, float | int]:
    """Policy reports omit None/nonfinite values and include p95."""
    return metric_distribution(
        (value for value in values if value is not None),
        include_p95=True,
    )


def basic_distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def latency_distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("distribution requires observations")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def _pearson_pairs(pairs: list[tuple[float, float]]) -> tuple[int, float | None]:
    if len(pairs) < 2:
        return len(pairs), None
    left_values, right_values = zip(*pairs, strict=True)
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_values)
        * sum((value - right_mean) ** 2 for value in right_values)
    )
    return len(pairs), None if denominator == 0 else numerator / denominator


def pearson_correlation(
    records: list[dict[str, Any]],
    left: str,
    right: str,
) -> tuple[int, float | None]:
    pairs = [
        (float(record[left]), float(record[right]))
        for record in records
        if record.get(left) is not None
        and record.get(right) is not None
        and math.isfinite(float(record[left]))
        and math.isfinite(float(record[right]))
    ]
    return _pearson_pairs(pairs)


def legacy_pearson_correlation(
    records: list[dict[str, Any]],
    left: str,
    right: str,
) -> float | None:
    pairs = [
        (float(record[left]), float(record[right]))
        for record in records
        if record[left] is not None and record[right] is not None
    ]
    return _pearson_pairs(pairs)[1]
