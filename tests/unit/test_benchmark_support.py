import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmarks import support
from benchmarks import report_statistics
from benchmarks.artifacts import atomic_output
from benchmarks.report_statistics import (
    basic_distribution,
    latency_distribution,
    legacy_pearson_correlation,
    metric_distribution,
    pearson_correlation,
    percentile,
    policy_distribution,
)


def test_atomic_tensor_sidecar_can_be_loaded(tmp_path: Path) -> None:
    path = tmp_path / "dense.pt"
    tensors = {"logits": torch.arange(6).reshape(2, 3)}
    with atomic_output(path, overwrite=True) as temporary:
        torch.save(tensors, temporary)
    restored = torch.load(path, weights_only=True)
    assert torch.equal(restored["logits"], tensors["logits"])


def test_benchmark_directory_does_not_shadow_stdlib_statistics() -> None:
    benchmark_directory = Path(report_statistics.__file__).parent
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import statistics; assert statistics.median([1, 3]) == 2",
            str(benchmark_directory),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [(0, "", False), (0, " M README.md\n", True), (1, "", None)],
)
def test_git_status_distinguishes_unavailable_metadata(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    expected: bool | None,
) -> None:
    monkeypatch.setattr(
        support.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, returncode, stdout, ""
        ),
    )
    assert support.git_is_dirty() is expected


def test_git_and_machine_metadata_handle_missing_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("executable unavailable")

    monkeypatch.setattr(support.subprocess, "run", missing)
    assert support.git_commit() == "unknown"
    assert support.git_is_dirty() is None
    metadata = support.machine_metadata(torch.device("cpu"))
    assert metadata["physical_memory_bytes"] is None
    assert metadata["cpu_brand"] is None


def test_synthetic_timer_preserves_warmup_and_synchronization_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    clock = iter([1.0, 1.002, 2.0, 2.004, 3.0, 3.006])
    monkeypatch.setattr(support, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(support.time, "perf_counter", lambda: next(clock))

    def operation() -> int:
        events.append("operation")
        return events.count("operation")

    latency, result = support.median_latency_ms(
        operation,
        warmups=2,
        repetitions=3,
        device=torch.device("cpu"),
    )
    assert latency == pytest.approx(4.0)
    assert result == 5
    assert (
        events == ["operation", "operation", "sync"] + ["sync", "operation", "sync"] * 3
    )


def test_synthetic_relative_error_preserves_input_precision_and_zero_cases() -> None:
    exact = torch.tensor([2e100], dtype=torch.float64)
    approximate = torch.tensor([1e100], dtype=torch.float64)
    assert support.relative_error(approximate, exact) == 0.5
    assert support.relative_error(torch.zeros(1), torch.zeros(1)) == 0.0
    assert math.isinf(support.relative_error(torch.ones(1), torch.zeros(1)))


def test_report_distribution_contracts_remain_distinct() -> None:
    values = [0.0, 10.0, 20.0, 30.0]
    expected = {
        "count": 4,
        "mean": 15.0,
        "median": 15.0,
        "min": 0.0,
        "p10": 3.0,
        "p25": 7.5,
        "p75": 22.5,
        "p90": 27.0,
        "max": 30.0,
    }
    assert metric_distribution([*values, float("inf"), float("nan")]) == pytest.approx(
        expected
    )
    assert policy_distribution([None, *values, float("inf")]) == pytest.approx(
        {**expected, "p95": 28.5},
    )
    assert basic_distribution(values) == {
        "mean": 15.0,
        "median": 15.0,
        "min": 0.0,
        "max": 30.0,
    }
    assert latency_distribution(values) == pytest.approx(
        {
            "count": 4,
            "mean": 15.0,
            "median": 15.0,
            "p90": 27.0,
            "p95": 28.5,
            "minimum": 0.0,
            "maximum": 30.0,
        }
    )
    assert metric_distribution([]) == {"count": 0}
    assert policy_distribution([None, float("nan")]) == {"count": 0}
    with pytest.raises(ValueError):
        latency_distribution([])
    with pytest.raises(TypeError):
        metric_distribution([None])


def test_percentiles_handle_singletons_endpoints_and_invalid_quantiles() -> None:
    assert percentile([5.0], 0.5) == 5.0
    assert percentile([4.0, 0.0], 0.25) == 1.0
    assert percentile([4.0, 0.0], 0.0) == 0.0
    assert percentile([4.0, 0.0], 1.0) == 4.0
    with pytest.raises(ValueError):
        percentile([], 0.5)
    for quantile in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            percentile([1.0], quantile)


def test_correlation_filtering_and_return_contracts() -> None:
    records = [{"x": 1.0, "y": 6.0}, {"x": 2.0, "y": 4.0}, {"x": 3.0, "y": 2.0}]
    assert pearson_correlation(records, "x", "y") == (3, -1.0)
    assert legacy_pearson_correlation(records, "x", "y") == -1.0
    contaminated = [*records, {"x": None, "y": 1.0}, {"x": float("inf"), "y": 1.0}]
    assert pearson_correlation([*contaminated, {}], "x", "y") == (3, -1.0)
    assert math.isnan(legacy_pearson_correlation(contaminated, "x", "y"))
    with pytest.raises(KeyError):
        legacy_pearson_correlation([{}], "x", "y")
    constant = [{"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 3.0}]
    assert pearson_correlation(constant, "x", "y") == (2, None)
    assert pearson_correlation(records[:1], "x", "y") == (1, None)
