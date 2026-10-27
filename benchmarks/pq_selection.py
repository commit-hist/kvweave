"""Phase 5B selection diagnostics; never run inside primary decode timings."""

from collections import Counter
from collections.abc import Callable
import cProfile
import hashlib
import math
import pstats
import random
import time
from typing import Any

import torch

from benchmarks.report_statistics import latency_distribution as distribution
from kvweave.indexes.pq.reference import build_pq_metadata, score_pq_codes
from kvweave.indexes.pq.selection import (
    deterministic_partial_selection,
    full_stable_ranking,
)


SelectionFunction = Callable[[torch.Tensor, int], torch.Tensor]


def full_selection(scores: torch.Tensor, budget: int) -> torch.Tensor:
    return full_stable_ranking(scores)[..., :budget]


def kthvalue_prototype(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Alternative threshold primitive; identical deterministic tie completion."""
    length = scores.shape[-1]
    if budget == length or torch.isnan(scores).any():
        return full_selection(scores, budget)
    threshold = torch.kthvalue(scores, length - budget + 1, dim=-1, keepdim=True).values
    above = scores > threshold
    tied = scores == threshold
    remaining = budget - above.sum(dim=-1, keepdim=True)
    selected = above | (tied & (tied.cumsum(dim=-1) <= remaining))
    indices = (
        selected.reshape(-1, length).nonzero()[:, 1].reshape(*scores.shape[:-1], budget)
    )
    order = scores.gather(-1, indices).argsort(dim=-1, descending=True, stable=True)
    return indices.gather(-1, order)


def integer_priority_prototype(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Partial selection on integer priority/ID keys, without modifying scores."""
    length = scores.shape[-1]
    if budget == length or torch.isnan(scores).any():
        return full_selection(scores, budget)
    threshold = torch.topk(scores, budget, dim=-1, sorted=False).values.amin(
        -1, keepdim=True
    )
    ids = torch.arange(length, device=scores.device)
    # Disjoint integer ranges prioritize > threshold, then == threshold by ID.
    priority = torch.where(
        scores > threshold,
        ids,
        torch.where(scores == threshold, ids + length, 2 * length),
    )
    indices = (
        priority.topk(budget, dim=-1, largest=False, sorted=False)
        .indices.sort(-1)
        .values
    )
    order = scores.gather(-1, indices).argsort(dim=-1, descending=True, stable=True)
    return indices.gather(-1, order)


def sorted_topk_tie_repair_prototype(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Repair only boundary membership, then sort exact integer group/ID keys."""
    s = scores.shape[-1]
    if k == s or torch.isnan(scores).any():
        return scores.argsort(dim=-1, descending=True, stable=True)[..., :k]
    values, ids = scores.topk(k, dim=-1, sorted=True)
    threshold = values[..., -1:]
    selected_ties = values == threshold
    count = selected_ties.sum(-1, keepdim=True)
    ties = scores == threshold
    chosen = ties & (ties.cumsum(-1) <= count)
    correct_ids = chosen.reshape(-1, s).nonzero()[:, 1]
    ids[selected_ties] = correct_ids
    starts = torch.ones_like(ids)
    starts[..., 1:] = values[..., 1:] != values[..., :-1]
    groups = starts.cumsum(-1)
    key = groups * s + ids
    order = key.argsort(dim=-1)
    return ids.gather(-1, order)


def update_digest(digest: Any, tensor: torch.Tensor) -> None:
    """Hash shape, dtype and exact bytes, retaining signed-zero distinctions."""
    digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
    digest.update(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )


def assert_tensor_exact(
    actual: torch.Tensor, expected: torch.Tensor, name: str
) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype mismatch")
    if not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        raise AssertionError(f"{name}: tensor bytes differ")


class TieStatistics:
    """Count searches [B,H,S], rows [S], and exact-score groups separately."""

    def __init__(self) -> None:
        self.searches = 0
        self.searches_with_ties = 0
        self.searches_with_boundary_ties = 0
        self.searches_with_cut_ties = 0
        self.rows = 0
        self.rows_with_ties = 0
        self.rows_with_boundary_ties = 0
        self.rows_with_cut_ties = 0
        self.group_sizes: Counter[int] = Counter()
        self.boundary_sizes: Counter[int] = Counter()

    def add(self, scores: torch.Tensor, budget: int) -> None:
        ranked = scores.gather(-1, full_stable_ranking(scores)).reshape(
            -1, scores.shape[-1]
        )
        search_tied = search_boundary = search_cut = False
        for row in ranked:
            _, counts = row.unique_consecutive(return_counts=True)
            sizes = counts[counts > 1].tolist()
            self.group_sizes.update(sizes)
            boundary = row[budget - 1]
            boundary_size = int((row == boundary).sum())
            cut = budget < row.numel() and bool(row[budget] == boundary)
            self.boundary_sizes[boundary_size] += 1
            self.rows += 1
            self.rows_with_ties += bool(sizes)
            self.rows_with_boundary_ties += boundary_size > 1
            self.rows_with_cut_ties += cut
            search_tied |= bool(sizes)
            search_boundary |= boundary_size > 1
            search_cut |= cut
        self.searches += 1
        self.searches_with_ties += search_tied
        self.searches_with_boundary_ties += search_boundary
        self.searches_with_cut_ties += search_cut

    def report(self) -> dict[str, Any]:
        def histogram_summary(histogram: Counter[int]) -> dict[str, Any]:
            values = [size for size, count in histogram.items() for _ in range(count)]
            return {
                "histogram": dict(sorted(histogram.items())),
                "distribution": distribution(values) if values else None,
            }

        return {
            "searches": self.searches,
            "rows": self.rows,
            "fraction_searches_with_any_tie": self.searches_with_ties / self.searches,
            "fraction_searches_with_boundary_tie": self.searches_with_boundary_ties
            / self.searches,
            "fraction_searches_cutting_boundary_tie": self.searches_with_cut_ties
            / self.searches,
            "fraction_rows_with_any_tie": self.rows_with_ties / self.rows,
            "fraction_rows_with_boundary_tie": self.rows_with_boundary_ties / self.rows,
            "fraction_rows_cutting_boundary_tie": self.rows_with_cut_ties / self.rows,
            "tied_group_sizes": histogram_summary(self.group_sizes),
            "boundary_group_sizes": histogram_summary(self.boundary_sizes),
            "definition": "boundary tie means the Kth score has multiplicity >1; cut means equal scores straddle K/K+1; full budget never cuts",
        }


def compare_methods(
    scores: torch.Tensor, budget: int, *, repetitions: int, prototypes: bool = False
) -> dict[str, Any]:
    methods: dict[str, SelectionFunction] = {
        "full_stable_sort_oracle": full_selection,
        "deterministic_partial_selection": deterministic_partial_selection,
    }
    if prototypes:
        methods.update(
            kthvalue_threshold=kthvalue_prototype,
            integer_priority=integer_priority_prototype,
            sorted_topk_tie_repair=sorted_topk_tie_repair_prototype,
        )
    expected = full_selection(scores, budget)
    for function in methods.values():
        actual = function(scores, budget)
        assert_tensor_exact(actual, expected, "ranked IDs")
        assert_tensor_exact(
            scores.gather(-1, actual), scores.gather(-1, expected), "ranked scores"
        )
        for _ in range(5):
            function(scores, budget)
    durations: dict[str, list[float]] = {name: [] for name in methods}
    order = list(methods)
    generator = random.Random(0)
    for _ in range(repetitions):
        generator.shuffle(order)
        for name in order:
            start = time.perf_counter_ns()
            methods[name](scores, budget)
            durations[name].append((time.perf_counter_ns() - start) / 1e6)
    digest = hashlib.sha256()
    update_digest(digest, expected)
    results = {name: distribution(values) for name, values in durations.items()}
    old = float(results["full_stable_sort_oracle"]["median"])
    new = float(results["deterministic_partial_selection"]["median"])
    return {
        "timing_ms": results,
        "raw_timing_ms": durations,
        "exact_ids_and_scores": True,
        "selection_sha256": digest.hexdigest(),
        "median_ms_saved": old - new,
        "median_speed_ratio": old / new,
        "median_reduction_percent": 100 * (old - new) / old,
    }


def scaling_benchmark(*, repetitions: int = 51) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)
    rows = []
    for batch, heads in ((1, 16), (2, 4)):
        for length in (512, 2048, 8192, 32768):
            for kind in ("continuous", "duplicated"):
                scores = (
                    torch.randn(batch, heads, length, generator=generator)
                    if kind == "continuous"
                    else torch.randint(
                        -128, 128, (batch, heads, length), generator=generator
                    ).float()
                )
                for fraction in (0.125, 0.25, 0.5, 1.0):
                    budget = math.ceil(length * fraction)
                    rows.append(
                        {
                            "B": batch,
                            "H": heads,
                            "S": length,
                            "K": budget,
                            "budget_fraction": fraction,
                            "score_distribution": kind,
                            **compare_methods(
                                scores, budget, repetitions=repetitions, prototypes=True
                            ),
                        }
                    )
            print(f"scaling B={batch} H={heads} S={length} complete", flush=True)
    reconstructed = []
    ties = TieStatistics()
    for length, heads, subspaces, centroids in (
        (512, 4, 2, 4),
        (1024, 16, 4, 8),
        (2048, 16, 4, 8),
    ):
        keys = torch.randn(1, heads, length, 64, generator=generator)
        metadata = build_pq_metadata(
            keys,
            num_subspaces=subspaces,
            num_centroids=centroids,
            max_iterations=8,
            seed=0,
        )
        for query_index in range(3):
            scores = score_pq_codes(
                torch.randn(1, heads, 64, generator=generator), metadata
            )
            for fraction in (0.125, 0.25, 0.5, 1.0):
                budget = math.ceil(length * fraction)
                ties.add(scores, budget)
                reconstructed.append(
                    {
                        "B": 1,
                        "H": heads,
                        "S": length,
                        "K": budget,
                        "M": subspaces,
                        "C": centroids,
                        "query": query_index,
                        "budget_fraction": fraction,
                        **compare_methods(
                            scores, budget, repetitions=repetitions, prototypes=True
                        ),
                    }
                )
    return {
        "warmup_calls_per_method": 5,
        "repetitions": repetitions,
        "order": "seed-0 shuffled method order within each repetition; single process, unchanged thread settings",
        "score_dtype": "float32",
        "device": "cpu",
        "rows": rows,
        "reconstructed_pq_rows": reconstructed,
        "reconstructed_pq_ties": ties.report(),
    }


def selection_profile(scores: torch.Tensor, budget: int) -> dict[str, Any]:
    """Separate operator/allocation and Python profiles of ranking only."""
    results = {}
    for name, function in (
        ("full_stable_sort_oracle", full_selection),
        ("deterministic_partial_selection", deterministic_partial_selection),
    ):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for _ in range(25):
                function(scores, budget)
        operators = [
            {
                "operator": event.key,
                "calls": event.count,
                "self_cpu_ms": event.self_cpu_time_total / 1000,
                "total_cpu_ms": event.cpu_time_total / 1000,
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "cpu_memory_bytes": event.cpu_memory_usage,
                "input_shapes": event.input_shapes,
            }
            for event in profiler.key_averages(group_by_input_shape=True)
        ]
        python = cProfile.Profile()
        python.enable()
        for _ in range(100):
            function(scores, budget)
        python.disable()
        stats = pstats.Stats(python)
        functions = [
            {
                "function": f"{file}:{line}({func})",
                "calls": total,
                "self_ms": self_time * 1000,
                "cumulative_ms": cumulative * 1000,
                "python_source": not file.startswith("~"),
            }
            for (file, line, func), (
                _,
                total,
                self_time,
                cumulative,
                _,
            ) in stats.stats.items()
        ]
        results[name] = {
            "operator_repetitions": 25,
            "python_repetitions": 100,
            "operators": sorted(
                operators, key=lambda row: row["self_cpu_ms"], reverse=True
            ),
            "functions": sorted(
                functions, key=lambda row: row["self_ms"], reverse=True
            ),
            "python_profile_total_ms": stats.total_tt * 1000,
        }
    return results


def selection_accounting(
    *, batch: int, heads: int, length: int, budget: int, score_bytes: int = 4
) -> dict[str, Any]:
    """Logical tensor payloads, not allocator peaks or measured DRAM traffic."""
    rows = batch * heads
    full = rows * length
    selected = rows * budget
    return {
        "B": batch,
        "H": heads,
        "S": length,
        "K": budget,
        "score_tensor_bytes_unchanged": full * score_bytes,
        "full_sort": {
            "full_ranked_indices_bytes": full * 8,
            "sort_values_bytes": full * score_bytes,
            "returned_id_view_retained_storage_bytes": full * 8,
            "final_selected_scores_bytes": selected * score_bytes,
        },
        "partial": (
            {"same_full_sort_path": True}
            if budget == length
            else {
                "nan_check_mask_bytes": full,
                "topk_values_bytes": selected * score_bytes,
                "topk_indices_bytes_unused": selected * 8,
                "threshold_bytes": rows * score_bytes,
                "above_and_tied_masks_bytes": 2 * full,
                "above_reduction_and_remaining_bytes": 2 * rows * 8,
                "tie_cumsum_bytes": full * 8,
                "boolean_to_int64_conversion_upper_estimate_bytes_each": full * 8,
                "tie_capacity_comparison_and_boolean_intermediates_bytes": 3 * full,
                "nonzero_coordinates_bytes": selected * 2 * 8,
                "candidate_score_gather_bytes": selected * score_bytes,
                "candidate_sort_values_bytes": selected * score_bytes,
                "candidate_sort_indices_bytes": selected * 8,
                "final_ranked_indices_bytes": selected * 8,
                "final_selected_scores_bytes": selected * score_bytes,
            }
        ),
        "traffic": {
            "full_sort": {
                "minimum_full_score_read_bytes": full * score_bytes,
                "full_index_write_bytes": full * 8,
                "final_gather_id_read_bytes": selected * 8,
                "final_score_read_plus_write_bytes": 2 * selected * score_bytes,
            },
            "partial": (
                {"same_full_sort_path": True}
                if budget == length
                else {
                    "full_score_scan_lower_bound_bytes": 4 * full * score_bytes,
                    "scans": "NaN check, topk threshold, > threshold, == threshold; topk internal rereads excluded",
                    "topk_index_write_bytes": selected * 8,
                    "nonzero_index_write_bytes": selected * 2 * 8,
                    "candidate_sort_index_and_final_id_write_bytes": 2 * selected * 8,
                    "two_candidate_gathers_score_read_plus_write_bytes": 4
                    * selected
                    * score_bytes,
                    "additional_traffic": "full masks, reductions, int64 tie prefix sums and K-candidate sorting; native workspace and repeated sorting passes not estimated",
                }
            ),
        },
        "limitations": "payloads are per operation, not additive peak-live storage; native topk/sort workspace and allocator reuse excluded; no hardware-bandwidth or asymptotic guarantee inferred",
    }


def compare_score_batches(
    scores: list[torch.Tensor], *, budget_fraction: float, repetitions: int
) -> dict[str, Any]:
    """Rank one captured score batch per layer, summing no other decode work."""
    methods: dict[str, SelectionFunction] = {
        "full_stable_sort_oracle": full_selection,
        "deterministic_partial_selection": deterministic_partial_selection,
        "kthvalue_threshold": kthvalue_prototype,
        "integer_priority": integer_priority_prototype,
        "sorted_topk_tie_repair": sorted_topk_tie_repair_prototype,
    }
    budgets = [math.ceil(score.shape[-1] * budget_fraction) for score in scores]
    digest = hashlib.sha256()
    for score, budget in zip(scores, budgets, strict=True):
        expected = full_selection(score, budget)
        update_digest(digest, expected)
        for function in methods.values():
            actual = function(score, budget)
            assert_tensor_exact(actual, expected, "prototype IDs")
            assert_tensor_exact(
                score.gather(-1, actual), score.gather(-1, expected), "prototype scores"
            )
    durations: dict[str, list[float]] = {name: [] for name in methods}
    for function in methods.values():
        for _ in range(5):
            for score, budget in zip(scores, budgets, strict=True):
                function(score, budget)
    generator = random.Random(0)
    for _ in range(repetitions):
        order = list(methods)
        generator.shuffle(order)
        for name in order:
            start = time.perf_counter_ns()
            for score, budget in zip(scores, budgets, strict=True):
                methods[name](score, budget)
            durations[name].append((time.perf_counter_ns() - start) / 1e6)
    return {
        "timing_ms": {name: distribution(values) for name, values in durations.items()},
        "raw_ms": durations,
        "selection_sha256": digest.hexdigest(),
        "exact_ids_and_scores": True,
        "layer_count": len(scores),
        "budget_fraction": budget_fraction,
        "warmup_replays": 5,
    }
