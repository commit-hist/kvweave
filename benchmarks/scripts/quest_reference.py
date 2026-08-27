#!/usr/bin/env python3
"""Reference full-attention versus Quest benchmark on synthetic tensors."""

import argparse
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from typing import TypeVar

import torch

from kvdb import BruteForceIndex, QuestIndex, TensorStorage
from kvdb.indexes.quest.reference import (
    candidate_recall,
    full_attention,
    selected_attention,
)


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[512, 2_048, 8_192, 32_768],
    )
    parser.add_argument("--page-sizes", nargs="+", type=int, default=[16, 64])
    parser.add_argument(
        "--budget-fractions",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.25, 0.125],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "batch_size": args.batch_size,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "repetitions": args.repetitions,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmups < 0:
        raise ValueError("warmups cannot be negative")
    if any(length <= 0 for length in args.context_lengths):
        raise ValueError("context lengths must be positive")
    if any(page_size <= 0 for page_size in args.page_sizes):
        raise ValueError("page sizes must be positive")
    if any(fraction <= 0.0 or fraction > 1.0 for fraction in args.budget_fractions):
        raise ValueError("budget fractions must be in (0, 1]")


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


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def git_is_dirty() -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def relative_error(approximate: torch.Tensor, exact: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(approximate - exact)
    denominator = torch.linalg.vector_norm(exact)
    if denominator.item() == 0:
        return 0.0 if numerator.item() == 0 else float("inf")
    return (numerator / denominator).item()


def make_tensors(
    context_length: int,
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed + context_length)
    tensor_shape = (
        args.batch_size,
        args.kv_heads,
        context_length,
        args.head_dim,
    )
    keys = torch.randn(*tensor_shape, generator=generator, device=device, dtype=dtype)
    values = torch.randn(
        *tensor_shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    query = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return query, keys, values


def run_ragged_regression(
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Measure the accepted mask boundary for ``S=65`` and page size 8."""
    context_length = 65
    page_size = 8
    requested_budget = 64
    batch_size = 2
    kv_heads = 2
    keys = torch.zeros(
        batch_size,
        kv_heads,
        context_length,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    query = torch.ones(
        batch_size,
        kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_id in range(batch_size):
        for head_id in range(kv_heads):
            tail_value = 10.0 if (batch_id + head_id) % 2 == 0 else -10.0
            keys[batch_id, head_id, -1] = tail_value
    generator = torch.Generator(device=device).manual_seed(args.seed + 65_008)
    values = torch.randn(
        batch_size,
        kv_heads,
        context_length,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    quest = QuestIndex(page_size=page_size)
    quest.build(keys)
    storage = TensorStorage()
    storage.put(keys, values)

    search_ms, selection = median_latency_ms(
        lambda: quest.search(query, requested_budget),
        warmups=args.warmups,
        repetitions=args.repetitions,
        device=device,
    )
    if selection.valid_mask is None:
        raise AssertionError("ragged Quest regression must produce a validity mask")
    fetch_ms, retrieved = median_latency_ms(
        lambda: storage.fetch(selection),
        warmups=args.warmups,
        repetitions=args.repetitions,
        device=device,
    )
    if retrieved.valid_mask is None:
        raise AssertionError("TensorStorage must preserve the ragged validity mask")
    attention_ms, _ = median_latency_ms(
        lambda: selected_attention(
            query,
            retrieved.keys,
            retrieved.values,
            retrieved.valid_mask,
        ),
        warmups=args.warmups,
        repetitions=args.repetitions,
        device=device,
    )
    counts = selection.valid_token_counts.to(torch.float32)
    print("ragged_regression=quest_mask_boundary")
    print(
        "context_length,page_size,requested_token_budget,"
        "actual_selected_tokens_min,actual_selected_tokens_max,"
        "actual_selected_tokens_mean,masked_placeholder_count,"
        "quest_search_ms,tensor_storage_fetch_ms,masked_selected_attention_ms"
    )
    print(
        f"{context_length},{page_size},{requested_budget},"
        f"{int(counts.min().item())},{int(counts.max().item())},"
        f"{counts.mean().item():.3f},"
        f"{int((~selection.valid_mask).sum().item())},"
        f"{search_ms:.6f},{fetch_ms:.6f},{attention_ms:.6f}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    print("benchmark=quest_reference_synthetic")
    print(
        "model=synthetic model_revision=not_applicable generated_tokens=0 "
        "baseline='exact_dot_product_topk_and_full_attention'"
    )
    print(
        f"torch_version={torch.__version__} git_commit={git_commit()} "
        f"git_dirty={git_is_dirty()} "
        f"hardware={hardware_name(device)!r}"
    )
    print(
        f"seed={args.seed} batch_size={args.batch_size} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} warmups={args.warmups} "
        f"repetitions={args.repetitions} device={device} dtype={dtype} "
        "budget_policy='requested_tokens;ceil_to_pages' "
        "tie_policy='descending_score;ascending_page_id'"
    )
    print(
        "context_length,page_size,requested_token_budget,budget_fraction,"
        "pages_selected,actual_selected_tokens_min,actual_selected_tokens_max,"
        "actual_selected_tokens_mean,selected_percentage,metadata_build_ms,"
        "metadata_bytes,quest_retrieval_ms,brute_force_retrieval_ms,"
        "brute_force_comparison_k,candidate_recall,full_attention_ms,"
        "selected_attention_ms,attention_relative_error"
    )

    for context_length in args.context_lengths:
        query, keys, values = make_tensors(
            context_length,
            args=args,
            device=device,
            dtype=dtype,
        )
        brute_force = BruteForceIndex()
        brute_force.build(keys)
        storage = TensorStorage()
        storage.put(keys, values)
        full_attention_ms, full_output = median_latency_ms(
            lambda: full_attention(query, keys, values),
            warmups=args.warmups,
            repetitions=args.repetitions,
            device=device,
        )

        for page_size in args.page_sizes:
            def build_quest() -> QuestIndex:
                index = QuestIndex(page_size=page_size)
                index.build(keys)
                return index

            metadata_build_ms, quest = median_latency_ms(
                build_quest,
                warmups=args.warmups,
                repetitions=args.repetitions,
                device=device,
            )
            metadata = quest.metadata
            metadata_bytes = (
                metadata.minimum.numel() * metadata.minimum.element_size()
                + metadata.maximum.numel() * metadata.maximum.element_size()
            )

            for budget_fraction in args.budget_fractions:
                requested_budget = max(1, math.ceil(context_length * budget_fraction))
                comparison_k = min(requested_budget, context_length)
                quest_retrieval_ms, quest_result = median_latency_ms(
                    lambda: quest.search_with_details(query, requested_budget),
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    device=device,
                )
                brute_force_ms, exact_topk = median_latency_ms(
                    lambda: brute_force.search(query, comparison_k),
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    device=device,
                )

                def fetch_and_run_selected_attention() -> torch.Tensor:
                    retrieved = storage.fetch(quest_result.selection)
                    return selected_attention(
                        query,
                        retrieved.keys,
                        retrieved.values,
                        retrieved.valid_mask,
                    )

                selected_attention_ms, approximate_output = median_latency_ms(
                    fetch_and_run_selected_attention,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    device=device,
                )

                actual_counts = quest_result.actual_token_counts.to(torch.float32)
                selected_percentage = (
                    100.0 * actual_counts.mean().item() / context_length
                )
                recall = candidate_recall(
                    quest_result.selection,
                    exact_topk,
                ).mean().item()
                output_error = relative_error(approximate_output, full_output)
                print(
                    f"{context_length},{page_size},{requested_budget},"
                    f"{budget_fraction:.6f},{quest_result.num_pages_to_select},"
                    f"{int(actual_counts.min().item())},"
                    f"{int(actual_counts.max().item())},"
                    f"{actual_counts.mean().item():.3f},{selected_percentage:.6f},"
                    f"{metadata_build_ms:.6f},{metadata_bytes},"
                    f"{quest_retrieval_ms:.6f},{brute_force_ms:.6f},"
                    f"{comparison_k},{recall:.6f},{full_attention_ms:.6f},"
                    f"{selected_attention_ms:.6f},{output_error:.8f}"
                )

    run_ragged_regression(args=args, device=device, dtype=dtype)


if __name__ == "__main__":
    main()
