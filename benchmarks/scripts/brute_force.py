#!/usr/bin/env python3
"""Smoke benchmark for exact Top-K retrieval on synthetic KV tensors."""

import argparse
import platform
import statistics
import subprocess
import time

import torch

from kvdb import BruteForceIndex


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-lengths", nargs="+", type=int, default=[128, 512, 2048, 8192]
    )
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


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


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "budget": args.budget,
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
    if any(args.budget > length for length in args.context_lengths):
        raise ValueError("budget cannot exceed a context length")


def measure_context(
    *,
    context_length: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, int]:
    generator = torch.Generator(device=device).manual_seed(args.seed + context_length)
    keys = torch.randn(
        args.batch_size,
        args.kv_heads,
        context_length,
        args.head_dim,
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
    index = BruteForceIndex()
    index.build(keys)

    for _ in range(args.warmups):
        index.search(query, args.budget)
    synchronize(device)

    latencies_ms: list[float] = []
    for _ in range(args.repetitions):
        synchronize(device)
        start = time.perf_counter()
        index.search(query, args.budget)
        synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1_000)

    score_count = args.batch_size * args.kv_heads * context_length
    selection_count = args.batch_size * args.kv_heads * args.budget
    estimated_tensor_bytes = (
        keys.numel() * keys.element_size()
        + query.numel() * query.element_size()
        + score_count * keys.element_size()
        + selection_count * keys.element_size()
        + selection_count * torch.tensor([], dtype=torch.int64).element_size()
    )
    return statistics.median(latencies_ms), estimated_tensor_bytes


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    print("benchmark=brute_force_retrieval_smoke")
    print(
        "model=synthetic model_revision=not_applicable generated_tokens=0 "
        "baseline=exact_dot_product_topk"
    )
    print(
        f"torch_version={torch.__version__} git_commit={git_commit()} "
        f"hardware={hardware_name(device)!r}"
    )
    print(
        f"seed={args.seed} batch_size={args.batch_size} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} warmups={args.warmups} repetitions={args.repetitions} "
        "index_parameters='score=raw_dot_product;selection=per_batch_head'"
    )
    print(
        "context_length,retrieval_budget,retrieval_latency_ms,device,dtype,"
        "estimated_tensor_bytes"
    )
    for context_length in args.context_lengths:
        latency_ms, estimated_tensor_bytes = measure_context(
            context_length=context_length,
            args=args,
            device=device,
            dtype=dtype,
        )
        print(
            f"{context_length},{args.budget},{latency_ms:.6f},"
            f"{device},{dtype},{estimated_tensor_bytes}"
        )


if __name__ == "__main__":
    main()
