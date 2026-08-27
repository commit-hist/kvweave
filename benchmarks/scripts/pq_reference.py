#!/usr/bin/env python3
"""Reference Quest-versus-PQ benchmark on deterministic synthetic tensors."""

import argparse
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from typing import TypeVar

import torch

from kvdb import BruteForceIndex, PQIndex, QuestIndex, TensorStorage
from kvdb.core.types import Selection
from kvdb.indexes.pq import reconstruct_keys
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
        default=[64, 256, 1_024],
    )
    parser.add_argument(
        "--budget-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0],
    )
    parser.add_argument("--page-sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--pq-subspaces", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--pq-centroids", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "batch_size": args.batch_size,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "kmeans_iterations": args.kmeans_iterations,
        "repetitions": args.repetitions,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.warmups < 0:
        raise ValueError("warmups cannot be negative")
    if any(length <= 0 for length in args.context_lengths):
        raise ValueError("context lengths must be positive")
    if any(page_size <= 0 for page_size in args.page_sizes):
        raise ValueError("page sizes must be positive")
    if any(subspaces <= 0 for subspaces in args.pq_subspaces):
        raise ValueError("PQ subspace counts must be positive")
    if any(args.head_dim % subspaces != 0 for subspaces in args.pq_subspaces):
        raise ValueError("head_dim must be divisible by every PQ subspace count")
    if any(centroids <= 0 for centroids in args.pq_centroids):
        raise ValueError("PQ centroid counts must be positive")
    if any(
        centroids > context_length
        for centroids in args.pq_centroids
        for context_length in args.context_lengths
    ):
        raise ValueError("PQ centroids cannot exceed a context length")
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
    shape = (
        args.batch_size,
        args.kv_heads,
        context_length,
        args.head_dim,
    )
    keys = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    values = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    query = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return query, keys, values


def logical_packed_code_bytes(codes: torch.Tensor, num_centroids: int) -> int:
    bits_per_code = (num_centroids - 1).bit_length()
    return math.ceil(codes.numel() * bits_per_code / 8)


def measure_selection_path(
    *,
    selection_operation: Callable[[], Selection],
    exact_topk: Selection,
    query: torch.Tensor,
    keys: torch.Tensor,
    storage: TensorStorage,
    full_output: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, Selection, float, float, float, float]:
    retrieval_ms, selection = median_latency_ms(
        selection_operation,
        warmups=args.warmups,
        repetitions=args.repetitions,
        device=device,
    )
    storage_fetch_ms, retrieved = median_latency_ms(
        lambda: storage.fetch(selection),
        warmups=args.warmups,
        repetitions=args.repetitions,
        device=device,
    )
    attention_ms, selected_output = median_latency_ms(
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
    recall = candidate_recall(selection, exact_topk).mean().item()
    output_error = relative_error(selected_output, full_output)
    if selection.indices.shape[-1] == keys.shape[2]:
        torch.testing.assert_close(
            selected_output,
            full_output,
            rtol=1e-4,
            atol=1e-5,
        )
    return (
        retrieval_ms,
        selection,
        storage_fetch_ms,
        attention_ms,
        recall,
        output_error,
    )


def print_row(
    *,
    strategy: str,
    context_length: int,
    requested_budget: int,
    budget_fraction: float,
    selection: Selection,
    build_ms: float,
    retrieval_ms: float,
    storage_fetch_ms: float,
    selected_attention_ms: float,
    full_attention_ms: float,
    brute_force_retrieval_ms: float,
    recall: float,
    output_error: float,
    page_size: int | None,
    num_subspaces: int | None,
    num_centroids: int | None,
    kmeans_iterations: int | None,
    code_storage_bytes: int,
    logical_code_storage_bytes: int,
    codebook_storage_bytes: int,
    reconstruction_relative_error: float | None,
) -> None:
    actual_counts = selection.valid_token_counts.to(torch.float32)
    selected_percentage = 100.0 * actual_counts.mean().item() / context_length
    print(
        f"{strategy},{context_length},{requested_budget},{budget_fraction:.6f},"
        f"{int(actual_counts.min().item())},"
        f"{int(actual_counts.max().item())},"
        f"{actual_counts.mean().item():.3f},{selected_percentage:.6f},"
        f"{build_ms:.6f},{retrieval_ms:.6f},{storage_fetch_ms:.6f},"
        f"{selected_attention_ms:.6f},{full_attention_ms:.6f},"
        f"{brute_force_retrieval_ms:.6f},{recall:.6f},{output_error:.8f},"
        f"{page_size if page_size is not None else 'not_applicable'},"
        f"{num_subspaces if num_subspaces is not None else 'not_applicable'},"
        f"{num_centroids if num_centroids is not None else 'not_applicable'},"
        f"{kmeans_iterations if kmeans_iterations is not None else 'not_applicable'},"
        f"{code_storage_bytes},{logical_code_storage_bytes},"
        f"{codebook_storage_bytes},"
        f"{reconstruction_relative_error if reconstruction_relative_error is not None else 'not_applicable'}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    print("benchmark=quest_pq_reference_synthetic")
    print(
        "model=synthetic model_revision=not_applicable generated_tokens=0 "
        "baseline='exact_dot_product_topk_and_full_attention'"
    )
    print(
        f"torch_version={torch.__version__} git_commit={git_commit()} "
        f"git_dirty={git_is_dirty()} hardware={hardware_name(device)!r}"
    )
    print(
        f"seed={args.seed} batch_size={args.batch_size} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} warmups={args.warmups} "
        f"repetitions={args.repetitions} device={device} dtype={dtype} "
        "timing_scope='readable_reference_python_pytorch'"
    )
    print(
        "strategy,context_length,requested_token_budget,budget_fraction,"
        "actual_selected_tokens_min,actual_selected_tokens_max,"
        "actual_selected_tokens_mean,selected_percentage,index_build_ms,"
        "retrieval_ms,storage_fetch_ms,selected_attention_ms,full_attention_ms,"
        "brute_force_retrieval_ms,candidate_recall,attention_relative_error,"
        "page_size,pq_subspaces,pq_centroids,kmeans_max_iterations,"
        "actual_code_storage_bytes,logical_packed_code_storage_bytes,"
        "codebook_storage_bytes,reconstruction_relative_error"
    )

    for context_length in args.context_lengths:
        query, keys, values = make_tensors(
            context_length,
            args=args,
            device=device,
            dtype=dtype,
        )
        storage = TensorStorage()
        storage.put(keys, values)
        brute_force = BruteForceIndex()
        brute_force.build(keys)
        full_attention_ms, full_output = median_latency_ms(
            lambda: full_attention(query, keys, values),
            warmups=args.warmups,
            repetitions=args.repetitions,
            device=device,
        )
        exact_by_budget: dict[int, tuple[float, Selection]] = {}
        for budget_fraction in args.budget_fractions:
            budget = max(1, math.ceil(context_length * budget_fraction))
            if budget not in exact_by_budget:
                exact_by_budget[budget] = median_latency_ms(
                    lambda budget=budget: brute_force.search(query, budget),
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    device=device,
                )

        for page_size in args.page_sizes:

            def build_quest() -> QuestIndex:
                index = QuestIndex(page_size=page_size)
                index.build(keys)
                return index

            build_ms, quest = median_latency_ms(
                build_quest,
                warmups=args.warmups,
                repetitions=args.repetitions,
                device=device,
            )
            for budget_fraction in args.budget_fractions:
                budget = max(1, math.ceil(context_length * budget_fraction))
                brute_force_ms, exact_topk = exact_by_budget[budget]
                (
                    retrieval_ms,
                    selection,
                    storage_fetch_ms,
                    attention_ms,
                    recall,
                    output_error,
                ) = measure_selection_path(
                    selection_operation=lambda budget=budget: quest.search(
                        query,
                        budget,
                    ),
                    exact_topk=exact_topk,
                    query=query,
                    keys=keys,
                    storage=storage,
                    full_output=full_output,
                    args=args,
                    device=device,
                )
                print_row(
                    strategy="quest",
                    context_length=context_length,
                    requested_budget=budget,
                    budget_fraction=budget_fraction,
                    selection=selection,
                    build_ms=build_ms,
                    retrieval_ms=retrieval_ms,
                    storage_fetch_ms=storage_fetch_ms,
                    selected_attention_ms=attention_ms,
                    full_attention_ms=full_attention_ms,
                    brute_force_retrieval_ms=brute_force_ms,
                    recall=recall,
                    output_error=output_error,
                    page_size=page_size,
                    num_subspaces=None,
                    num_centroids=None,
                    kmeans_iterations=None,
                    code_storage_bytes=0,
                    logical_code_storage_bytes=0,
                    codebook_storage_bytes=0,
                    reconstruction_relative_error=None,
                )

        for num_subspaces in args.pq_subspaces:
            for num_centroids in args.pq_centroids:

                def build_pq() -> PQIndex:
                    index = PQIndex(
                        num_subspaces=num_subspaces,
                        num_centroids=num_centroids,
                        max_iterations=args.kmeans_iterations,
                        seed=args.seed,
                    )
                    index.build(keys)
                    return index

                build_ms, pq = median_latency_ms(
                    build_pq,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    device=device,
                )
                metadata = pq.metadata
                code_storage_bytes = (
                    metadata.codes.numel() * metadata.codes.element_size()
                )
                packed_code_bytes = logical_packed_code_bytes(
                    metadata.codes,
                    num_centroids,
                )
                codebook_storage_bytes = (
                    metadata.codebooks.numel() * metadata.codebooks.element_size()
                )
                reconstruction_error = relative_error(
                    reconstruct_keys(metadata),
                    keys,
                )
                for budget_fraction in args.budget_fractions:
                    budget = max(1, math.ceil(context_length * budget_fraction))
                    brute_force_ms, exact_topk = exact_by_budget[budget]
                    (
                        retrieval_ms,
                        selection,
                        storage_fetch_ms,
                        attention_ms,
                        recall,
                        output_error,
                    ) = measure_selection_path(
                        selection_operation=lambda budget=budget: pq.search(
                            query,
                            budget,
                        ),
                        exact_topk=exact_topk,
                        query=query,
                        keys=keys,
                        storage=storage,
                        full_output=full_output,
                        args=args,
                        device=device,
                    )
                    print_row(
                        strategy="pq",
                        context_length=context_length,
                        requested_budget=budget,
                        budget_fraction=budget_fraction,
                        selection=selection,
                        build_ms=build_ms,
                        retrieval_ms=retrieval_ms,
                        storage_fetch_ms=storage_fetch_ms,
                        selected_attention_ms=attention_ms,
                        full_attention_ms=full_attention_ms,
                        brute_force_retrieval_ms=brute_force_ms,
                        recall=recall,
                        output_error=output_error,
                        page_size=None,
                        num_subspaces=num_subspaces,
                        num_centroids=num_centroids,
                        kmeans_iterations=args.kmeans_iterations,
                        code_storage_bytes=code_storage_bytes,
                        logical_code_storage_bytes=packed_code_bytes,
                        codebook_storage_bytes=codebook_storage_bytes,
                        reconstruction_relative_error=reconstruction_error,
                    )


if __name__ == "__main__":
    main()
