#!/usr/bin/env python3
"""Run the pinned Pythia Phase 3B stateful autoregressive decode experiment."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
from pathlib import Path
import platform
import statistics
from typing import Any, Iterable

import torch

from benchmarks.artifacts import (
    atomic_output,
    prepare_report,
    write_content_addressed,
    write_json,
)
from benchmarks.support import git_commit, git_is_dirty, hardware_name
from benchmarks.decode import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TRANSFORMERS_VERSION,
    DEFAULT_TRANSFORMERS_REVISION,
    build_dense_trace,
    validate_hugging_face_generation,
    assert_full_budget_step,
)
from benchmarks.phase3a import (
    TEXT_FIXTURES,
    TextFixture,
    build_deterministic_fixture,
)
from kvweave.integrations.transformers import (
    DecodeMode,
    DecodeStrategy,
    DensePrefillSnapshot,
    GPTNeoXDecodeRunner,
    GPTNeoXDecodeStep,
    QuestMetadataUpdateMode,
    attention_mass_captured,
    generation_divergence_metrics,
    logit_comparison_metrics,
    per_head_relative_error,
    relative_tensor_error,
    select_decode_input,
    validate_gpt_neox_config,
)


DEFAULT_OUTPUT = Path("benchmarks/results/pythia-410m-phase3b-decode.json")
DEFAULT_DENSE_TENSORS_OUTPUT = Path(
    "benchmarks/results/pythia-410m-phase3b-dense-tensors.pt"
)
DEFAULT_FIXTURES = (
    "narrative_prose",
    "technical_exposition",
    "code_like",
    "list_table",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--fixture-ids", nargs="+", default=list(DEFAULT_FIXTURES))
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[256, 512, 1_024],
    )
    parser.add_argument("--generated-tokens", type=int, default=32)
    parser.add_argument(
        "--budget-fractions",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.25],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in DecodeMode],
        default=[mode.value for mode in DecodeMode],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[DecodeStrategy.QUEST.value, DecodeStrategy.PQ.value],
        default=[DecodeStrategy.QUEST.value, DecodeStrategy.PQ.value],
    )
    parser.add_argument("--quest-page-size", type=int, default=64)
    parser.add_argument("--pq-subspaces", type=int, default=4)
    parser.add_argument("--pq-centroids", type=int, default=8)
    parser.add_argument("--pq-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-hf-validation", action="store_true")
    parser.add_argument("--full-budget-rtol", type=float, default=1e-4)
    parser.add_argument("--full-budget-atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dense-tensors-output",
        type=Path,
        default=DEFAULT_DENSE_TENSORS_OUTPUT,
        help="sidecar base name; actual filename includes its SHA-256 and is recorded in JSON",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[TextFixture, ...]:
    if args.model_id != DEFAULT_MODEL_ID:
        raise ValueError("Phase 3B must use the accepted pinned model ID")
    if args.model_revision != DEFAULT_MODEL_REVISION:
        raise ValueError("Phase 3B must use the accepted pinned model revision")
    if args.generated_tokens < 2:
        raise ValueError("generated_tokens must be at least two to exercise decode")
    if any(length <= 0 for length in args.sequence_lengths):
        raise ValueError("sequence lengths must be positive")
    if len(set(args.sequence_lengths)) != len(args.sequence_lengths):
        raise ValueError("sequence lengths must be unique")
    if any(not 0.0 < fraction <= 1.0 for fraction in args.budget_fractions):
        raise ValueError("budget fractions must be in (0, 1]")
    if len(set(args.budget_fractions)) != len(args.budget_fractions):
        raise ValueError("budget fractions must be unique")
    if 1.0 not in args.budget_fractions:
        raise ValueError("the 100% correctness control is required")
    if args.quest_page_size != 64:
        raise ValueError("Phase 3B Quest configuration is frozen at page size 64")
    if (args.pq_subspaces, args.pq_centroids) != (4, 8):
        raise ValueError("Phase 3B PQ configuration is frozen at M4/C8")
    if args.pq_iterations != 8:
        raise ValueError("Phase 3B retains the accepted eight PQ iterations")
    if args.seed != 0:
        raise ValueError("Phase 3B retains the accepted deterministic seed zero")
    available = {fixture.fixture_id: fixture for fixture in TEXT_FIXTURES}
    if len(set(args.fixture_ids)) != len(args.fixture_ids):
        raise ValueError("fixture IDs must be unique")
    unknown = sorted(set(args.fixture_ids) - set(available))
    if unknown:
        raise ValueError(f"unknown fixture IDs: {', '.join(unknown)}")
    fixtures = tuple(available[fixture_id] for fixture_id in args.fixture_ids)
    required_structures = {
        "narrative_prose",
        "technical_exposition",
        "code_like",
        "list_table",
    }
    if (
        args.fixture_ids == list(DEFAULT_FIXTURES)
        and {fixture.fixture_id for fixture in fixtures} != required_structures
    ):
        raise AssertionError("default fixture coverage changed unexpectedly")
    return fixtures


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def layer_metrics(
    approximate: GPTNeoXDecodeStep,
    dense: GPTNeoXDecodeStep,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for approximate_layer, dense_layer in zip(
        approximate.layers,
        dense.layers,
        strict=True,
    ):
        selection = approximate_layer.selection
        if selection is None:
            raise RuntimeError("approximate layer observation has no selection")
        attention_errors = per_head_relative_error(
            approximate_layer.attention_output,
            dense_layer.attention_output,
        )[0]
        attention_mass = attention_mass_captured(
            dense_layer.attention_weights,
            selection,
        )[0]
        results.append(
            {
                "layer": approximate_layer.layer_index,
                "sequence_length": approximate_layer.sequence_length,
                "head_indices": list(range(attention_errors.shape[0])),
                "selected_candidate_count_by_head": (
                    approximate_layer.selected_token_counts[0].cpu().tolist()
                ),
                "newest_token_included_for_every_head": (
                    approximate_layer.newest_token_included
                ),
                "attention_mass_captured_by_head": attention_mass.float()
                .cpu()
                .tolist(),
                "attention_output_relative_error_by_head": attention_errors.float()
                .cpu()
                .tolist(),
                "mean_attention_mass_captured": float(
                    attention_mass.float().mean().item()
                ),
                "mean_attention_output_relative_error": float(
                    attention_errors.float().mean().item()
                ),
                "maximum_attention_output_relative_error": float(
                    attention_errors.float().max().item()
                ),
                "residual_stream_relative_error": relative_tensor_error(
                    approximate_layer.residual_output,
                    dense_layer.residual_output,
                ),
                "timing_ms": {
                    "index_update_or_rebuild": (approximate_layer.index_update_time_ms),
                    "retrieval_search_and_policy": (
                        approximate_layer.retrieval_time_ms
                    ),
                    "storage_fetch": approximate_layer.storage_fetch_time_ms,
                    "selected_attention": (
                        approximate_layer.selected_attention_time_ms
                    ),
                    "remaining_layer_computation": (
                        approximate_layer.remaining_layer_time_ms
                    ),
                },
                "memory_bytes": {
                    "dense_kv": approximate_layer.dense_kv_bytes,
                    "quest_metadata": approximate_layer.quest_metadata_bytes,
                    "pq_codes_actual_int64": approximate_layer.pq_code_bytes,
                    "pq_codes_logical_packed": (
                        approximate_layer.pq_logical_code_bytes
                    ),
                    "pq_codebooks": approximate_layer.pq_codebook_bytes,
                    "selected_full_precision_kv": (approximate_layer.selected_kv_bytes),
                },
            }
        )
    return results


def run_approximate_path(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    dense_tokens: list[int],
    dense_steps: list[GPTNeoXDecodeStep],
    *,
    strategy: DecodeStrategy,
    budget_fraction: float,
    mode: DecodeMode,
    args: argparse.Namespace,
) -> dict[str, Any]:
    state = runner.initialize_state(
        snapshot,
        strategy=strategy,
        budget_fraction=budget_fraction,
        quest_page_size=args.quest_page_size,
        pq_num_subspaces=args.pq_subspaces,
        pq_num_centroids=args.pq_centroids,
        pq_max_iterations=args.pq_iterations,
        seed=args.seed,
        quest_metadata_update_mode=QuestMetadataUpdateMode.FULL_REBUILD,
    )
    path_tokens = [int(snapshot.next_token_logits.argmax(dim=-1).item())]
    records: list[dict[str, Any]] = []
    for generation_position, dense_step in enumerate(dense_steps, start=1):
        dense_previous = torch.tensor(
            [[dense_tokens[generation_position - 1]]],
            dtype=torch.int64,
            device=snapshot.input_ids.device,
        )
        path_previous = torch.tensor(
            [[path_tokens[-1]]],
            dtype=torch.int64,
            device=snapshot.input_ids.device,
        )
        input_token = select_decode_input(
            mode,
            dense_token=dense_previous,
            path_token=path_previous,
        )
        approximate_step = runner.step(state, input_token)
        if budget_fraction == 1.0:
            assert_full_budget_step(
                approximate_step,
                dense_step,
                rtol=args.full_budget_rtol,
                atol=args.full_budget_atol,
            )
        path_tokens.append(int(approximate_step.next_token.item()))
        compared_logits = logit_comparison_metrics(
            approximate_step.next_token_logits,
            dense_step.next_token_logits,
        )
        compared_layers = layer_metrics(approximate_step, dense_step)
        records.append(
            {
                "generation_position": generation_position,
                "cache_length": approximate_step.layers[0].sequence_length,
                "input_token": int(input_token.item()),
                "dense_input_token": int(dense_previous.item()),
                "path_input_token": int(path_previous.item()),
                **compared_logits,
                "timing_ms": {
                    "total_decode_step": approximate_step.total_time_ms,
                    "index_update_or_rebuild": sum(
                        layer["timing_ms"]["index_update_or_rebuild"]
                        for layer in compared_layers
                    ),
                    "retrieval_search_and_policy": sum(
                        layer["timing_ms"]["retrieval_search_and_policy"]
                        for layer in compared_layers
                    ),
                    "storage_fetch": sum(
                        layer["timing_ms"]["storage_fetch"] for layer in compared_layers
                    ),
                    "selected_attention": sum(
                        layer["timing_ms"]["selected_attention"]
                        for layer in compared_layers
                    ),
                    "remaining_model_computation": (
                        approximate_step.remaining_model_time_ms
                    ),
                },
                "memory_bytes": {
                    key: sum(layer["memory_bytes"][key] for layer in compared_layers)
                    for key in compared_layers[0]["memory_bytes"]
                },
                "layers": compared_layers,
            }
        )
    return {
        "strategy": strategy.value,
        "configuration": (
            f"p{args.quest_page_size}"
            if strategy is DecodeStrategy.QUEST
            else f"M{args.pq_subspaces}/C{args.pq_centroids}"
        ),
        "budget_fraction": budget_fraction,
        "mode": mode.value,
        "generated_token_ids": path_tokens,
        "generation_metrics": generation_divergence_metrics(
            dense_tokens,
            path_tokens,
        ),
        "index_update_policy": state.index_update_policy,
        "pq_codebook_policy": state.codebook_policy,
        "initial_index_build_time_ms_by_layer": [
            layer.initial_index_build_time_ms for layer in state.layers
        ],
        "initial_index_build_time_ms_total": sum(
            layer.initial_index_build_time_ms for layer in state.layers
        ),
        "steps": records,
    }


def dense_artifact_entry(
    logits: list[torch.Tensor],
    steps: list[GPTNeoXDecodeStep],
) -> dict[str, torch.Tensor]:
    return {
        "logits": torch.stack([tensor[0].detach().cpu() for tensor in logits]),
        "attention_outputs": torch.stack(
            [
                torch.stack(
                    [layer.attention_output[0].detach().cpu() for layer in step.layers]
                )
                for step in steps
            ]
        ),
        "residual_streams": torch.stack(
            [
                torch.stack(
                    [
                        layer.residual_output[0, 0].detach().cpu()
                        for layer in step.layers
                    ]
                )
                for step in steps
            ]
        ),
        "cache_lengths": torch.tensor(
            [steps[0].layers[0].sequence_length - 1]
            + [step.layers[0].sequence_length for step in steps],
            dtype=torch.int64,
        ),
    }


def _means(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def analyze(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    generation_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    layer_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    timing_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (
            run["mode"],
            run["strategy"],
            run["configuration"],
            run["budget_fraction"],
        )
        generation_grouped[key].append(run["generation_metrics"])
        for step in run["steps"]:
            grouped[key].append(step)
            timing_grouped[key].append(step["timing_ms"])
            for layer in step["layers"]:
                layer_grouped[(*key, layer["layer"])].append(layer)

    logit_summary = []
    generation_summary = []
    timing_summary = []
    for key, steps in sorted(grouped.items()):
        mode, strategy, configuration, budget = key
        logit_summary.append(
            {
                "mode": mode,
                "strategy": strategy,
                "configuration": configuration,
                "budget_fraction": budget,
                "decode_step_count": len(steps),
                "top_1_agreement_rate": _means(
                    float(step["top_1_agreement"]) for step in steps
                ),
                "mean_top_5_overlap_fraction": _means(
                    step["top_5_overlap_fraction"] for step in steps
                ),
                "mean_logit_cosine_similarity": _means(
                    step["logit_cosine_similarity"] for step in steps
                ),
                "mean_logit_relative_error": _means(
                    step["logit_relative_error"] for step in steps
                ),
                "mean_kl_divergence": _means(
                    step["kl_divergence_dense_to_approximate"] for step in steps
                ),
                "maximum_kl_divergence": max(
                    step["kl_divergence_dense_to_approximate"] for step in steps
                ),
                "mean_dense_top_1_rank": _means(
                    step["dense_top_1_rank_under_approximate_logits"] for step in steps
                ),
                "first_half_mean_logit_relative_error": _means(
                    step["logit_relative_error"]
                    for step in steps
                    if step["generation_position"] <= 16
                ),
                "second_half_mean_logit_relative_error": _means(
                    step["logit_relative_error"]
                    for step in steps
                    if step["generation_position"] > 16
                ),
            }
        )
        generation_rows = generation_grouped[key]
        distribution = Counter(
            "none"
            if row["first_divergence_position"] is None
            else str(row["first_divergence_position"])
            for row in generation_rows
        )
        generation_summary.append(
            {
                "mode": mode,
                "strategy": strategy,
                "configuration": configuration,
                "budget_fraction": budget,
                "run_count": len(generation_rows),
                "first_divergence_distribution": dict(sorted(distribution.items())),
                "mean_token_agreement_rate": _means(
                    row["token_agreement_rate"] for row in generation_rows
                ),
                "mean_longest_common_prefix_tokens": _means(
                    row["longest_common_prefix_tokens"] for row in generation_rows
                ),
                "reconvergence_rate_after_divergence": _means(
                    float(row["reconverged_after_first_divergence"])
                    for row in generation_rows
                    if row["first_divergence_position"] is not None
                ),
            }
        )
        timings = timing_grouped[key]
        timing_fields = timings[0]
        timing_summary.append(
            {
                "mode": mode,
                "strategy": strategy,
                "configuration": configuration,
                "budget_fraction": budget,
                **{
                    f"mean_{field}_ms": _means(row[field] for row in timings)
                    for field in timing_fields
                },
            }
        )

    layer_summary = []
    for key, layers in sorted(layer_grouped.items()):
        mode, strategy, configuration, budget, layer = key
        layer_summary.append(
            {
                "mode": mode,
                "strategy": strategy,
                "configuration": configuration,
                "budget_fraction": budget,
                "layer": layer,
                "mean_attention_mass_captured": _means(
                    row["mean_attention_mass_captured"] for row in layers
                ),
                "mean_attention_output_relative_error": _means(
                    row["mean_attention_output_relative_error"] for row in layers
                ),
                "mean_residual_stream_relative_error": _means(
                    row["residual_stream_relative_error"] for row in layers
                ),
            }
        )
    return {
        "logit_metrics": logit_summary,
        "generation_metrics": generation_summary,
        "layer_metrics": layer_summary,
        "timings": timing_summary,
    }


def run_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    fixtures = validate_args(args)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    except ImportError as error:
        raise RuntimeError(
            "install the optional model experiment dependency: "
            "pip install -e '.[model-experiment]'"
        ) from error
    if __version__ != DEFAULT_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {DEFAULT_TRANSFORMERS_VERSION}, found {__version__}"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    architecture = validate_gpt_neox_config(model.config)
    if max(args.sequence_lengths) + args.generated_tokens > (
        architecture.max_position_embeddings
    ):
        raise ValueError("prompt plus generated tokens exceeds model context limit")
    resolved_revision = getattr(model.config, "_commit_hash", args.model_revision)
    if resolved_revision != args.model_revision:
        raise RuntimeError("resolved model revision did not match the exact pin")
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    if attention_implementation != "eager":
        raise RuntimeError("Phase 3B requires eager attention")

    runner = GPTNeoXDecodeRunner(model)
    runs: list[dict[str, Any]] = []
    dense_cases: list[dict[str, Any]] = []
    dense_tensors: dict[str, Any] = {}
    total_cases = len(fixtures) * len(args.sequence_lengths)
    case_number = 0
    for fixture in fixtures:
        for sequence_length in args.sequence_lengths:
            case_number += 1
            tokenized = build_deterministic_fixture(
                tokenizer,
                fixture,
                sequence_length,
            )
            input_ids = tokenized.input_ids.to(device)
            snapshot = runner.dense_prefill(input_ids)
            dense_tokens, dense_logits, dense_steps = build_dense_trace(
                runner,
                snapshot,
                generated_tokens=args.generated_tokens,
            )
            hf_validation = (
                {"skipped": True}
                if args.skip_hf_validation
                else validate_hugging_face_generation(
                    model,
                    input_ids,
                    generated_tokens=args.generated_tokens,
                    custom_tokens=dense_tokens,
                    custom_logits=dense_logits,
                )
            )
            case_id = f"{fixture.fixture_id}-s{sequence_length}"
            dense_tensors[case_id] = dense_artifact_entry(
                dense_logits,
                dense_steps,
            )
            dense_cases.append(
                {
                    "case_id": case_id,
                    "fixture_id": fixture.fixture_id,
                    "fixture_structure": fixture.structure,
                    "fixture_text_sha256": text_sha256(fixture.text),
                    "prompt_length": sequence_length,
                    "base_token_count": tokenized.base_token_count,
                    "repetitions": tokenized.repetitions,
                    "token_ids_sha256": tokenized.token_ids_sha256,
                    "generated_tokens": args.generated_tokens,
                    "generated_token_ids": dense_tokens,
                    "generated_text": tokenizer.decode(dense_tokens),
                    "dense_prefill_time_ms": snapshot.prefill_time_ms,
                    "dense_decode_step_time_ms": [
                        step.total_time_ms for step in dense_steps
                    ],
                    "cache_lengths": [sequence_length]
                    + [step.layers[0].sequence_length for step in dense_steps],
                    "hugging_face_validation": hf_validation,
                }
            )
            print(
                f"case {case_number}/{total_cases} {case_id}: dense/HF gate passed",
                flush=True,
            )
            ordered_budgets = sorted(args.budget_fractions, reverse=True)
            for strategy_value in args.strategies:
                strategy = DecodeStrategy(strategy_value)
                for budget_fraction in ordered_budgets:
                    for mode_value in args.modes:
                        mode = DecodeMode(mode_value)
                        run = run_approximate_path(
                            runner,
                            snapshot,
                            dense_tokens,
                            dense_steps,
                            strategy=strategy,
                            budget_fraction=budget_fraction,
                            mode=mode,
                            args=args,
                        )
                        run.update(
                            {
                                "case_id": case_id,
                                "fixture_id": fixture.fixture_id,
                                "prompt_length": sequence_length,
                                "generated_text": tokenizer.decode(
                                    run["generated_token_ids"]
                                ),
                            }
                        )
                        runs.append(run)
                        print(
                            f"  {strategy.value} budget={budget_fraction:g} "
                            f"mode={mode.value} passed",
                            flush=True,
                        )

    artifact = {
        "schema_version": 1,
        "phase": "3B autoregressive decode validation",
        "status": "complete",
        "provenance": {
            "model_id": args.model_id,
            "model_revision": resolved_revision,
            "transformers_version": __version__,
            "transformers_source_revision": DEFAULT_TRANSFORMERS_REVISION,
            "transformers_attention_implementation": attention_implementation,
            "torch_version": torch.__version__,
            "dtype": "float32",
            "device": str(device),
            "hardware": hardware_name(device),
            "platform": platform.platform(),
            "git_commit": git_commit(),
            "git_dirty_before_result_write": git_is_dirty(),
            "seed": args.seed,
        },
        "architecture": asdict(architecture),
        "protocol": {
            "dense_prefill": True,
            "first_generated_token_source": "dense prefill final-position logits",
            "approximate_decode_steps_per_run": args.generated_tokens - 1,
            "prompt_lengths": args.sequence_lengths,
            "generated_tokens": args.generated_tokens,
            "fixture_ids": args.fixture_ids,
            "strategies": [DecodeStrategy.DENSE.value, *args.strategies],
            "budget_fractions": args.budget_fractions,
            "modes": args.modes,
            "quest_configuration": f"p{args.quest_page_size}",
            "pq_configuration": f"M{args.pq_subspaces}/C{args.pq_centroids}",
            "quest_update_policy": "rebuild page metadata after every KV append",
            "pq_update_policy": (
                "train codebooks on dense-prefill keys, freeze codebooks, and "
                "encode each appended key against them"
            ),
            "newest_token_policy": (
                "integration-level forced inclusion by replacement of the final "
                "ranked candidate when absent, followed by causal-order sorting"
            ),
            "newest_token_policy_classification": (
                "runtime/Quest-inspired integration policy; not a mathematical "
                "necessity and not part of either index ranking"
            ),
            "static_layer_head_policy": {
                "included": False,
                "reason": (
                    "mixing per-head Quest and PQ selections requires concurrent "
                    "indexes and heterogeneous fetch assembly, adding substantial "
                    "integration complexity beyond the A/B/C correctness gate"
                ),
            },
            "learned_adaptive_policy_included": False,
            "shared_interface_changes": {
                "KVIndex": False,
                "Selection": False,
                "KVStorage": False,
                "RetrievedKV": False,
                "KVCache": False,
            },
            "full_budget_tolerance": {
                "rtol": args.full_budget_rtol,
                "atol": args.full_budget_atol,
            },
        },
        "dense_cases": dense_cases,
        "runs": runs,
        "analysis": analyze(runs),
        "limitations": [
            "one 410M standard-MHA model with maximum context 2048",
            "deterministically repeated local fixtures rather than an external corpus",
            "reference PyTorch/Python timings are diagnostic costs, not speed claims",
            "dense prompt prefill only; approximate prefill was not evaluated",
            "no GQA/MQA, fused kernel, downstream-task, or perplexity claim",
            "the first generated token comes from dense prefill, so each 32-token "
            "run contains 31 approximate retrieval steps",
        ],
    }
    return artifact, dense_tensors


def main() -> None:
    args = parse_args()
    artifact, dense_tensors = run_experiment(args)
    report = prepare_report(artifact)

    def write_tensors(temporary: Path) -> None:
        # A stream avoids embedding the random temporary filename in the ZIP
        # archive, so identical tensor payloads keep reproducible serialization.
        with temporary.open("wb") as output_file:
            torch.save(dense_tensors, output_file)

    # Open the report destination first to reject invalid/symlink paths before
    # publishing a sidecar. Immutable sidecars keep old reports valid even if
    # final report publication fails or a concurrent run wins the report race.
    with atomic_output(args.output, overwrite=True) as temporary_report:
        sidecar, digest = write_content_addressed(
            args.dense_tensors_output, write_tensors
        )
        report["dense_tensor_artifact"] = {
            "path": str(sidecar),
            "sha256": digest,
            "contents": (
                "per-case dense logits, per-decode-step/per-layer attention outputs, "
                "residual streams, and cache lengths"
            ),
        }
        write_json(temporary_report, report, overwrite=True, sort_keys=False)
    print(f"wrote {args.output}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
