#!/usr/bin/env python3
"""Run the pinned Pythia Phase 3B stateful autoregressive decode experiment."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any, Iterable

import torch

from benchmarks.phase3a import (
    TEXT_FIXTURES,
    TextFixture,
    build_deterministic_fixture,
)
from kvdb.integrations.transformers import (
    DecodeMode,
    DecodeStrategy,
    DensePrefillSnapshot,
    GPTNeoXDecodeRunner,
    GPTNeoXDecodeStep,
    attention_mass_captured,
    generation_divergence_metrics,
    logit_comparison_metrics,
    per_head_relative_error,
    relative_tensor_error,
    select_decode_input,
    validate_gpt_neox_config,
)


DEFAULT_MODEL_ID = "EleutherAI/pythia-410m"
DEFAULT_MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
DEFAULT_TRANSFORMERS_VERSION = "5.15.1"
DEFAULT_TRANSFORMERS_REVISION = "550d7b3834670483a4df436541272c055dc364bf"
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
    return None if completed.returncode != 0 else bool(completed.stdout.strip())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"{platform.machine()} Apple MPS"
    return platform.processor() or platform.machine() or "unknown"


def build_dense_trace(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    *,
    generated_tokens: int,
) -> tuple[list[int], list[torch.Tensor], list[GPTNeoXDecodeStep]]:
    state = runner.initialize_state(snapshot, strategy=DecodeStrategy.DENSE)
    generated = [int(snapshot.next_token_logits.argmax(dim=-1).item())]
    logits = [snapshot.next_token_logits]
    steps: list[GPTNeoXDecodeStep] = []
    for _ in range(1, generated_tokens):
        input_token = torch.tensor(
            [[generated[-1]]],
            dtype=torch.int64,
            device=snapshot.input_ids.device,
        )
        step = runner.step(state, input_token)
        steps.append(step)
        logits.append(step.next_token_logits)
        generated.append(int(step.next_token.item()))
    return generated, logits, steps


def validate_hugging_face_generation(
    model: Any,
    input_ids: torch.Tensor,
    *,
    generated_tokens: int,
    custom_tokens: list[int],
    custom_logits: list[torch.Tensor],
) -> dict[str, Any]:
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=generated_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    hugging_face_tokens = generated.sequences[0, -generated_tokens:].tolist()
    if len(hugging_face_tokens) != generated_tokens:
        raise RuntimeError("Hugging Face generation stopped before requested length")
    if hugging_face_tokens != custom_tokens:
        first_difference = next(
            index
            for index, (dense, reference) in enumerate(
                zip(custom_tokens, hugging_face_tokens, strict=True)
            )
            if dense != reference
        )
        raise RuntimeError(
            "custom dense decode diverged from Hugging Face at generated position "
            f"{first_difference}: custom={custom_tokens[first_difference]}, "
            f"hf={hugging_face_tokens[first_difference]}"
        )
    if len(generated.scores) != len(custom_logits):
        raise RuntimeError("Hugging Face did not return one score tensor per token")
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for custom, reference in zip(custom_logits, generated.scores, strict=True):
        torch.testing.assert_close(custom, reference, rtol=1e-4, atol=1e-5)
        difference = torch.linalg.vector_norm(custom.float() - reference.float())
        denominator = torch.linalg.vector_norm(reference.float())
        maximum_absolute_error = max(
            maximum_absolute_error,
            float((custom.float() - reference.float()).abs().max().item()),
        )
        maximum_relative_error = max(
            maximum_relative_error,
            0.0
            if denominator.item() == 0 and difference.item() == 0
            else float((difference / denominator).item()),
        )
    return {
        "passed": True,
        "generated_tokens": generated_tokens,
        "token_sequence_exact_match": True,
        "logit_rtol": 1e-4,
        "logit_atol": 1e-5,
        "maximum_logit_absolute_error": maximum_absolute_error,
        "maximum_logit_relative_error": maximum_relative_error,
    }


def assert_full_budget_step(
    approximate: GPTNeoXDecodeStep,
    dense: GPTNeoXDecodeStep,
    *,
    rtol: float,
    atol: float,
) -> None:
    torch.testing.assert_close(
        approximate.next_token_logits,
        dense.next_token_logits,
        rtol=rtol,
        atol=atol,
    )
    if not torch.equal(approximate.next_token, dense.next_token):
        raise AssertionError("100% retrieval changed the greedy next token")
    for approximate_layer, dense_layer in zip(
        approximate.layers,
        dense.layers,
        strict=True,
    ):
        if approximate_layer.selection is None:
            raise AssertionError("100% approximate path did not expose a selection")
        if not approximate_layer.newest_token_included:
            raise AssertionError("100% selection omitted the newest token")
        expected = torch.arange(
            approximate_layer.sequence_length,
            device=approximate_layer.selection.indices.device,
        )
        valid_mask = approximate_layer.selection.valid_mask
        if valid_mask is None:
            valid_mask = torch.ones_like(
                approximate_layer.selection.indices,
                dtype=torch.bool,
            )
        for head_index in range(valid_mask.shape[1]):
            actual = approximate_layer.selection.indices[0, head_index][
                valid_mask[0, head_index]
            ]
            if not torch.equal(actual, expected):
                raise AssertionError("100% selection did not contain causal KV exactly")
        torch.testing.assert_close(
            approximate_layer.attention_output,
            dense_layer.attention_output,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            approximate_layer.residual_output,
            dense_layer.residual_output,
            rtol=rtol,
            atol=atol,
        )


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
    args.dense_tensors_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dense_tensors, args.dense_tensors_output)
    artifact["dense_tensor_artifact"] = {
        "path": str(args.dense_tensors_output),
        "sha256": file_sha256(args.dense_tensors_output),
        "contents": (
            "per-case dense logits, per-decode-step/per-layer attention outputs, "
            "residual streams, and cache lengths"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.dense_tensors_output}")


if __name__ == "__main__":
    main()
