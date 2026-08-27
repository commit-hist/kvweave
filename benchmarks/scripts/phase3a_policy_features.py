#!/usr/bin/env python3
"""Collect leakage-safe pre-retrieval features for Phase 3A policy feasibility."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import torch

from benchmarks.phase3a import build_deterministic_fixture, calculate_query_positions
from benchmarks.policy_feasibility import (
    FEATURE_DEFINITIONS,
    FIXTURE_SPLITS,
    build_key_feature_state,
    extract_pre_retrieval_feature_rows,
    fixture_manifest,
    maintained_key_metadata_bytes,
    validate_fixture_lock,
    validate_tokenized_fixture_lock,
)
from kvdb.integrations.transformers import (
    capture_gpt_neox_activations,
    causal_slice,
    validate_gpt_neox_config,
    validate_layer_indices,
)


DEFAULT_MODEL_ID = "EleutherAI/pythia-410m"
DEFAULT_MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
DEFAULT_TRANSFORMERS_VERSION = "5.15.1"
DEFAULT_OUTPUTS = {
    "development": Path(
        "benchmarks/results/pythia-410m-phase3a-policy-development-features.json"
    ),
    "held_out": Path(
        "benchmarks/results/pythia-410m-phase3a-policy-held-out-features.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-split",
        choices=sorted(FIXTURE_SPLITS),
        default="development",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[512, 2048])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 12, 23])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--print-lock-material",
        action="store_true",
        help="print held-out content/token hashes without collecting model features",
    )
    return parser.parse_args()


def git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def load_transformers() -> tuple[Any, Any, str]:
    try:
        from transformers import AutoModel, AutoTokenizer, __version__
    except ImportError as error:
        raise RuntimeError(
            "install the optional model experiment dependency: "
            "pip install -e '.[model-experiment]'"
        ) from error
    if __version__ != DEFAULT_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {DEFAULT_TRANSFORMERS_VERSION}, found {__version__}"
        )
    return AutoModel, AutoTokenizer, __version__


def print_lock_material(args: argparse.Namespace) -> None:
    _, auto_tokenizer, version = load_transformers()
    tokenizer = auto_tokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
    )
    fixtures = FIXTURE_SPLITS["held_out"]
    token_hashes: dict[str, dict[int, str]] = {}
    for fixture in fixtures:
        token_hashes[fixture.fixture_id] = {}
        for sequence_length in sorted(args.sequence_lengths):
            tokenized = build_deterministic_fixture(
                tokenizer,
                fixture,
                sequence_length,
            )
            token_hashes[fixture.fixture_id][sequence_length] = (
                tokenized.token_ids_sha256
            )
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "transformers_version": version,
                "fixture_manifest": fixture_manifest(fixtures),
                "token_id_sha256": token_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )


def collect_features(args: argparse.Namespace) -> dict[str, Any]:
    validate_fixture_lock(args.fixture_split)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if any(length not in {512, 2048} for length in args.sequence_lengths):
        raise ValueError("policy feasibility is frozen to lengths 512 and 2048")
    if len(set(args.sequence_lengths)) != len(args.sequence_lengths):
        raise ValueError("sequence lengths must be unique")

    auto_model, auto_tokenizer, transformers_version = load_transformers()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = auto_tokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
    )
    model = auto_model.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    architecture = validate_gpt_neox_config(model.config)
    layers = validate_layer_indices(
        args.layers,
        num_hidden_layers=architecture.num_hidden_layers,
    )
    if tuple(layers) != (0, 12, 23):
        raise ValueError("policy feasibility is frozen to layers 0, 12, and 23")
    resolved_revision = getattr(model.config, "_commit_hash", args.model_revision)
    if resolved_revision != args.model_revision:
        raise RuntimeError("resolved model revision did not match the pinned revision")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("feature collection requires eager model attention")

    feature_rows: list[dict[str, Any]] = []
    tokenizations: list[dict[str, Any]] = []
    fixtures = FIXTURE_SPLITS[args.fixture_split]
    for fixture_index, fixture in enumerate(fixtures, start=1):
        for sequence_length in sorted(args.sequence_lengths):
            print(
                f"features split={args.fixture_split} fixture={fixture.fixture_id} "
                f"length={sequence_length} ({fixture_index}/{len(fixtures)})",
                flush=True,
            )
            tokenized = build_deterministic_fixture(
                tokenizer,
                fixture,
                sequence_length,
            )
            validate_tokenized_fixture_lock(
                args.fixture_split,
                fixture,
                sequence_length,
                tokenized,
            )
            tokenizations.append(
                {
                    "text_fixture_id": fixture.fixture_id,
                    "sequence_length": sequence_length,
                    "base_token_count": tokenized.base_token_count,
                    "repetitions_before_truncation": tokenized.repetitions,
                    "resulting_token_count": tokenized.input_ids.shape[1],
                    "token_ids_sha256": tokenized.token_ids_sha256,
                }
            )
            input_ids = tokenized.input_ids.to(device)
            capture = capture_gpt_neox_activations(
                model,
                input_ids,
                layer_indices=layers,
                attention_mask=torch.ones_like(input_ids, device=device),
                capture_device="cpu",
                capture_dtype=torch.float32,
            )
            for layer in layers:
                activations = capture.layers[layer]
                for position in calculate_query_positions(sequence_length):
                    sliced = causal_slice(activations, position.token_index)
                    metadata_start = time.perf_counter()
                    state = build_key_feature_state(sliced.keys)
                    metadata_seconds = time.perf_counter() - metadata_start
                    feature_start = time.perf_counter()
                    rows = extract_pre_retrieval_feature_rows(
                        sliced.query,
                        state,
                        text_fixture_id=fixture.fixture_id,
                        sequence_length=sequence_length,
                        query_position_label=position.label,
                        query_position=position.token_index,
                        layer_id=layer,
                    )
                    feature_seconds = time.perf_counter() - feature_start
                    for row in rows:
                        row["metadata_construction_seconds_batch"] = metadata_seconds
                        row["feature_extraction_seconds_batch"] = feature_seconds
                        row["batch_head_count"] = architecture.num_attention_heads
                        row["persistent_metadata_bytes_batch"] = (
                            maintained_key_metadata_bytes(
                                head_dimension=architecture.head_dimension,
                                num_heads=architecture.num_attention_heads,
                            )
                        )
                    feature_rows.extend(rows)
            del capture

    expected_count = (
        len(fixtures)
        * len(args.sequence_lengths)
        * 4
        * len(layers)
        * architecture.num_attention_heads
    )
    if len(feature_rows) != expected_count:
        raise AssertionError("pre-retrieval feature matrix is incomplete")
    return {
        "schema_version": 1,
        "artifact": "pythia_410m_phase3a_pre_retrieval_features",
        "fixture_split": args.fixture_split,
        "scope": "pre-retrieval features only; contains no retrieval or attention labels",
        "provenance": {
            "model_id": args.model_id,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": resolved_revision,
            "transformers_version": transformers_version,
            "transformers_source_revision": (
                "550d7b3834670483a4df436541272c055dc364bf"
            ),
            "transformers_attention_implementation": "eager",
            "torch_version": torch.__version__,
            "model_dtype": "float32",
            "capture_dtype": "float32",
            "device": str(device),
            "hardware": platform.platform(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "seed": args.seed,
        },
        "architecture": asdict(architecture),
        "fixture_manifest": fixture_manifest(fixtures),
        "tokenizations": tokenizations,
        "configuration": {
            "sequence_lengths": sorted(args.sequence_lengths),
            "query_positions": ["25_percent", "50_percent", "75_percent", "final"],
            "layers": list(layers),
            "heads": list(range(architecture.num_attention_heads)),
        },
        "feature_definitions": [
            asdict(definition) for definition in FEATURE_DEFINITIONS
        ],
        "cost_semantics": {
            "metadata_construction_seconds_batch": (
                "observed O(H*S*D) reconstruction of incrementally maintainable state; "
                "not an inference-time requirement"
            ),
            "feature_extraction_seconds_batch": (
                "observed batched query-time extraction from maintained state for all heads"
            ),
            "persistent_metadata_bytes_batch": (
                "D+4 float32 values plus one int64 count per head"
            ),
        },
        "feature_row_count": len(feature_rows),
        "feature_rows": feature_rows,
    }


def main() -> None:
    args = parse_args()
    if args.print_lock_material:
        print_lock_material(args)
        return
    output = args.output or DEFAULT_OUTPUTS[args.fixture_split]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature artifact: {output}")
    result = collect_features(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    print(f"feature_rows={result['feature_row_count']}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
