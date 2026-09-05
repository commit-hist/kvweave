"""Offline support for the Phase 3A policy-feasibility experiment.

This module is research/benchmark code.  It deliberately does not define a
public policy, planner, router, or adaptive-index abstraction in ``kvweave``.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as torch_functional

from benchmarks.statistics import policy_distribution as distribution, percentile
from benchmarks.phase3a import TEXT_FIXTURES, TextFixture, TokenizedFixture


DEVELOPMENT_FIXTURES = TEXT_FIXTURES
HELD_OUT_FIXTURES = (
    TextFixture(
        fixture_id="contract_cross_references",
        structure=(
            "Legalistic clauses with defined terms, exceptions, and cross-references."
        ),
        text=(
            "Section 2.1 defines the Delivery Window as five business days after "
            "Notice. Subject to Section 4(b), the Supplier shall inspect each sealed "
            "crate; provided, however, that a documented Force Event suspends the "
            "Window only for the period stated in Exhibit C. Clause 7 survives "
            "termination notwithstanding any contrary schedule. "
        ),
    ),
    TextFixture(
        fixture_id="lab_notebook",
        structure=(
            "Timestamped experimental notes with measurements, interventions, and "
            "revisions."
        ),
        text=(
            "08:10 sample R4: temperature 19.6 C, pressure 100.8 kPa, color clear.\n"
            "08:35 added 2.0 mL buffer; the sensor drifted +0.07 before calibration.\n"
            "09:05 repeat R4b: temperature 21.1 C, pressure 100.5 kPa, color amber.\n"
            "09:40 note: reject channel three; retain the raw trace and calibration ID.\n"
        ),
    ),
    TextFixture(
        fixture_id="nested_configuration",
        structure=(
            "Nested JSON-like configuration with repeated keys, arrays, booleans, "
            "and overrides."
        ),
        text=(
            '{"service":{"name":"atlas","retries":3,"regions":["west","north"],'
            '"limits":{"soft":128,"hard":256}},"flags":{"audit":true,'
            '"shadow":false},"override":{"region":"north","retries":1}}\n'
        ),
    ),
    TextFixture(
        fixture_id="operations_log",
        structure=(
            "Shell and service log transcript with paths, status codes, and retries."
        ),
        text=(
            "$ sync-cache --shard 07 --checkpoint /var/tmp/cp-19\n"
            "[14:03:11] scan=ok blocks=184 stale=6\n"
            "[14:03:12] upload shard=07 attempt=1 status=503\n"
            "[14:03:14] upload shard=07 attempt=2 status=200 bytes=917504\n"
            "$ verify-cache --shard 07 ; result=clean digest=9f2a\n"
        ),
    ),
    TextFixture(
        fixture_id="poetic_stanzas",
        structure=(
            "Line-broken verse with recurring imagery, internal rhyme, and refrain."
        ),
        text=(
            "Under the copper moon the tide withdrew,\n"
            "leaving a silver ladder on the sand.\n"
            "A bell in the fog rang once, then twice;\n"
            "remember the harbor, remember the light.\n"
            "By dawn the ladder vanished from the shore,\n"
            "yet salt kept the shape of every rung.\n"
        ),
    ),
    TextFixture(
        fixture_id="mathematical_derivation",
        structure=(
            "Stepwise symbolic derivation with assumptions, equations, and a boundary "
            "case."
        ),
        text=(
            "Assume n >= 2 and let a_0 = 1. From a_(k+1) = r a_k + c, substitute "
            "twice to obtain a_2 = r^2 + rc + c. Therefore a_n = r^n + "
            "c(1-r^n)/(1-r) when r != 1. For r = 1, take the boundary case "
            "a_n = 1 + nc and verify it by induction. "
        ),
    ),
    TextFixture(
        fixture_id="bilingual_glossary",
        structure=(
            "Paired bilingual glossary entries with grammatical tags and usage notes."
        ),
        text=(
            "luz [es, noun] = light; plural luces; usage: luz tenue.\n"
            "puente [es, noun] = bridge; plural puentes; usage: cruzar el puente.\n"
            "ouvrir [fr, verb] = to open; past participle ouvert.\n"
            "rivage [fr, noun] = shore; note: distinct from rive in this context.\n"
        ),
    ),
    TextFixture(
        fixture_id="incident_bulletins",
        structure=(
            "Numbered news-style bulletins with corrections, locations, and status "
            "transitions."
        ),
        text=(
            "Bulletin 17 — North Junction: the 06:20 departure is delayed 14 minutes. "
            "Bulletin 18 — East Market: platform four reopened after inspection. "
            "Correction to Bulletin 17: the delay is 11 minutes, not 14. "
            "Bulletin 19 — North Junction: service resumed; passengers should retain "
            "tickets issued before 07:00. "
        ),
    ),
)

FIXTURE_SPLITS: dict[str, tuple[TextFixture, ...]] = {
    "development": DEVELOPMENT_FIXTURES,
    "held_out": HELD_OUT_FIXTURES,
}

# These hashes freeze authored UTF-8 fixture content independently of a tokenizer.
# Values are populated once before any held-out retrieval evaluation.
LOCKED_FIXTURE_TEXT_SHA256: dict[str, str] = {
    "contract_cross_references": (
        "997201f8c0f7ac784916d16e0a73c4db53f7c589609eb67bffa24507faf7441f"
    ),
    "lab_notebook": (
        "33ccb08da60f8883384492c01f2dadf573bbd84bc3cb00dd7a1ff715bc26946a"
    ),
    "nested_configuration": (
        "ff5be0b548073edecf576e5900bd2893384f5e251b82bcdfd7617a5d766ffbb5"
    ),
    "operations_log": (
        "ed7f66e129c4f4b058f9ec1d66e456a674d97b34b5e002b4108d92f92511a548"
    ),
    "poetic_stanzas": (
        "38f4921d0df36e92fc891ec971ce395b877a9bf2e576aa703c350334fb4b06f1"
    ),
    "mathematical_derivation": (
        "cadd959690282a1fb55ba547d88c541d856b470e2911c13445843fbe99aed2f7"
    ),
    "bilingual_glossary": (
        "1ae634ecbbc9f1a51c57a732d84865ffaf93cef7398ae008a4a896108ab758e5"
    ),
    "incident_bulletins": (
        "868107d7dd006127694a10d7992352b82ee3bc470705516eda99067e06614a01"
    ),
}

# These hashes freeze the exact pinned-tokenizer IDs at the two accepted lengths.
# Values are populated once before any held-out retrieval evaluation.
LOCKED_TOKEN_ID_SHA256: dict[str, dict[int, str]] = {
    "contract_cross_references": {
        512: "24f729c3de328a8eefa7d7d5b306d370b9c7e62f7aea5098d97cd001b3de5a2a",
        2048: "0a4180e77d4f766b917e33254ca591f4f2078f832fccf2e248a2b379cb0f5034",
    },
    "lab_notebook": {
        512: "b13ecb2e7dc2169b4b7a64efdca5f14a2a1b79dc2bc9a04db4d1664bd038ffd4",
        2048: "d171d9ad7ba50f2ca9f0b03a472940e4997b2856d684ea9f6209e6a1e53f5b78",
    },
    "nested_configuration": {
        512: "f265fb20b213277f9562a5a05dd0535dcd1a422f7bacb0f1098b90d4213b01cc",
        2048: "f9d30bfa480234b0e22061cfa0e1f9ff7d6542cee3eaa3c5eee3af7d8a880789",
    },
    "operations_log": {
        512: "330296e6912b60418d535c0118cf56817d2fe0d8d77cd5a0a607db8bf7e88494",
        2048: "cedd7d42c6a1da6788e05e96b0756ff52f25da2f92b0b6727e127ef150d304d0",
    },
    "poetic_stanzas": {
        512: "87efbf332765496aadd38f8a1f8447a8030942d3094a78369e76dec2a1738590",
        2048: "e19f9d80319586978a400fca3ae196dea9fa218ede628bb0ae4d960c90c08321",
    },
    "mathematical_derivation": {
        512: "893f16b5802a1f9b2af4d2c9ddc6767c9f73e7650e0bc48b90a765b77232a693",
        2048: "9ca10f6d6e479bf942a013882a8b08296ed1837a209eaa3508a32a6716f633ed",
    },
    "bilingual_glossary": {
        512: "263b6f04db2fdb1b9d412ef022a7cf8888ea85c02f5e7a01438f44215c5f4e71",
        2048: "fdbbbba38a2ead76ac23b66f1bc7bf656138000b24ed011151c5b409dffddb9a",
    },
    "incident_bulletins": {
        512: "069a36a004560ab6199e8f608b82b8204dc91ea455dcd417c2d75be2d320833e",
        2048: "92c5dd5c433926f33d0cd973afa9201cdfe5335a80ad1b25c116dd66cbcb15d4",
    },
}

DEVELOPMENT_TOKEN_ID_SHA256: dict[str, dict[int, str]] = {
    "repetitive_prose": {
        512: "658e6689ef37d763c66102870376385bf105c30e1c608c7d86fa263fb72529e1",
        2048: "98412d0bc89cd39136b581eacf7a02a57f711c4b4b7407a1f395a22d1375bcdb",
    },
    "narrative_prose": {
        512: "686c0d89f3f9bec74ae3c548faf3fa2c0b313ca8924c9f0c2416a11f6c97b12f",
        2048: "720056b5d4fa562d4042525482363751c9c0b4b9998be06d83c86dfb10de6cc4",
    },
    "technical_exposition": {
        512: "6cb41a7b11378f53d95194813ced063fbd550a95866b1ee9c10ca2375b89469c",
        2048: "a98b382aeda74271f776a1d8d4f5fa4e5c974a63382c21bb0a33c9273d71aed3",
    },
    "code_like": {
        512: "b9b95960afe007960454bf1a78665d4d9332564517464bce78b155226b997722",
        2048: "53c9270f4bd15582d2bfa41ec56bc1695046a179716ea8fe7b27e5b864e4ff01",
    },
    "list_table": {
        512: "7f8e21cf06b63bda989354f4c74457ee461fcbe8f1ec476e70d413f98cd12d3c",
        2048: "8272d9e74860a45e6e300f630661dc2455d17f36cf2f7a95f6eeb00f3db5e9ca",
    },
    "dialogue_qa": {
        512: "2b2ecb74ea7dc9d041be0f0ed7368bb4d6539ab6d6c49f56fc72bd6f93810b88",
        2048: "0098661f312dcbdfcf333ffc91170f90c81727292ce2143e77c2f14e090d2729",
    },
    "mixed_sentence_lengths": {
        512: "22441d6cad39fc364376c5cae2ec13e44c105be6862ee2d3e8edf3a4f22110a1",
        2048: "64c342b5772a33ec1e42fd9ba221a99251de2a8b1080283ce86f6474c519a88a",
    },
    "symbolic_pattern": {
        512: "d16064facf4c0259d61810d0ba548ec2dc7c19fd676b619ec2e9f6bc5ba6ef60",
        2048: "fc1aeb5421ff5d149a20991044f9adcfd9923c8791405095055b14d74d9513e5",
    },
}


CANDIDATE_CONFIGURATIONS = (
    "quest:page_size=16",
    "quest:page_size=64",
    "pq:M=2,C=4,iterations=8",
    "pq:M=4,C=8,iterations=8",
)
PARTIAL_BUDGETS = (0.125, 0.25, 0.5)
OBSERVATION_KEY_FIELDS = (
    "text_fixture_id",
    "sequence_length",
    "query_position_label",
    "layer",
    "head",
)

# Outcome-only fields are denied at the feature boundary.  The feature
# extractor has no argument through which any of them can enter.
FORBIDDEN_INFERENCE_FEATURES = frozenset(
    {
        "exact_qk_scores",
        "exact_topk",
        "full_attention_weights",
        "attention_entropy_nats",
        "normalized_attention_entropy",
        "top_1_attention_mass",
        "top_4_attention_mass",
        "top_16_attention_mass",
        "quest_selection",
        "pq_selection",
        "candidate_recall",
        "attention_mass_captured",
        "relative_attention_output_error",
        "mass_oracle_configuration",
        "error_oracle_configuration",
    }
)


@dataclass(frozen=True)
class FeatureDefinition:
    """One frozen legal inference feature and its cost/provenance contract."""

    name: str
    available: str
    computational_complexity: str
    persistent_metadata: str
    approximate_storage: str
    strategy_specific: bool


FEATURE_DEFINITIONS = (
    FeatureDefinition(
        "layer_id",
        "model execution schedule before retrieval",
        "O(1)",
        "none",
        "0 bytes",
        False,
    ),
    FeatureDefinition(
        "head_id",
        "model execution schedule before retrieval",
        "O(1)",
        "none",
        "0 bytes",
        False,
    ),
    FeatureDefinition(
        "causal_context_length",
        "KV-cache cursor before retrieval",
        "O(1)",
        "token count",
        "8 bytes per head (shared count may reduce this)",
        False,
    ),
    FeatureDefinition(
        "normalized_query_position",
        "query position and captured sequence length before retrieval",
        "O(1)",
        "none beyond token count",
        "0 additional bytes",
        False,
    ),
    FeatureDefinition(
        "query_l2_norm",
        "when the post-RoPE query vector is produced",
        "O(D)",
        "none",
        "0 bytes",
        False,
    ),
    FeatureDefinition(
        "query_mean",
        "when the post-RoPE query vector is produced",
        "O(D)",
        "none",
        "0 bytes",
        False,
    ),
    FeatureDefinition(
        "query_standard_deviation",
        "when the post-RoPE query vector is produced",
        "O(D)",
        "none",
        "0 bytes",
        False,
    ),
    FeatureDefinition(
        "query_max_absolute_value",
        "when the post-RoPE query vector is produced",
        "O(D)",
        "none",
        "0 bytes",
        False,
    ),
    FeatureDefinition(
        "query_positive_fraction",
        "when the post-RoPE query vector is produced",
        "O(D)",
        "none",
        "0 bytes",
        False,
    ),
    FeatureDefinition(
        "key_scalar_mean",
        "from incrementally maintainable prefix statistics before retrieval",
        "O(1) query-time; O(D) per appended key",
        "key vector sum and token count",
        "included in D+4 float32 values plus one int64 count per head",
        False,
    ),
    FeatureDefinition(
        "key_scalar_standard_deviation",
        "from incrementally maintainable prefix statistics before retrieval",
        "O(1) query-time; O(D) per appended key",
        "key vector sum, element-square sum, and token count",
        "included in D+4 float32 values plus one int64 count per head",
        False,
    ),
    FeatureDefinition(
        "key_l2_norm_mean",
        "from incrementally maintainable prefix statistics before retrieval",
        "O(1) query-time; O(D) per appended key",
        "sum of per-key L2 norms",
        "included in D+4 float32 values plus one int64 count per head",
        False,
    ),
    FeatureDefinition(
        "key_l2_norm_standard_deviation",
        "from incrementally maintainable prefix statistics before retrieval",
        "O(1) query-time; O(D) per appended key",
        "sum and square-sum of per-key L2 norms",
        "included in D+4 float32 values plus one int64 count per head",
        False,
    ),
    FeatureDefinition(
        "key_max_absolute_value",
        "from incrementally maintainable prefix statistics before retrieval",
        "O(1) query-time; O(D) per appended key",
        "running absolute maximum",
        "included in D+4 float32 values plus one int64 count per head",
        False,
    ),
    FeatureDefinition(
        "mean_key_vector_l2_norm",
        "from incrementally maintainable prefix statistics before retrieval",
        "O(D) query-time; O(D) per appended key",
        "key vector sum",
        "D float32 values per head",
        False,
    ),
    FeatureDefinition(
        "query_mean_key_cosine",
        "after the query exists, using the maintained mean-key vector",
        "O(D)",
        "key vector sum",
        "D float32 values per head",
        False,
    ),
)

FEATURE_NAMES = tuple(definition.name for definition in FEATURE_DEFINITIONS)
NUMERIC_MODEL_FEATURES = tuple(
    name for name in FEATURE_NAMES if name not in {"layer_id", "head_id"}
)
MODEL_LAYERS = (0, 12, 23)
MODEL_HEADS = tuple(range(16))


@dataclass(frozen=True)
class KeyFeatureState:
    """Incrementally maintainable strategy-independent prefix metadata."""

    vector_sum: torch.Tensor
    element_square_sum: torch.Tensor
    norm_sum: torch.Tensor
    norm_square_sum: torch.Tensor
    maximum_absolute_value: torch.Tensor
    token_count: int


def fixture_text_sha256(fixture: TextFixture) -> str:
    return hashlib.sha256(fixture.text.encode("utf-8")).hexdigest()


def fixture_manifest(fixtures: Sequence[TextFixture]) -> list[dict[str, str]]:
    return [
        {
            "text_fixture_id": fixture.fixture_id,
            "structure": fixture.structure,
            "text_sha256": fixture_text_sha256(fixture),
        }
        for fixture in fixtures
    ]


def validate_fixture_lock(split: str) -> None:
    """Reject fixture edits after the checked-in content lock is established."""
    if split not in FIXTURE_SPLITS:
        raise ValueError(f"unknown fixture split {split!r}")
    fixtures = FIXTURE_SPLITS[split]
    expected_ids = {fixture.fixture_id for fixture in fixtures}
    if split == "held_out":
        if set(LOCKED_FIXTURE_TEXT_SHA256) != expected_ids:
            raise RuntimeError("held-out fixture text hash lock is incomplete")
        for fixture in fixtures:
            actual = fixture_text_sha256(fixture)
            expected = LOCKED_FIXTURE_TEXT_SHA256[fixture.fixture_id]
            if actual != expected:
                raise RuntimeError(
                    f"held-out fixture {fixture.fixture_id!r} changed after locking"
                )


def validate_tokenized_fixture_lock(
    split: str,
    fixture: TextFixture,
    sequence_length: int,
    tokenized: TokenizedFixture,
) -> None:
    expected_by_split = (
        DEVELOPMENT_TOKEN_ID_SHA256
        if split == "development"
        else LOCKED_TOKEN_ID_SHA256
    )
    try:
        expected = expected_by_split[fixture.fixture_id][sequence_length]
    except KeyError as error:
        raise RuntimeError(
            f"missing locked token hash for {split} fixture={fixture.fixture_id} "
            f"length={sequence_length}"
        ) from error
    if tokenized.token_ids_sha256 != expected:
        raise RuntimeError(
            f"token hash mismatch for {split} fixture={fixture.fixture_id} "
            f"length={sequence_length}: expected {expected}, got "
            f"{tokenized.token_ids_sha256}"
        )


def build_key_feature_state(keys: torch.Tensor) -> KeyFeatureState:
    """Build the maintained-statistics equivalent for keys ``[B,H,S,D]``."""
    if not isinstance(keys, torch.Tensor) or keys.ndim != 4:
        raise ValueError("keys must be a tensor with shape [B, H, S, D]")
    if not torch.is_floating_point(keys) or not torch.isfinite(keys).all().item():
        raise ValueError("keys must contain finite floating-point values")
    if any(size <= 0 for size in keys.shape):
        raise ValueError("keys dimensions must be positive")
    key_norms = torch.linalg.vector_norm(keys, dim=-1)
    return KeyFeatureState(
        vector_sum=keys.sum(dim=-2),
        element_square_sum=keys.square().sum(dim=(-2, -1)),
        norm_sum=key_norms.sum(dim=-1),
        norm_square_sum=key_norms.square().sum(dim=-1),
        maximum_absolute_value=keys.abs().amax(dim=(-2, -1)),
        token_count=keys.shape[-2],
    )


def maintained_key_metadata_bytes(
    *,
    head_dimension: int,
    num_heads: int = 1,
    float_bytes: int = 4,
) -> int:
    """Return D+4 float scalars and one int64 count for each head."""
    if head_dimension <= 0 or num_heads <= 0 or float_bytes <= 0:
        raise ValueError("metadata dimensions and scalar width must be positive")
    return num_heads * ((head_dimension + 4) * float_bytes + 8)


def extract_pre_retrieval_feature_rows(
    query: torch.Tensor,
    state: KeyFeatureState,
    *,
    text_fixture_id: str,
    sequence_length: int,
    query_position_label: str,
    query_position: int,
    layer_id: int,
) -> list[dict[str, Any]]:
    """Extract only frozen, pre-retrieval features for query ``[B,H,D]``.

    Exact token scores, attention, retrieval selections, and outcome labels are
    unavailable to this function by construction.
    """
    if not isinstance(query, torch.Tensor) or query.ndim != 3:
        raise ValueError("query must be a tensor with shape [B, H, D]")
    if query.shape[:2] != state.vector_sum.shape[:2]:
        raise ValueError("query batch/head dimensions must match key metadata")
    if query.shape[-1] != state.vector_sum.shape[-1]:
        raise ValueError("query dimension must match key metadata")
    if state.token_count != query_position + 1:
        raise ValueError("key metadata must cover the inclusive causal prefix")
    if sequence_length <= 0 or not 0 <= query_position < sequence_length:
        raise ValueError("query position must be inside the captured sequence")

    query_norm = torch.linalg.vector_norm(query, dim=-1)
    query_mean = query.mean(dim=-1)
    query_std = query.std(dim=-1, correction=0)
    query_max_abs = query.abs().amax(dim=-1)
    query_positive_fraction = (query > 0).to(query.dtype).mean(dim=-1)

    token_count = state.token_count
    dimension = query.shape[-1]
    mean_key_vector = state.vector_sum / token_count
    key_scalar_mean = state.vector_sum.sum(dim=-1) / (token_count * dimension)
    key_scalar_second_moment = state.element_square_sum / (token_count * dimension)
    key_scalar_variance = torch.clamp(
        key_scalar_second_moment - key_scalar_mean.square(),
        min=0,
    )
    key_norm_mean = state.norm_sum / token_count
    key_norm_variance = torch.clamp(
        state.norm_square_sum / token_count - key_norm_mean.square(),
        min=0,
    )
    mean_key_norm = torch.linalg.vector_norm(mean_key_vector, dim=-1)
    cosine_denominator = query_norm * mean_key_norm
    query_mean_key_cosine = torch.where(
        cosine_denominator > 0,
        (query * mean_key_vector).sum(dim=-1) / cosine_denominator,
        torch.zeros_like(cosine_denominator),
    )

    tensors = {
        "query_l2_norm": query_norm,
        "query_mean": query_mean,
        "query_standard_deviation": query_std,
        "query_max_absolute_value": query_max_abs,
        "query_positive_fraction": query_positive_fraction,
        "key_scalar_mean": key_scalar_mean,
        "key_scalar_standard_deviation": torch.sqrt(key_scalar_variance),
        "key_l2_norm_mean": key_norm_mean,
        "key_l2_norm_standard_deviation": torch.sqrt(key_norm_variance),
        "key_max_absolute_value": state.maximum_absolute_value,
        "mean_key_vector_l2_norm": mean_key_norm,
        "query_mean_key_cosine": query_mean_key_cosine,
    }
    rows: list[dict[str, Any]] = []
    for batch_index in range(query.shape[0]):
        if query.shape[0] != 1:
            raise ValueError("the Phase 3A experiment requires batch size one")
        for head_index in range(query.shape[1]):
            row: dict[str, Any] = {
                "text_fixture_id": text_fixture_id,
                "sequence_length": sequence_length,
                "query_position_label": query_position_label,
                "query_position": query_position,
                "layer": layer_id,
                "head": head_index,
                "layer_id": float(layer_id),
                "head_id": float(head_index),
                "causal_context_length": float(token_count),
                "normalized_query_position": float(
                    (query_position + 1) / sequence_length
                ),
            }
            for name, values in tensors.items():
                value = float(values[batch_index, head_index].item())
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite pre-retrieval feature {name}")
                row[name] = value
            rows.append(row)
    return rows


def audit_feature_schema(rows: Iterable[Mapping[str, Any]]) -> None:
    """Reject label leakage and any unregistered numeric model feature."""
    required = set(OBSERVATION_KEY_FIELDS) | {"query_position"} | set(FEATURE_NAMES)
    for row in rows:
        leaked = FORBIDDEN_INFERENCE_FEATURES.intersection(row)
        if leaked:
            raise ValueError(f"forbidden inference features present: {sorted(leaked)}")
        missing = required - set(row)
        if missing:
            raise ValueError(f"feature row is missing fields: {sorted(missing)}")


def candidate_id(strategy: str, configuration: str) -> str:
    value = f"{strategy}:{configuration}"
    if value not in CANDIDATE_CONFIGURATIONS:
        raise ValueError(f"unexpected candidate configuration {value!r}")
    return value


def observation_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in OBSERVATION_KEY_FIELDS)


def assemble_policy_examples(
    feature_rows: Sequence[Mapping[str, Any]],
    retrieval_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join legal feature rows to post-hoc labels without crossing the boundary."""
    audit_feature_schema(feature_rows)
    feature_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in feature_rows:
        key = observation_key(source)
        if key in feature_by_key:
            raise ValueError(f"duplicate pre-retrieval feature row for {key!r}")
        feature_by_key[key] = {
            field: source[field]
            for field in (
                *OBSERVATION_KEY_FIELDS,
                "query_position",
                *FEATURE_NAMES,
            )
        }

    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in retrieval_records:
        if record.get("strategy") not in {"quest", "pq"}:
            continue
        budget = float(record["budget_fraction"])
        if budget not in PARTIAL_BUDGETS:
            continue
        key = (*observation_key(record), budget)
        configuration = candidate_id(
            str(record["strategy"]),
            str(record["configuration"]),
        )
        if configuration in grouped[key]:
            raise ValueError(f"duplicate candidate outcome for {key!r}")
        grouped[key][configuration] = record

    expected_example_count = len(feature_by_key) * len(PARTIAL_BUDGETS)
    if len(grouped) != expected_example_count:
        raise ValueError(
            "retrieval outcomes do not cover every feature observation and budget"
        )
    examples: list[dict[str, Any]] = []
    for joined_key, candidate_records in sorted(grouped.items(), key=repr):
        key = joined_key[:-1]
        budget = float(joined_key[-1])
        if key not in feature_by_key:
            raise ValueError(f"outcome has no matching feature row: {key!r}")
        if set(candidate_records) != set(CANDIDATE_CONFIGURATIONS):
            raise ValueError(
                f"outcome group is missing frozen candidates: {joined_key!r}"
            )
        masses = {
            configuration: float(record["attention_mass_captured"])
            for configuration, record in candidate_records.items()
        }
        errors = {
            configuration: float(record["relative_attention_output_error"])
            for configuration, record in candidate_records.items()
        }
        example = {
            **{field: value for field, value in zip(OBSERVATION_KEY_FIELDS, key)},
            "query_position": feature_by_key[key]["query_position"],
            "budget_fraction": budget,
            "features": feature_by_key[key],
            "outcomes": {
                configuration: {
                    "attention_mass_captured": masses[configuration],
                    "relative_attention_output_error": errors[configuration],
                }
                for configuration in CANDIDATE_CONFIGURATIONS
            },
            "post_hoc_diagnostics": {
                "normalized_attention_entropy": float(
                    next(iter(candidate_records.values()))[
                        "normalized_attention_entropy"
                    ]
                ),
                "candidate_recall": {
                    configuration: float(record["candidate_recall"])
                    for configuration, record in candidate_records.items()
                },
            },
            "mass_oracle_configuration": choose_configuration(
                masses,
                maximize=True,
            ),
            "error_oracle_configuration": choose_configuration(
                errors,
                maximize=False,
            ),
        }
        examples.append(example)
    return examples


def fit_mass_lookup(
    examples: Sequence[Mapping[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Fit a development-only fixed choice using mean attention mass."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        group_key = (
            float(example["budget_fraction"]),
            *(example[field] for field in group_fields),
        )
        groups[group_key].append(example)
    rows: list[dict[str, Any]] = []
    for group_key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        means = {
            configuration: statistics.fmean(
                float(example["outcomes"][configuration]["attention_mass_captured"])
                for example in group
            )
            for configuration in CANDIDATE_CONFIGURATIONS
        }
        chosen = choose_configuration(means, maximize=True)
        row = {
            "budget_fraction": group_key[0],
            **dict(zip(group_fields, group_key[1:], strict=True)),
            "configuration": chosen,
            "development_mean_attention_mass": means[chosen],
            "development_configuration_means": means,
            "development_sample_count": len(group),
        }
        rows.append(row)
    return rows


def predict_mass_lookup(
    lookup_rows: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> list[str]:
    mapping = {
        (
            float(row["budget_fraction"]),
            *(row[field] for field in group_fields),
        ): str(row["configuration"])
        for row in lookup_rows
    }
    predictions: list[str] = []
    for example in examples:
        key = (
            float(example["budget_fraction"]),
            *(example[field] for field in group_fields),
        )
        if key not in mapping:
            raise ValueError(f"lookup has no development-fitted choice for {key!r}")
        predictions.append(mapping[key])
    return predictions


def choose_configuration(
    values: Mapping[str, float],
    *,
    maximize: bool,
) -> str:
    """Choose by outcome value, then frozen candidate order on exact ties."""
    if set(values) != set(CANDIDATE_CONFIGURATIONS):
        raise ValueError("values must contain exactly the four frozen candidates")
    ordered = list(CANDIDATE_CONFIGURATIONS)
    if maximize:
        best_value = max(float(values[name]) for name in ordered)
        return next(name for name in ordered if float(values[name]) == best_value)
    best_value = min(float(values[name]) for name in ordered)
    return next(name for name in ordered if float(values[name]) == best_value)


def mass_regret(oracle_mass: float, predicted_mass: float) -> float:
    return oracle_mass - predicted_mass


def error_regret(predicted_error: float, oracle_error: float) -> float:
    return predicted_error - oracle_error


def oracle_gap_recovery(
    predicted_mass: float,
    fixed_mass: float,
    oracle_mass: float,
    *,
    epsilon: float = 1e-8,
) -> float | None:
    denominator = oracle_mass - fixed_mass
    if abs(denominator) <= epsilon:
        return None
    return (predicted_mass - fixed_mass) / denominator


def bootstrap_fixture_mean_interval(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    seed: int,
    samples: int = 2_000,
) -> dict[str, float | int]:
    """Return a fixture-cluster bootstrap 95% interval for a mean."""
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    by_fixture: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and math.isfinite(float(value)):
            by_fixture[str(row["text_fixture_id"])].append(float(value))
    fixture_ids = sorted(by_fixture)
    if not fixture_ids or any(not by_fixture[fixture_id] for fixture_id in fixture_ids):
        raise ValueError("bootstrap requires finite values in every fixture cluster")
    generator = torch.Generator().manual_seed(seed)
    bootstrap_means: list[float] = []
    for _ in range(samples):
        sampled_indices = torch.randint(
            len(fixture_ids),
            (len(fixture_ids),),
            generator=generator,
        ).tolist()
        sampled_values = [
            value
            for index in sampled_indices
            for value in by_fixture[fixture_ids[index]]
        ]
        bootstrap_means.append(statistics.fmean(sampled_values))
    all_values = [value for values in by_fixture.values() for value in values]
    return {
        "cluster_count": len(fixture_ids),
        "bootstrap_samples": samples,
        "mean": statistics.fmean(all_values),
        "lower_95": percentile(bootstrap_means, 0.025),
        "upper_95": percentile(bootstrap_means, 0.975),
    }


@dataclass(frozen=True)
class Standardizer:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Standardizer":
        return cls(
            feature_names=tuple(str(name) for name in payload["feature_names"]),
            means=tuple(float(value) for value in payload["means"]),
            scales=tuple(float(value) for value in payload["scales"]),
        )


def fit_standardizer(rows: Sequence[Mapping[str, Any]]) -> Standardizer:
    if not rows:
        raise ValueError("standardizer requires training rows")
    means = tuple(
        statistics.fmean(float(row[name]) for row in rows)
        for name in NUMERIC_MODEL_FEATURES
    )
    scales: list[float] = []
    for name, mean in zip(NUMERIC_MODEL_FEATURES, means, strict=True):
        variance = statistics.fmean((float(row[name]) - mean) ** 2 for row in rows)
        scale = math.sqrt(variance)
        scales.append(scale if scale > 1e-12 else 1.0)
    return Standardizer(NUMERIC_MODEL_FEATURES, means, tuple(scales))


def encode_feature_rows(
    rows: Sequence[Mapping[str, Any]],
    standardizer: Standardizer,
) -> torch.Tensor:
    """Encode numeric features plus fixed layer/head one-hot identity."""
    encoded: list[list[float]] = []
    for row in rows:
        numeric = [
            (float(row[name]) - mean) / scale
            for name, mean, scale in zip(
                standardizer.feature_names,
                standardizer.means,
                standardizer.scales,
                strict=True,
            )
        ]
        layer = int(float(row["layer_id"]))
        head = int(float(row["head_id"]))
        if layer not in MODEL_LAYERS or head not in MODEL_HEADS:
            raise ValueError("feature row has an unexpected layer/head identity")
        layer_one_hot = [float(layer == value) for value in MODEL_LAYERS]
        head_one_hot = [float(head == value) for value in MODEL_HEADS]
        encoded.append([1.0, *numeric, *layer_one_hot, *head_one_hot])
    return torch.tensor(encoded, dtype=torch.float64)


@dataclass(frozen=True)
class LogisticModel:
    """Serialized budget-specific multinomial logistic regression."""

    budget_fraction: float
    standardizer: Standardizer
    weights: tuple[tuple[float, ...], ...]
    l2_penalty: float
    learning_rate: float
    epochs: int

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["standardizer"] = self.standardizer.to_json()
        payload["candidate_order"] = list(CANDIDATE_CONFIGURATIONS)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "LogisticModel":
        if tuple(payload["candidate_order"]) != CANDIDATE_CONFIGURATIONS:
            raise ValueError("serialized model candidate order does not match the lock")
        return cls(
            budget_fraction=float(payload["budget_fraction"]),
            standardizer=Standardizer.from_json(payload["standardizer"]),
            weights=tuple(
                tuple(float(value) for value in row) for row in payload["weights"]
            ),
            l2_penalty=float(payload["l2_penalty"]),
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
        )


def train_logistic_model(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    *,
    budget_fraction: float,
    l2_penalty: float,
    learning_rate: float = 0.05,
    epochs: int = 250,
    seed: int = 0,
) -> LogisticModel:
    if len(rows) != len(labels) or not rows:
        raise ValueError("training rows and labels must be non-empty and aligned")
    if any(label not in range(len(CANDIDATE_CONFIGURATIONS)) for label in labels):
        raise ValueError("labels must use the frozen candidate indices")
    torch.manual_seed(seed)
    standardizer = fit_standardizer(rows)
    features = encode_feature_rows(rows, standardizer)
    targets = torch.tensor(labels, dtype=torch.int64)
    weights = torch.zeros(
        (features.shape[1], len(CANDIDATE_CONFIGURATIONS)),
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = features @ weights
        loss = torch_functional.cross_entropy(logits, targets)
        # Do not penalize the intercept row.
        loss = loss + l2_penalty * weights[1:].square().sum()
        loss.backward()
        optimizer.step()
    frozen_weights = tuple(
        tuple(float(value) for value in row) for row in weights.detach().tolist()
    )
    return LogisticModel(
        budget_fraction=budget_fraction,
        standardizer=standardizer,
        weights=frozen_weights,
        l2_penalty=l2_penalty,
        learning_rate=learning_rate,
        epochs=epochs,
    )


def predict_logistic(
    model: LogisticModel,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    audit_feature_schema(rows)
    features = encode_feature_rows(rows, model.standardizer)
    weights = torch.tensor(model.weights, dtype=torch.float64)
    predicted = torch.argmax(features @ weights, dim=-1).tolist()
    return [CANDIDATE_CONFIGURATIONS[index] for index in predicted]


def measure_prediction_latency(
    model: LogisticModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int = 50,
) -> dict[str, float | int]:
    if repetitions <= 0 or not rows:
        raise ValueError("latency measurement requires rows and repetitions")
    # Warm construction/matmul paths before timing.
    predict_logistic(model, rows)
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        predict_logistic(model, rows)
        samples.append((time.perf_counter() - start) / len(rows))
    return distribution(samples)


def _oracle_labels(examples: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        CANDIDATE_CONFIGURATIONS.index(str(example["mass_oracle_configuration"]))
        for example in examples
    ]


def development_cross_validate_logistic(
    examples: Sequence[Mapping[str, Any]],
    *,
    budget_fraction: float,
    l2_candidates: Sequence[float] = (0.0, 0.0001, 0.001, 0.01),
    learning_rate: float = 0.05,
    epochs: int = 250,
    seed: int = 0,
) -> dict[str, Any]:
    """Select L2 by leave-one-development-fixture-out mean attention mass."""
    budget_examples = [
        example
        for example in examples
        if float(example["budget_fraction"]) == budget_fraction
    ]
    fixture_ids = sorted(
        {str(example["text_fixture_id"]) for example in budget_examples}
    )
    if set(fixture_ids) != {fixture.fixture_id for fixture in DEVELOPMENT_FIXTURES}:
        raise ValueError("cross-validation requires exactly the development fixtures")
    candidate_rows: list[dict[str, Any]] = []
    for l2_penalty in l2_candidates:
        fold_rows: list[dict[str, Any]] = []
        for fold_index, held_fixture in enumerate(fixture_ids):
            training = [
                example
                for example in budget_examples
                if example["text_fixture_id"] != held_fixture
            ]
            validation = [
                example
                for example in budget_examples
                if example["text_fixture_id"] == held_fixture
            ]
            model = train_logistic_model(
                [example["features"] for example in training],
                _oracle_labels(training),
                budget_fraction=budget_fraction,
                l2_penalty=float(l2_penalty),
                learning_rate=learning_rate,
                epochs=epochs,
                seed=seed + fold_index,
            )
            predictions = predict_logistic(
                model,
                [example["features"] for example in validation],
            )
            masses = [
                float(example["outcomes"][prediction]["attention_mass_captured"])
                for example, prediction in zip(validation, predictions, strict=True)
            ]
            accuracies = [
                float(prediction == example["mass_oracle_configuration"])
                for example, prediction in zip(validation, predictions, strict=True)
            ]
            fold_rows.append(
                {
                    "held_out_development_fixture": held_fixture,
                    "sample_count": len(validation),
                    "mean_attention_mass": statistics.fmean(masses),
                    "oracle_configuration_accuracy": statistics.fmean(accuracies),
                }
            )
        candidate_rows.append(
            {
                "l2_penalty": float(l2_penalty),
                "mean_cross_validated_attention_mass": statistics.fmean(
                    float(row["mean_attention_mass"]) for row in fold_rows
                ),
                "mean_cross_validated_oracle_configuration_accuracy": (
                    statistics.fmean(
                        float(row["oracle_configuration_accuracy"]) for row in fold_rows
                    )
                ),
                "folds": fold_rows,
            }
        )
    # Attention mass, not classification accuracy, is the selection criterion.
    selected = max(
        candidate_rows,
        key=lambda row: (
            float(row["mean_cross_validated_attention_mass"]),
            -float(row["l2_penalty"]),
        ),
    )
    return {
        "budget_fraction": budget_fraction,
        "selection_metric": "leave-one-development-fixture-out attention mass",
        "selected_l2_penalty": selected["l2_penalty"],
        "candidates": candidate_rows,
    }


def train_development_logistic_models(
    examples: Sequence[Mapping[str, Any]],
    *,
    learning_rate: float = 0.05,
    epochs: int = 250,
    seed: int = 0,
) -> tuple[list[LogisticModel], list[dict[str, Any]]]:
    models: list[LogisticModel] = []
    cross_validation: list[dict[str, Any]] = []
    for budget_index, budget in enumerate(PARTIAL_BUDGETS):
        cv = development_cross_validate_logistic(
            examples,
            budget_fraction=budget,
            learning_rate=learning_rate,
            epochs=epochs,
            seed=seed + budget_index * 100,
        )
        budget_examples = [
            example
            for example in examples
            if float(example["budget_fraction"]) == budget
        ]
        model = train_logistic_model(
            [example["features"] for example in budget_examples],
            _oracle_labels(budget_examples),
            budget_fraction=budget,
            l2_penalty=float(cv["selected_l2_penalty"]),
            learning_rate=learning_rate,
            epochs=epochs,
            seed=seed + budget_index,
        )
        models.append(model)
        cross_validation.append(cv)
    return models, cross_validation


def predict_budget_specific_logistic(
    models: Sequence[LogisticModel],
    examples: Sequence[Mapping[str, Any]],
) -> list[str]:
    by_budget = {model.budget_fraction: model for model in models}
    if set(by_budget) != set(PARTIAL_BUDGETS):
        raise ValueError("one logistic model is required for every partial budget")
    predictions: list[str] = []
    for example in examples:
        budget = float(example["budget_fraction"])
        predictions.extend(predict_logistic(by_budget[budget], [example["features"]]))
    return predictions


def index_memory_bytes(
    *,
    sequence_length: int,
    head_dimension: int,
    num_heads: int,
    num_layers: int,
) -> dict[str, dict[str, int]]:
    """Estimate resident reference-index bytes, excluding shared full KV tensors."""
    if min(sequence_length, head_dimension, num_heads, num_layers) <= 0:
        raise ValueError("index memory dimensions must be positive")
    multiplier = num_heads * num_layers
    quest_p16 = 2 * math.ceil(sequence_length / 16) * head_dimension * 4 * multiplier
    quest_p64 = 2 * math.ceil(sequence_length / 64) * head_dimension * 4 * multiplier

    def pq_bytes(num_subspaces: int, num_centroids: int) -> tuple[int, int]:
        codebook = num_centroids * head_dimension * 4 * multiplier
        actual_codes = sequence_length * num_subspaces * 8 * multiplier
        logical_code_bits = math.ceil(math.log2(num_centroids))
        logical_codes = (
            math.ceil(sequence_length * num_subspaces * logical_code_bits / 8)
            * multiplier
        )
        return codebook + actual_codes, codebook + logical_codes

    pq_m2_actual, pq_m2_logical = pq_bytes(2, 4)
    pq_m4_actual, pq_m4_logical = pq_bytes(4, 8)
    actual = {
        "quest_p16": quest_p16,
        "quest_p64": quest_p64,
        "pq_m2_c4": pq_m2_actual,
        "pq_m4_c8": pq_m4_actual,
    }
    logical = {
        "quest_p16": quest_p16,
        "quest_p64": quest_p64,
        "pq_m2_c4": pq_m2_logical,
        "pq_m4_c8": pq_m4_logical,
    }
    for values in (actual, logical):
        values["quest_p16_plus_p64"] = values["quest_p16"] + values["quest_p64"]
        values["pq_m2_c4_plus_m4_c8"] = values["pq_m2_c4"] + values["pq_m4_c8"]
        values["all_four"] = (
            values["quest_p16_plus_p64"] + values["pq_m2_c4_plus_m4_c8"]
        )
    return {"actual_reference_bytes": actual, "logical_packed_pq_bytes": logical}


def sha256_json_payload(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
