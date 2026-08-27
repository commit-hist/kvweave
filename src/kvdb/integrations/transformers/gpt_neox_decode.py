"""Explicit GPT-NeoX autoregressive decode for Phase 3B experiments.

This is a correctness-oriented model integration, not a serving runtime. It
uses a loaded Hugging Face GPT-NeoX model's existing projections, norms, MLPs,
rotary embedding, and LM head while making every one-token decode operation
and KVDB retrieval boundary explicit. Transformers remains an optional
dependency and is not imported at module load time.
"""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import torch
from torch.nn import functional as functional

from kvdb import KVCache, PQIndex, QuestIndex, TensorStorage
from kvdb.core.types import Selection, validate_kv_tensors
from kvdb.integrations.transformers.gpt_neox import (
    GPTNeoXArchitecture,
    ReferenceAttention,
    apply_gpt_neox_rope,
    reference_attention,
    split_gpt_neox_qkv,
    validate_gpt_neox_config,
)


class DecodeMode(str, Enum):
    """How the next token fed to an approximate path is selected."""

    TEACHER_FORCED = "teacher_forced"
    FREE_RUNNING = "free_running"


class DecodeStrategy(str, Enum):
    """Phase 3B attention strategies."""

    DENSE = "dense"
    QUEST = "quest"
    PQ = "pq"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed(operation: Any, *, device: torch.device) -> tuple[float, Any]:
    _synchronize(device)
    start = time.perf_counter()
    result = operation()
    _synchronize(device)
    return (time.perf_counter() - start) * 1_000.0, result


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Return physical tensor storage represented by its logical elements."""
    return tensor.numel() * tensor.element_size()


def append_causal_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append canonical KV entries without changing their causal order."""
    validate_kv_tensors(keys, values)
    validate_kv_tensors(new_keys, new_values)
    if keys.shape[:2] + keys.shape[3:] != new_keys.shape[:2] + new_keys.shape[3:]:
        raise ValueError("new KV must match existing batch, heads, and head dimension")
    return (
        torch.cat((keys, new_keys), dim=2),
        torch.cat((values, new_values), dim=2),
    )


def prepare_decode_selection(
    selection: Selection,
    *,
    newest_token_index: int,
) -> Selection:
    """Force current-token inclusion and sort valid candidates causally.

    The integration policy preserves each row's selected-token count. If the
    newest token was absent, it replaces the final ranked valid candidate.
    Sorting happens only after retrieval diagnostics and before storage fetch.
    Neither behavior is part of ``QuestIndex`` or ``PQIndex`` ranking.
    """
    if (
        not isinstance(newest_token_index, int)
        or isinstance(newest_token_index, bool)
        or newest_token_index < 0
    ):
        raise ValueError("newest_token_index must be a non-negative integer")
    if torch.any(selection.indices > newest_token_index).item():
        raise ValueError("selection contains a future token")

    indices = selection.indices.clone()
    scores = None if selection.scores is None else selection.scores.clone()
    valid_mask = (
        torch.ones_like(indices, dtype=torch.bool)
        if selection.valid_mask is None
        else selection.valid_mask.clone()
    )
    for batch_index in range(indices.shape[0]):
        for head_index in range(indices.shape[1]):
            row_mask = valid_mask[batch_index, head_index]
            row_indices = indices[batch_index, head_index]
            if torch.any(row_indices[row_mask] == newest_token_index).item():
                continue
            last_valid_position = int(row_mask.nonzero(as_tuple=False)[-1].item())
            row_indices[last_valid_position] = newest_token_index
            if scores is not None:
                scores[batch_index, head_index, last_valid_position] = 0.0

    sentinel = torch.full_like(indices, torch.iinfo(torch.int64).max)
    sort_keys = torch.where(valid_mask, indices, sentinel)
    order = torch.argsort(sort_keys, dim=-1, stable=True)
    indices = torch.gather(indices, dim=-1, index=order)
    valid_mask = torch.gather(valid_mask, dim=-1, index=order)
    if scores is not None:
        scores = torch.gather(scores, dim=-1, index=order)
    return Selection(
        indices=indices,
        scores=scores,
        valid_mask=None if torch.all(valid_mask).item() else valid_mask,
    )


def update_decode_cache(
    strategy: DecodeStrategy,
    cache: KVCache | None,
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    new_keys: torch.Tensor,
) -> None:
    """Apply the Phase 3B reference index/storage update policy."""
    validate_kv_tensors(keys, values)
    if strategy is DecodeStrategy.DENSE:
        if cache is not None:
            raise ValueError("dense decode must not own a retrieval cache")
        return
    if cache is None:
        raise ValueError("approximate decode requires a KVDB cache")
    if strategy is DecodeStrategy.QUEST:
        if not isinstance(cache.index, QuestIndex):
            raise ValueError("Quest decode requires QuestIndex")
        cache.build(keys, values)
        return
    if strategy is DecodeStrategy.PQ:
        if not isinstance(cache.index, PQIndex):
            raise ValueError("PQ decode requires PQIndex")
        cache.index.append(new_keys)
        cache.storage.put(keys, values)
        return
    raise ValueError(f"unsupported strategy {strategy!r}")


def select_decode_input(
    mode: DecodeMode,
    *,
    dense_token: torch.Tensor,
    path_token: torch.Tensor,
) -> torch.Tensor:
    """Choose the token consumed by a path at the next decode iteration."""
    if dense_token.shape != path_token.shape or dense_token.dtype != torch.int64:
        raise ValueError("dense and path tokens must share an int64 shape")
    if mode is DecodeMode.TEACHER_FORCED:
        return dense_token
    if mode is DecodeMode.FREE_RUNNING:
        return path_token
    raise ValueError(f"unsupported decode mode {mode!r}")


def relative_tensor_error(
    approximate: torch.Tensor,
    dense: torch.Tensor,
) -> float:
    """Return float32 L2 relative error with a defined zero-denominator case."""
    if approximate.shape != dense.shape:
        raise ValueError("compared tensors must have the same shape")
    approximate_float = approximate.float()
    dense_float = dense.float()
    numerator = torch.linalg.vector_norm(approximate_float - dense_float)
    denominator = torch.linalg.vector_norm(dense_float)
    if denominator.item() == 0:
        return 0.0 if numerator.item() == 0 else float("inf")
    return float((numerator / denominator).item())


def logit_comparison_metrics(
    approximate_logits: torch.Tensor,
    dense_logits: torch.Tensor,
) -> dict[str, float | int | bool]:
    """Compare one next-token logit vector in float32."""
    if approximate_logits.shape != dense_logits.shape:
        raise ValueError("logit tensors must have the same shape")
    if approximate_logits.ndim != 2 or approximate_logits.shape[0] != 1:
        raise ValueError("Phase 3B logits must have shape [1, vocabulary]")
    approximate = approximate_logits.float()
    dense = dense_logits.float()
    difference = approximate - dense
    l2_error = torch.linalg.vector_norm(difference)
    dense_norm = torch.linalg.vector_norm(dense)
    relative_error = (
        0.0
        if dense_norm.item() == 0 and l2_error.item() == 0
        else (
            float("inf")
            if dense_norm.item() == 0
            else float((l2_error / dense_norm).item())
        )
    )
    cosine = functional.cosine_similarity(approximate, dense, dim=-1)[0].clamp(
        min=-1.0,
        max=1.0,
    )
    dense_log_probabilities = functional.log_softmax(dense, dim=-1)
    approximate_log_probabilities = functional.log_softmax(approximate, dim=-1)
    dense_probabilities = dense_log_probabilities.exp()
    kl_divergence = torch.sum(
        dense_probabilities * (dense_log_probabilities - approximate_log_probabilities),
        dim=-1,
    )[0].clamp_min(0.0)
    dense_top_1 = int(dense.argmax(dim=-1).item())
    approximate_top_1 = int(approximate.argmax(dim=-1).item())
    approximate_dense_token_logit = approximate[0, dense_top_1]
    dense_top_1_rank = int(
        (approximate[0] > approximate_dense_token_logit).sum().item() + 1
    )
    top_count = min(5, dense.shape[-1])
    dense_top_5 = set(torch.topk(dense[0], k=top_count).indices.tolist())
    approximate_top_5 = set(torch.topk(approximate[0], k=top_count).indices.tolist())
    top_5_overlap_count = len(dense_top_5 & approximate_top_5)
    return {
        "logit_cosine_similarity": float(cosine.item()),
        "logit_l2_error": float(l2_error.item()),
        "logit_relative_error": relative_error,
        "kl_divergence_dense_to_approximate": float(kl_divergence.item()),
        "dense_top_1_token": dense_top_1,
        "approximate_top_1_token": approximate_top_1,
        "dense_top_1_rank_under_approximate_logits": dense_top_1_rank,
        "top_1_agreement": dense_top_1 == approximate_top_1,
        "top_5_overlap_count": top_5_overlap_count,
        "top_5_overlap_fraction": top_5_overlap_count / top_count,
    }


def generation_divergence_metrics(
    dense_tokens: list[int],
    approximate_tokens: list[int],
) -> dict[str, Any]:
    """Summarize free-running token divergence at matching positions."""
    if len(dense_tokens) != len(approximate_tokens):
        raise ValueError("generation sequences must have equal length")
    if not dense_tokens:
        raise ValueError("generation sequences must not be empty")
    differing_positions = [
        index
        for index, (dense_token, approximate_token) in enumerate(
            zip(dense_tokens, approximate_tokens, strict=True)
        )
        if dense_token != approximate_token
    ]
    first_divergence = differing_positions[0] if differing_positions else None
    longest_common_prefix = (
        len(dense_tokens) if first_divergence is None else first_divergence
    )
    cumulative_differences: list[int] = []
    difference_count = 0
    for dense_token, approximate_token in zip(
        dense_tokens,
        approximate_tokens,
        strict=True,
    ):
        difference_count += int(dense_token != approximate_token)
        cumulative_differences.append(difference_count)
    reconverged = bool(
        first_divergence is not None
        and any(
            dense_tokens[index] == approximate_tokens[index]
            for index in range(first_divergence + 1, len(dense_tokens))
        )
    )
    return {
        "first_divergence_position": first_divergence,
        "token_agreement_rate": 1.0 - (len(differing_positions) / len(dense_tokens)),
        "longest_common_prefix_tokens": longest_common_prefix,
        "reconverged_after_first_divergence": reconverged,
        "differing_positions": differing_positions,
        "cumulative_difference_count_by_position": cumulative_differences,
        "total_differing_tokens": len(differing_positions),
    }


@dataclass(frozen=True)
class PrefillLayerKV:
    """One dense-prefill layer's post-RoPE KV state."""

    keys: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        validate_kv_tensors(self.keys, self.values)


@dataclass(frozen=True)
class DensePrefillSnapshot:
    """Dense prompt result used to initialize independent decode paths."""

    input_ids: torch.Tensor
    next_token_logits: torch.Tensor
    layers: tuple[PrefillLayerKV, ...]
    prefill_time_ms: float


@dataclass
class DecodeLayerState:
    """Mutable per-layer causal KV and optional KVDB coordinator."""

    keys: torch.Tensor
    values: torch.Tensor
    cache: KVCache | None
    initial_index_build_time_ms: float


@dataclass
class GPTNeoXDecodeState:
    """One independent dense or approximate generation branch."""

    strategy: DecodeStrategy
    budget_fraction: float
    layers: list[DecodeLayerState]
    next_token_logits: torch.Tensor
    current_length: int
    index_update_policy: str
    codebook_policy: str | None


@dataclass(frozen=True)
class LayerDecodeObservation:
    """Observable outputs and reference costs for one decode layer."""

    layer_index: int
    sequence_length: int
    query: torch.Tensor
    attention_output: torch.Tensor
    attention_weights: torch.Tensor
    residual_output: torch.Tensor
    selection: Selection | None
    selected_token_counts: torch.Tensor
    newest_token_included: bool
    index_update_time_ms: float
    retrieval_time_ms: float
    storage_fetch_time_ms: float
    selected_attention_time_ms: float
    remaining_layer_time_ms: float
    dense_kv_bytes: int
    quest_metadata_bytes: int
    pq_code_bytes: int
    pq_logical_code_bytes: int
    pq_codebook_bytes: int
    selected_kv_bytes: int


@dataclass(frozen=True)
class GPTNeoXDecodeStep:
    """One explicit token-consumption step and its resulting next logits."""

    input_token: torch.Tensor
    next_token_logits: torch.Tensor
    next_token: torch.Tensor
    layers: tuple[LayerDecodeObservation, ...]
    total_time_ms: float
    remaining_model_time_ms: float


class GPTNeoXDecodeRunner:
    """Reference one-token GPT-NeoX decoder with inspectable KVDB attention."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.architecture: GPTNeoXArchitecture = validate_gpt_neox_config(
            getattr(model, "config", None)
        )
        base_model = getattr(model, "gpt_neox", None)
        lm_head = getattr(model, "lm_head", None)
        if base_model is None or lm_head is None:
            raise ValueError("GPT-NeoX causal-LM modules are unavailable")
        if len(base_model.layers) != self.architecture.num_hidden_layers:
            raise ValueError("GPT-NeoX layer count does not match config")
        self.base_model = base_model
        self.lm_head = lm_head

    def dense_prefill(self, input_ids: torch.Tensor) -> DensePrefillSnapshot:
        """Run dense Hugging Face prefill and snapshot its canonical KV cache."""
        if input_ids.ndim != 2 or input_ids.dtype != torch.int64:
            raise ValueError("input_ids must have int64 shape [B, S]")
        if input_ids.shape[0] != 1:
            raise ValueError("Phase 3B reference decode requires batch size one")
        if input_ids.shape[1] >= self.architecture.max_position_embeddings:
            raise ValueError("prompt must leave room for at least one generated token")
        device = input_ids.device

        def run_prefill() -> Any:
            with torch.no_grad():
                return self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                    logits_to_keep=1,
                )

        prefill_time_ms, outputs = _timed(run_prefill, device=device)
        cache_layers = getattr(outputs.past_key_values, "layers", None)
        if (
            cache_layers is None
            or len(cache_layers) != self.architecture.num_hidden_layers
        ):
            raise RuntimeError("dense prefill did not expose one KV cache per layer")
        layers: list[PrefillLayerKV] = []
        for cache_layer in cache_layers:
            keys = getattr(cache_layer, "keys", None)
            values = getattr(cache_layer, "values", None)
            if keys is None or values is None:
                raise RuntimeError("dense prefill returned an uninitialized KV layer")
            layers.append(
                PrefillLayerKV(
                    keys=keys.detach().clone(),
                    values=values.detach().clone(),
                )
            )
        return DensePrefillSnapshot(
            input_ids=input_ids.detach().clone(),
            next_token_logits=outputs.logits[:, -1, :].detach().clone(),
            layers=tuple(layers),
            prefill_time_ms=prefill_time_ms,
        )

    def initialize_state(
        self,
        snapshot: DensePrefillSnapshot,
        *,
        strategy: DecodeStrategy,
        budget_fraction: float = 1.0,
        quest_page_size: int = 64,
        pq_num_subspaces: int = 4,
        pq_num_centroids: int = 8,
        pq_max_iterations: int = 8,
        seed: int = 0,
    ) -> GPTNeoXDecodeState:
        """Create an independent path and build its initial reference indexes."""
        if not math.isfinite(budget_fraction) or not 0.0 < budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be in (0, 1]")
        layers: list[DecodeLayerState] = []
        for layer in snapshot.layers:
            cache: KVCache | None = None
            build_time_ms = 0.0
            if strategy is DecodeStrategy.QUEST:
                cache = KVCache(
                    index=QuestIndex(page_size=quest_page_size),
                    storage=TensorStorage(),
                )
            elif strategy is DecodeStrategy.PQ:
                cache = KVCache(
                    index=PQIndex(
                        num_subspaces=pq_num_subspaces,
                        num_centroids=pq_num_centroids,
                        max_iterations=pq_max_iterations,
                        seed=seed,
                    ),
                    storage=TensorStorage(),
                )
            elif strategy is not DecodeStrategy.DENSE:
                raise ValueError(f"unsupported strategy {strategy!r}")
            if cache is not None:
                build_time_ms, _ = _timed(
                    lambda: cache.build(layer.keys, layer.values),
                    device=layer.keys.device,
                )
            layers.append(
                DecodeLayerState(
                    keys=layer.keys,
                    values=layer.values,
                    cache=cache,
                    initial_index_build_time_ms=build_time_ms,
                )
            )
        return GPTNeoXDecodeState(
            strategy=strategy,
            budget_fraction=budget_fraction,
            layers=layers,
            next_token_logits=snapshot.next_token_logits,
            current_length=snapshot.input_ids.shape[1],
            index_update_policy=(
                "none"
                if strategy is DecodeStrategy.DENSE
                else (
                    "rebuild_page_metadata_after_each_append"
                    if strategy is DecodeStrategy.QUEST
                    else "encode_appended_key_with_frozen_prefill_codebooks"
                )
            ),
            codebook_policy=(
                "trained_on_dense_prefill_keys_and_frozen_during_decode"
                if strategy is DecodeStrategy.PQ
                else None
            ),
        )

    def _update_index(
        self,
        state: GPTNeoXDecodeState,
        layer_state: DecodeLayerState,
        new_keys: torch.Tensor,
    ) -> None:
        update_decode_cache(
            state.strategy,
            layer_state.cache,
            keys=layer_state.keys,
            values=layer_state.values,
            new_keys=new_keys,
        )

    @staticmethod
    def _index_memory(layer_state: DecodeLayerState) -> tuple[int, int, int, int]:
        cache = layer_state.cache
        if cache is None:
            return 0, 0, 0, 0
        if isinstance(cache.index, QuestIndex):
            metadata = cache.index.metadata
            return (
                tensor_bytes(metadata.minimum) + tensor_bytes(metadata.maximum),
                0,
                0,
                0,
            )
        if isinstance(cache.index, PQIndex):
            metadata = cache.index.metadata
            logical_code_bytes = math.ceil(
                metadata.codes.numel() * (metadata.num_centroids - 1).bit_length() / 8
            )
            return (
                0,
                tensor_bytes(metadata.codes),
                logical_code_bytes,
                tensor_bytes(metadata.codebooks),
            )
        raise RuntimeError("unsupported KVDB index in decode state")

    def step(
        self,
        state: GPTNeoXDecodeState,
        input_token: torch.Tensor,
    ) -> GPTNeoXDecodeStep:
        """Consume one token and produce logits for the following token."""
        if input_token.shape != (1, 1) or input_token.dtype != torch.int64:
            raise ValueError("input_token must have int64 shape [1, 1]")
        if state.current_length >= self.architecture.max_position_embeddings:
            raise ValueError("decode would exceed max_position_embeddings")
        device = input_token.device
        _synchronize(device)
        step_start = time.perf_counter()
        position_ids = torch.tensor(
            [[state.current_length]],
            dtype=torch.int64,
            device=device,
        )
        with torch.no_grad():
            hidden_states = self.base_model.embed_in(input_token)
            hidden_states = self.base_model.emb_dropout(hidden_states)
            cosine, sine = self.base_model.rotary_emb(
                hidden_states,
                position_ids=position_ids,
            )
            observations: list[LayerDecodeObservation] = []
            for layer_index, (layer, layer_state) in enumerate(
                zip(self.base_model.layers, state.layers, strict=True)
            ):
                _synchronize(device)
                layer_start = time.perf_counter()
                layer_input = hidden_states
                normalized = layer.input_layernorm(layer_input)
                projected_qkv = layer.attention.query_key_value(normalized)
                query, new_keys, new_values = split_gpt_neox_qkv(
                    projected_qkv,
                    num_attention_heads=self.architecture.num_attention_heads,
                )
                query, new_keys = apply_gpt_neox_rope(
                    query,
                    new_keys,
                    cosine,
                    sine,
                )
                layer_state.keys, layer_state.values = append_causal_kv(
                    layer_state.keys,
                    layer_state.values,
                    new_keys,
                    new_values,
                )
                query_single = query[:, :, 0, :]
                index_update_time_ms, _ = _timed(
                    lambda: self._update_index(state, layer_state, new_keys),
                    device=device,
                )

                selection: Selection | None = None
                retrieval_time_ms = 0.0
                storage_fetch_time_ms = 0.0
                if state.strategy is DecodeStrategy.DENSE:
                    retrieved_keys = layer_state.keys
                    retrieved_values = layer_state.values
                    retrieved_mask = None
                    selected_counts = torch.full(
                        layer_state.keys.shape[:2],
                        layer_state.keys.shape[2],
                        dtype=torch.int64,
                        device=device,
                    )
                    newest_token_included = True
                else:
                    if layer_state.cache is None:
                        raise RuntimeError("approximate layer has no KVDB cache")
                    token_budget = max(
                        1,
                        math.ceil(layer_state.keys.shape[2] * state.budget_fraction),
                    )

                    def retrieve_selection() -> Selection:
                        ranked = layer_state.cache.index.search(
                            query_single,
                            token_budget,
                        )
                        return prepare_decode_selection(
                            ranked,
                            newest_token_index=layer_state.keys.shape[2] - 1,
                        )

                    retrieval_time_ms, selection = _timed(
                        retrieve_selection,
                        device=device,
                    )
                    storage_fetch_time_ms, retrieved = _timed(
                        lambda: layer_state.cache.storage.fetch(selection),
                        device=device,
                    )
                    retrieved_keys = retrieved.keys
                    retrieved_values = retrieved.values
                    retrieved_mask = retrieved.valid_mask
                    selected_counts = selection.valid_token_counts
                    newest_token_included = bool(
                        torch.all(
                            (
                                (selection.indices == (layer_state.keys.shape[2] - 1))
                                & (
                                    torch.ones_like(
                                        selection.indices,
                                        dtype=torch.bool,
                                    )
                                    if selection.valid_mask is None
                                    else selection.valid_mask
                                )
                            ).any(dim=-1)
                        ).item()
                    )

                selected_attention_time_ms, attention = _timed(
                    lambda: reference_attention(
                        query_single,
                        retrieved_keys,
                        retrieved_values,
                        valid_mask=retrieved_mask,
                        scale=self.architecture.attention_scale,
                    ),
                    device=device,
                )
                if not isinstance(attention, ReferenceAttention):
                    raise RuntimeError("reference attention returned an invalid result")
                attention_projection = layer.attention.dense(
                    attention.output.reshape(1, 1, -1)
                )
                attention_projection = layer.post_attention_dropout(
                    attention_projection
                )
                if layer.use_parallel_residual:
                    mlp_output = layer.mlp(layer.post_attention_layernorm(layer_input))
                    mlp_output = layer.post_mlp_dropout(mlp_output)
                    hidden_states = mlp_output + attention_projection + layer_input
                else:
                    attention_residual = attention_projection + layer_input
                    mlp_output = layer.mlp(
                        layer.post_attention_layernorm(attention_residual)
                    )
                    mlp_output = layer.post_mlp_dropout(mlp_output)
                    hidden_states = mlp_output + attention_residual
                _synchronize(device)
                total_layer_time_ms = (time.perf_counter() - layer_start) * 1_000.0
                accounted_time_ms = (
                    index_update_time_ms
                    + retrieval_time_ms
                    + storage_fetch_time_ms
                    + selected_attention_time_ms
                )
                quest_bytes, pq_code_bytes, pq_logical_bytes, pq_codebook_bytes = (
                    self._index_memory(layer_state)
                )
                semantic_selected = int(selected_counts.sum().item())
                selected_kv_bytes = (
                    semantic_selected
                    * layer_state.keys.shape[-1]
                    * layer_state.keys.element_size()
                    * 2
                )
                observations.append(
                    LayerDecodeObservation(
                        layer_index=layer_index,
                        sequence_length=layer_state.keys.shape[2],
                        query=query_single.detach(),
                        attention_output=attention.output.detach(),
                        attention_weights=attention.weights.detach(),
                        residual_output=hidden_states.detach(),
                        selection=selection,
                        selected_token_counts=selected_counts.detach(),
                        newest_token_included=newest_token_included,
                        index_update_time_ms=index_update_time_ms,
                        retrieval_time_ms=retrieval_time_ms,
                        storage_fetch_time_ms=storage_fetch_time_ms,
                        selected_attention_time_ms=selected_attention_time_ms,
                        remaining_layer_time_ms=max(
                            0.0,
                            total_layer_time_ms - accounted_time_ms,
                        ),
                        dense_kv_bytes=(
                            tensor_bytes(layer_state.keys)
                            + tensor_bytes(layer_state.values)
                        ),
                        quest_metadata_bytes=quest_bytes,
                        pq_code_bytes=pq_code_bytes,
                        pq_logical_code_bytes=pq_logical_bytes,
                        pq_codebook_bytes=pq_codebook_bytes,
                        selected_kv_bytes=selected_kv_bytes,
                    )
                )
            hidden_states = self.base_model.final_layer_norm(hidden_states)
            next_token_logits = self.lm_head(hidden_states)[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        _synchronize(device)
        total_time_ms = (time.perf_counter() - step_start) * 1_000.0
        accounted_step_ms = sum(
            observation.index_update_time_ms
            + observation.retrieval_time_ms
            + observation.storage_fetch_time_ms
            + observation.selected_attention_time_ms
            for observation in observations
        )
        state.next_token_logits = next_token_logits.detach()
        state.current_length += 1
        return GPTNeoXDecodeStep(
            input_token=input_token.detach().clone(),
            next_token_logits=next_token_logits.detach(),
            next_token=next_token.detach(),
            layers=tuple(observations),
            total_time_ms=total_time_ms,
            remaining_model_time_ms=max(0.0, total_time_ms - accounted_step_ms),
        )
