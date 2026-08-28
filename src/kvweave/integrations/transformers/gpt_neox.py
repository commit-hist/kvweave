"""Minimal GPT-NeoX activation extraction for reference experiments.

The module intentionally has no import-time dependency on Transformers. The
capture entry point accepts a loaded GPT-NeoX model structurally, while the
shape conversion, RoPE, causal slicing, and attention helpers remain testable
offline with ordinary PyTorch tensors.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.nn import functional as functional

from kvweave.core.types import (
    RetrievedKV,
    Selection,
    validate_kv_tensors,
    validate_query,
)


@dataclass(frozen=True)
class GPTNeoXArchitecture:
    """Verified GPT-NeoX properties needed at the KVWeave adapter boundary."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dimension: int
    max_position_embeddings: int
    rotary_dimensions: int
    attention_scale: float


def _positive_config_integer(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"GPT-NeoX config {name} must be a positive integer")
    return value


def validate_gpt_neox_config(config: Any) -> GPTNeoXArchitecture:
    """Validate standard-MHA GPT-NeoX and return its canonical dimensions.

    GQA/MQA configurations are rejected because Phase 3A intentionally keeps
    one query head per KV head. Both legacy ``rotary_pct`` and current
    Transformers ``rope_parameters.partial_rotary_factor`` are understood.
    """
    if config is None or getattr(config, "model_type", None) != "gpt_neox":
        raise ValueError("only model_type='gpt_neox' is supported")

    hidden_size = _positive_config_integer(config, "hidden_size")
    num_hidden_layers = _positive_config_integer(config, "num_hidden_layers")
    num_attention_heads = _positive_config_integer(config, "num_attention_heads")
    max_position_embeddings = _positive_config_integer(
        config,
        "max_position_embeddings",
    )
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    configured_kv_heads = getattr(config, "num_key_value_heads", None)
    num_key_value_heads = (
        num_attention_heads if configured_kv_heads is None else configured_kv_heads
    )
    if (
        not isinstance(num_key_value_heads, int)
        or isinstance(num_key_value_heads, bool)
        or num_key_value_heads <= 0
    ):
        raise ValueError("num_key_value_heads must be a positive integer when set")
    if num_key_value_heads != num_attention_heads:
        raise ValueError("GQA/MQA is outside the GPT-NeoX Phase 3A adapter")

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        rotary_fraction = rope_parameters.get("partial_rotary_factor", 1.0)
    else:
        rotary_fraction = getattr(config, "rotary_pct", 0.25)
    if not isinstance(rotary_fraction, (int, float)) or isinstance(
        rotary_fraction,
        bool,
    ):
        raise ValueError("GPT-NeoX rotary fraction must be numeric")
    if not 0.0 < float(rotary_fraction) <= 1.0:
        raise ValueError("GPT-NeoX rotary fraction must be in (0, 1]")

    head_dimension = hidden_size // num_attention_heads
    rotary_dimensions = int(head_dimension * float(rotary_fraction))
    if rotary_dimensions <= 0 or rotary_dimensions > head_dimension:
        raise ValueError("GPT-NeoX rotary dimensions must be within the head")
    if rotary_dimensions % 2 != 0:
        raise ValueError("GPT-NeoX rotary dimensions must be even")

    return GPTNeoXArchitecture(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dimension=head_dimension,
        max_position_embeddings=max_position_embeddings,
        rotary_dimensions=rotary_dimensions,
        attention_scale=head_dimension**-0.5,
    )


def validate_layer_indices(
    layer_indices: Sequence[int],
    *,
    num_hidden_layers: int,
) -> tuple[int, ...]:
    """Validate a non-empty, unique layer selection without reordering it."""
    if isinstance(layer_indices, (str, bytes)) or not isinstance(
        layer_indices,
        Sequence,
    ):
        raise TypeError("layer_indices must be a sequence of integers")
    normalized = tuple(layer_indices)
    if not normalized:
        raise ValueError("at least one layer must be selected")
    for layer_index in normalized:
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError("layer indices must be integers")
        if layer_index < 0 or layer_index >= num_hidden_layers:
            raise ValueError(
                f"layer index {layer_index} is outside [0, {num_hidden_layers})"
            )
    if len(set(normalized)) != len(normalized):
        raise ValueError("layer indices must be unique")
    return normalized


def split_gpt_neox_qkv(
    projected_qkv: torch.Tensor,
    *,
    num_attention_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert GPT-NeoX fused projection output to ``[B, H, S, D]``.

    GPT-NeoX lays out the final projection dimension as ``[H, 3 * D]``;
    splitting the raw ``[B, S, 3 * hidden]`` output into three contiguous
    hidden-sized blocks would therefore be incorrect.
    """
    if not isinstance(projected_qkv, torch.Tensor):
        raise TypeError("projected_qkv must be a torch.Tensor")
    if projected_qkv.ndim != 3:
        raise ValueError("projected_qkv must have shape [B, S, 3 * hidden]")
    if any(size <= 0 for size in projected_qkv.shape):
        raise ValueError("projected_qkv dimensions must be positive")
    if not torch.is_floating_point(projected_qkv):
        raise TypeError("projected_qkv must use a floating-point dtype")
    if (
        not isinstance(num_attention_heads, int)
        or isinstance(num_attention_heads, bool)
        or num_attention_heads <= 0
    ):
        raise ValueError("num_attention_heads must be a positive integer")

    fused_dimension = projected_qkv.shape[-1]
    divisor = 3 * num_attention_heads
    if fused_dimension % divisor != 0:
        raise ValueError(
            "projected_qkv final dimension must equal 3 * H * D for an integer D"
        )
    head_dimension = fused_dimension // divisor
    batch_size, sequence_length, _ = projected_qkv.shape
    qkv = projected_qkv.reshape(
        batch_size,
        sequence_length,
        num_attention_heads,
        3 * head_dimension,
    ).transpose(1, 2)
    return qkv.chunk(3, dim=-1)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first_half = tensor[..., : tensor.shape[-1] // 2]
    second_half = tensor[..., tensor.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_gpt_neox_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply GPT-NeoX partial RoPE to Q/K shaped ``[B, H, S, D]``.

    ``cosine`` and ``sine`` have shape ``[B, S, R]``. Only the leading ``R``
    dimensions are rotated; the remainder passes through unchanged. Values are
    not accepted because GPT-NeoX does not apply RoPE to V.
    """
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
        raise TypeError("query and key must be torch.Tensor instances")
    if query.ndim != 4 or key.ndim != 4 or query.shape != key.shape:
        raise ValueError("query and key must have identical shape [B, H, S, D]")
    if not torch.is_floating_point(query) or not torch.is_floating_point(key):
        raise TypeError("query and key must use floating-point dtypes")
    if query.dtype != key.dtype or query.device != key.device:
        raise ValueError("query and key must share dtype and device")
    if not isinstance(cosine, torch.Tensor) or not isinstance(sine, torch.Tensor):
        raise TypeError("cosine and sine must be torch.Tensor instances")
    if cosine.ndim != 3 or cosine.shape != sine.shape:
        raise ValueError("cosine and sine must have identical shape [B, S, R]")
    if cosine.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError("RoPE batch and sequence dimensions must match Q/K")
    if cosine.dtype != query.dtype or sine.dtype != query.dtype:
        raise ValueError("RoPE tensors and Q/K must use the same dtype")
    if cosine.device != query.device or sine.device != query.device:
        raise ValueError("RoPE tensors and Q/K must share a device")
    rotary_dimensions = cosine.shape[-1]
    if rotary_dimensions <= 0 or rotary_dimensions > query.shape[-1]:
        raise ValueError("RoPE dimension must be positive and no larger than D")
    if rotary_dimensions % 2 != 0:
        raise ValueError("RoPE dimension must be even")

    broadcast_cosine = cosine.unsqueeze(1)
    broadcast_sine = sine.unsqueeze(1)
    query_rotary, query_pass = (
        query[..., :rotary_dimensions],
        query[..., rotary_dimensions:],
    )
    key_rotary, key_pass = (
        key[..., :rotary_dimensions],
        key[..., rotary_dimensions:],
    )
    rotated_query = (query_rotary * broadcast_cosine) + (
        _rotate_half(query_rotary) * broadcast_sine
    )
    rotated_key = (key_rotary * broadcast_cosine) + (
        _rotate_half(key_rotary) * broadcast_sine
    )
    return (
        torch.cat((rotated_query, query_pass), dim=-1),
        torch.cat((rotated_key, key_pass), dim=-1),
    )


@dataclass(frozen=True)
class GPTNeoXLayerActivations:
    """Post-RoPE Q/K, unchanged V, and model attention projection for a layer."""

    layer_index: int
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    projected_attention_output: torch.Tensor
    dense_weight: torch.Tensor
    dense_bias: torch.Tensor | None

    def __post_init__(self) -> None:
        validate_kv_tensors(self.key, self.value)
        if self.query.shape != self.key.shape:
            raise ValueError("full-sequence query and key tensors must match")
        if self.query.dtype != self.key.dtype or self.query.device != self.key.device:
            raise ValueError("query and KV tensors must share dtype and device")
        if self.projected_attention_output.shape != (
            self.query.shape[0],
            self.query.shape[2],
            self.query.shape[1] * self.query.shape[3],
        ):
            raise ValueError("projected attention output must have shape [B, S, H * D]")
        hidden_size = self.query.shape[1] * self.query.shape[3]
        if self.dense_weight.shape != (hidden_size, hidden_size):
            raise ValueError("dense weight must have shape [H * D, H * D]")
        if self.dense_bias is not None and self.dense_bias.shape != (hidden_size,):
            raise ValueError("dense bias must have shape [H * D]")


@dataclass(frozen=True)
class GPTNeoXCapture:
    """Captured selected-layer activations and verified architecture."""

    architecture: GPTNeoXArchitecture
    layers: dict[int, GPTNeoXLayerActivations]


def _capture_tensor(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    captured = tensor.detach().to(device=device, dtype=dtype or tensor.dtype)
    return captured.clone()


def capture_gpt_neox_activations(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layer_indices: Sequence[int],
    attention_mask: torch.Tensor | None = None,
    capture_device: torch.device | str = "cpu",
    capture_dtype: torch.dtype | None = None,
) -> GPTNeoXCapture:
    """Run one model forward and construct selected-layer post-RoPE Q/K/V.

    Hooks observe the fused QKV projection, the shared rotary embedding output,
    and the attention module's post-dense output. Q and K are then constructed
    with the same partial-RoPE equations as GPT-NeoX; V is copied unchanged.
    No model attention path is patched or replaced.
    """
    config = getattr(model, "config", None)
    architecture = validate_gpt_neox_config(config)
    selected_layers = validate_layer_indices(
        layer_indices,
        num_hidden_layers=architecture.num_hidden_layers,
    )
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim != 2 or input_ids.dtype != torch.int64:
        raise ValueError("input_ids must use torch.int64 with shape [B, S]")
    if any(size <= 0 for size in input_ids.shape):
        raise ValueError("input_ids dimensions must be positive")
    if input_ids.shape[1] > architecture.max_position_embeddings:
        raise ValueError("input sequence exceeds max_position_embeddings")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        if not torch.all(attention_mask != 0).item():
            raise ValueError(
                "Phase 3A capture requires an unpadded, fully valid sequence"
            )

    base_model = getattr(model, "gpt_neox", model)
    layers = getattr(base_model, "layers", None)
    rotary_embedding = getattr(base_model, "rotary_emb", None)
    if layers is None or rotary_embedding is None:
        raise ValueError("loaded model does not expose the GPT-NeoX layer structure")
    if len(layers) != architecture.num_hidden_layers:
        raise ValueError("loaded GPT-NeoX layer count does not match its config")

    destination = torch.device(capture_device)
    projected_qkv: dict[int, torch.Tensor] = {}
    projected_attention: dict[int, torch.Tensor] = {}
    rotary_state: dict[str, torch.Tensor] = {}
    hook_handles: list[Any] = []

    def capture_rotary(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("GPT-NeoX rotary embedding did not return (cos, sin)")
        rotary_state["cosine"] = _capture_tensor(
            output[0],
            device=destination,
            dtype=capture_dtype,
        )
        rotary_state["sine"] = _capture_tensor(
            output[1],
            device=destination,
            dtype=capture_dtype,
        )

    hook_handles.append(rotary_embedding.register_forward_hook(capture_rotary))

    for layer_index in selected_layers:
        attention = getattr(layers[layer_index], "attention", None)
        qkv_projection = getattr(attention, "query_key_value", None)
        dense_projection = getattr(attention, "dense", None)
        if qkv_projection is None or dense_projection is None:
            raise ValueError("GPT-NeoX attention projection modules are unavailable")
        if (
            getattr(qkv_projection, "out_features", None)
            != 3 * architecture.hidden_size
        ):
            raise ValueError("GPT-NeoX fused QKV projection has an unsupported layout")

        def capture_qkv(
            _module: Any,
            _inputs: Any,
            output: torch.Tensor,
            *,
            selected_layer: int = layer_index,
        ) -> None:
            projected_qkv[selected_layer] = _capture_tensor(
                output,
                device=destination,
                dtype=capture_dtype,
            )

        def capture_attention(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            selected_layer: int = layer_index,
        ) -> None:
            attention_output = output[0] if isinstance(output, tuple) else output
            projected_attention[selected_layer] = _capture_tensor(
                attention_output,
                device=destination,
                dtype=capture_dtype,
            )

        hook_handles.append(qkv_projection.register_forward_hook(capture_qkv))
        hook_handles.append(attention.register_forward_hook(capture_attention))

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for handle in hook_handles:
            handle.remove()
        if was_training:
            model.train()

    if set(projected_qkv) != set(selected_layers):
        raise RuntimeError("not every selected GPT-NeoX QKV projection was captured")
    if set(projected_attention) != set(selected_layers):
        raise RuntimeError("not every selected GPT-NeoX attention output was captured")
    if set(rotary_state) != {"cosine", "sine"}:
        raise RuntimeError("GPT-NeoX rotary positional state was not captured")

    captured_layers: dict[int, GPTNeoXLayerActivations] = {}
    for layer_index in selected_layers:
        raw_query, raw_key, value = split_gpt_neox_qkv(
            projected_qkv[layer_index],
            num_attention_heads=architecture.num_attention_heads,
        )
        query, key = apply_gpt_neox_rope(
            raw_query,
            raw_key,
            rotary_state["cosine"],
            rotary_state["sine"],
        )
        dense = layers[layer_index].attention.dense
        captured_layers[layer_index] = GPTNeoXLayerActivations(
            layer_index=layer_index,
            query=query,
            key=key,
            value=value,
            projected_attention_output=projected_attention[layer_index],
            dense_weight=_capture_tensor(
                dense.weight,
                device=destination,
                dtype=capture_dtype,
            ),
            dense_bias=(
                None
                if dense.bias is None
                else _capture_tensor(
                    dense.bias,
                    device=destination,
                    dtype=capture_dtype,
                )
            ),
        )
    return GPTNeoXCapture(architecture=architecture, layers=captured_layers)


@dataclass(frozen=True)
class DecodeActivationSlice:
    """One query position and its valid causal KV prefix."""

    query_position: int
    query: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


def causal_slice(
    activations: GPTNeoXLayerActivations,
    query_position: int,
) -> DecodeActivationSlice:
    """Return Q at ``t`` and only K/V positions ``0..t`` (inclusive)."""
    if not isinstance(query_position, int) or isinstance(query_position, bool):
        raise TypeError("query_position must be an integer")
    sequence_length = activations.key.shape[2]
    if query_position < 0 or query_position >= sequence_length:
        raise ValueError("query_position is outside the captured sequence")
    return DecodeActivationSlice(
        query_position=query_position,
        query=activations.query[:, :, query_position, :],
        keys=activations.key[:, :, : query_position + 1, :],
        values=activations.value[:, :, : query_position + 1, :],
    )


@dataclass(frozen=True)
class ReferenceAttention:
    """Model-scaled attention output and full token probabilities."""

    output: torch.Tensor
    weights: torch.Tensor


def _validate_attention_scale(scale: float | None, head_dimension: int) -> float:
    attention_scale = head_dimension**-0.5 if scale is None else scale
    if (
        not isinstance(attention_scale, (int, float))
        or isinstance(attention_scale, bool)
        or not math.isfinite(float(attention_scale))
        or attention_scale <= 0
    ):
        raise ValueError("attention scale must be a positive finite number")
    return float(attention_scale)


def reference_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> ReferenceAttention:
    """Reproduce eager GPT-NeoX attention for one query position.

    Retrieval indexes continue to rank unscaled raw dot products. This helper
    applies the model scale only when forming attention probabilities and, like
    Transformers eager attention, performs softmax in float32 before casting
    probabilities back to the query dtype.
    """
    retrieved = RetrievedKV(keys=keys, values=values, valid_mask=valid_mask)
    validate_query(query, keys)
    attention_scale = _validate_attention_scale(scale, keys.shape[-1])

    logits = torch.einsum("bhd,bhsd->bhs", query, retrieved.keys) * attention_scale
    if retrieved.valid_mask is not None:
        logits = logits.masked_fill(~retrieved.valid_mask, float("-inf"))
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_values = retrieved.values
    if retrieved.valid_mask is not None:
        attention_values = torch.where(
            retrieved.valid_mask.unsqueeze(-1),
            attention_values,
            torch.zeros_like(attention_values),
        )
    output = torch.einsum("bhs,bhsd->bhd", weights, attention_values)
    return ReferenceAttention(output=output, weights=weights)


def reference_causal_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Reproduce eager GPT-NeoX attention for every causal query position.

    Q/K/V all use ``[B, H, S, D]``. Computing the complete score matrix with
    ``torch.matmul`` intentionally matches the operation shape used by the
    model's eager attention implementation, making it the reconstruction check
    for captured activations. Retrieval experiments still use the one-query
    helper above after explicit causal slicing.
    """
    validate_kv_tensors(keys, values)
    if not isinstance(query, torch.Tensor) or query.shape != keys.shape:
        raise ValueError("full query and KV tensors must share shape [B, H, S, D]")
    if query.dtype != keys.dtype or query.device != keys.device:
        raise ValueError("full query and KV tensors must share dtype and device")
    attention_scale = _validate_attention_scale(scale, keys.shape[-1])

    logits = torch.matmul(query, keys.transpose(2, 3)) * attention_scale
    causal_mask = torch.ones(
        keys.shape[2],
        keys.shape[2],
        dtype=torch.bool,
        device=keys.device,
    ).triu(diagonal=1)
    logits = logits.masked_fill(causal_mask, float("-inf"))
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, values)


def project_head_outputs(
    head_outputs: torch.Tensor,
    dense_weight: torch.Tensor,
    dense_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Concatenate ``[B, H, D]`` heads and apply GPT-NeoX's output projection."""
    if not isinstance(head_outputs, torch.Tensor) or head_outputs.ndim != 3:
        raise ValueError("head_outputs must have shape [B, H, D]")
    hidden_size = head_outputs.shape[1] * head_outputs.shape[2]
    if dense_weight.shape != (hidden_size, hidden_size):
        raise ValueError("dense_weight must have shape [H * D, H * D]")
    if (
        dense_weight.dtype != head_outputs.dtype
        or dense_weight.device != head_outputs.device
    ):
        raise ValueError("head outputs and dense projection must share dtype/device")
    if dense_bias is not None:
        if dense_bias.shape != (hidden_size,):
            raise ValueError("dense_bias must have shape [H * D]")
        if (
            dense_bias.dtype != head_outputs.dtype
            or dense_bias.device != head_outputs.device
        ):
            raise ValueError("dense bias and head outputs must share dtype/device")
    return functional.linear(
        head_outputs.reshape(head_outputs.shape[0], hidden_size),
        dense_weight,
        dense_bias,
    )


def per_head_relative_error(
    approximate: torch.Tensor,
    exact: torch.Tensor,
) -> torch.Tensor:
    """Return vector relative error independently for every ``[B, H]``."""
    if approximate.shape != exact.shape or approximate.ndim != 3:
        raise ValueError("attention outputs must share shape [B, H, D]")
    numerator = torch.linalg.vector_norm(approximate - exact, dim=-1)
    denominator = torch.linalg.vector_norm(exact, dim=-1)
    infinite = torch.full_like(numerator, float("inf"))
    return torch.where(
        denominator == 0,
        torch.where(numerator == 0, torch.zeros_like(numerator), infinite),
        numerator / denominator,
    )


def attention_mass_captured(
    full_attention_weights: torch.Tensor,
    selection: Selection,
) -> torch.Tensor:
    """Sum exact full-attention probability on selected token IDs per ``[B, H]``."""
    if not isinstance(full_attention_weights, torch.Tensor):
        raise TypeError("full_attention_weights must be a torch.Tensor")
    if full_attention_weights.ndim != 3:
        raise ValueError("full_attention_weights must have shape [B, H, S]")
    if selection.indices.shape[:2] != full_attention_weights.shape[:2]:
        raise ValueError("selection and attention weights must share B and H")
    if selection.indices.device != full_attention_weights.device:
        raise ValueError("selection and attention weights must share a device")
    valid_mask = selection.valid_mask
    safe_indices = selection.indices
    if valid_mask is not None:
        safe_indices = torch.where(
            valid_mask, safe_indices, torch.zeros_like(safe_indices)
        )
    if torch.any(safe_indices >= full_attention_weights.shape[-1]).item():
        raise IndexError("selection index exceeds the full attention distribution")
    selected_mass = torch.gather(
        full_attention_weights,
        dim=-1,
        index=safe_indices,
    )
    if valid_mask is not None:
        selected_mass = torch.where(
            valid_mask, selected_mass, torch.zeros_like(selected_mass)
        )
    return selected_mass.sum(dim=-1)
