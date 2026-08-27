"""Reference Hugging Face Transformers integration utilities."""

from kvdb.integrations.transformers.gpt_neox import (
    DecodeActivationSlice,
    GPTNeoXArchitecture,
    GPTNeoXCapture,
    GPTNeoXLayerActivations,
    ReferenceAttention,
    apply_gpt_neox_rope,
    attention_mass_captured,
    capture_gpt_neox_activations,
    causal_slice,
    per_head_relative_error,
    project_head_outputs,
    reference_attention,
    reference_causal_attention,
    split_gpt_neox_qkv,
    validate_gpt_neox_config,
    validate_layer_indices,
)

__all__ = [
    "DecodeActivationSlice",
    "GPTNeoXArchitecture",
    "GPTNeoXCapture",
    "GPTNeoXLayerActivations",
    "ReferenceAttention",
    "apply_gpt_neox_rope",
    "attention_mass_captured",
    "capture_gpt_neox_activations",
    "causal_slice",
    "per_head_relative_error",
    "project_head_outputs",
    "reference_attention",
    "reference_causal_attention",
    "split_gpt_neox_qkv",
    "validate_gpt_neox_config",
    "validate_layer_indices",
]
