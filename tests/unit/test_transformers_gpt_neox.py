from types import SimpleNamespace

import pytest
import torch

from kvdb.core.types import Selection
from kvdb.integrations.transformers import (
    GPTNeoXLayerActivations,
    apply_gpt_neox_rope,
    attention_mass_captured,
    causal_slice,
    per_head_relative_error,
    project_head_outputs,
    reference_attention,
    reference_causal_attention,
    split_gpt_neox_qkv,
    validate_gpt_neox_config,
    validate_layer_indices,
)


def pythia_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_type": "gpt_neox",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "max_position_embeddings": 2048,
        "rotary_pct": 0.25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_activations(sequence_length: int = 5) -> GPTNeoXLayerActivations:
    query = torch.arange(
        2 * 2 * sequence_length * 4,
        dtype=torch.float32,
    ).reshape(2, 2, sequence_length, 4)
    key = query + 100.0
    value = query + 200.0
    hidden_size = 8
    return GPTNeoXLayerActivations(
        layer_index=3,
        query=query,
        key=key,
        value=value,
        projected_attention_output=torch.zeros(
            2,
            sequence_length,
            hidden_size,
        ),
        dense_weight=torch.eye(hidden_size),
        dense_bias=torch.zeros(hidden_size),
    )


def test_validate_pythia_architecture_and_standard_mha() -> None:
    architecture = validate_gpt_neox_config(pythia_config())

    assert architecture.hidden_size == 1024
    assert architecture.num_hidden_layers == 24
    assert architecture.num_attention_heads == 16
    assert architecture.num_key_value_heads == 16
    assert architecture.head_dimension == 64
    assert architecture.max_position_embeddings == 2048
    assert architecture.rotary_dimensions == 16
    assert architecture.attention_scale == pytest.approx(0.125)


def test_validate_current_rope_parameter_format() -> None:
    architecture = validate_gpt_neox_config(
        pythia_config(
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 0.5,
            }
        )
    )
    assert architecture.rotary_dimensions == 32


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_type": "llama"}, "only model_type"),
        ({"num_key_value_heads": 4}, "GQA/MQA"),
        ({"hidden_size": 1025}, "divisible"),
        ({"rotary_pct": 0.3}, "even"),
    ],
)
def test_unsupported_or_malformed_architecture_is_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_gpt_neox_config(pythia_config(**overrides))


def test_layer_selection_preserves_requested_early_middle_late_order() -> None:
    assert validate_layer_indices(
        [0, 12, 23],
        num_hidden_layers=24,
    ) == (0, 12, 23)


@pytest.mark.parametrize(
    ("layers", "error_type"),
    [
        ([], ValueError),
        ([24], ValueError),
        ([-1], ValueError),
        ([1, 1], ValueError),
        ([True], TypeError),
        ([1.0], TypeError),
    ],
)
def test_malformed_layer_selection_is_rejected(
    layers: list[object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        validate_layer_indices(layers, num_hidden_layers=24)  # type: ignore[arg-type]


def test_fused_gpt_neox_qkv_layout_converts_to_canonical_heads() -> None:
    # Encode [head, q/k/v, dimension] directly so an incorrect split into
    # contiguous hidden-sized blocks is immediately visible.
    source = torch.empty(1, 2, 2, 3, 3)
    for token in range(2):
        for head in range(2):
            for component in range(3):
                source[0, token, head, component] = 100 * component + 10 * head + token
    projected = source.reshape(1, 2, 18)

    query, key, value = split_gpt_neox_qkv(
        projected,
        num_attention_heads=2,
    )

    assert query.shape == (1, 2, 2, 3)
    assert key.shape == (1, 2, 2, 3)
    assert value.shape == (1, 2, 2, 3)
    torch.testing.assert_close(query[0, 1, 1], torch.full((3,), 11.0))
    torch.testing.assert_close(key[0, 1, 1], torch.full((3,), 111.0))
    torch.testing.assert_close(value[0, 1, 1], torch.full((3,), 211.0))


@pytest.mark.parametrize(
    "projected",
    [
        torch.ones(2, 6),
        torch.ones(1, 2, 17),
        torch.ones(1, 0, 18),
        torch.ones(1, 2, 18, dtype=torch.int64),
    ],
)
def test_malformed_fused_qkv_is_rejected(projected: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        split_gpt_neox_qkv(projected, num_attention_heads=2)


def test_partial_rope_rotates_only_the_leading_dimensions_and_not_values() -> None:
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]]]])
    key = torch.tensor([[[[5.0, 6.0, 7.0, 8.0, 11.0, 12.0]]]])
    value = torch.tensor([[[[13.0, 14.0, 15.0, 16.0, 17.0, 18.0]]]])
    cosine = torch.zeros(1, 1, 4)
    sine = torch.ones(1, 1, 4)

    rotated_query, rotated_key = apply_gpt_neox_rope(
        query,
        key,
        cosine,
        sine,
    )

    torch.testing.assert_close(
        rotated_query,
        torch.tensor([[[[-3.0, -4.0, 1.0, 2.0, 9.0, 10.0]]]]),
    )
    torch.testing.assert_close(
        rotated_key,
        torch.tensor([[[[-7.0, -8.0, 5.0, 6.0, 11.0, 12.0]]]]),
    )
    # The integration API never passes V through the RoPE function.
    torch.testing.assert_close(
        value,
        torch.tensor([[[[13.0, 14.0, 15.0, 16.0, 17.0, 18.0]]]]),
    )


def test_causal_slice_extracts_query_and_excludes_future_tokens() -> None:
    activations = make_activations(sequence_length=5)

    sliced = causal_slice(activations, query_position=2)

    assert sliced.query.shape == (2, 2, 4)
    assert sliced.keys.shape == (2, 2, 3, 4)
    assert sliced.values.shape == (2, 2, 3, 4)
    torch.testing.assert_close(sliced.query, activations.query[:, :, 2, :])
    torch.testing.assert_close(sliced.keys, activations.key[:, :, :3, :])
    assert not torch.any(sliced.keys == activations.key[:, :, 3:, :].amin()).item()


@pytest.mark.parametrize("query_position", [-1, 5, True, 1.5])
def test_invalid_query_position_is_rejected(query_position: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        causal_slice(make_activations(), query_position)  # type: ignore[arg-type]


def test_reference_attention_matches_manual_scaled_float32_softmax() -> None:
    query = torch.tensor([[[2.0, 0.0]]])
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    values = torch.tensor([[[[3.0, 1.0], [1.0, 5.0]]]])

    result = reference_attention(query, keys, values, scale=0.5)
    expected_weights = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    expected_output = expected_weights @ values[0, 0]

    torch.testing.assert_close(result.weights[0, 0], expected_weights)
    torch.testing.assert_close(result.output[0, 0], expected_output)


def test_full_causal_reconstruction_excludes_every_future_position() -> None:
    query = torch.ones(1, 1, 3, 2)
    keys = torch.ones(1, 1, 3, 2)
    values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]]]])

    output = reference_causal_attention(query, keys, values, scale=1.0)

    torch.testing.assert_close(output[0, 0, 0], values[0, 0, 0])
    torch.testing.assert_close(output[0, 0, 1], torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(output[0, 0, 2], torch.tensor([10.0 / 3, 11.0 / 3]))


def test_positive_attention_scaling_preserves_retrieval_ranking() -> None:
    query = torch.tensor([[[2.0, -1.0]]])
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]]])
    raw_scores = torch.einsum("bhd,bhsd->bhs", query, keys)

    assert torch.equal(
        raw_scores.argsort(dim=-1, descending=True),
        (raw_scores * 0.125).argsort(dim=-1, descending=True),
    )


def test_reference_attention_respects_rectangular_validity() -> None:
    query = torch.ones(1, 1, 2)
    keys = torch.tensor([[[[1.0, 1.0], [100.0, 100.0]]]])
    values = torch.tensor([[[[2.0, 3.0], [999.0, 999.0]]]])

    result = reference_attention(
        query,
        keys,
        values,
        valid_mask=torch.tensor([[[True, False]]]),
    )

    torch.testing.assert_close(result.weights, torch.tensor([[[1.0, 0.0]]]))
    torch.testing.assert_close(result.output, torch.tensor([[[2.0, 3.0]]]))


def test_head_concatenation_and_dense_projection_reconstruct_model_layout() -> None:
    head_outputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ]
    )
    bias = torch.ones(4)

    projected = project_head_outputs(head_outputs, weight, bias)

    torch.testing.assert_close(projected, torch.tensor([[2.0, 3.0, 7.0, 9.0]]))


def test_per_head_relative_error_is_not_collapsed_to_an_average() -> None:
    exact = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    approximate = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])

    errors = per_head_relative_error(approximate, exact)

    torch.testing.assert_close(errors, torch.tensor([[1.0, 0.0]]))


def test_attention_mass_capture_is_distinct_from_candidate_recall() -> None:
    weights = torch.tensor([[[0.7, 0.2, 0.08, 0.02]]])
    selection = Selection(indices=torch.tensor([[[0, 3]]], dtype=torch.int64))

    mass = attention_mass_captured(weights, selection)

    torch.testing.assert_close(mass, torch.tensor([[0.72]]))


def test_attention_mass_ignores_masked_quest_placeholders() -> None:
    weights = torch.tensor([[[0.5, 0.3, 0.2]]])
    selection = Selection(
        indices=torch.tensor([[[1, 0]]], dtype=torch.int64),
        valid_mask=torch.tensor([[[True, False]]]),
    )

    torch.testing.assert_close(
        attention_mass_captured(weights, selection),
        torch.tensor([[0.3]]),
    )
