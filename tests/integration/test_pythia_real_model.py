import pytest
import torch

from kvdb import PQIndex, QuestIndex, TensorStorage
from kvdb.integrations.transformers import (
    capture_gpt_neox_activations,
    causal_slice,
    project_head_outputs,
    reference_attention,
    reference_causal_attention,
)


MODEL_ID = "EleutherAI/pythia-410m"
MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"


def test_real_model_validation_is_pinned_and_opt_in() -> None:
    """Keep Pants' per-file pytest shard valid when the model test is deselected."""
    assert MODEL_ID == "EleutherAI/pythia-410m"
    assert len(MODEL_REVISION) == 40


def assert_full_selection(selection: object, sequence_length: int) -> None:
    indices = selection.indices  # type: ignore[attr-defined]
    valid_mask = selection.valid_mask  # type: ignore[attr-defined]
    if valid_mask is None:
        valid_mask = torch.ones_like(indices, dtype=torch.bool)
    expected = torch.arange(sequence_length, dtype=torch.int64)
    for head_index in range(indices.shape[1]):
        actual = indices[0, head_index][valid_mask[0, head_index]].sort().values
        torch.testing.assert_close(actual, expected)


@pytest.mark.model_download
def test_pinned_pythia_attention_reconstruction_and_full_budget_indexes() -> None:
    try:
        from transformers import AutoModel, AutoTokenizer, __version__
    except ImportError as error:
        pytest.fail(
            "install the model-experiment optional dependency before running "
            "model_download tests",
            pytrace=False,
        )
        raise AssertionError from error

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    token_ids = tokenizer(
        "A deterministic Pythia activation validation sequence for KVDB. ",
        add_special_tokens=False,
    )["input_ids"]
    sequence_length = 64
    input_ids = torch.tensor(
        [(token_ids * (sequence_length // len(token_ids) + 1))[:sequence_length]],
        dtype=torch.int64,
    )

    capture = capture_gpt_neox_activations(
        model,
        input_ids,
        layer_indices=[0],
        attention_mask=torch.ones_like(input_ids),
        capture_dtype=torch.float32,
    )

    assert model.config._commit_hash == MODEL_REVISION
    assert capture.architecture.hidden_size == 1024
    assert capture.architecture.num_hidden_layers == 24
    assert capture.architecture.num_attention_heads == 16
    assert capture.architecture.num_key_value_heads == 16
    assert capture.architecture.max_position_embeddings == 2048
    assert capture.architecture.rotary_dimensions == 16

    activations = capture.layers[0]
    sliced = causal_slice(activations, query_position=sequence_length - 1)
    full_causal = reference_causal_attention(
        activations.query,
        activations.key,
        activations.value,
        scale=capture.architecture.attention_scale,
    )
    independently_projected = project_head_outputs(
        full_causal[:, :, -1, :],
        activations.dense_weight,
        activations.dense_bias,
    )
    torch.testing.assert_close(
        independently_projected,
        activations.projected_attention_output[:, -1],
        rtol=1e-4,
        atol=1e-5,
    )

    full = reference_attention(
        sliced.query,
        sliced.keys,
        sliced.values,
        scale=capture.architecture.attention_scale,
    )

    storage = TensorStorage()
    storage.put(sliced.keys, sliced.values)
    indexes = [
        QuestIndex(page_size=16),
        PQIndex(
            num_subspaces=2,
            num_centroids=4,
            max_iterations=4,
            seed=0,
        ),
    ]
    for index in indexes:
        index.build(sliced.keys)
        selection = index.search(sliced.query, sequence_length)
        assert_full_selection(selection, sequence_length)
        retrieved = storage.fetch(selection)
        selected = reference_attention(
            sliced.query,
            retrieved.keys,
            retrieved.values,
            valid_mask=retrieved.valid_mask,
            scale=capture.architecture.attention_scale,
        )
        torch.testing.assert_close(selected.output, full.output, rtol=1e-4, atol=1e-5)

    # Keep the optional dependency entirely inside this test's selected path.
    assert __version__ == "5.15.1"
