import pytest
import torch

from kvweave.integrations.transformers import (
    DecodeStrategy,
    GPTNeoXDecodeRunner,
    QuestMetadataUpdateMode,
)


MODEL_ID = "EleutherAI/pythia-410m"
MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
TRANSFORMERS_VERSION = "5.15.1"


def assert_decode_steps_bit_exact(actual: object, expected: object) -> None:
    for name in ("input_token", "next_token_logits", "next_token"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    for actual_layer, expected_layer in zip(
        getattr(actual, "layers"),
        getattr(expected, "layers"),
        strict=True,
    ):
        for name in (
            "query",
            "attention_output",
            "attention_weights",
            "residual_output",
            "selected_token_counts",
        ):
            assert torch.equal(
                getattr(actual_layer, name),
                getattr(expected_layer, name),
            )
        actual_selection = actual_layer.selection
        expected_selection = expected_layer.selection
        assert actual_selection is not None
        assert expected_selection is not None
        for name in ("indices", "scores", "valid_mask"):
            actual_value = getattr(actual_selection, name)
            expected_value = getattr(expected_selection, name)
            assert (actual_value is None) == (expected_value is None)
            if actual_value is not None and expected_value is not None:
                assert torch.equal(actual_value, expected_value)


def test_decode_validation_is_pinned_and_opt_in() -> None:
    assert MODEL_ID == "EleutherAI/pythia-410m"
    assert len(MODEL_REVISION) == 40


def repeated_prompt(tokenizer: object, sequence_length: int) -> torch.Tensor:
    # This legacy pre-KVWeave literal is frozen for stable decode controls.
    token_ids = tokenizer(  # type: ignore[operator]
        "A deterministic stateful decode fixture for the KVDB reference loop. ",
        add_special_tokens=False,
    )["input_ids"]
    return torch.tensor(
        [(token_ids * (sequence_length // len(token_ids) + 1))[:sequence_length]],
        dtype=torch.int64,
    )


@pytest.mark.model_download
def test_custom_dense_decode_matches_hugging_face_greedy_generation() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    except ImportError as error:
        pytest.fail(
            "install the model-experiment optional dependency before running "
            "model_download tests",
            pytrace=False,
        )
        raise AssertionError from error

    assert __version__ == TRANSFORMERS_VERSION
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    runner = GPTNeoXDecodeRunner(model)

    for sequence_length in (16, 64):
        input_ids = repeated_prompt(tokenizer, sequence_length)
        snapshot = runner.dense_prefill(input_ids)
        state = runner.initialize_state(snapshot, strategy=DecodeStrategy.DENSE)
        custom_tokens = [int(snapshot.next_token_logits.argmax(dim=-1).item())]
        custom_logits = [snapshot.next_token_logits]
        for _ in range(3):
            step = runner.step(
                state,
                torch.tensor([[custom_tokens[-1]]], dtype=torch.int64),
            )
            custom_tokens.append(int(step.next_token.item()))
            custom_logits.append(step.next_token_logits)

        generated = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=4,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        assert generated.sequences[0, -4:].tolist() == custom_tokens
        assert len(generated.scores) == len(custom_logits)
        for custom, hugging_face in zip(
            custom_logits,
            generated.scores,
            strict=True,
        ):
            torch.testing.assert_close(
                custom,
                hugging_face,
                rtol=1e-4,
                atol=1e-5,
            )


@pytest.mark.model_download
def test_full_budget_quest_and_pq_decode_match_dense() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        pytest.fail(
            "install the model-experiment optional dependency before running "
            "model_download tests",
            pytrace=False,
        )
        raise AssertionError from error

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    runner = GPTNeoXDecodeRunner(model)
    snapshot = runner.dense_prefill(repeated_prompt(tokenizer, 32))
    dense = runner.initialize_state(snapshot, strategy=DecodeStrategy.DENSE)
    approximate_states = [
        runner.initialize_state(
            snapshot,
            strategy=DecodeStrategy.QUEST,
            budget_fraction=1.0,
            quest_page_size=64,
        ),
        runner.initialize_state(
            snapshot,
            strategy=DecodeStrategy.PQ,
            budget_fraction=1.0,
            pq_num_subspaces=4,
            pq_num_centroids=8,
            pq_max_iterations=8,
            seed=0,
        ),
    ]
    quest_oracle_pairs = [
        (
            runner.initialize_state(
                snapshot,
                strategy=DecodeStrategy.QUEST,
                budget_fraction=budget_fraction,
                quest_page_size=64,
                quest_metadata_update_mode=(QuestMetadataUpdateMode.FULL_REBUILD),
            ),
            runner.initialize_state(
                snapshot,
                strategy=DecodeStrategy.QUEST,
                budget_fraction=budget_fraction,
                quest_page_size=64,
                quest_metadata_update_mode=QuestMetadataUpdateMode.INCREMENTAL,
            ),
        )
        for budget_fraction in (0.5, 1.0)
    ]
    dense_token = snapshot.next_token_logits.argmax(dim=-1, keepdim=True)
    for _ in range(3):
        dense_step = runner.step(dense, dense_token)
        for state in approximate_states:
            approximate_step = runner.step(state, dense_token)
            torch.testing.assert_close(
                approximate_step.next_token_logits,
                dense_step.next_token_logits,
                rtol=1e-4,
                atol=1e-5,
            )
            assert int(approximate_step.next_token.item()) == int(
                dense_step.next_token.item()
            )
            for approximate_layer, dense_layer in zip(
                approximate_step.layers,
                dense_step.layers,
                strict=True,
            ):
                assert approximate_layer.newest_token_included
                assert torch.all(
                    approximate_layer.selected_token_counts
                    == approximate_layer.sequence_length
                ).item()
                torch.testing.assert_close(
                    approximate_layer.attention_output,
                    dense_layer.attention_output,
                    rtol=1e-4,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    approximate_layer.residual_output,
                    dense_layer.residual_output,
                    rtol=1e-4,
                    atol=1e-5,
                )
        for oracle_state, incremental_state in quest_oracle_pairs:
            oracle_step = runner.step(oracle_state, dense_token)
            incremental_step = runner.step(incremental_state, dense_token)
            assert_decode_steps_bit_exact(incremental_step, oracle_step)
            for oracle_layer_state, incremental_layer_state in zip(
                oracle_state.layers,
                incremental_state.layers,
                strict=True,
            ):
                assert oracle_layer_state.cache is not None
                assert incremental_layer_state.cache is not None
                assert torch.equal(
                    incremental_layer_state.cache.index.metadata.minimum,
                    oracle_layer_state.cache.index.metadata.minimum,
                )
                assert torch.equal(
                    incremental_layer_state.cache.index.metadata.maximum,
                    oracle_layer_state.cache.index.metadata.maximum,
                )
        dense_token = dense_step.next_token
