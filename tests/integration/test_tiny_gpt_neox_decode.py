"""Exercise the complete decoder offline with random, tiny model weights."""

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from benchmarks.decode import (
    assert_full_budget_step,
    build_dense_trace,
    validate_hugging_face_generation,
)
from kvweave import QuestIndex
from kvweave.integrations.transformers import (
    DecodeStrategy,
    GPTNeoXDecodeRunner,
    QuestMetadataUpdateMode,
)


@pytest.fixture
def tiny_runner(monkeypatch: pytest.MonkeyPatch) -> GPTNeoXDecodeRunner:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        config = GPTNeoXConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bos_token_id=0,
            eos_token_id=None,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        model = GPTNeoXForCausalLM(config).eval()
    return GPTNeoXDecodeRunner(model)


def test_offline_dense_trace_matches_hugging_face(
    tiny_runner: GPTNeoXDecodeRunner,
) -> None:
    input_ids = torch.tensor([[1, 7, 3, 9, 2, 5, 4]])
    snapshot = tiny_runner.dense_prefill(input_ids)
    tokens, logits, steps = build_dense_trace(
        tiny_runner,
        snapshot,
        generated_tokens=5,
    )
    result = validate_hugging_face_generation(
        tiny_runner.model,
        input_ids,
        generated_tokens=5,
        custom_tokens=tokens,
        custom_logits=logits,
    )
    assert result["passed"]
    assert [step.layers[0].sequence_length for step in steps] == [8, 9, 10, 11]


def test_offline_full_budget_quest_and_pq_match_dense(
    tiny_runner: GPTNeoXDecodeRunner,
) -> None:
    snapshot = tiny_runner.dense_prefill(torch.tensor([[1, 7, 3, 9, 2, 5, 4]]))
    dense = tiny_runner.initialize_state(snapshot, strategy=DecodeStrategy.DENSE)
    paths = [
        tiny_runner.initialize_state(
            snapshot,
            strategy=DecodeStrategy.QUEST,
            quest_page_size=4,
        ),
        tiny_runner.initialize_state(
            snapshot,
            strategy=DecodeStrategy.PQ,
            pq_num_subspaces=2,
            pq_num_centroids=2,
            pq_max_iterations=2,
        ),
    ]
    input_token = snapshot.next_token_logits.argmax(dim=-1, keepdim=True)
    for expected_length in (8, 9, 10, 11):
        dense_step = tiny_runner.step(dense, input_token)
        for path in paths:
            step = tiny_runner.step(path, input_token)
            assert_full_budget_step(step, dense_step, rtol=1e-4, atol=1e-5)
            assert path.current_length == expected_length
        input_token = dense_step.next_token
    # Independent paths must not append to or overwrite the shared prefill.
    assert all(layer.keys.shape[2] == 7 for layer in snapshot.layers)


@pytest.mark.parametrize("budget_fraction", [0.5, 1.0])
def test_offline_incremental_quest_matches_rebuild_across_page_boundary(
    tiny_runner: GPTNeoXDecodeRunner,
    budget_fraction: float,
) -> None:
    snapshot = tiny_runner.dense_prefill(torch.tensor([[1, 7, 3, 9, 2, 5, 4]]))
    states = [
        tiny_runner.initialize_state(
            snapshot,
            strategy=DecodeStrategy.QUEST,
            budget_fraction=budget_fraction,
            quest_page_size=4,
            quest_metadata_update_mode=mode,
        )
        for mode in (
            QuestMetadataUpdateMode.FULL_REBUILD,
            QuestMetadataUpdateMode.INCREMENTAL,
        )
    ]
    # Teacher forcing isolates each path from token-choice divergence.
    for token in (6, 8, 10, 12):
        oracle, incremental = [
            tiny_runner.step(state, torch.tensor([[token]])) for state in states
        ]
        assert torch.equal(incremental.next_token_logits, oracle.next_token_logits)
        assert torch.equal(incremental.next_token, oracle.next_token)
        for actual, expected in zip(incremental.layers, oracle.layers, strict=True):
            for name in (
                "query",
                "attention_output",
                "attention_weights",
                "residual_output",
            ):
                assert torch.equal(getattr(actual, name), getattr(expected, name))
            assert actual.selection is not None and expected.selection is not None
            for name in ("indices", "scores", "valid_mask"):
                left, right = (
                    getattr(actual.selection, name),
                    getattr(expected.selection, name),
                )
                assert (left is None) == (right is None)
                if left is not None:
                    assert torch.equal(left, right)
        for left, right in zip(states[0].layers, states[1].layers, strict=True):
            assert left.cache is not None and right.cache is not None
            assert isinstance(left.cache.index, QuestIndex)
            assert isinstance(right.cache.index, QuestIndex)
            assert torch.equal(
                left.cache.index.metadata.minimum, right.cache.index.metadata.minimum
            )
            assert torch.equal(
                left.cache.index.metadata.maximum, right.cache.index.metadata.maximum
            )
