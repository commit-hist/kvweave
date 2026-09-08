"""Validate ranking evidence helpers independently of the downloaded model."""

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from benchmarks.decode import build_dense_trace
from benchmarks.pq_selection import TieStatistics, compare_methods, assert_tensor_exact
from benchmarks.scripts.phase5b_pq_partial_selection import correctness_replay
from kvweave.integrations.transformers import GPTNeoXDecodeRunner


def test_tie_statistics_distinguish_boundary_from_cut() -> None:
    scores = torch.tensor(
        [[[5.0, 2.0, 5.0, 2.0, 5.0, 2.0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]]
    )
    stats = TieStatistics()
    stats.add(scores, 3)
    report = stats.report()
    assert report["fraction_searches_with_any_tie"] == 1
    assert report["fraction_rows_with_any_tie"] == 0.5
    assert report["fraction_rows_with_boundary_tie"] == 0.5
    assert report["fraction_searches_cutting_boundary_tie"] == 0
    stats.add(scores, 2)
    assert stats.report()["fraction_searches_cutting_boundary_tie"] == 0.5
    stats.add(scores, 6)
    assert stats.report()["fraction_searches_cutting_boundary_tie"] == 1 / 3


def test_benchmark_alternatives_match_large_boundary_tie() -> None:
    scores = torch.tensor([[[5.0, 2.0, 5.0, 2.0, 5.0, 2.0]]])
    result = compare_methods(scores, 2, repetitions=2, prototypes=True)
    assert result["exact_ids_and_scores"]
    assert len(result["timing_ms"]) == 5
    assert all(row["count"] == 2 for row in result["timing_ms"].values())


def test_exact_comparison_detects_signed_zero() -> None:
    with pytest.raises(AssertionError, match="bytes differ"):
        assert_tensor_exact(torch.tensor([0.0]), torch.tensor([-0.0]), "zeros")


@pytest.mark.parametrize("budget", [0.5, 1.0])
def test_complete_capture_replay_with_tiny_model(budget: float) -> None:
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
        runner = GPTNeoXDecodeRunner(GPTNeoXForCausalLM(config).eval())
    snapshot = runner.dense_prefill(torch.tensor([[1, 7, 3, 9, 2, 5, 4, 6]]))
    tokens, _, steps = build_dense_trace(runner, snapshot, generated_tokens=32)
    result, quality, scores = correctness_replay(
        runner, snapshot, tokens, steps, fixture_id="tiny", budget=budget
    )
    assert result["passed"]
    assert result["layer_steps"] == 62
    assert len(quality) == 31
    assert (
        scores is None
    )  # The representative score sample is pinned to real-model layer 12.
    assert len(set(result["hashes"].values())) == 1
