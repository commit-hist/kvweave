"""Pinned decode benchmark controls shared by the Phase 3B–5A entry points."""

from typing import Any

import torch

from kvweave.integrations.transformers import (
    DecodeStrategy,
    DensePrefillSnapshot,
    GPTNeoXDecodeRunner,
    GPTNeoXDecodeStep,
)
from kvweave.metrics import relative_l2_error
from kvweave.metrics.reference import selection_mask


DEFAULT_MODEL_ID = "EleutherAI/pythia-410m"
DEFAULT_MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
DEFAULT_TRANSFORMERS_VERSION = "5.15.1"
DEFAULT_TRANSFORMERS_REVISION = "550d7b3834670483a4df436541272c055dc364bf"


def build_dense_trace(
    runner: GPTNeoXDecodeRunner,
    snapshot: DensePrefillSnapshot,
    *,
    generated_tokens: int,
) -> tuple[list[int], list[torch.Tensor], list[GPTNeoXDecodeStep]]:
    state = runner.initialize_state(snapshot, strategy=DecodeStrategy.DENSE)
    generated = [int(snapshot.next_token_logits.argmax(dim=-1).item())]
    logits = [snapshot.next_token_logits]
    steps: list[GPTNeoXDecodeStep] = []
    for _ in range(1, generated_tokens):
        input_token = torch.tensor(
            [[generated[-1]]],
            dtype=torch.int64,
            device=snapshot.input_ids.device,
        )
        step = runner.step(state, input_token)
        steps.append(step)
        logits.append(step.next_token_logits)
        generated.append(int(step.next_token.item()))
    return generated, logits, steps


def validate_hugging_face_generation(
    model: Any,
    input_ids: torch.Tensor,
    *,
    generated_tokens: int,
    custom_tokens: list[int],
    custom_logits: list[torch.Tensor],
) -> dict[str, Any]:
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=generated_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    hugging_face_tokens = generated.sequences[0, -generated_tokens:].tolist()
    if len(hugging_face_tokens) != generated_tokens:
        raise RuntimeError("Hugging Face generation stopped before requested length")
    if hugging_face_tokens != custom_tokens:
        first_difference = next(
            index
            for index, (dense, reference) in enumerate(
                zip(custom_tokens, hugging_face_tokens, strict=True)
            )
            if dense != reference
        )
        raise RuntimeError(
            "custom dense decode diverged from Hugging Face at generated position "
            f"{first_difference}: custom={custom_tokens[first_difference]}, "
            f"hf={hugging_face_tokens[first_difference]}"
        )
    if len(generated.scores) != len(custom_logits):
        raise RuntimeError("Hugging Face did not return one score tensor per token")
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for custom, reference in zip(custom_logits, generated.scores, strict=True):
        torch.testing.assert_close(custom, reference, rtol=1e-4, atol=1e-5)
        maximum_absolute_error = max(
            maximum_absolute_error,
            float((custom.float() - reference.float()).abs().max().item()),
        )
        maximum_relative_error = max(
            maximum_relative_error,
            relative_l2_error(custom, reference, dtype=torch.float32),
        )
    return {
        "passed": True,
        "generated_tokens": generated_tokens,
        "token_sequence_exact_match": True,
        "logit_rtol": 1e-4,
        "logit_atol": 1e-5,
        "maximum_logit_absolute_error": maximum_absolute_error,
        "maximum_logit_relative_error": maximum_relative_error,
    }


def assert_full_budget_step(
    approximate: GPTNeoXDecodeStep,
    dense: GPTNeoXDecodeStep,
    *,
    rtol: float,
    atol: float,
) -> None:
    torch.testing.assert_close(
        approximate.next_token_logits,
        dense.next_token_logits,
        rtol=rtol,
        atol=atol,
    )
    if not torch.equal(approximate.next_token, dense.next_token):
        raise AssertionError("100% retrieval changed the greedy next token")
    for approximate_layer, dense_layer in zip(
        approximate.layers,
        dense.layers,
        strict=True,
    ):
        if approximate_layer.selection is None:
            raise AssertionError("100% approximate path did not expose a selection")
        if not approximate_layer.newest_token_included:
            raise AssertionError("100% selection omitted the newest token")
        expected = torch.arange(
            approximate_layer.sequence_length,
            device=approximate_layer.selection.indices.device,
        )
        valid_mask = selection_mask(approximate_layer.selection)
        for head_index in range(valid_mask.shape[1]):
            actual = approximate_layer.selection.indices[0, head_index][
                valid_mask[0, head_index]
            ]
            if not torch.equal(actual, expected):
                raise AssertionError("100% selection did not contain causal KV exactly")
        torch.testing.assert_close(
            approximate_layer.attention_output,
            dense_layer.attention_output,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            approximate_layer.residual_output,
            dense_layer.residual_output,
            rtol=rtol,
            atol=atol,
        )
