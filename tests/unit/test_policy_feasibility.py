import json

import pytest
import torch

from benchmarks.policy_feasibility import (
    CANDIDATE_CONFIGURATIONS,
    DEVELOPMENT_FIXTURES,
    FEATURE_NAMES,
    FORBIDDEN_INFERENCE_FEATURES,
    HELD_OUT_FIXTURES,
    LOCKED_FIXTURE_TEXT_SHA256,
    LOCKED_TOKEN_ID_SHA256,
    LogisticModel,
    assemble_policy_examples,
    audit_feature_schema,
    build_key_feature_state,
    choose_configuration,
    error_regret,
    extract_pre_retrieval_feature_rows,
    fixture_text_sha256,
    index_memory_bytes,
    maintained_key_metadata_bytes,
    mass_regret,
    oracle_gap_recovery,
    predict_logistic,
    train_logistic_model,
    validate_fixture_lock,
)


def feature_rows() -> list[dict[str, object]]:
    query = torch.tensor([[[1.0, -2.0], [-1.0, 3.0]]])
    keys = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]],
                [[-2.0, 1.0], [1.0, 1.0], [0.0, -1.0]],
            ]
        ]
    )
    return extract_pre_retrieval_feature_rows(
        query,
        build_key_feature_state(keys),
        text_fixture_id="fixture",
        sequence_length=4,
        query_position_label="75_percent",
        query_position=2,
        layer_id=12,
    )


def retrieval_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    records = []
    for row in rows:
        for budget in (0.125, 0.25, 0.5):
            for index, candidate in enumerate(CANDIDATE_CONFIGURATIONS):
                strategy, configuration = candidate.split(":", maxsplit=1)
                records.append(
                    {
                        **{
                            field: row[field]
                            for field in (
                                "text_fixture_id",
                                "sequence_length",
                                "query_position_label",
                                "layer",
                                "head",
                            )
                        },
                        "budget_fraction": budget,
                        "strategy": strategy,
                        "configuration": configuration,
                        "attention_mass_captured": 0.4 + index * 0.1,
                        "relative_attention_output_error": 0.8 - index * 0.1,
                        "candidate_recall": 0.2 + index * 0.1,
                        "normalized_attention_entropy": 0.3,
                    }
                )
    return records


def test_held_out_fixture_content_and_token_hashes_are_complete_and_locked() -> None:
    validate_fixture_lock("held_out")
    held_out_ids = {fixture.fixture_id for fixture in HELD_OUT_FIXTURES}

    assert len(held_out_ids) == 8
    assert held_out_ids == set(LOCKED_FIXTURE_TEXT_SHA256)
    assert held_out_ids == set(LOCKED_TOKEN_ID_SHA256)
    assert all(
        set(lengths) == {512, 2048} for lengths in LOCKED_TOKEN_ID_SHA256.values()
    )
    assert all(
        fixture_text_sha256(fixture) == LOCKED_FIXTURE_TEXT_SHA256[fixture.fixture_id]
        for fixture in HELD_OUT_FIXTURES
    )


def test_development_and_held_out_fixture_membership_is_disjoint() -> None:
    development_ids = {fixture.fixture_id for fixture in DEVELOPMENT_FIXTURES}
    held_out_ids = {fixture.fixture_id for fixture in HELD_OUT_FIXTURES}
    development_text_hashes = {
        fixture_text_sha256(fixture) for fixture in DEVELOPMENT_FIXTURES
    }
    held_out_text_hashes = {
        fixture_text_sha256(fixture) for fixture in HELD_OUT_FIXTURES
    }

    assert development_ids.isdisjoint(held_out_ids)
    assert development_text_hashes.isdisjoint(held_out_text_hashes)


def test_feature_extraction_is_deterministic_and_contains_no_labels() -> None:
    first = feature_rows()
    second = feature_rows()

    assert first == second
    audit_feature_schema(first)
    assert set(FEATURE_NAMES).isdisjoint(FORBIDDEN_INFERENCE_FEATURES)
    assert all(set(row).isdisjoint(FORBIDDEN_INFERENCE_FEATURES) for row in first)


def test_feature_audit_rejects_post_hoc_attention_label() -> None:
    row = feature_rows()[0]
    row["attention_mass_captured"] = 0.9

    with pytest.raises(ValueError, match="forbidden inference"):
        audit_feature_schema([row])


def test_policy_examples_join_labels_after_preserving_feature_boundary() -> None:
    rows = feature_rows()
    examples = assemble_policy_examples(rows, retrieval_records(rows))

    assert len(examples) == len(rows) * 3
    assert all(
        set(example["features"]).isdisjoint(FORBIDDEN_INFERENCE_FEATURES)
        for example in examples
    )
    assert all(
        example["mass_oracle_configuration"] == CANDIDATE_CONFIGURATIONS[-1]
        for example in examples
    )


def test_tie_breaking_uses_frozen_candidate_order() -> None:
    tied = {configuration: 0.5 for configuration in CANDIDATE_CONFIGURATIONS}

    assert choose_configuration(tied, maximize=True) == CANDIDATE_CONFIGURATIONS[0]
    assert choose_configuration(tied, maximize=False) == CANDIDATE_CONFIGURATIONS[0]


def test_regret_and_oracle_gap_recovery_definitions() -> None:
    assert mass_regret(0.9, 0.7) == pytest.approx(0.2)
    assert error_regret(0.4, 0.1) == pytest.approx(0.3)
    assert oracle_gap_recovery(0.8, 0.6, 0.9) == pytest.approx(2.0 / 3.0)
    assert oracle_gap_recovery(0.9, 0.9, 0.9) is None


def test_logistic_model_serialization_and_predictions_are_deterministic() -> None:
    rows = feature_rows()
    # Duplicate rows with one query statistic perturbed to make two label classes.
    training = [dict(rows[0]), dict(rows[1]), dict(rows[0]), dict(rows[1])]
    training[2]["query_mean"] = float(training[2]["query_mean"]) + 1.0
    training[3]["query_mean"] = float(training[3]["query_mean"]) - 1.0
    labels = [0, 1, 0, 1]

    first = train_logistic_model(
        training,
        labels,
        budget_fraction=0.125,
        l2_penalty=0.001,
        epochs=20,
        seed=4,
    )
    second = train_logistic_model(
        training,
        labels,
        budget_fraction=0.125,
        l2_penalty=0.001,
        epochs=20,
        seed=4,
    )
    restored = LogisticModel.from_json(json.loads(json.dumps(first.to_json())))

    assert first.weights == second.weights
    assert restored == first
    assert predict_logistic(first, training) == predict_logistic(restored, training)


def test_feature_and_index_metadata_costs_are_explicit() -> None:
    assert maintained_key_metadata_bytes(head_dimension=64) == (64 + 4) * 4 + 8
    memory = index_memory_bytes(
        sequence_length=512,
        head_dimension=64,
        num_heads=16,
        num_layers=1,
    )

    actual = memory["actual_reference_bytes"]
    logical = memory["logical_packed_pq_bytes"]
    assert actual["quest_p16_plus_p64"] == (actual["quest_p16"] + actual["quest_p64"])
    assert actual["all_four"] == (
        actual["quest_p16_plus_p64"] + actual["pq_m2_c4_plus_m4_c8"]
    )
    assert logical["pq_m2_c4"] < actual["pq_m2_c4"]
