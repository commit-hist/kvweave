import torch

from benchmarks.phase4 import (
    allocation_estimates,
    build_layer_step_records,
    rank_retrieval_bottlenecks,
    tensor_traffic_estimates,
)
from kvweave import PQIndex, QuestIndex
from kvweave.profiling import (
    ComponentProfiler,
    aggregate_component_timings,
    estimate_tensor_bytes,
    profile_component,
    profile_context,
)


def test_component_timer_records_nested_context_and_aggregates() -> None:
    profiler = ComponentProfiler()
    with (
        profiler.activate(),
        profile_context(
            phase="decode",
            strategy="quest",
            layer=3,
        ),
    ):
        with profile_component("quest.metadata.page_minimum"):
            torch.arange(4).sum()
        with profile_component("quest.metadata.page_minimum"):
            torch.arange(2).sum()

    assert len(profiler.records) == 2
    assert all(record.duration_ms >= 0 for record in profiler.records)
    assert profiler.records[0].context == {
        "phase": "decode",
        "strategy": "quest",
        "layer": 3,
    }
    summary = aggregate_component_timings(
        profiler.records,
        group_fields=("strategy", "layer"),
    )
    assert len(summary) == 1
    assert summary[0]["component"] == "quest.metadata.page_minimum"
    assert summary[0]["call_count"] == 2
    assert summary[0]["median_ms"] >= 0
    assert summary[0]["p95_ms"] >= summary[0]["median_ms"]


def test_byte_estimates_and_accounting_preserve_shapes() -> None:
    assert estimate_tensor_bytes((1, 2, 3), torch.float32) == 24
    allocations = allocation_estimates(
        strategy="pq",
        sequence_length=17,
        budget_fraction=0.5,
        batch_size=1,
        heads=2,
        head_dimension=8,
        selection_width=9,
        pq_subspaces=2,
        pq_centroids=4,
    )
    codes = next(row for row in allocations if row["name"] == "pq_appended_code_tensor")
    assert codes["shape"] == [1, 2, 17, 2]
    assert codes["estimated_bytes"] == 1 * 2 * 17 * 2 * 8
    traffic = tensor_traffic_estimates(
        strategy="pq",
        sequence_length=17,
        batch_size=1,
        heads=2,
        head_dimension=8,
        selection_width=9,
        pq_subspaces=2,
        pq_centroids=4,
    )
    assert all(
        row["estimated_total_bytes"]
        == row["estimated_read_bytes"] + row["estimated_write_bytes"]
        for row in traffic
    )


def test_layer_aggregation_keeps_retrieval_and_model_compute_separate() -> None:
    profiler = ComponentProfiler()
    with (
        profiler.activate(),
        profile_context(
            phase="decode",
            fixture_id="fixture",
            strategy="quest",
            budget_fraction=0.5,
            decode_step=1,
            layer=0,
        ),
    ):
        with profile_component("quest.metadata.page_minimum"):
            pass
        with profile_component("model.qkv_projection"):
            pass
    rows = build_layer_step_records(
        profiler.records,
        [
            {
                "fixture_id": "fixture",
                "strategy": "quest",
                "budget_fraction": 0.5,
                "decode_step": 1,
                "layer": 0,
                "total_layer_time_ms": 1.0,
            }
        ],
    )
    assert len(rows) == 1
    categories = rows[0]["component_categories_ms"]
    assert categories["metadata_rebuild"] >= 0
    assert categories["qkv_projection"] >= 0
    assert categories["miscellaneous_layer_overhead"] >= 0


def test_profiling_scopes_do_not_change_quest_or_pq_selection() -> None:
    torch.manual_seed(11)
    keys = torch.randn(1, 2, 16, 8)
    query = torch.randn(1, 2, 8)
    for index in (
        QuestIndex(page_size=4),
        PQIndex(
            num_subspaces=2,
            num_centroids=4,
            max_iterations=3,
            seed=0,
        ),
    ):
        index.build(keys)
        plain = index.search(query, budget=8)
        profiler = ComponentProfiler()
        with profiler.activate(), profile_context(phase="test"):
            profiled = index.search(query, budget=8)
        assert torch.equal(profiled.indices, plain.indices)
        assert torch.equal(profiled.scores, plain.scores)
        if plain.valid_mask is None:
            assert profiled.valid_mask is None
        else:
            assert torch.equal(profiled.valid_mask, plain.valid_mask)
        assert profiler.records


def test_bottleneck_ranking_uses_measured_component_time() -> None:
    rows = []
    for strategy, dominant in (("quest", "metadata_rebuild"), ("pq", "ranking_topk")):
        categories = {
            "metadata_rebuild": 4.0 if strategy == "quest" else 0.0,
            "query_page_scoring": 1.0,
            "page_ranking": 0.5,
            "page_to_token_expansion": 0.25,
            "frozen_append": 2.0,
            "lookup_table_construction": 1.0,
            "approximate_score_reconstruction": 3.0,
            "ranking_topk": 5.0 if strategy == "pq" else 0.0,
            "newest_token_inclusion": 0.5,
            "causal_reordering": 0.5,
            "storage_fetch_gather": 1.0,
        }
        rows.append(
            {
                "strategy": strategy,
                "budget_fraction": 0.5,
                "component_categories_ms": categories,
            }
        )
        assert dominant in categories
    ranked = rank_retrieval_bottlenecks(rows)
    assert ranked["quest"][0]["component"] == "metadata_rebuild"
    assert ranked["pq"][0]["component"] == "ranking_topk"
