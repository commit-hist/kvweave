# KVDB

Experimental indexing and retrieval infrastructure for LLM KV caches.

See `DESIGN.md` for the current technical hypothesis and architecture.

> Status: early research prototype.

KVDB is currently validating whether storage and retrieval strategies can share
a small interface without erasing the properties that make each strategy
useful. It is not an inference framework.

## Phase 1 reference validation

The initial internal KV layout is:

```text
[B, Hkv, S, D]
```

The current decode-query layout is `[B, Hkv, D]`. Selection indices, scores, and
optional validity masks use `[B, Hkv, K]`, so each batch item and KV head
receives its own token selection. Model-native layouts and grouped-query
attention will be handled by future integration code, not by the core package.

Current retrieval implementations are an exact dot-product Top-K oracle and an
independent readable PyTorch Quest-style page index. Both operate on synthetic
or caller-provided tensors through the same `KVIndex` interface. PQ retrieval is
not implemented yet.

## Development

```bash
python -m pip install -e '.[test]'
pytest
python benchmarks/scripts/brute_force.py
python benchmarks/scripts/quest_reference.py
```

The brute-force script is a harness smoke test. The Quest script compares exact
Top-K/full attention with page selection/selected attention on deterministic
synthetic tensors. Their output is not a performance or model-quality claim.
Quest research provenance and the independent implementation boundary are
recorded in `docs/RESEARCH.md`.
