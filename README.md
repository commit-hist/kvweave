# KVDB

Experimental indexing and retrieval infrastructure for LLM KV caches.

See `DESIGN.md` for the current technical hypothesis and architecture.

> Status: early research prototype.

KVDB is currently validating whether storage and retrieval strategies can share
a small interface without erasing the properties that make each strategy
useful. It is not an inference framework.

## Phase 0 foundation

The initial internal KV layout is:

```text
[B, Hkv, S, D]
```

The current decode-query layout is `[B, Hkv, D]`. Selection indices and scores
use `[B, Hkv, K]`, so each batch item and KV head receives its own token
selection. Model-native layouts and grouped-query attention will be handled by
future integration code, not by the core package.

The only retrieval implementation in this phase is an exact dot-product Top-K
oracle over synthetic or caller-provided tensors. Quest and PQ retrieval are not
implemented yet.

## Development

```bash
python -m pip install -e '.[test]'
pytest
python benchmarks/scripts/brute_force.py
```

The benchmark is a smoke test for the harness. Its output is not a performance
claim. Quest research provenance and the independent implementation boundary
are recorded in `docs/RESEARCH.md`.
