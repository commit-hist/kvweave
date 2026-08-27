# KVDB

Experimental indexing and retrieval infrastructure for LLM KV caches.

See `DESIGN.md` for the current technical hypothesis and architecture.

> Status: early research prototype.

KVDB is currently validating whether storage and retrieval strategies can share
a small interface without erasing the properties that make each strategy
useful. It is not an inference framework.

## Phase 2 reference validation

The initial internal KV layout is:

```text
[B, Hkv, S, D]
```

The current decode-query layout is `[B, Hkv, D]`. Selection indices, scores, and
optional validity masks use `[B, Hkv, K]`, so each batch item and KV head
receives its own token selection. Model-native layouts and grouped-query
attention will be handled by future integration code, not by the core package.

Current retrieval implementations are an exact dot-product Top-K oracle, an
independent readable PyTorch Quest-style page index, and an independent
deterministic product-quantization token index. All three operate on synthetic
or caller-provided tensors through the same `KVIndex` interface. Quest returns
page-expanded token selections and may use the common validity mask; PQ returns
exact-budget individual token selections without a mask.

The PQ reference uses bounded Lloyd-style K-means, one codebook per contiguous
subspace, nearest-centroid codes, and query-to-centroid lookup-table scoring.
It is intentionally not the complete PQCache runtime: there is no CPU offload,
adaptive overlap policy, GQA aggregation, GPU cache, packed-code kernel, or
model integration.

## Development

Install [Mise](https://mise.jdx.dev/getting-started.html), then let it install
Python and the Pants launcher declared by this repository. `pants.toml` pins the
Pants version, and Pants installs the pinned Ruff tool, so neither Pants nor
Ruff needs a separate project installation.

```bash
mise install
mise run fmt
mise run check
mise run package
mise run bench:brute-force
mise run bench:quest
mise run bench:pq
```

The underlying Pants commands remain available with `mise exec -- pants ...`.

The brute-force script is a harness smoke test. The Quest script compares exact
Top-K/full attention with page selection/selected attention on deterministic
synthetic tensors. Their output is not a performance or model-quality claim.
The PQ script compares Quest and PQ at equal requested token budgets and records
actual selected counts separately. These readable reference timings do not
justify claims that either algorithm is faster or better. Quest and PQCache
research provenance and the independent implementation boundaries are recorded
in `docs/RESEARCH.md`.
