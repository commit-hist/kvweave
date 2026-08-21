# KVDB — Technical Design

## Status

**Experimental**

This document describes the initial architecture for KVDB.

The architecture is intentionally minimal and should evolve based on empirical results.

---

# 1. Problem

Transformer inference stores key/value tensors for previous tokens in the KV cache.

For long contexts, this cache becomes expensive in:

* GPU memory
* memory bandwidth
* attention computation
* storage
* transfer cost

Traditional attention effectively asks:

> Which previous KV entries matter for the current query?

At sufficiently large context lengths, this begins to resemble an information-retrieval problem.

Recent research explores multiple ways to avoid attending over every cached token:

* eviction
* token selection
* page selection
* sparse attention
* clustering
* approximate nearest-neighbor retrieval
* product quantization
* hierarchical indexing
* offloaded KV storage

These approaches currently tend to be implemented as isolated research systems.

KVDB investigates whether they can share a common infrastructure layer.

---

# 2. Hypothesis

KVDB's central hypothesis is:

> KV-cache retrieval can be decomposed into storage, indexing, retrieval policy, and inference integration sufficiently cleanly that multiple research algorithms can operate behind one common interface.

If true, this enables something analogous to a vector-search engine specialized for transformer KV state.

---

# 3. Goals

## Primary

Create a reusable KV retrieval abstraction capable of supporting multiple algorithms.

Initially:

* Quest-style page retrieval
* PQCache-style approximate retrieval

## Secondary

Provide reproducible benchmarking infrastructure.

## Long-Term

Potentially support:

* SnapKV
* H2O
* RetrievalAttention
* Squeezed Attention
* ClusterKV
* newer hierarchical retrieval methods

and runtime integrations such as:

* Hugging Face Transformers
* vLLM
* SGLang
* llama.cpp
* MLX

---

# 4. Non-Goals

KVDB is not initially:

* an inference server
* a distributed KV-cache service
* a vector database for document embeddings
* a training framework
* a model implementation
* an agent memory framework
* a RAG library

---

# 5. Conceptual Model

Standard attention:

```text
query
  │
  ▼
┌──────────────────────────────┐
│ every key in the KV cache    │
└──────────────────────────────┘
  │
  ▼
attention
```

KVDB:

```text
                    query
                      │
                      ▼
                 ┌─────────┐
                 │  Index  │
                 └─────────┘
                      │
                candidate IDs
                      │
                      ▼
                 ┌─────────┐
                 │ Storage │
                 └─────────┘
                      │
                 selected KV
                      │
                      ▼
                  attention
```

The important architectural boundary is:

```text
Index chooses WHAT
Storage determines WHERE
Integration determines HOW attention consumes it
```

---

# 6. Components

## KVIndex

Responsible for indexing and candidate selection.

Conceptual interface:

```python
class KVIndex:
    def build(
        self,
        keys: torch.Tensor,
    ) -> None:
        ...

    def search(
        self,
        query: torch.Tensor,
        budget: int,
    ) -> "Selection":
        ...
```

Potential implementations:

```text
QuestIndex
PQIndex
ClusterIndex
HierarchicalIndex
BruteForceIndex
```

---

## KVStorage

Responsible for storage and retrieval of keys and values.

```python
class KVStorage:
    def put(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        ...

    def fetch(
        self,
        selection: "Selection",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

Potential implementations:

```text
CPUStorage
GPUStorage
PinnedCPUStorage
SSDStorage
```

Only CPU/GPU storage should be considered initially.

---

## Selection

Algorithms should return a common selection representation.

Possible initial form:

```python
@dataclass
class Selection:
    indices: torch.Tensor
    scores: torch.Tensor | None = None
```

Future representations may include:

```text
token IDs
page IDs
ranges
hierarchical nodes
```

Do not generalize prematurely.

---

## KVCache

High-level coordinator.

```python
class KVCache:
    def __init__(
        self,
        index: KVIndex,
        storage: KVStorage,
    ):
        ...

    def build(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        ...

    def retrieve(
        self,
        query: torch.Tensor,
        budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

---

# 7. Tensor Layout

Tensor layout should be made explicit throughout the code.

Typical KV tensors may have shapes similar to:

```text
[batch, heads, sequence, head_dim]
```

or:

```text
[batch, sequence, kv_heads, head_dim]
```

KVDB should establish one canonical internal layout during Phase 1.

Suggested:

```text
[B, Hkv, S, D]
```

where:

```text
B   = batch
Hkv = number of KV heads
S   = sequence length
D   = head dimension
```

Adapters should transform model-native layouts into the canonical representation.

---

# 8. Multi-Head Behavior

This is an important design question.

Possible indexing strategies include:

```text
one index per KV head

one shared index across heads

grouped indexes

index over compressed representations
```

Phase 1 should implement the form closest to the chosen Quest reference algorithm.

Do not lock the public API around one strategy.

---

# 9. Quest-Style Index

Quest-style retrieval is the first experimental implementation.

Simplified conceptual algorithm:

```text
KV keys
   │
   ▼
divide sequence into pages
   │
   ▼
compute page statistics
   │
   ▼
query arrives
   │
   ▼
score each page using page statistics
   │
   ▼
select Top-K pages
   │
   ▼
run attention on selected KV
```

Initial page metadata may contain min/max values across keys according to the algorithm being reproduced.

The reference implementation should favor readability over kernel efficiency.

---

# 10. Quest Data Structures

Potential representation:

```python
@dataclass
class QuestMetadata:
    minimum: torch.Tensor
    maximum: torch.Tensor
    page_size: int
```

Example dimensions:

```text
minimum: [Hkv, num_pages, D]
maximum: [Hkv, num_pages, D]
```

Exact dimensions should follow algorithmic requirements.

---

# 11. Brute-Force Baseline

Before evaluating approximate retrieval, implement a brute-force oracle.

Conceptually:

```python
scores = query @ keys.transpose(-1, -2)
topk = scores.topk(k)
```

This is not necessarily equivalent to complete attention output, but it provides a useful retrieval oracle.

Comparisons can include:

```text
Quest candidate recall@K
vs
brute-force candidate top-K
```

This separates:

```text
retrieval error
```

from:

```text
attention approximation error
```

---

# 12. PQ Retrieval

Product quantization is the second planned strategy.

Conceptually:

```text
keys
  │
  ▼
split vector into M subspaces
  │
  ▼
learn codebook for each subspace
  │
  ▼
represent each key using M code IDs
  │
  ▼
query
  │
  ▼
lookup-table distance approximation
  │
  ▼
Top-K KV candidates
```

The first implementation should use an independent, readable PQ implementation.

Do not immediately integrate external ANN libraries unless necessary.

The purpose of Phase 2 is to understand the algorithm and abstraction boundary.

---

# 13. Retrieval Budget

Algorithms expose different natural budgets:

```text
tokens
pages
bytes
percentage of cache
GPU memory
```

The internal system needs a common way to reason about budget.

For Phase 1, use:

```text
selected token count
```

where possible.

Quest may internally convert:

```text
token budget
    ↓
number of pages
```

Future API might support:

```python
RetrievalBudget(
    max_tokens=4096,
)
```

or:

```python
RetrievalBudget(
    max_bytes=512 * 1024 * 1024,
)
```

Do not implement both yet.

---

# 14. Attention Integration

Phase 1 should avoid rewriting model attention kernels.

Instead:

```text
model produces/query provides Q
             │
             ▼
          KVDB
             │
       selected K/V
             │
             ▼
reference attention
```

This allows independent validation.

Later integrations may fuse:

```text
retrieval
+
KV fetch
+
attention
```

for performance.

---

# 15. Model Integration

Initial integration should target a small Hugging Face causal model.

Requirements:

* permissively available model
* manageable on commodity hardware
* supports standard attention
* easy extraction of Q/K/V tensors

Initial correctness tests may use synthetic tensors before any model integration.

Preferred progression:

```text
synthetic tensors
      ↓
single attention layer
      ↓
small transformer
      ↓
real long-context model
```

---

# 16. Benchmark Architecture

Benchmark results should distinguish at least four costs.

## Index Build

```text
keys → index
```

Measure:

```text
seconds
tokens/sec indexed
temporary memory
index memory
```

---

## Retrieval

```text
query → candidate selection
```

Measure:

```text
latency
throughput
candidate recall
```

---

## Attention

```text
selected KV → attention output
```

Measure:

```text
attention latency
attention memory
```

---

## End-to-End

Eventually:

```text
prompt
  ↓
prefill
  ↓
index creation
  ↓
decode loop
  ↓
tokens/sec
```

End-to-end performance is the metric that ultimately matters.

---

# 17. Initial Metrics

Phase 1 benchmark should report:

```text
Context length
Page size
Retrieval budget
Selected percentage
Index build time
Index size
Retrieval latency
Candidate recall@K
Attention-output error
Peak memory where practical
```

Potential error metric:

```python
relative_error = (
    torch.norm(approx_output - full_output)
    / torch.norm(full_output)
)
```

Quality metrics should not rely only on tensor error once model-level testing begins.

---

# 18. Dataset Strategy

Phase 1 should not begin with a huge benchmark suite.

Start with:

### Synthetic data

Useful for correctness and scaling behavior.

Then:

### Small long-context task

Choose one benchmark where relevant context can be identified or model output can be evaluated reproducibly.

Potential later suites include long-context retrieval and reasoning benchmarks, but selecting them should be based on what property we are testing.

Do not add a benchmark simply because it is popular.

---

# 19. Benchmark Matrix

Eventually evaluate:

```text
context:
8K
16K
32K
64K
128K+

budget:
100%
50%
25%
12.5%
6.25%

strategy:
Full
Quest
PQ
others
```

The most interesting plot will likely be:

```text
quality
   ▲
   │            full attention
   │        ●
   │     ●
   │   ●
   │ ●
   └──────────────────────► memory / latency
```

We want the Pareto frontier.

---

# 20. Hardware Strategy

## Phase 1

PyTorch reference implementation.

CPU and/or CUDA depending on available development hardware.

## Phase 2

Profile real bottlenecks.

## Phase 3

Potential optimization backends:

```text
Triton
CUDA
C++
Rust
Metal
MLX
```

Backend decisions must come from benchmark evidence.

---

# 21. Apple Silicon Strategy

Long term, Apple Silicon may be a meaningful differentiator.

The unified-memory model creates a different set of constraints than discrete GPU systems.

Potential opportunity:

```text
very large KV state
     ↓
unified memory
     ↓
indexed retrieval
     ↓
selective attention
```

Possible future backends:

```text
PyTorch MPS
Metal
MLX
```

Do not start here unless the reference implementation is already validated.

---

# 22. Persistence

Persistence is strategically interesting but out of scope initially.

Potential future API:

```python
cache.save("repo-context.kvdb")

cache = KVCache.load("repo-context.kvdb")
```

This becomes particularly valuable for fixed reusable context such as:

```text
code repositories
large documents
agent memory
knowledge bases
```

But persistence should not affect Phase 1 architecture unless necessary.

---

# 23. Reusable Context

A future KVDB mode may distinguish:

```text
static context
+
dynamic context
```

Example:

```text
100K-token codebase
        │
   indexed once
        │
        ├── question 1
        ├── question 2
        ├── question 3
        └── question 4
```

This can amortize index construction and prefill costs.

This idea should be benchmarked separately from normal one-shot inference.

---

# 24. Correctness Strategy

Approximate KV retrieval introduces several possible errors:

```text
candidate selection error
page approximation error
quantization error
attention-output error
generation-quality degradation
```

Tests should isolate them.

Example progression:

```text
exact attention
       ↓
exact Top-K token selection
       ↓
approximate Top-K retrieval
       ↓
page-level retrieval
       ↓
model generation
```

This makes failures diagnosable.

---

# 25. Research Tracking

Create:

```text
docs/RESEARCH.md
```

For each algorithm record:

```text
Name
Paper title
Authors
Venue/year
Paper URL
Official implementation
Implementation license
Algorithm summary
Reported benchmark
Reproduction status
KVDB implementation status
Notes
```

Example:

```markdown
## Quest

Paper:
...

Official implementation:
...

License:
...

KVDB status:
Reference implementation in progress.

Differences from upstream:
Independent implementation based on paper description.
```

---

# 26. Performance Claims

README claims must include enough information to reproduce them.

Bad:

> KVDB is 5× faster.

Good:

> On MODEL / GPU / 64K context / batch=1 / specified retrieval budget, KVDB Quest retrieval reduced measured attention latency from X ms to Y ms in commit ABC.

Never extrapolate benchmark results beyond tested conditions.

---

# 27. Phase Plan

## Phase 0 — Research

* review Quest
* review PQCache
* record licenses
* document algorithms
* choose baseline tensors/model

## Phase 1 — Quest Reference

* canonical tensor format
* brute-force oracle
* page metadata
* Quest retrieval
* correctness tests
* benchmark harness

Exit condition:

> Full-vs-Quest benchmark runs reproducibly.

## Phase 2 — PQ Reference

* PQ encoder
* codebooks
* approximate search
* recall benchmark
* KV integration

Exit condition:

> Quest and PQ operate through the same KVIndex interface.

## Phase 3 — Model Integration

* Hugging Face adapter
* real Q/K/V extraction
* decode experiment
* end-to-end quality evaluation

## Phase 4 — Profiling

Identify bottlenecks.

No optimization backend is chosen before this phase.

## Phase 5 — Optimization

Potential:

```text
Triton
CUDA
C++
Rust
Metal
```

## Phase 6 — Runtime Integration

Potential first targets:

```text
vLLM
SGLang
MLX
```

---

# 28. Key Research Questions

The project should continuously try to answer:

### RQ1

Can multiple KV retrieval algorithms genuinely share one abstraction without significant overhead?

### RQ2

At what context length does indexed retrieval outperform scanning?

### RQ3

What fraction of KV entries must be retrieved to maintain model quality?

### RQ4

Which indexing strategy wins under different:

```text
context lengths
models
hardware
memory limits
query patterns
```

### RQ5

Can index construction be amortized across repeated queries?

### RQ6

Is persistent indexed KV context useful enough to become a standalone systems primitive?

### RQ7

Does Apple unified memory materially change the best KV indexing/storage architecture?

---

# 29. Kill Criteria

We should be willing to stop or substantially change direction.

Reconsider the project if experiments show:

* shared abstractions impose unacceptable overhead
* modern fused attention makes retrieval consistently slower
* retrieval quality requires nearly the full KV cache
* index construction overwhelms savings
* newer inference frameworks already provide equivalent abstractions
* algorithms fail to reproduce meaningful real-world gains

Negative results are useful.

Do not protect the original hypothesis from evidence.

---

# 30. Initial Definition of Success

The first meaningful success is **not stars, downloads, or a large codebase**.

It is this:

```text
One repository
      │
      ├── Full-attention baseline
      │
      └── Quest-style retrieval
                 │
                 ▼
      same tensors / same benchmark
                 │
                 ▼
        reproducible tradeoff
       between retrieval budget,
         quality and latency
```

Then:

```text
add PQ
   ↓
same interface
   ↓
compare Pareto frontiers
```

If that works convincingly, KVDB graduates from an interesting idea into a real systems project.

