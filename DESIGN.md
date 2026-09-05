# KVWeave — Technical Design

## Status

**Experimental Research Preview**

This document describes the current implemented reference architecture and
retains future ideas where they help explain the original design direction.

The architecture is intentionally minimal and should evolve based on empirical results.

Repository maintenance keeps benchmark support outside the runtime contracts:
shared environment/timing, decode controls, report statistics, and artifact
publication live under `benchmarks/`. Strategy-independent synthetic recall
and attention controls live under `src/kvweave/metrics/`, with compatibility
imports at their historical Quest location. The default offline suite exercises
the full decoder using a tiny randomly initialized model; pinned Pythia tests
remain opt-in research controls.

Phases 0–3 validated research provenance, synthetic Quest/PQ behavior, shared
interfaces, real Pythia activations, and stateful multi-token decode. Phase 4
completed profiling of that unchanged path. Phase 5A then validated exact
incremental Quest metadata maintenance as the first narrow optimization
experiment; other Quest and PQ optimization remains future work. All current
timing and approximation results remain correctness/diagnostic evidence, not
production-runtime, speedup, or general model-quality claims.

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

KVWeave investigates whether they can share a common infrastructure layer.

---

# 2. Hypothesis

KVWeave's central hypothesis is:

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

KVWeave is not initially:

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

KVWeave:

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

The current reference implementation is `TensorStorage`, which retains tensors
on their existing PyTorch device. Potential future implementations include:

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
    ) -> "RetrievedKV":
        ...
```

Potential implementations:

```text
CPUStorage
GPUStorage
PinnedCPUStorage
SSDStorage
```

Separate CPU/GPU/offloaded storage policies remain future work.

---

## Selection

Algorithms should return a common selection representation.

Current form:

```python
@dataclass
class Selection:
    indices: torch.Tensor
    scores: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
```

The optional mask is required by demonstrated Phase 1 behavior: independently
ranked page sets can contain the partial final page for some batch/head entries
but not others, producing different valid candidate counts. The rectangular
index tensor uses valid in-range placeholders only, while `valid_mask` defines
which entries are semantically selected and exposes the actual counts.

`Selection` remains the index result: it describes which sequence positions
were selected. Storage consumes it but does not add algorithm-specific page
IDs, scores, budgets, or page counts to the retrieved tensors.

Future representations may include:

```text
token IDs
page IDs
ranges
hierarchical nodes
```

Do not generalize prematurely.

---

## RetrievedKV

Storage returns a minimal mask-preserving representation:

```python
@dataclass
class RetrievedKV:
    keys: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor | None = None
```

Keys and values have rectangular shape `[B, Hkv, K, D]`. When valid candidate
counts differ across batch items or KV heads, `valid_mask` has shape
`[B, Hkv, K]` and false positions are padding that must not participate in
attention. `valid_mask=None` means every retrieved position is valid.

Rectangular tensors plus an explicit validity mask are preferred now because
they preserve batch/head tensor operations and device placement while handling
the ragged counts already demonstrated by partial Quest pages. Ragged Python
lists would move shape handling and per-row loops into storage consumers and
make reference attention less representative of tensor execution. This is a
single retrieval result type, not the start of a generic tensor-container
hierarchy.

Algorithm-specific metadata remains on algorithm results such as
`QuestSearchResult`; it does not belong in `RetrievedKV`.

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
    ) -> RetrievedKV:
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

KVWeave established the following canonical internal layout in Phase 1:

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

The current Quest reference follows the paper-level per-head form. One query
corresponds to one KV head. Queries
therefore have shape:

```text
[B, Hkv, D]
```

Page selection is independent for every batch item and KV head. Grouped-query
attention (GQA), including query-head aggregation or shared selection within a
KV-head group, is explicitly out of scope for this reference implementation.
Those choices belong at the future model-integration boundary.

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

A partial final page is an ordinary indexed page. `QuestIndex` neither drops it
nor force-includes it; its score alone determines whether it is selected.

For reproducibility, KVWeave ranks equal page scores by ascending page ID. Selected
pages otherwise follow descending score order, and token IDs within each page
are ascending. This is a KVWeave reference policy, not a claim of upstream tie
compatibility.

The reference implementation should favor readability over kernel efficiency.

---

# 10. Quest Data Structures

Current representation:

```python
@dataclass
class QuestMetadata:
    minimum: torch.Tensor
    maximum: torch.Tensor
    page_size: int
    sequence_length: int
```

Phase 1 dimensions are:

```text
minimum: [B, Hkv, P, D]
maximum: [B, Hkv, P, D]
```

where `P = ceil(S / page_size)`. Metadata must retain the original sequence
length and page size so every page's valid token range, including a short final
page, can be reconstructed without treating padding as data.

Dense prefill constructs this representation with the unchanged full metadata
build. During append-only Quest decode, the index updates only the affected
final page: a partial page takes elementwise minima/maxima with the new key,
while an append after a full page creates one metadata page. The implementation
returns replacement tensors and a new frozen metadata value so earlier state is
not mutated; the full build remains available as a decode-time oracle. This is
Quest-specific state and does not extend `KVIndex` with a generic mutable-index
protocol.

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

Product quantization is the second implemented reference strategy.

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

Phase 2 used this implementation to validate the algorithm and abstraction
boundary.

## Phase 2 reference evidence

The independent PQ reference now partitions canonical keys `[B, Hkv, S, D]`
into `M` equal contiguous subspaces, trains codebooks
`[B, Hkv, M, C, D / M]`, and encodes keys as centroid IDs
`[B, Hkv, S, M]`. Query lookup-table scoring produces approximate raw dot
products `[B, Hkv, S]` and selects exactly the requested number of individual
tokens per batch item and KV head.

PQ operates through the existing `KVIndex.build/search -> Selection ->
TensorStorage.fetch -> RetrievedKV -> reference attention` path. It required no
changes to `KVIndex`, `Selection`, `KVStorage`, `RetrievedKV`, `KVCache`, or
reference attention. Normal PQ selections have `valid_mask=None`; that is
already the common representation for a rectangular, all-valid result. At full
budget, PQ ranking is a permutation of every sequence token, so fetched
attention reproduces full attention independently of quantization quality.

This evidence is limited to deterministic synthetic reference behavior. It
does not validate the complete PQCache system, model quality, packed-code
memory savings, offload scheduling, GQA policy, or optimized latency.

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
requested token count
```

at the public `KVIndex` boundary. Quest converts a positive requested token
budget to a page budget with:

```text
num_pages_to_select = ceil(token_budget / page_size)
```

and caps that value at the number of indexed pages. A request greater than or
equal to the sequence length selects every page and recovers every valid token.
Because page selection is indivisible and the final page may be partial, Quest
must separately expose the actual number of valid candidate tokens selected for
each batch item and KV head. It must not report the requested token budget as
the actual count unless they are equal.

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

The current reference path avoids rewriting model attention kernels.

Instead:

```text
model produces/query provides Q
             │
             ▼
           index
             │
          Selection
             │
             ▼
       storage.fetch()
             │
             ▼
 RetrievedKV(K, V, valid_mask)
             │
             ▼
reference attention
```

Reference selected attention masks invalid logits to negative infinity before
softmax and excludes invalid values from the weighted sum. Rows with no valid
retrieved token are rejected before softmax, preventing silent NaNs. The dense
path needs no mask when every retrieved position is valid.

This allows independent validation while preserving the retrieval semantics
across the storage boundary.

Later integrations may fuse:

```text
retrieval
+
KV fetch
+
attention
```

for performance.

The standalone `QuestIndex` remains model-agnostic. The following are
integration or decode-runtime policies, not intrinsic search behavior:

* forced inclusion of the newest, possibly partial page;
* dense attention in early transformer layers;
* RoPE placement and selection of the full-oracle versus incremental Quest
  metadata-update path; and
* GQA query-head aggregation or shared-selection policy.

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

## Phase 3A evidence boundary

The first real-model adapter targets standard-MHA GPT-NeoX. It observes the
model's fused QKV projection and rotary-embedding state, constructs Q/K after
the model's partial RoPE transform, leaves V unchanged, and converts all three
to canonical `[B, Hkv, S, D]`. For one query position `t`, the adapter exposes
query `[B, Hkv, D]` and the causal K/V prefix `0..t` only.

This adapter is intentionally outside `KVIndex`, storage, and cache
coordination. Real Pythia activations required no change to `KVIndex`,
`Selection`, `KVStorage`, `RetrievedKV`, or `KVCache`. Quest and PQ therefore
continue through the same selection and storage path established synthetically.

The activation experiment does not patch model attention. An independent eager
attention reconstruction must match the model's attention output before any
retrieval measurements are accepted. Retrieval indexes rank unscaled raw dot
products; the positive model scale does not change that ranking. Reference
attention applies the model scale, causal semantics, float32 softmax, and value
aggregation separately.

Phase 3A does not support perplexity, downstream quality,
generation-equivalence, end-to-end latency, or speedup claims. Stateful
approximate decode/generation integration was subsequently implemented and
validated in Phase 3B under the separate evidence boundary below.

## Phase 3B evidence boundary

The experimental GPT-NeoX decode runner performs dense Hugging Face prefill,
then explicitly executes each one-token layer step using the model's existing
embeddings, layer norms, fused QKV projection, partial RoPE, output projection,
parallel residual, MLP, final norm, and LM head. Model-specific control remains
under `integrations/transformers`; neither Hugging Face generation globally nor
the model class is patched.

Every layer retains canonical causal KV `[B, Hkv, S, D]`. After the new post-
RoPE K and unchanged V are appended, the historical Phase 3B Quest path
rebuilds its readable page metadata and PQ assigns the new key to the codebooks
trained on dense-prefill keys.
Those PQ codebooks remain frozen during decode. Both paths continue through the
existing index, `Selection`, tensor storage, and retrieved-KV boundary. Stateful
decode required no change to `KVIndex`, `Selection`, `KVStorage`, `RetrievedKV`,
or `KVCache`.

The integration, not either index, force-includes the newest token by replacing
the final ranked candidate when necessary. It then sorts valid token IDs into
causal order before storage fetch and attention. This is a reference runtime
policy inspired by decode integrations, not a mathematical requirement or a
change to Quest/PQ ranking.

On the pinned Pythia-410M model, all custom dense traces matched Hugging Face
greedy generation, and every 100% Quest/PQ decode step recovered every causal
token and matched dense logits, attention outputs, residual streams, and token
choices within the recorded tolerance. Partial budgets produced material,
strategy-, layer-, head-, fixture-, and step-dependent errors; free-running
divergence was substantially worse than teacher-forced disagreement. Phase 3B
therefore validates the architectural capability, not the adequacy of any
partial-retrieval quality policy.

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

## Current reference and completed profiling

Phases 1–3 use readable PyTorch reference implementations. Phase 4 profiled the
CPU/float32 Pythia-410M decode path on the recorded Apple Silicon machine and
selected specific future experiments; it did not choose or implement an
optimization backend.

## Future optimization backends

Only after a correctness-preserving before/after experiment and new profiling,
potential backends may include:

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
cache.save("repo-context.kvweave")

cache = KVCache.load("repo-context.kvweave")
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

A future KVWeave mode may distinguish:

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
KVWeave implementation status
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

KVWeave status:
Reference implementation in progress.

Differences from upstream:
Independent implementation based on paper description.
```

---

# 26. Performance Claims

README claims must include enough information to reproduce them.

Bad:

> KVWeave is 5× faster.

Good:

> On MODEL / GPU / 64K context / batch=1 / specified retrieval budget, KVWeave Quest retrieval reduced measured attention latency from X ms to Y ms in commit ABC.

Never extrapolate benchmark results beyond tested conditions.

---

# 27. Phase Plan

## Phase 0 — Research (complete)

* review Quest
* review PQCache
* record licenses
* document algorithms
* choose baseline tensors/model

## Phase 1 — Quest Reference (complete)

* canonical tensor format
* brute-force oracle
* page metadata
* Quest retrieval
* correctness tests
* benchmark harness

Exit condition:

> Full-vs-Quest benchmark runs reproducibly.

## Phase 2 — PQ Reference (complete)

* PQ encoder
* codebooks
* approximate search
* recall benchmark
* KV integration

Exit condition:

> Quest and PQ operate through the same KVIndex interface.

The reference implementation and synthetic end-to-end tests now satisfy this
exit condition without modifying the shared interfaces. Model-level and
optimized-runtime questions remain for later phases.

## Phase 3A — Real-Activation Validation (complete)

* minimal Hugging Face GPT-NeoX adapter
* post-RoPE real Q/K/V extraction
* causal single-query retrieval experiment
* independent full-attention reconstruction
* per-layer/per-head recall, attention-mass, and output-error measurements

Exit condition:

> Quest and PQ preserve their common path on validated real activations, and
> every full-budget experiment recovers the complete causal KV set.

## Phase 3B — Decode/Generation Integration (complete)

* complete: explicit one-token GPT-NeoX decode path after dense prefill
* complete: dense/Hugging Face generation equivalence
* complete: teacher-forced and free-running Quest/PQ evaluation
* complete: reference timing and memory accounting

Phase 3B must not inherit quality or speed claims from Phase 3A activation
metrics.

Exit evidence:

> Stateful multi-token decode preserved the shared KVWeave abstraction and exact
> full-budget behavior. Partial retrieval errors can compound. Phase 4
> subsequently profiled this proven path without assuming that one fixed sparse
> setting was quality-safe or choosing an optimization backend.

## Phase 4 — Profiling (complete)

The accepted reference decode path was profiled without changing ranking,
decode, storage, attention, or model-integration semantics. The measured Quest
target is metadata rebuild; the measured PQ target is stable full-score
ranking. These are inputs to future experiments, not completed optimizations or
speedup claims.

## Phase 5A — Exact incremental Quest metadata (complete)

The unchanged full build remains the oracle. Initial prefill uses that build;
each decode append either updates the existing final page extrema or adds one
new metadata page. The pinned Phase 4 Quest matrix produced bit-exact metadata,
selection, attention, residual, and logit equivalence. Same-run metadata,
retrieval, and total-decode medians improved, with absolute timing drift against
the historical Phase 4 artifact explicitly retained as a limitation. No shared
interface or root public API changed.

## Phase 5 — Further optimization (future; not started)

Potential:

```text
Triton
CUDA
C++
Rust
Metal
```

## Phase 6 — Runtime Integration (future; not started)

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

If that works convincingly, KVWeave graduates from an interesting idea into a real systems project.
