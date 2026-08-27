# KVDB — Agent Instructions

## Mission

KVDB is an experimental high-performance indexing and retrieval layer for LLM KV caches.

The long-term hypothesis is:

> Long-context KV-cache management increasingly resembles a database/search problem, and a reusable storage/indexing abstraction can support multiple retrieval strategies across inference runtimes.

KVDB is **not** intended to become another full LLM inference framework.

Our job is to build a focused layer between inference engines and KV storage/retrieval strategies.

---

## Current Phase

We are in **Phase 0 / Phase 1: technical validation**.

Do not optimize for feature completeness.

The immediate goal is to determine whether multiple KV retrieval strategies can operate behind one clean interface without sacrificing the performance characteristics that make those strategies useful.

Initial strategies:

1. Quest-style page-level KV retrieval
2. PQCache-style product-quantized KV retrieval

Initial implementation priority:

1. Correctness
2. Reproducibility
3. Clean abstractions
4. Benchmarkability
5. Performance
6. Integrations

---

## Non-Goals

Do NOT attempt to build all of the following yet:

* full vLLM integration
* full SGLang integration
* llama.cpp integration
* distributed inference
* multi-node KV storage
* production networking layer
* cloud service
* UI
* Kubernetes support
* custom CUDA kernels unless justified by profiling
* every KV compression paper
* training infrastructure
* model serving platform

Avoid framework-building for hypothetical future requirements.

---

# Architecture Principles

## 1. Separate Storage From Retrieval

Storage answers:

> Where are the KV tensors?

Retrieval answers:

> Which KV entries should participate in attention?

These must remain separate abstractions.

Example:

```python
storage = GPUStorage(...)
index = QuestIndex(...)

cache = KVCache(
    storage=storage,
    index=index,
)
```

Do not make retrieval algorithms responsible for persistence or device management unless technically unavoidable.

---

## 2. Separate Model Integration From KVDB Core

Model-specific behavior belongs under integrations.

Core indexing code must not directly depend on a particular Hugging Face model.

Prefer:

```text
integrations/
    transformers/
        llama.py
```

over:

```text
indexes/
    quest/
        llama_quest.py
```

---

## 3. Reference Before Optimization

Each algorithm should ideally have:

```text
reference implementation
        ↓
correctness tests
        ↓
benchmark
        ↓
profiling
        ↓
optimized implementation
```

Do not write CUDA kernels before establishing that the algorithm itself works.

---

## 4. Algorithms Are Plugins, Not Architecture

Quest, PQCache, SnapKV, RetrievalAttention, Squeezed Attention, etc. are strategies.

KVDB is the infrastructure.

The core architecture must not assume that Quest is the canonical method.

---

## 5. Benchmarks Are First-Class Code

Every performance claim must be reproducible.

A benchmark should record at minimum:

* model
* model revision
* dtype
* device
* GPU/CPU hardware
* context length
* generated token count
* retrieval budget
* batch size
* index parameters
* baseline
* latency
* memory usage
* quality/recall metric where applicable
* git commit

Never place an undocumented performance number in README.md.

---

# Repository Structure

Target structure:

```text
kvdb/
├── AGENTS.md
├── DESIGN.md
├── README.md
├── LICENSE
├── pyproject.toml
│
├── src/
│   └── kvdb/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── cache.py
│       │   ├── types.py
│       │   └── interfaces.py
│       │
│       ├── indexes/
│       │   ├── base.py
│       │   ├── quest/
│       │   │   ├── index.py
│       │   │   └── reference.py
│       │   └── pq/
│       │       ├── index.py
│       │       └── reference.py
│       │
│       ├── storage/
│       │   ├── base.py
│       │   ├── cpu.py
│       │   └── gpu.py
│       │
│       ├── integrations/
│       │   └── transformers/
│       │
│       └── metrics/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── correctness/
│
├── benchmarks/
│   ├── README.md
│   ├── configs/
│   ├── scripts/
│   └── results/
│
├── docs/
│   ├── RESEARCH.md
│   ├── BENCHMARKING.md
│   └── algorithms/
│
└── examples/
```

Modify this only when there is a concrete engineering reason.

---

# Initial Public API

Keep the API small.

Possible direction:

```python
from kvdb import KVCache

cache = KVCache(
    index="quest",
    budget_tokens=4096,
)

cache.build(keys, values)

selection = cache.retrieve(query)
```

Internally we may eventually expose:

```python
index.build(keys)
index.search(query, budget)
storage.fetch(selection)
```

Do not stabilize public APIs prematurely.

---

# Core Interfaces

Prefer minimal interfaces resembling:

```python
from typing import Protocol

class KVIndex(Protocol):
    def build(self, keys): ...
    def search(self, query, budget: int): ...


class KVStorage(Protocol):
    def put(self, keys, values): ...
    def fetch(self, selection): ...


class KVCache:
    ...
```

Exact signatures may change based on experiments.

Do not add abstractions solely because they might be useful later.

---

# Phase 1 Milestone

Phase 1 is complete when a fresh checkout can:

```bash
pants lint ::
pants test ::
pants run benchmarks/scripts:quest_reference
```

and the benchmark compares:

```text
full attention
vs
Quest-style page selection
```

using the same model/data.

The output must include at least:

* context length
* retrieval budget
* index build time
* retrieval latency
* selected-token/page recall
* memory estimate
* end-to-end timing where meaningful

---

# Quest Implementation Guidance

Initial Quest implementation should be a readable PyTorch reference version.

Start with:

1. partition KV keys into fixed-size pages
2. calculate per-page statistics needed by the algorithm
3. score pages for a query
4. select top-k pages
5. retrieve corresponding KV entries
6. compare approximate attention against full attention

Correctness test should answer:

> Does increasing the retrieval budget converge toward full attention?

Do not optimize page scoring until this behavior is validated.

---

# PQ Implementation Guidance

PQ is Phase 2 unless specifically requested.

When implemented:

1. establish a brute-force retrieval baseline
2. implement product quantization as a standalone component
3. measure approximate nearest-neighbor recall
4. integrate with KV selection
5. compare against Quest under identical budgets

PQ code should not initially be fused into attention kernels.

---

# Testing Rules

Every algorithm requires:

### Unit tests

Test individual components and shapes.

### Correctness tests

Compare approximate retrieval with brute-force/full-attention behavior.

### Regression tests

Any fixed bug should receive a test when practical.

### Performance tests

Performance benchmarks are separate from normal unit tests.

Tests should avoid downloading large models by default.

---

# Research Integrity

Before implementing an algorithm from a paper:

1. read the paper
2. locate the official repository
3. identify its license
4. record it in `docs/RESEARCH.md`
5. distinguish:

   * algorithm described in paper
   * implementation derived from upstream code
   * independent implementation

Never copy code from an upstream repository without confirming licensing.

Never remove upstream attribution.

Do not imply that KVDB invented algorithms originating in research papers.

---

# Licensing

Preferred project license:

**Apache-2.0**, unless project leadership changes this decision.

Dependencies with restrictive or incompatible licenses must not be introduced silently.

If unsure, document the dependency and stop before incorporating its code.

---

# Dependencies

Prefer:

* Python 3.11+
* PyTorch
* NumPy only where needed
* pytest
* minimal benchmarking dependencies

Avoid adding:

* large orchestration frameworks
* LangChain
* LlamaIndex
* web frameworks
* databases
* distributed systems dependencies

unless the current milestone explicitly needs them.

---

# Coding Style

Use:

* Python type hints
* descriptive variable names
* small functions
* explicit tensor shapes in docstrings where useful
* deterministic seeds in benchmarks/tests
* comments explaining algorithmic reasoning rather than syntax

Prefer:

```python
def compute_page_bounds(
    keys: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

over overly generic helpers.

---

# Performance Work

Never optimize based purely on intuition.

Use:

```text
benchmark
   ↓
profile
   ↓
identify bottleneck
   ↓
optimize
   ↓
benchmark again
```

Potential future optimization targets include:

* vectorized PyTorch
* Triton
* CUDA
* Rust/C++
* AVX2
* AVX-512
* ARM NEON
* Metal
* MLX

But none should be introduced before evidence justifies them.

---

# Apple Silicon

Apple Silicon support is a strategic future objective.

Design assumptions must therefore avoid unnecessary CUDA coupling.

Where possible, maintain:

```text
algorithm
    ↓
tensor/backend abstraction
    ↓
CUDA / CPU / MPS / MLX
```

However, portability must not compromise the initial experiment.

---

# Commit Discipline

Each meaningful commit should represent one coherent change.

Good examples:

```text
feat: add reference Quest page index
test: verify Quest recall approaches full attention
bench: add full-vs-Quest latency benchmark
docs: record PQCache implementation license
```

Avoid commits such as:

```text
updates
fix things
misc
```

---

# Before Making Large Changes

Before implementing something that changes architecture substantially:

1. inspect `DESIGN.md`
2. determine whether the change is needed for the current milestone
3. explain the proposed architecture in the active task
4. choose the smallest implementation that tests the hypothesis

---

# Definition of Done

For every task:

* implementation exists
* tests pass
* relevant documentation is updated
* benchmark is run when performance-related
* no undocumented copied code is introduced
* results are reported accurately
* limitations are stated

Do not report work as successful merely because the code executes.

We care about whether the underlying technical hypothesis survives measurement.
