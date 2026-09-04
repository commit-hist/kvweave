# KVWeave

KVWeave explores whether different KV-cache retrieval and indexing algorithms
can share a common infrastructure layer for transformer inference.

> **Status: Experimental Research Preview**
>
> KVWeave is a correctness-first research project. Current reference
> implementations are not optimized for production inference, and current CPU
> timings are not performance claims.
>
> **KVWeave is experimental research software. APIs may change without notice.**
> Benchmark results from reference implementations should not be interpreted as
> production performance claims.

## Why KVWeave?

Long-context transformer inference increasingly uses KV eviction, sparse token
retrieval, page retrieval, approximate nearest-neighbor (ANN) and product-
quantized search, clustering, hierarchical indexing, and storage offload. These
techniques often ship as isolated research implementations.

KVWeave investigates whether they can instead share one small boundary between
an inference runtime and KV storage/retrieval strategies. It is not an inference
engine, vector database, or replacement for vLLM or SGLang.

## Architecture

```text
LLM runtime
    |
    v
 KVWeave
    |
 +--+--+
 |     |
Quest  PQ
 |     |
 +--+--+
    |
 Storage
    |
Attention
```

- The index chooses **what** cached positions participate.
- Storage determines **where** the corresponding K/V tensors live and fetches
  them.
- Model integration determines **how** the selected K/V is consumed by
  attention.

The current canonical tensor layout is `[B, Hkv, S, D]`. Selection remains
per batch item and KV head, with an explicit validity mask when page expansion
produces different candidate counts.

## Current capabilities

Implemented:

- Quest-style page retrieval as an independent, readable PyTorch reference;
- PQ-style token retrieval as an independent, deterministic PyTorch reference;
- a common `KVIndex` / `KVStorage` / `KVCache` path;
- rectangular masked retrieval for logically ragged selections;
- synthetic retrieval and attention correctness benchmarks;
- post-RoPE Q/K and unchanged-V validation on pinned Pythia-410M activations;
- explicit stateful GPT-NeoX autoregressive decode after dense prefill;
- exact append-time Quest metadata maintenance with the full rebuild retained as
  an internal oracle;
- dense and 100%-retrieval correctness controls; and
- opt-in component, operator, allocation, and tensor-traffic profiling.

Not yet implemented or productionized:

- optimized attention/retrieval kernels, including custom CUDA, Triton,
  MLX, or Metal kernels;
- vLLM or SGLang runtime integration;
- GQA or MQA retrieval policy/integration;
- long-context real-model validation beyond the tested Pythia context; or
- a stable public API or optimized end-to-end inference runtime.

## Quick start

KVWeave requires Python 3.11 or newer. Repository development is driven by the
pinned Mise/Pants toolchain. The default test suite is offline and does not
download a model.

```bash
mise install
mise run test
mise run package
```

`mise run package` creates the wheel and source distribution under `dist/`.
Install the wheel in a virtual environment to use KVWeave outside the
repository build graph.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install dist/kvweave-0.0.1-py3-none-any.whl
```

A minimal synthetic retrieval using the current experimental public surface:

```python
import torch

from kvweave import KVCache, QuestIndex, TensorStorage

torch.manual_seed(0)
keys = torch.randn(1, 2, 16, 4)
values = torch.randn(1, 2, 16, 4)
query = torch.randn(1, 2, 4)

cache = KVCache(index=QuestIndex(page_size=4), storage=TensorStorage())
cache.build(keys, values)
retrieved = cache.retrieve(query, budget=8)

print(retrieved.keys.shape)  # torch.Size([1, 2, 8, 4])
```

The root package explicitly exports `KVCache`, `Selection`, `RetrievedKV`,
`BruteForceIndex`, `QuestIndex`, `PQIndex`, and `TensorStorage`. This surface is
intentional but unstable. GPT-NeoX integration and profiling helpers remain
experimental submodule APIs and are not exported from the package root.

Formatting and lint checks use the same pinned toolchain:

```bash
mise run lint
```

## Research status

- **Phase 0:** reviewed the Quest and PQCache papers, official repositories,
  and licensing/provenance boundaries.
- **Phase 1:** validated Quest-style page selection and full-attention
  convergence on deterministic synthetic tensors.
- **Phase 2:** validated independent PQ-style token retrieval through the same
  storage and attention boundary.
- **Phase 3A:** validated real Pythia activations, replicated structural
  diagnostics, and preserved a negative result for cheap adaptive policy
  prediction.
- **Phase 3B:** validated stateful multi-token decode and exact dense recovery at
  100% retrieval; partial retrieval showed compounding error.
- **Phase 4:** profiled the unchanged reference path and identified measured
  bottlenecks. It did not implement an optimization or establish a speedup.
- **Phase 5A:** validated exact incremental Quest metadata maintenance as a
  narrow optimization experiment. All oracle selections and decode outputs
  remained bit-exact in the pinned matrix; broader Quest and PQ optimization
  has not begun.

See [DESIGN.md](DESIGN.md) for the architecture and phase plan,
[docs/RESEARCH.md](docs/RESEARCH.md) for detailed evidence and provenance, and
[benchmarks/README.md](benchmarks/README.md) for reproducible experiment
commands and evidence boundaries.

## Important limitations

- Retrieval and decode paths are readable CPU/PyTorch reference
  implementations, not production kernels.
- Real-model validation currently covers one small standard-MHA model:
  `EleutherAI/pythia-410m` at one pinned revision.
- Pythia-410M has a 2,048-token context limit, so these experiments do not
  establish long-context production behavior.
- GQA and MQA are not supported by the current model integration.
- There is no optimized end-to-end runtime or production serving integration.
- Partial retrieval can introduce errors that compound across layers and
  autoregressive decode steps.
- Reference CPU timings are for diagnosis and reproducibility, not speed or
  throughput claims.
- Append-time Quest metadata still uses ordinary eager PyTorch and copies its
  small contiguous metadata tensors to preserve simple ownership semantics.

## Roadmap

Near-term research directions, without dates or promised outcomes:

- evaluate any further narrow Quest experiment separately against the next
  measured bottleneck while preserving all correctness and quality controls;
- run the independently scoped Phase 5B PQ experiment only after its own
  protocol is reviewed;
- validate longer-context models and additional hardware configurations; and
- evaluate additional hardware backends only when profiling supports them.

Possible later work includes GQA, MLX/Metal, CUDA/Triton, and inference-runtime
integration. None is a commitment to production readiness or a particular
speedup.

## Attribution and acknowledgments

KVWeave independently implements and studies published ideas. Quest-style page
retrieval originates with **QUEST: Query-Aware Sparsity for Efficient
Long-Context LLM Inference** by Tang et al. PQ-style retrieval is studied in the
context of **PQCache: Product Quantization-based KVCache for Long Context LLM
Inference** by Zhang et al., building on standard product quantization.

The real-model work uses EleutherAI's Pythia project through Hugging Face
Transformers. KVWeave is an independent project and does not imply endorsement,
affiliation, or authorship of those algorithms and projects. Exact papers,
repositories, revisions, licenses, and implementation boundaries are recorded
in [docs/RESEARCH.md](docs/RESEARCH.md).

## License

KVWeave is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE)
for the project copyright notice.

## Contributing

Contributions are welcome within the experimental research scope. See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Suspected
vulnerabilities must follow [SECURITY.md](SECURITY.md), and community
participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Citation
metadata is available in [CITATION.cff](CITATION.cff).
