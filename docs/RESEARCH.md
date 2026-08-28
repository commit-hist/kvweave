# Research Notes

This document records the research provenance used by KVWeave. Algorithmic ideas,
observations from upstream implementations, and KVWeave-authored code are kept
separate so that attribution and licensing remain explicit.

## Contents

- [Public-preview provenance and dependency audit](#public-preview-provenance-and-dependency-audit)
- [Phase 4 profiling](#pythia-410m-phase-4-reference-decode-profiling)
- [Phase 3B stateful decode](#pythia-410m-phase-3b-autoregressive-decode-validation)
- [Phase 3A real activations](#pythia-410m-phase-3a-activation-validation)
- [Phase 3A structural replication](#pythia-410m-phase-3a-structural-replication)
- [Phase 3A policy-feasibility negative result](#pythia-410m-phase-3a-policy-feasibility-validation)
- [Phase 2 PQ/PQCache research](#phase-2-research-and-implementation-pqcache)
- [Phase 0/1 Quest research](#phase-0phase-1-research-and-implementation-quest)

## Public-preview provenance and dependency audit

This audit was refreshed on 2026-08-28 against Git commit
`8aefeb8e85876f33dfd74652c5d4d81d3b9b0e91` before the public-preview
documentation changes. It records repository evidence and implementation
boundaries; it is not a legal compatibility opinion.

No model weights, upstream Quest/PQCache source, third-party source trees, or
generated benchmark/profiler artifacts are tracked. Direct Python packages are
depended upon through package metadata or development tooling rather than
copied into KVWeave. The Pants lockfile is a universal resolver lock and lists
platform-specific transitive artifacts, including CUDA packages on platforms
where PyTorch selects them; those artifacts are not bundled in this source
repository.

| Component | Role in KVWeave | Recorded license | Bundled? | Evidence / boundary |
| --- | --- | --- | --- | --- |
| KVWeave | Project source | Apache-2.0 | yes | Canonical license text in [`LICENSE`](../LICENSE) and project notice in [`NOTICE`](../NOTICE). |
| PyTorch (`torch>=2.2`) | Required tensor/runtime dependency | BSD-3-Clause | no | Required by `pyproject.toml`; upstream [license](https://github.com/pytorch/pytorch/blob/main/LICENSE). |
| Transformers (`5.15.1`) | Optional pinned model-experiment dependency | Apache-2.0 | no | Optional extra only; exact source revision and model boundary are recorded in the Phase 3A section below; upstream [license](https://github.com/huggingface/transformers/blob/v5.15.1/LICENSE). |
| EleutherAI Pythia-410M | Opt-in experiment model | Apache-2.0 as declared by the pinned model card | no | Downloaded only for opt-in tests/benchmarks; exact model revision is recorded below. |
| pytest (`>=8,<9`) | Optional/default offline test dependency | MIT | no | `test` extra; upstream [license](https://github.com/pytest-dev/pytest/blob/main/LICENSE). |
| setuptools (`>=77`) | PEP 517 build backend | MIT | no | Build-system dependency; upstream [license](https://github.com/pypa/setuptools/blob/main/LICENSE). |
| Pants (`2.33.0`) and scie-pants (`0.13.2`) | Locked development/build/test orchestration | Apache-2.0 | no | Tool versions are pinned in `pants.toml` and `mise.toml`; upstream [Pants license](https://github.com/pantsbuild/pants/blob/main/LICENSE) and [scie-pants license](https://github.com/pantsbuild/scie-pants/blob/main/LICENSE). |
| Ruff (managed by Pants) | Formatting and linting | MIT | no | Tool environment is managed by Pants; upstream [license](https://github.com/astral-sh/ruff/blob/main/LICENSE). |
| Mise (developer-installed) | Toolchain/task launcher | MIT | no | Required by the documented development workflow but not packaged with KVWeave; upstream [license](https://github.com/jdx/mise/blob/main/LICENSE). |
| Quest paper/repository | Origin of the page-retrieval idea and implementation cross-check | Official repository MIT, with file/submodule caveats | no | KVWeave implementation is independent; the exact upstream revision and exclusions are recorded below. |
| PQCache paper/repository | Research context for PQ-style KV retrieval | No top-level upstream repository license at the inspected revision | no | No upstream PQCache source may be copied or adapted; KVWeave uses an independent standard-PQ implementation based on the paper-level description. |

The existing `NOTICE` intentionally contains only the KVWeave project notice.
Research citation is distinct from attribution for copied or redistributed
software: studying a paper or depending on an installed package does not by
itself add that project's text to KVWeave's `NOTICE`. Any future vendoring,
source adaptation, or binary redistribution requires a new file-by-file notice
and license audit.

## Pythia-410M Phase 4 reference decode profiling

### Frozen methodology and evidence boundary

Phase 4 profiles the accepted Phase 3B execution path without optimizing or
changing it. The fixed matrix is the pinned Pythia-410M model revision
`9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`, Transformers 5.15.1 source
revision `550d7b3834670483a4df436541272c055dc364bf`, CPU, float32, eager/reference
execution, exact 1,024-token `technical_exposition` and `code_like` fixtures,
32 generated-token positions with the accepted 31 explicit decode steps,
teacher forcing only, Quest p64, PQ M4/C8 with eight initial Lloyd iterations
and seed zero, and 50% plus 100% control budgets. The structured gitignored
artifact is `benchmarks/results/pythia-410m-phase4-profile.json`. Three large
raw PyTorch traces are under `benchmarks/results/profile/pythia-410m-phase4/`
and remain uncommitted.

The measurement machine was an Apple M1 Max with 64 GiB physical memory,
macOS 26.6.2 arm64, CPython 3.11.16, and PyTorch 2.13.0. The existing/default
thread configuration was eight intra-op threads and ten inter-op threads. No
thread tuning or lower-thread comparison was performed.

Every measured path followed one complete uninstrumented 31-step warmup replay.
The measured replay then contributed 62 decode-step and 1,488 layer-step
observations per strategy/budget cell across the two fixtures. Named scopes use
`time.perf_counter_ns`; the median calibrated empty-scope cost was 1.292
microseconds total and 0.084 microseconds inside the recorded body. This cost
was not subtracted. CPU execution required no synchronization. Initialization
was measured once per fixture and strategy before the 50% path and excluded
from steady-state distributions.

The instrumentation binds intermediate values and adds scopes around the
existing expressions. It does not change stable full `argsort`, ranking,
newest-token replacement, causal ordering, QKV splitting, partial RoPE,
storage fetch, attention, residual math, or shared interfaces. Each profiled
replay was compared with a separately initialized uninstrumented replay.
Logits, queries, attention outputs, residual streams, and selection IDs/scores/
masks were bit-exact for every one of 310 profiled strategy/budget/fixture
steps. Dense generation matched Hugging Face with zero observed logit error on
both fixtures. Quest and PQ 100% selected every causal token and matched dense
at every step/layer.

The 50% Phase 4 logit metrics also exactly reproduce the matching frozen Phase
3B cells. Quest's 62-step Top-1 agreement, Top-5 overlap, mean relative logit
error, and KL were `0.983871`, `0.706452`, `0.426711`, and `0.050359`; PQ's
were `0.967742`, `0.664516`, `0.514389`, and `0.033720`. The corresponding
100% values were exact. Profiling therefore changed no accepted quality or
correctness result.

### Dense baseline and model/retrieval boundary

Times below are milliseconds per complete 24-layer decode step, summarized
over 62 observations. `Total layers` excludes embedding/rotary setup, final
normalization, LM head, and other step-level work. `Misc.` is the residual
inside layer wall time after all named non-overlapping scopes.

| Dense component | Median | p90 | p95 |
| --- | ---: | ---: | ---: |
| QKV projection | 12.279 | 12.673 | 12.754 |
| RoPE / QK preparation | 2.357 | 2.662 | 2.697 |
| causal K/V append | 7.406 | 9.668 | 9.942 |
| dense attention | 13.727 | 17.577 | 17.699 |
| attention output projection | 5.060 | 5.282 | 5.297 |
| MLP | 31.316 | 32.575 | 32.676 |
| layer norm / residual | 1.382 | 1.709 | 1.722 |
| miscellaneous layer overhead | 2.577 | 3.024 | 3.048 |
| total layers | 76.902 | 83.667 | 85.266 |
| total decode step | 86.175 | 92.827 | 94.964 |

KVWeave retrieval overhead is defined here as index update/rebuild, search,
selection policy, and `TensorStorage.fetch`; selected attention remains model
attention whose cost depends on `K`. Normal model compute is QKV/RoPE, the
common full-precision K/V append, output projection, MLP, layer norms/residuals,
and miscellaneous layer work. This keeps the index/storage boundary separate
from ordinary model computation.

Matched approximate-minus-dense medians show that Quest 50% added 49.561 ms of
retrieval work, changed selected attention by -0.249 ms, changed comparable
normal model work by +5.810 ms, and raised total step wall time by 63.961 ms.
PQ 50% added 50.183 ms of retrieval, changed selected attention by -6.813 ms,
changed comparable normal work by +4.959 ms, and raised total by 50.563 ms.
The matched residuals were below 0.7 ms. These reference deltas describe where
time went; they are not speedup-potential estimates, and medians of individual
components need not add exactly.

### Initialization costs

Quest initial p64 construction across 24 layers had a median total of 6.902 ms
(range 6.858--6.946 ms across the two fixtures). Its per-layer median/p90/p95
was 0.287/0.335/0.366 ms. Per-layer component distributions were:

| Quest initialization component | Median | p90 | p95 | Calls |
| --- | ---: | ---: | ---: | ---: |
| page reshape/padding | 0.004 | 0.012 | 0.014 | 48 |
| page minimum | 0.153 | 0.177 | 0.203 | 48 |
| page maximum | 0.102 | 0.117 | 0.120 | 48 |
| metadata object | 0.005 | 0.008 | 0.013 | 48 |
| validation/miscellaneous | 0.022 | 0.034 | 0.041 | 48 |

PQ initial M4/C8 construction across 24 layers had a median total of 6.937
seconds (6.713--7.161 seconds). Its per-layer median/p90/p95 was
284.127/308.211/311.553 ms. Initial training is not included in decode
overhead.

| PQ initialization component | Median/layer | p90 | p95 | Calls |
| --- | ---: | ---: | ---: | ---: |
| codebook input reshape/allocation | 0.015 | 0.019 | 0.019 | 48 |
| K-means training | 279.611 | 302.225 | 305.581 | 48 layer totals |
| prefill encoding | 2.366 | 2.958 | 3.402 | 48 |
| initial code storage/metadata | 0.384 | 0.501 | 0.544 | 48 |
| validation/miscellaneous | 1.834 | 2.293 | 2.505 | 48 |

Each K-means layer total contains 64 independently timed head/subspace groups;
the full artifact retains their 3,072 atomic calls. The representative largest
initial PQ temporary is the 32 MiB float32 broadcast-difference tensor
`[1,16,1024,4,8,16]`; distances are 2 MiB, persistent int64 codes 512 KiB,
and codebooks 32 KiB per layer. At S=1,024, Quest needs no tail padding and
writes two 64 KiB metadata tensors `[1,16,16,64]` per layer.

### Quest steady-state breakdown

Each row below is summed across all 24 layers for one step. `Share` is the
median fraction of retrieval overhead only; selected attention and total decode
are deliberately outside that denominator. Each budget has 62 step totals and
1,488 underlying layer calls.

| Quest component | 50% median / p90 / p95 | 50% share | 100% median / p90 / p95 | 100% share |
| --- | ---: | ---: | ---: | ---: |
| metadata rebuild | 21.097 / 26.451 / 30.108 | 41.9% | 21.323 / 25.121 / 26.562 | 35.3% |
| query-page scoring | 0.953 / 1.048 / 1.081 | 1.9% | 0.910 / 1.078 / 1.087 | 1.6% |
| page ranking / IDs | 0.892 / 1.019 / 1.050 | 1.8% | 0.905 / 1.044 / 1.053 | 1.6% |
| page-to-token expansion/mask | 5.446 / 5.674 / 5.718 | 11.0% | 8.950 / 9.399 / 9.482 | 15.2% |
| newest-token inclusion | 6.608 / 7.355 / 7.514 | 13.4% | 5.518 / 5.685 / 5.712 | 9.5% |
| causal reordering | 3.818 / 3.942 / 3.949 | 7.5% | 6.926 / 7.365 / 7.416 | 11.6% |
| `TensorStorage` fetch/gather | 10.676 / 12.774 / 14.483 | 21.7% | 14.544 / 18.569 / 20.742 | 24.8% |
| total retrieval overhead | 49.561 / 56.361 / 58.418 | 100% | 58.784 / 66.124 / 69.754 | 100% |
| selected attention | 13.721 / 16.920 / 18.323 | n/a | 12.405 / 17.654 / 17.796 | n/a |
| total decode step | 143.023 / 153.974 / 162.240 | n/a | 144.740 / 166.613 / 170.120 | n/a |

The median Quest layer took 5.413 ms at 50% and 5.794 ms at 100%. Metadata
rebuild was 0.804/0.801 ms per layer, fetch was 0.400/0.518 ms, and selected
attention was 0.572/0.559 ms. The raw artifact retains every step/layer value.

Metadata rebuild itself separates as follows, again as 24-layer step totals:

| Quest metadata operation | 50% median / p90 / p95 | 100% median / p90 / p95 | Atomic calls/cell |
| --- | ---: | ---: | ---: |
| page reshape/padding and copies | 12.906 / 15.968 / 20.250 | 13.011 / 15.857 / 16.510 | 1,488 |
| page minimum | 3.803 / 4.643 / 5.711 | 3.820 / 4.583 / 5.389 | 1,488 |
| page maximum | 3.531 / 5.333 / 6.417 | 3.425 / 4.971 / 5.800 | 1,488 |
| metadata object construction | 0.205 / 0.261 / 0.288 | 0.220 / 0.256 / 0.263 | 1,488 |

At the representative final S=1,055 step, Quest reads an estimated 8,642,560
bytes of full keys twice for min/max and writes 139,264 metadata bytes per
layer, for 8,781,824 logical bytes total. Page scoring moves an estimated
349,248 bytes. At 50%, the rectangular selection width is 576 because nine
p64 pages are selected: K/V gather reads and writes 4,718,592 bytes each
(9,437,184 total), and selected attention consumes 4,718,592 K/V bytes. At
100%, gather traffic rises to 17,285,120 bytes and attention K/V consumption to
8,642,560 bytes. These are analytical logical estimates, not cache or allocator
counters.

Allocation hot spots agree with that traffic. The partial tail creates two
132 KiB padding tensors and two 4.25 MiB padded K inputs per layer; reshape is a
view. Persistent min/max metadata is 68 KiB each. The three score-dimension
intermediates are 68 KiB each. Expanded 50% page-token IDs are 72 KiB plus a
9 KiB mask. Gathered K and V are 2.25 MiB each at 50% and 4.12 MiB each at
100%. The separate operator trace records 421.6 MB of `cat` allocations across
the complete 24-layer Quest step, followed by 108.5 MB in the dominant gather
shape.

The Quest 50% operator trace confirms a copy/reduction/gather workload rather
than a page-scoring bottleneck. Top retrieval-relevant operators by self CPU
time were `cat` 15.316 ms (193 calls including baseline K/V and metadata
copies), `gather` 9.746 ms for the dominant selected-K/V shape, `sort` 4.987
ms for selection/mask ordering, `amin` 3.130 ms, and `where` 2.895 ms. Normal
model `addmm` calls remain large but are outside Quest retrieval overhead.

The 50-to-100 comparison separates budget-independent work: metadata changed
only +1.1%, scoring -4.5%, and page ranking +1.4%, while page expansion rose
64%, causal reordering 81%, and fetch 36%. Total decode was nearly unchanged
in the pooled medians (+1.2%), demonstrating that this reference path does not
turn a smaller selected budget into a measured end-to-end benefit. This is not
an optimized-runtime or speedup claim.

### PQ steady-state breakdown

| PQ component | 50% median / p90 / p95 | 50% share | 100% median / p90 / p95 | 100% share |
| --- | ---: | ---: | ---: | ---: |
| frozen append | 10.154 / 11.824 / 13.206 | 20.4% | 10.563 / 11.608 / 12.131 | 17.2% |
| lookup-table construction | 0.964 / 1.036 / 1.143 | 1.9% | 1.030 / 1.228 / 1.236 | 1.7% |
| approximate score reconstruction | 2.177 / 2.240 / 2.273 | 4.4% | 2.297 / 2.476 / 2.496 | 3.8% |
| stable full-score ranking / IDs | 15.762 / 16.251 / 16.282 | 31.4% | 15.947 / 16.767 / 16.898 | 26.0% |
| newest-token inclusion | 5.047 / 5.183 / 5.218 | 10.0% | 5.608 / 5.971 / 6.009 | 9.3% |
| causal reordering | 5.327 / 5.513 / 5.557 | 10.7% | 10.545 / 11.235 / 11.340 | 17.3% |
| `TensorStorage` fetch/gather | 10.282 / 13.120 / 14.979 | 20.5% | 15.012 / 18.230 / 21.632 | 24.2% |
| total retrieval overhead | 50.183 / 53.845 / 55.577 | 100% | 62.070 / 65.897 / 67.971 | 100% |
| selected attention | 6.968 / 7.510 / 11.265 | n/a | 11.697 / 20.262 / 20.635 | n/a |
| total decode step | 130.953 / 151.199 / 152.946 | n/a | 151.061 / 176.313 / 177.400 | n/a |

The median PQ layer took 5.049 ms at 50% and 5.969 ms at 100%. Ranking was
0.656/0.668 ms per layer, frozen append 0.416/0.428 ms, fetch 0.378/0.551 ms,
and selected attention 0.284/0.503 ms.

Frozen append is mostly reference code concatenation and metadata validation,
not centroid math. At 50%, its 24-layer median is 9.120 ms for code append and
metadata construction, 0.568 ms for centroid distances, 0.233 ms for centroid
assignment, and 0.227 ms for subspace preparation. The corresponding 100%
values are 9.459, 0.604, 0.244, and 0.236 ms. Ranking is almost entirely the
stable full `argsort`: 15.516 ms plus 0.243 ms ID handling at 50%, and 15.595
plus 0.370 ms at 100%. Each atomic operation has 1,488 calls per cell.

At S=1,055, the frozen assignment moves an estimated 37,376 logical bytes per
layer, but the reference int64 code `cat` moves 1,080,320 bytes and scales with
S. Lookup-table construction moves 38,912 bytes. Approximate score
reconstruction moves an estimated 1,147,840 bytes and scales with `H*S*M`.
The 50% K/V gather moves 8,650,752 bytes and attention consumes 4,325,376 K/V
bytes; both double approximately at 100%.

Major per-layer PQ allocations are the 32 KiB append difference tensor, 2 KiB
distances, 527.5 KiB replacement int64 code tensor, 2 KiB lookup table, 66 KiB
approximate score tensor, four 66 KiB subspace lookup results, and 132 KiB
ranked-ID tensor. Gathered K/V are 2.06 MiB each at 50% and 4.12 MiB each at
100%. The operator trace records 103.8 MB in the dominant K/V gather shape,
220.6 MB of `cat` allocations across the full step, and 4.86 MB for the full
score sort shape.

The PQ operator trace makes the selected bottleneck unambiguous: stable sort of
`[1,16,1055]` scores was the top operator at 15.823 ms self CPU time across 24
calls (16.149 ms total). The next retrieval-specific operators were gather at
9.639 ms, `cat` at 8.715 ms, the separate causal-order sort at 4.277 ms, and
full-code validation reductions such as `any` at 3.141 ms. Score reconstruction
did not dominate.

From 50% to 100%, append, lookup construction, score reconstruction, and full
score ranking changed only +4.0%, +6.9%, +5.6%, and +1.2%. They scan or update
the full context independently of K. Causal ordering rose 98%, storage fetch
46%, and selected attention 68%, as expected for K-dependent work. Total decode
rose 15.4%. Full-context ranking at both budgets is direct evidence that this
reference operation erases part of the theoretical sparse-attention benefit.

### Python overhead, scaling, and bottleneck ranking

Separate cProfile replays are diagnostic and excluded from primary timing.
For Quest 50%, `prepare_decode_selection` had 5.954 ms Python self time across
24 calls, `GPTNeoXDecodeRunner.step` 3.788 ms, and page expansion 1.640 ms. For
PQ 50%, preparation used 4.914 ms, `PQMetadata.__post_init__` 3.197 ms, the
runner 3.145 ms, and the Python score loop 1.077 ms. The named-scope/context
machinery itself was below a millisecond per representative step. These
cProfile self times include interpreter/dispatch accounting and must not be
summed with separately profiled native operator times. They show meaningful
control-flow and validation overhead, but the top selected targets are also
confirmed by tensor-operator evidence.

Expected tensor-dimension scaling is:

| Work | Approximate scaling |
| --- | --- |
| Quest metadata rebuild | `O(H*S*D)` |
| Quest query-page scoring | `O(H*ceil(S/P)*D)` |
| Quest page ranking | reference `O(H*(S/P)*log(S/P))` |
| Quest page expansion/policy | `O(H*K)` plus causal sort `O(H*K*log K)` |
| PQ frozen centroid assignment | `O(H*M*C*(D/M)) = O(H*C*D)` |
| PQ reference code append | `O(H*S*M)` |
| PQ lookup table | `O(H*M*C*(D/M)) = O(H*C*D)` |
| PQ score reconstruction | `O(H*S*M)` |
| PQ stable full ranking | reference `O(H*S*log S)` |
| storage fetch / selected attention | `O(H*K*D)` |

These are shape-level expectations, not exact hardware complexities; Python,
validation, allocation, copies, cache behavior, and PyTorch thread scheduling
are material in this reference implementation.

The measured pooled top three, across both budgets and 124 steps per strategy,
are:

1. Quest metadata rebuild: 21.237 ms median, 28.444 ms p95, 38.9% median
   retrieval-overhead share. It is repeated `O(H*S*D)` reference-runtime work;
   an exact incremental formulation should not affect quality semantics.
2. Quest storage fetch/gather: 13.020 ms, 18.968 ms p95, 23.6%. It is shared
   `O(H*K*D)` gather-bound work and is not Quest-specific.
3. Quest page-to-token expansion: 7.075 ms, 9.398 ms p95, 12.1%. It is
   `O(H*K)` reference representation/mask work.

1. PQ stable full-score ranking: 15.846 ms median, 16.765 ms p95, 27.6%
   median retrieval-overhead share. It is `O(H*S*log S)` in this reference and
   repeats independently of K; exact selected IDs and tie semantics are at risk
   if changed carelessly.
2. PQ storage fetch/gather: 13.409 ms, 18.228 ms p95, 23.3%. This is shared
   gather-bound `O(H*K*D)` work.
3. PQ frozen append: 10.303 ms, 12.156 ms p95, 19.1%. This mixes small
   algorithmic centroid assignment with `O(H*S*M)` reference concatenation and
   validation overhead.

### Selected targets, backend fit, and Apple Silicon interpretation

**QUEST TARGET: metadata rebuild.** It is the largest repeated Quest retrieval
component at both budgets, is independent of K, reads the full key cache twice,
and is corroborated by `cat`, `amin`, and `amax` operator timings. The exact
next experiment is an incremental eager-PyTorch semantic oracle that updates
only the newest page after each append, preserves exact metadata and selection
IDs, reruns every Phase 3B/4 correctness gate, and compares before/after
component and total timings against this artifact. That experiment has not
started.

**PQ TARGET: stable full-score ranking.** It is the largest repeated PQ
retrieval component at both budgets and the top retrieval operator by self CPU
time. The exact next experiment is deterministic partial selection against the
existing stable full `argsort`, including explicit ascending-token-ID tie
repair if needed, with bit-exact selection/newest/causal/full-budget controls
before timing. No sort was replaced in Phase 4.

For Quest, portable eager PyTorch incremental state is the best first semantic
experiment because the primary opportunity is eliminating an `O(S)` rebuild,
not choosing a lower-level backend for the same waste. Portable C++ or Rust
with ARM NEON/SIMD is second if the reduced update remains material. MLX/Metal
may fit a later device-resident Apple cache; Triton/CUDA fit a later GPU profile.
For PQ ranking, a PyTorch partial-selection prototype with deterministic tie
repair is the smallest first experiment; portable C++/Rust selection is next,
then MLX/Metal for a device-resident Apple path and Triton/CUDA for a separately
profiled GPU path. `torch.compile` is not selected for either target: the
measured Quest issue is algorithmic full-cache work and the PQ issue is one
specific stable selection operation. These are ranked experiment directions,
not backend commitments.

On this Apple M1 Max CPU, Quest metadata appears primarily allocation/copy- and
bandwidth-bound with two reductions; ARM NEON or portable SIMD could help the
remaining reduction only after incremental state removes the full scan.
Accelerate may help dense linear algebra but is not a direct answer to page
metadata construction. Quest/PQ fetch are gather- and bandwidth-bound. PQ
ranking is sort/selection-bound; NEON is less directly applicable than a
deterministic selection algorithm or library primitive. PQ code append is
allocation/copy/validation-bound. Query-page scoring, PQ lookup construction,
and PQ score reconstruction are small here; claiming Metal or MLX will improve
the profile would be unsupported without a device-resident replay.

### Decision and limitations

Phase 4 required no change to `KVIndex`, `Selection`, `KVStorage`,
`RetrievedKV`, `KVCache`, Quest/PQ ranking, storage semantics, attention, or
model integration architecture. `DESIGN.md` therefore remains unchanged. The
only code changes are opt-in instrumentation, accounting/aggregation helpers,
the fixed profile runner, and offline tests. No external source was copied and
no profiler dependency was added.

Phase 5 is justified only as two narrow before/after experiments against the
selected targets above. It is not yet justified as a backend migration, a
general speedup claim, or simultaneous optimization of other visible costs.
Limitations are one model, one prompt length, two deterministic fixtures, one
Apple CPU/thread configuration, 62 steps per cell, eager reference execution,
timer perturbation, analytical rather than hardware byte counters, and
separate operator/cProfile replays. Partial-retrieval quality remains sensitive;
performance work must preserve the exact quality gates rather than treating
speed as sufficient evidence.

## Pythia-410M Phase 3B autoregressive decode validation

### Frozen boundary and decode architecture

Phase 3B retains the exact Pythia-410M model revision
`9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`, Transformers 5.15.1 source
revision `550d7b3834670483a4df436541272c055dc364bf`, fused-QKV interpretation,
partial RoPE, attention scale `0.125`, Quest ranking, PQ ranking, and shared
KVWeave interfaces accepted in Phase 3A. The full gitignored structured result is
`benchmarks/results/pythia-410m-phase3b-decode.json`; exact dense logits,
per-layer attention outputs, residual streams, and cache lengths are stored in
`benchmarks/results/pythia-410m-phase3b-dense-tensors.pt`, SHA-256
`408c35d4b801bf8fb12b76bc66700d471bf13fa3c1d16ab52e54141decb7162e`.

Dense prompt prefill runs through the pinned model and snapshots each layer's
post-RoPE K and unchanged V. Every later token is executed explicitly through
the model's existing embedding, layer norms, QKV projection, partial RoPE,
attention output projection, parallel residual, MLP, final norm, and LM head.
Hugging Face generation is not globally patched. The first generated token is
selected from dense-prefill logits; a 32-token generation therefore contains
31 one-token retrieval steps.

At each layer, the current K/V entry is appended in causal order before search.
Quest p64 rebuilds page metadata from the complete current key state at every
step. PQ M4/C8 trains eight-iteration codebooks on dense-prefill keys, freezes
those codebooks, and encodes each appended key against them. Encoding new
database vectors against a trained quantizer is standard PQ behavior and also
matches the previously documented upstream observation that ordinary short
decode does not retrain initial codebooks. No PQCache source was copied.

The integration force-includes the newest token, replacing the final ranked
valid candidate only when it is absent, and then sorts valid token positions
into causal order before storage fetch and attention. This is a reference
runtime/Quest-inspired policy, not a mathematical necessity and not part of
either index ranking. At 100%, it is a no-op because every token is present.

The static development-derived layer/head table was deferred: mixing Quest and
PQ independently per head would require both index families plus heterogeneous
selection/fetch assembly, adding a second integration question to the A/B/C
correctness gate. The failed learned query-adaptive predictor was not used and
query-adaptive work remains parked.

### Matrix and correctness gates

The matrix uses four existing Phase 3A development fixtures: narrative prose,
technical exposition, code-like text, and list/table text. Each is repeated and
truncated with the pinned tokenizer to prompt lengths 256, 512, and 1,024. All
12 cases generate 32 tokens without exceeding the model's 2,048-token limit.
Quest p64 and PQ M4/C8 run at 25%, 50%, and 100% in teacher-forced and free-
running modes. This yields 144 approximate runs and 4,464 approximate decode
steps, with 372 steps per strategy/budget/mode cell.

All 12 custom dense traces matched Hugging Face greedy token sequences and
per-step logits exactly at `rtol=1e-4, atol=1e-5`; observed maximum absolute
and relative logit differences were both zero. For each strategy, all 24 full-
budget runs and 744 decode steps selected every valid causal KV token and
included the newest token. Quest and PQ both had zero observed logit, attention-
output, and residual-stream error against dense; attention-mass deviation from
one was at most `2.5034e-6` from float32 summation.

This answers the existential Phase 3B question positively: KVWeave retrieval can
operate in a stateful multi-token loop without architectural failure. Stateful
decode required no change to `KVIndex`, `Selection`, `KVStorage`, `RetrievedKV`,
or `KVCache`. The only strategy-specific addition is PQ code append against
frozen codebooks; it does not change search ranking or the shared protocol.

### Teacher-forced logit results

Means below cover all 372 approximate decode steps in each cell. KL is
`KL(dense || approximate)` in float32.

| Strategy | Budget | Top-1 agreement | Top-5 overlap | Logit cosine | Logit relative error | Mean KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quest p64 | 25% | 0.798 | 0.510 | 0.747 | 0.654 | 0.826 |
| Quest p64 | 50% | 0.927 | 0.660 | 0.853 | 0.479 | 0.294 |
| PQ M4/C8 | 25% | 0.868 | 0.497 | 0.734 | 0.687 | 0.533 |
| PQ M4/C8 | 50% | 0.987 | 0.671 | 0.877 | 0.470 | 0.045 |
| Quest/PQ | 100% | 1.000 | 1.000 | 1.000 | 0 | 0 |

Increasing the budget from 25% to 50% improves every listed mean for both
strategies. No one partial strategy dominates every diagnostic: Quest p64 has
slightly lower 25% logit relative error and higher Top-5 overlap, while PQ has
higher Top-1 agreement and lower KL at both budgets and is clearly stronger on
50% logit metrics.

### Free-running generation divergence

| Strategy | Budget | No divergence | Mean token agreement | Mean longest common prefix |
| --- | ---: | ---: | ---: | ---: |
| Quest p64 | 25% | 1/12 | 0.299 | 6.33 tokens |
| Quest p64 | 50% | 5/12 | 0.518 | 15.92 tokens |
| PQ M4/C8 | 25% | 1/12 | 0.362 | 10.75 tokens |
| PQ M4/C8 | 50% | 8/12 | 0.857 | 27.42 tokens |
| Quest/PQ | 100% | 12/12 | 1.000 | 32.00 tokens |

Quest 25% first divergences occurred at positions 2/3/4/5/6/8 (plus one no-
divergence case); Quest 50% at 2/3/4/5/6/7 (plus five no-divergence cases).
PQ 25% first divergences occurred at 2/6/7/8/9/12/31 (plus one no-divergence
case); PQ 50% at 2/16/26/29 (plus eight no-divergence cases). Some paths later
matched the dense token at individual positions, but this is not state
reconvergence: free-running histories and KV states remain different.

The gap between teacher-forced and free-running results is material. At 50%,
PQ's mean top-1 agreement is `0.987` under dense-token inputs but free-running
sequence agreement is `0.857`; Quest changes from `0.927` to `0.518`.
Teacher forcing is therefore necessary to separate approximation-state damage
from the amplification caused by consuming a different token history.

### Attention, hidden-state, and compounding behavior

Teacher-forced representative layer means were:

| Strategy | Budget | Layer | Attention mass | Attention-output error | Residual-stream error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quest p64 | 25% | 0 / 12 / 23 | 0.675 / 0.725 / 0.792 | 0.315 / 0.705 / 1.121 | 0.107 / 0.640 / 1.098 |
| Quest p64 | 50% | 0 / 12 / 23 | 0.829 / 0.846 / 0.845 | 0.189 / 0.464 / 0.682 | 0.064 / 0.430 / 0.695 |
| PQ M4/C8 | 25% | 0 / 12 / 23 | 0.820 / 0.688 / 0.508 | 0.262 / 0.994 / 2.366 | 0.080 / 0.737 / 1.089 |
| PQ M4/C8 | 50% | 0 / 12 / 23 | 0.951 / 0.877 / 0.661 | 0.083 / 0.539 / 1.482 | 0.032 / 0.334 / 0.657 |

Residual error increases substantially from layer 0 through layer 23, directly
showing error accumulation across layers. Layer 23 remains difficult and head-
specific. Quest p64 captures much more layer-23 mass than PQ M4/C8 at both 25%
(`0.792` versus `0.508`) and 50% (`0.845` versus `0.661`) and has much lower
layer-23 attention-output error. Quest heads 15/7/1 had the lowest 25% layer-23
mass; PQ heads 2/1/13 were lowest, with head 2 mean error `5.847`.

Teacher-forced state error also grows across decode steps, although logit damage
is not monotonic. From decode step 1 to 31, layer-23 residual error changes from
`0.329` to `0.860` for Quest 25%, `0.192` to `0.525` for Quest 50%, `0.721` to
`1.052` for PQ 25%, and `0.452` to `0.606` for PQ 50%. Corresponding logit
relative error changes from `0.244` to `0.596`, `0.146` to `0.440`, `0.504` to
`0.744`, and `0.365` to `0.449`. Free-running step-31 logit relative errors are
larger still: `0.994/0.879/1.110/0.750` in the same order.

Layer-23 attention-output error has moderate teacher-forced correlation with
logit relative error for Quest (`0.459` at 25%, `0.601` at 50%) but weak
correlation for PQ (`0.218`, `0.128`). Late-layer error is therefore predictive
for some paths, not a complete strategy-independent explanation of logit
divergence. These correlations pool fixtures, lengths, and steps and are
descriptive only.

### Reference timing and memory accounting

All timing numbers are unoptimized CPU/Python means and identify work for later
profiling; they are not speedups. Dense prefill means were `255.7`, `443.3`, and
`1,033.0` ms for prompts 256/512/1,024. Dense decode-step means were `71.0`,
`77.9`, and `82.0` ms at those prompt lengths.

Across teacher-forced contexts and steps, total per-step reference costs summed
over all 24 layers were:

| Strategy | Budget | Update/rebuild ms | Search/policy ms | Fetch ms | Attention ms | Remaining model ms | Total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quest p64 | 25% | 17.60 | 14.05 | 6.94 | 8.09 | 66.20 | 112.88 |
| Quest p64 | 50% | 18.45 | 16.65 | 8.94 | 11.16 | 69.12 | 124.31 |
| PQ M4/C8 | 25% | 10.78 | 20.12 | 7.23 | 5.04 | 69.86 | 113.04 |
| PQ M4/C8 | 50% | 9.84 | 21.47 | 7.94 | 6.27 | 69.13 | 114.64 |

Quest initial reference index construction across all layers averaged
`4.60/6.33/6.97` ms for prompt lengths 256/512/1,024. PQ prefill training and
encoding averaged `2.73/4.94/6.87` seconds. Frozen-codebook append reduces PQ
decode updates to roughly 10 ms per step across all layers; retraining at every
step was neither required nor performed.

At the final decode step for a 1,024-token prompt (causal length 1,055), dense
KV is `197.812 MiB` across 24 layers. Quest metadata is `3.188 MiB`; selected KV
is `56.765 MiB` at a requested 25% and `104.596 MiB` at 50%. Page rounding makes
actual mean selected fractions `28.7%` and `52.9%` at this length (and as high
as `40.2%/61.8%` at the 256-token prompt). PQ stores `12.363 MiB` of actual
reference int64 codes, a `0.580 MiB` logical packed-code estimate, and `0.750
MiB` of codebooks; selected KV is `49.5/99.0 MiB` at 25%/50%. These figures
exclude allocator overhead and do not establish production memory savings.

### Evidence decision and exact next experiment

Phase 3B strengthens the central KVWeave hypothesis: two distinct indexes retain
one storage/selection/cache boundary through causal KV growth and multi-token
state evolution, and both exactly recover dense decode at full budget. It also
weakens any claim that a fixed partial budget is automatically generation-safe:
errors compound across layers and steps, and free-running divergence can amplify
otherwise high teacher-forced top-1 agreement.

Phase 4 profiling is justified because the path is now correctness-gated and
the reference breakdown exposes large, distinct costs. Backend optimization is
not yet justified without profiling. The exact proposed next experiment is a
preregistered profiling-only Phase 4 run using the unchanged pinned model,
1024-token technical and code-like prompts, 32 teacher-forced decode tokens,
Quest p64 and frozen-codebook PQ M4/C8 at 50% plus the 100% controls. Profile
per-layer index update, search/policy, fetch, selected attention, and remaining
model computation; record allocations and tensor traffic; then identify one
measured bottleneck per strategy. Do not change retrieval semantics, add a
backend, or claim speedup until those profiles select a target and a subsequent
before/after benchmark preserves every Phase 3B correctness control.

## Pythia-410M Phase 3A activation validation

### Model, license, and dependency provenance

- **Model:**
  [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m),
  current retrained (non-v0) release.
- **Exact model revision:**
  [`9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`](https://huggingface.co/EleutherAI/pythia-410m/tree/9879c9b5f8bea9051dcb0e68dff21493d67e9d4f),
  resolved from `main` and pinned before model download on 2026-08-27. The
  model card says branch `step143000` is the same final checkpoint as `main`.
- **Model license:** Apache-2.0, as declared by the pinned model-card metadata
  and its Model Details section. This is a license for the model release; KVWeave
  does not copy model or Transformers source into the repository.
- **Pinned Transformers experiment dependency:** `transformers==5.15.1`,
  Apache-2.0. Tag `v5.15.1` resolves to source commit
  [`550d7b3834670483a4df436541272c055dc364bf`](https://github.com/huggingface/transformers/tree/550d7b3834670483a4df436541272c055dc364bf).
  It is an optional `model-experiment` dependency, not a core runtime
  dependency.

The pinned downloaded `config.json` and the live `model.config` both verify:

```text
architecture                  GPTNeoXForCausalLM / GPT-NeoX
model_type                    gpt_neox
hidden_size                   1024
num_hidden_layers             24
num_attention_heads           16
head_dimension               64
num_key_value_heads           16 (standard MHA; no GQA/MQA reduction)
max_position_embeddings       2048
rotary_pct                    0.25
rotary_dimensions per head    16
rotary base                   10000
attention scale               64**-0.5 = 0.125
parallel residual             true
```

The raw config does not declare a separate `num_key_value_heads`; GPT-NeoX's
fused projection produces Q, K, and V for all 16 heads. The loaded attention
module's projection has output width `3 * hidden_size`, confirming ordinary
multi-head attention rather than GQA or MQA. No concrete incompatibility was
found, so the requested model choice was retained.

### Extraction, RoPE, causality, and reconstruction

KVWeave does not patch the model's attention path. For selected layers, a forward
hook observes `GPTNeoXAttention.query_key_value`, whose native fused output is
`[B, S, 3 * hidden]`. GPT-NeoX interprets this as `[B, S, H, 3 * D]`, then
transposes and chunks the final dimension into Q/K/V `[B, H, S, D]`. Splitting
the fused output into three contiguous hidden-sized blocks would be wrong and
is covered by an offline unit test.

The model-level rotary module is also observed. Its cosine/sine tensors are
used to independently reproduce Transformers 5.15.1 partial RoPE: only the
leading 16 of each head's 64 Q/K dimensions are rotated, while the remaining
48 dimensions pass through unchanged. V receives no positional transform.
Thus the indexed keys and search queries are the post-RoPE representations that
actually participate in model attention, not raw fused-projection Q/K.

Each tested sequence length receives its own deterministic, unpadded model
forward. For query position `t = S - 1`, Q is sliced to `[B, H, D]` and only
K/V positions `0..t` are exposed to retrieval. Future positions cannot enter
the index or storage. The input is a locally authored text tokenized once and
repeated to exact lengths 256, 512, 1,024, and 2,048; no external dataset is
used. This deliberately narrow activation distribution is a limitation.

The model is forced to eager attention. Independent reconstruction computes
the full causal QK matrix with scale `0.125`, masks future tokens, applies
softmax in float32, multiplies by V, concatenates heads, and applies the model's
dense attention projection. All 12 layer/length checks (layers 0, 12, and 23)
passed at `rtol=1e-4, atol=1e-5`. The worst relative reconstruction error was
`7.4232e-7`; the worst absolute element error was `1.4306e-6`.

Quest, PQ, and exact Top-K still rank unscaled raw QK dot products. Multiplying
every token score for one head/query by the same positive `0.125` does not
change ranking. Only the final attention comparison applies the model scale.

### Phase 3A experiment and observed behavior

The deterministic matrix covers all 16 heads for layers 0, 12, and 23; sequence
lengths 256/512/1,024/2,048; token budgets 12.5%/25%/50%/100%; Quest page sizes
16 and 64; and PQ `(M=2,C=4)` and `(M=4,C=8)` with eight Lloyd iterations and
seed zero. It produces 3,840 per-head records. The structured local result is
`benchmarks/results/pythia-410m-phase3a-reference.json` (benchmark outputs are
gitignored by repository policy).

Across heads, layers, and lengths, the partial-budget mean metrics were:

| Strategy/config | Budget | Candidate recall | Attention mass | Relative output error |
| --- | ---: | ---: | ---: | ---: |
| Exact Top-K | 12.5% | 1.000 | 0.887 | 0.125 |
| Exact Top-K | 25% | 1.000 | 0.943 | 0.067 |
| Exact Top-K | 50% | 1.000 | 0.984 | 0.022 |
| Quest page 16 | 12.5% | 0.328 | 0.467 | 1.134 |
| Quest page 16 | 25% | 0.444 | 0.582 | 0.910 |
| Quest page 16 | 50% | 0.619 | 0.734 | 0.568 |
| Quest page 64 | 12.5% | 0.378 | 0.661 | 0.370 |
| Quest page 64 | 25% | 0.431 | 0.727 | 0.274 |
| Quest page 64 | 50% | 0.606 | 0.843 | 0.140 |
| PQ M2/C4 | 12.5% | 0.282 | 0.318 | 1.325 |
| PQ M2/C4 | 25% | 0.478 | 0.548 | 0.963 |
| PQ M2/C4 | 50% | 0.677 | 0.768 | 0.602 |
| PQ M4/C8 | 12.5% | 0.454 | 0.508 | 1.011 |
| PQ M4/C8 | 25% | 0.567 | 0.655 | 0.756 |
| PQ M4/C8 | 50% | 0.713 | 0.834 | 0.430 |

These averages hide extreme layer/head variation. Layer 23 was much harder for
approximate ranking than layers 0 and 12. For example, Quest page 16 at layer
23 had mean recall/mass/error `0.111/0.139/2.702` at 12.5% budget, while Quest
page 64 had `0.214/0.707/0.325`. Some late-layer heads assigned essentially all
attention mass to tokens selected by page 64 despite low raw Top-K candidate
recall; other heads captured almost no mass and produced very large relative
errors. The sample therefore does not support reporting averages alone.

Budget increases improved mean recall for both Quest and PQ and made attention
mass nondecreasing in every one of 384 strategy/config/context/layer/head
groups. Candidate recall itself was individually monotonic in 306/384 Quest
groups and 347/384 PQ groups because its exact Top-K target also expands with
budget. Output error was nonincreasing in 365/384 Quest groups and 342/384 PQ
groups. Every strategy reached full coverage at 100%.

At equal actual candidate counts, smaller Quest pages did **not** reliably win:
page 16 beat page 64 for candidate recall in 247/528 comparisons, lost 251,
and tied 30. Page 64 slightly more often captured greater attention mass
(264 versus 240, 24 ties) and produced lower output error (268 versus 238, 22
ties). The synthetic tendency toward smaller-page recall therefore did not
survive robustly in this one real-activation sample.

PQ M4/C8 had lower mean key-reconstruction error than M2/C4 (`0.294` versus
`0.340`). At fixed context/layer/head/budget, the higher-reconstruction-quality
configuration improved candidate recall in 459/576 comparisons, attention mass
in 440/576, and output error in 409/576. This is a tendency, not a reliable
per-head rule.

For partial budgets, pooled Pearson correlation between candidate recall and
output error was `-0.398`, `-0.380`, and `-0.331` at 12.5%, 25%, and 50%.
Attention-mass correlation with output error was substantially stronger at
`-0.645`, `-0.705`, and `-0.760`. Attention mass is therefore the more useful
diagnostic in this matrix, but the single repeated corpus is too small for a
general model-quality conclusion. Pooled PQ reconstruction correlations are
confounded by context/layer differences and are not treated as causal evidence.

All 60 full-budget strategy/config/layer/length checks covered every causal KV
token. Ranked full selections permute token order, so float32 reduction order
left a worst per-head relative residual of `8.1063e-4` and worst absolute
residual of `4.4169e-4`; both are within the explicitly recorded `1e-3` and
`5e-4` permutation-equivalence bounds. Quest/PQ full-budget coverage spans all
16 heads in every tested layer and length.

This evidence strengthens the shared-interface hypothesis: no changes were
required to `KVIndex`, `Selection`, `KVStorage`, `RetrievedKV`, `KVCache`,
Quest ranking, PQ ranking, or storage. It weakens any assumption that synthetic
average recall alone predicts real attention behavior. This phase makes no
generation, perplexity, downstream-quality, or speed claim.

## Pythia-410M Phase 3A structural replication

### Replication methodology

The follow-up retained the exact model, revision, `transformers==5.15.1`, eager
attention, fused-QKV interpretation, partial-RoPE construction, causal slicing,
raw-dot-product ranking, attention scale, Quest ranking, PQ ranking, bounded
eight-iteration K-means, and seed zero described above. No external dataset was
downloaded. The structured result is the gitignored local artifact
`benchmarks/results/pythia-410m-phase3a-replication.json`.

Eight independently tokenized local fixtures intentionally vary structure:

| Fixture | Structural purpose |
| --- | --- |
| `repetitive_prose` | recurring nouns, verbs, and clause order |
| `narrative_prose` | chronological characters, places, and changing events |
| `technical_exposition` | definitions, causal claims, and numeric terms |
| `code_like` | Python-like indentation, identifiers, branches, and literals |
| `list_table` | labeled rows, delimiters, fields, and quantities |
| `dialogue_qa` | alternating speakers and explicit questions/answers |
| `mixed_sentence_lengths` | alternating very short and multi-clause sentences |
| `symbolic_pattern` | cyclic markers, symbols, fields, and sparse changes |

Each fixture is tokenized without special tokens, repeated independently, and
truncated deterministically to exactly 512 and 2,048 tokens. The artifact
records the authored text, base token count, repetition count, resulting token
count, and token-ID SHA-256 for every fixture/length pair.

Fractional queries use `ceil(sequence_length * fraction) - 1`, with the causal
prefix including positions `0..t`. Exact positions and valid causal lengths are:

| Captured length | 25% | 50% | 75% | Final |
| ---: | ---: | ---: | ---: | ---: |
| 512 | `t=127`, 128 tokens | `t=255`, 256 | `t=383`, 384 | `t=511`, 512 |
| 2,048 | `t=511`, 512 tokens | `t=1023`, 1,024 | `t=1535`, 1,536 | `t=2047`, 2,048 |

Every causal prefix evaluates layers 0, 12, and 23; all 16 heads; requested
budgets 12.5%, 25%, 50%, and 100%; exact Top-K; Quest page sizes 16 and 64; and
PQ `(M=2,C=4)` and `(M=4,C=8)`. The complete matrix contains 3,072 unique
layer/head/query attention observations, 61,440 strategy/budget records, and
15,360 per-head full-budget invariant records.

### Diagnostic definitions

- Attention entropy is Shannon entropy `-sum(p * ln(p))` of the exact
  full-attention distribution, in natural-log units (nats).
- Normalized entropy is entropy divided by `ln(S_causal)`.
- Effective attention support is `exp(entropy)`, in effective-token units. A
  one-hot distribution therefore has support one; a uniform distribution over
  `S` tokens has support `S`.
- Top-1/Top-4/Top-16 mass is the sum of the largest one/four/sixteen exact
  attention probabilities.
- Quest bound looseness is the Quest upper-bound page score minus the maximum
  exact unscaled `q dot k` token score within that page. Means and maxima are
  recorded separately for selected, non-selected, and all pages.
- PQ score MAE/RMSE compare approximate and exact unscaled token scores.
  Tie-aware Spearman correlation measures rank agreement. The exact top-token
  score error and MAE restricted to the exact Top-16 tokens isolate errors on
  the most important scores.
- Candidate recall, attention mass, per-head relative output error, actual
  candidate count, and full-budget coverage/mass/output invariants retain their
  accepted meanings. No diagnostic is collapsed into a combined score.

Natural entropy varies with valid prefix length, so empirical low/middle/high
strata use terciles of normalized entropy across the 3,072 unique observations.
The boundaries were `0.2739605` and `0.6143728`. These are descriptive strata,
not universal sparsity thresholds.

### Permutation-order correctness regression

The first expanded run found a benchmark-evaluation issue, not an extraction or
retrieval bug. For narrative prose at sequence length 2,048, query 50%, layer
23/head 4, full-budget exact Top-K covered all 1,024 causal tokens and captured
mass `1.000000119`, but strategy-ranked token order changed the float32 value
reduction enough to produce maximum absolute residual `5.796e-4`, just outside
the established `5e-4` bound. A diagnostic continuation also found PQ head 9
with relative residual `1.270e-3` despite full coverage and mass one.

Attention is mathematically invariant to candidate permutation. The benchmark
now retains original strategy order for every ranking/candidate diagnostic but
sorts the same valid selected token IDs into ascending causal order before the
storage fetch and attention reduction. This removes non-semantic reduction
order from the correctness check and partial-budget output comparison. A
regression test covers masked/ragged selection canonicalization. No Q/K/V,
RoPE, causality, attention, index, selection, storage, retrieved-KV, or cache
implementation changed.

All 192 independent full-causal reconstruction checks passed the original
`rtol=1e-4, atol=1e-5`; the worst relative and absolute residuals were
`1.0520e-6` and `4.5300e-6`. All 15,360 full-budget per-head invariants covered
every causal token, captured mass within `1e-5` of one (observed range
`0.99999845` to `1.00000250`), and matched canonical full reference attention
exactly after ordering.

### Attention sparsity and layer variability

The pooled entropy distribution was broad: mean/median `2.692/3.070` nats,
10th/25th/75th/90th percentiles `0.0109/0.7496/4.1646/4.9532`. Effective
support had mean/median `52.47/21.55` tokens and 10th/25th/75th/90th
percentiles `1.011/2.116/64.37/141.62` tokens. Pooled mean Top-1/Top-4/Top-16
mass was `0.435/0.606/0.753`; the corresponding Top-16 10th/median/90th
percentiles were `0.392/0.784/0.9999995`.

| Layer | Entropy mean / median (nats) | Normalized entropy mean / median | Effective support mean / median | Top-1 / Top-4 / Top-16 mean mass |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4.141 / 4.151 | 0.659 / 0.678 | 97.46 / 63.48 | 0.148 / 0.331 / 0.575 |
| 12 | 3.372 / 3.369 | 0.533 / 0.545 | 56.74 / 29.05 | 0.339 / 0.527 / 0.694 |
| 23 | 0.562 / 0.229 | 0.089 / 0.037 | 3.21 / 1.26 | 0.817 / 0.959 / 0.990 |

Layer 23 is therefore qualitatively distinct across the expanded input set,
not merely an average shift: its median exact attention distribution has only
`1.26` effective tokens and Top-1 mass near one.

### Retrieval results and layer-23 replication

Partial-budget pooled means were:

| Strategy/config | Budget | Recall | Attention mass | Relative output error |
| --- | ---: | ---: | ---: | ---: |
| Exact Top-K | 12.5% | 1.000 | 0.890 | 0.128 |
| Exact Top-K | 25% | 1.000 | 0.945 | 0.069 |
| Exact Top-K | 50% | 1.000 | 0.985 | 0.023 |
| Quest p16 | 12.5% | 0.342 | 0.505 | 0.930 |
| Quest p16 | 25% | 0.447 | 0.632 | 0.690 |
| Quest p16 | 50% | 0.627 | 0.791 | 0.398 |
| Quest p64 | 12.5% | 0.381 | 0.643 | 0.456 |
| Quest p64 | 25% | 0.457 | 0.735 | 0.311 |
| Quest p64 | 50% | 0.599 | 0.831 | 0.193 |
| PQ M2/C4 | 12.5% | 0.280 | 0.305 | 1.442 |
| PQ M2/C4 | 25% | 0.471 | 0.538 | 1.012 |
| PQ M2/C4 | 50% | 0.673 | 0.769 | 0.600 |
| PQ M4/C8 | 12.5% | 0.445 | 0.497 | 1.120 |
| PQ M4/C8 | 25% | 0.568 | 0.665 | 0.788 |
| PQ M4/C8 | 50% | 0.718 | 0.813 | 0.491 |

At layer 23 and 12.5% budget, exact Top-K captured at least 90%, 95%, and 99%
mass in `1,020/1,024`, `1,014/1,024`, and `993/1,024` observations. Across the
four approximate configurations, attention mass was below 50%, 75%, and 90%
in `2,633/4,096`, `2,728/4,096`, and `2,813/4,096` observations. Less-than-50%
rates were Quest p16 `767/1,024`, Quest p64 `380/1,024`, PQ M2/C4 `766/1,024`,
and PQ M4/C8 `720/1,024`.

This behavior persisted in every text: exact Top-K reached 99% in 118 to 128
of 128 observations per fixture, while approximate less-than-50% counts ranged
from `250/512` for symbolic patterns to `371/512` for list/table text. Position
dependence was smaller: approximate less-than-50% counts were `697/1,024` at
25%, then `647`, `645`, and `644` at 50%, 75%, and final. Head specificity was
large: head 12 fell below 50% in `232/256` approximate cases, while heads 8 and
11 did so in `133/256`. Thus the late-layer effect is persistent and strongly
head-specific, with real but secondary input and position dependence.

At 12.5%, low-entropy-tercile heads were fragile for every configuration.
Recall/mass/error were `0.175/0.335/1.952` for Quest p16,
`0.244/0.706/0.509` for Quest p64, `0.185/0.275/2.290` for PQ M2/C4, and
`0.243/0.334/2.138` for PQ M4/C8. High-entropy values were respectively
`0.413/0.491/0.437`, `0.487/0.554/0.385`, `0.318/0.295/0.943`, and
`0.547/0.535/0.594`. Low entropy consistently meant lower recall and larger
output error, but not uniformly lower mass: coarse Quest pages often happened
to include the critical sparse token and raised low-entropy mass. This supports
the hypothesis that concentrated heads are cheap for an oracle but fragile for
approximate ranking, without implying entropy alone selects a strategy.

### Quest page size and bound quality

Across 9,216 equal-requested-partial-budget comparisons, p16 versus p64 outcomes
were recall `4,272/4,292/652`, mass `3,988/4,594/634`, and lower output error
`4,376/4,295/545` (p16 better / p64 better / tie). Actual candidate counts were
equal in 7,296 comparisons and differed due to page rounding in 1,920. Entropy
strata reversed the tendency: p64 captured more mass in 1,608 versus 1,074
low-entropy and 1,726 versus 1,198 high-entropy comparisons, whereas p16 won
1,716 versus 1,260 middle-entropy comparisons. A fixed smaller page is not a
reliable winner; coarse pages can help when important tokens are spatially
co-selected, but actual candidate-count differences remain a confound.

Absolute selected-page mean looseness increased sharply by layer. For p16 it
was `76.3`, `150.8`, and `7,156.1` at layers 0/12/23; for p64 it was `98.9`,
`202.6`, and `8,906.7`. Pooled descriptive correlations of selected-page mean
looseness with recall/mass/error were `-0.469/-0.284/+0.322`, consistent with
looser bounds harming retrieval. The within-layer-23 values were much weaker
(`-0.149/-0.093/+0.077`), and layer 0 even reversed sign. Absolute score scale
strongly confounds pooled results. Loose bounds contribute diagnostic signal
but do not by themselves explain the late-layer failures or establish cause.

### PQ capacity and score quality

M4/C8 lowered key-reconstruction error in all 3,072 fixed
text/length/query/layer/head comparisons. Across 9,216 partial-budget retrieval
comparisons it improved recall `7,149` times (M2/C4 improved `1,855`, 212 ties),
mass `7,016` times (`1,990`, 210 ties), and output error `6,421` times (`2,705`,
90 ties). The advantage was strong in middle/high-entropy strata but much less
consistent in the low-entropy stratum, where M4/C8 improved output error only
`1,610/3,072` times.

Mean reconstruction errors for M2/C4 versus M4/C8 were `0.600/0.505` at layer
0, `0.270/0.230` at layer 12, and `0.118/0.108` at layer 23. Better global
reconstruction did not guarantee better query scores: layer-23 score RMSE was
`184.46` for M2/C4 and `196.49` for M4/C8, although Spearman agreement improved
slightly from `0.201` to `0.216` and exact-Top-16 MAE improved from `192.42` to
`184.07`.

Pooled score-rank correlation related more strongly to recall/mass/error
(`+0.647/+0.403/-0.436`) than score RMSE (`-0.458/-0.289/+0.287`). Exact
top-token absolute score error related to mass/error at `-0.363/+0.437`.
Reconstruction-error correlations were mixed or weak after stratifying by
layer, demonstrating why pooled reconstruction correlations are not causal
evidence. PQ score approximation, especially ordering and critical-token
error, explains some failure better than global key reconstruction, but large
unexplained per-head variation remains.

### Correlations, fixed strategies, and policy evidence

For all approximate partial-budget records, candidate-recall correlation with
output error was `-0.398`, `-0.373`, and `-0.299` at 12.5%, 25%, and 50%.
Attention-mass correlation was consistently stronger at `-0.647`, `-0.718`,
and `-0.771`. The same ordering held within every tested layer at 12.5%.
Attention mass remains the better diagnostic of output damage in this matrix.

Pooled entropy correlations are confounded by layer: entropy versus
recall/mass/error was `+0.417/+0.184/-0.363`, while within-layer associations
were weaker and sometimes reversed. These are descriptive Pearson
correlations, not causal estimates.

No single approximate configuration won every head/query. At 12.5%, the best
fixed mean-mass configuration was Quest p64 at `0.643`, while a retrospective
per-head/query oracle over the same four configurations reached `0.763`. Mean
output error fell from the best fixed `0.456` to oracle `0.242`. At layer 23
alone, mass rose from `0.632` to `0.779` and error fell from `0.549` to `0.240`.
All four configurations won at least 244 mass cases and 290 error cases in the
pooled 3,072-observation comparison (ties included). This oracle is unavailable
at runtime and ignores feature and switching cost. It supports a separate
held-out policy-feasibility experiment, not adding an adaptive subsystem to the
architecture yet.

### Architecture result, limitations, and next experiment

The expanded experiment required no change to `KVIndex`, `Selection`,
`KVStorage`, `RetrievedKV`, or `KVCache`; it also required no change to model
extraction, RoPE, causality, Quest ranking, PQ ranking, or reference attention.
`DESIGN.md` therefore remains unchanged. The evidence strengthens the claim
that one fixed approximate strategy/configuration is insufficient across these
layers and heads, but is not yet enough to justify a production adaptive-policy
interface.

Limitations remain substantial: one 410M standard-MHA model; eight authored,
deterministically repeated texts rather than natural corpora; only two captured
lengths, four positions, and three layers; single-query activation analysis;
tiny reference PQ configurations; no sink/local policy; no GQA; no decode or
generation; no perplexity/downstream metric; no optimized kernel; no runtime or
memory-cost comparison; absolute Quest looseness and PQ score errors are scale
dependent; pooled correlations mix known confounders; and the retrospective
oracle uses exact outcome labels unavailable to a real router.

The exact proposed next experiment is a held-out Phase 3A policy-feasibility
test, not decode integration: freeze these four configurations and budgets,
use the current eight fixtures only as development data, author eight new
unseen structural fixtures as a locked test set, and predict the best existing
configuration per layer/head/query using only pre-retrieval features available
without full attention or exact Top-K (layer/head ID, query/key norms, page-bound
score dispersion, PQ approximate-score dispersion, and reconstruction error).
Report held-out regret in attention mass and output error versus both the best
fixed configuration and the unattainable retrospective oracle, plus feature
and index costs. Do not add a public planner/policy interface unless that
prospective held-out test recovers a material fraction of the oracle gap.

## Pythia-410M Phase 3A policy-feasibility validation

### Frozen protocol before held-out evaluation

This experiment asks only whether features available before approximate
retrieval can select among the four already accepted configurations more
effectively than a development-selected fixed choice on unseen structural
inputs after feature/prediction cost. It is not an adaptive planner
implementation and introduces no public/core API.

The candidate order, which is also the exact-tie order, is frozen as Quest p16,
Quest p64, PQ M2/C4, and PQ M4/C8. Budgets are evaluated independently at
12.5%, 25%, and 50%; 100% remains a retrieval correctness control and is not a
training target. The accepted eight structural-replication fixtures are
development-only. Their 16 token-ID hashes are those already recorded by the
accepted replication artifact.

| Development fixture | Text SHA-256 | 512-token ID SHA-256 | 2,048-token ID SHA-256 |
| --- | --- | --- | --- |
| `repetitive_prose` | `d2544673aafde381d44732562d438cd06565a466a73d4d870b20c5e08a77e3fe` | `658e6689ef37d763c66102870376385bf105c30e1c608c7d86fa263fb72529e1` | `98412d0bc89cd39136b581eacf7a02a57f711c4b4b7407a1f395a22d1375bcdb` |
| `narrative_prose` | `74aaa3525e1203e03dd26dea4a753a1d3836ca205e325990e8633f22b5768fd5` | `686c0d89f3f9bec74ae3c548faf3fa2c0b313ca8924c9f0c2416a11f6c97b12f` | `720056b5d4fa562d4042525482363751c9c0b4b9998be06d83c86dfb10de6cc4` |
| `technical_exposition` | `a78c8b7e5cd1fc6fe7a092328dafca9925c7ad487940c2de96b76df9b5356a95` | `6cb41a7b11378f53d95194813ced063fbd550a95866b1ee9c10ca2375b89469c` | `a98b382aeda74271f776a1d8d4f5fa4e5c974a63382c21bb0a33c9273d71aed3` |
| `code_like` | `99fa8ec18a43006ec2f41f7b4cc691a775c77410713adf44fec3e5b0ea75c4b7` | `b9b95960afe007960454bf1a78665d4d9332564517464bce78b155226b997722` | `53c9270f4bd15582d2bfa41ec56bc1695046a179716ea8fe7b27e5b864e4ff01` |
| `list_table` | `d91616050fc016fab0e7e5fac34c7ac6eac017cd3c2d89f4a2c549c077e52e61` | `7f8e21cf06b63bda989354f4c74457ee461fcbe8f1ec476e70d413f98cd12d3c` | `8272d9e74860a45e6e300f630661dc2455d17f36cf2f7a95f6eeb00f3db5e9ca` |
| `dialogue_qa` | `9acb52a774d59e5484ae47760a4d991dd94af87ea8cb7642913fea2d8f4a2ea4` | `2b2ecb74ea7dc9d041be0f0ed7368bb4d6539ab6d6c49f56fc72bd6f93810b88` | `0098661f312dcbdfcf333ffc91170f90c81727292ce2143e77c2f14e090d2729` |
| `mixed_sentence_lengths` | `9612825285b92ca8f5dad948c9755735370cd1e15762b14f74856371e4d1e02a` | `22441d6cad39fc364376c5cae2ec13e44c105be6862ee2d3e8edf3a4f22110a1` | `64c342b5772a33ec1e42fd9ba221a99251de2a8b1080283ce86f6474c519a88a` |
| `symbolic_pattern` | `ed13132ca3566670ba1340c6edefae02763750bdd40183f03c5cb622632cab7d` | `d16064facf4c0259d61810d0ba548ec2dc7c19fd676b619ec2e9f6bc5ba6ef60` | `fc1aeb5421ff5d149a20991044f9adcfd9923c8791405095055b14d74d9513e5` |

Before any held-out outcome was generated, eight independently authored inputs
were frozen by UTF-8 text hash and pinned-tokenizer token-ID hash:

| Held-out fixture | Structural purpose | Text SHA-256 | 512-token ID SHA-256 | 2,048-token ID SHA-256 |
| --- | --- | --- | --- | --- |
| `contract_cross_references` | legalistic clauses, definitions, exceptions, and cross-references | `997201f8c0f7ac784916d16e0a73c4db53f7c589609eb67bffa24507faf7441f` | `24f729c3de328a8eefa7d7d5b306d370b9c7e62f7aea5098d97cd001b3de5a2a` | `0a4180e77d4f766b917e33254ca591f4f2078f832fccf2e248a2b379cb0f5034` |
| `lab_notebook` | timestamped measurements, interventions, and revisions | `33ccb08da60f8883384492c01f2dadf573bbd84bc3cb00dd7a1ff715bc26946a` | `b13ecb2e7dc2169b4b7a64efdca5f14a2a1b79dc2bc9a04db4d1664bd038ffd4` | `d171d9ad7ba50f2ca9f0b03a472940e4997b2856d684ea9f6209e6a1e53f5b78` |
| `nested_configuration` | nested JSON-like objects, arrays, booleans, and overrides | `ff5be0b548073edecf576e5900bd2893384f5e251b82bcdfd7617a5d766ffbb5` | `f265fb20b213277f9562a5a05dd0535dcd1a422f7bacb0f1098b90d4213b01cc` | `f9d30bfa480234b0e22061cfa0e1f9ff7d6542cee3eaa3c5eee3af7d8a880789` |
| `operations_log` | shell/service transcript, paths, status codes, and retries | `ed7f66e129c4f4b058f9ec1d66e456a674d97b34b5e002b4108d92f92511a548` | `330296e6912b60418d535c0118cf56817d2fe0d8d77cd5a0a607db8bf7e88494` | `cedd7d42c6a1da6788e05e96b0756ff52f25da2f92b0b6727e127ef150d304d0` |
| `poetic_stanzas` | line-broken verse, imagery, rhyme, and refrain | `38f4921d0df36e92fc891ec971ce395b877a9bf2e576aa703c350334fb4b06f1` | `87efbf332765496aadd38f8a1f8447a8030942d3094a78369e76dec2a1738590` | `e19f9d80319586978a400fca3ae196dea9fa218ede628bb0ae4d960c90c08321` |
| `mathematical_derivation` | assumptions, equations, substitutions, and boundary case | `cadd959690282a1fb55ba547d88c541d856b470e2911c13445843fbe99aed2f7` | `893f16b5802a1f9b2af4d2c9ddc6767c9f73e7650e0bc48b90a765b77232a693` | `9ca10f6d6e479bf942a013882a8b08296ed1837a209eaa3508a32a6716f633ed` |
| `bilingual_glossary` | paired multilingual entries, grammar tags, and usage notes | `1ae634ecbbc9f1a51c57a732d84865ffaf93cef7398ae008a4a896108ab758e5` | `263b6f04db2fdb1b9d412ef022a7cf8888ea85c02f5e7a01438f44215c5f4e71` | `fdbbbba38a2ead76ac23b66f1bc7bf656138000b24ed011151c5b409dffddb9a` |
| `incident_bulletins` | numbered bulletins, corrections, locations, and status changes | `868107d7dd006127694a10d7992352b82ee3bc470705516eda99067e06614a01` | `069a36a004560ab6199e8f608b82b8204dc91ea455dcd417c2d75be2d320833e` | `92c5dd5c433926f33d0cd973afa9201cdfe5335a80ad1b25c116dd66cbcb15d4` |

The exact model inputs are constructed identically to development data:
tokenize each authored fixture without special tokens, repeat independently,
and truncate to exactly 512 or 2,048 tokens. Source and token locks are enforced
by offline tests and by both feature and outcome collectors.

### Frozen feature and predictor contract

The model receives exactly: layer ID; head ID; causal context length; normalized
query position; query L2 norm, mean, population standard deviation, maximum
absolute value, and positive-sign fraction; key scalar mean and population
standard deviation; mean and population standard deviation of per-key L2 norm;
key maximum absolute value; mean-key-vector L2 norm; and query/mean-key cosine.
Layer/head are one-hot encoded; all other features are standardized using
development means/scales only.

Query statistics cost `O(D)`. Key statistics require a running key-vector sum,
element-square sum, key-norm sum/square-sum, absolute maximum, and token count.
They are maintainable in `O(D)` per appended key and occupy `D+4` float32 values
plus one int64 count per head (280 bytes at `D=64`). Every feature is available
before retrieval and is strategy-independent. No Quest or PQ query outcome is a
feature, and the feature function has no parameter for exact scores, Top-K,
attention weights/entropy/mass, retrieval selections, recall, or output error.

The learned model is budget-specific multinomial logistic regression implemented
only with PyTorch. Targets are retrospective maximum-attention-mass winners;
output-error winners are recorded only for analysis. Four L2 values
`0, 1e-4, 1e-3, 1e-2` were compared by leave-one-development-fixture-out mean
attention mass, not configuration accuracy. Selected L2 values were `1e-2`,
`1e-4`, and `0` at 12.5%, 25%, and 50%; learning rate `0.05`, 250 epochs, and
seed zero were then frozen. The development-selected best global configuration
was Quest p64 at every partial budget, with development mean mass
`0.643124/0.734611/0.830729`. Layer and layer/head lookups were also frozen from
development data. The serialized freeze artifact SHA-256 is
`57a00bbf5368ed96121f40140106ea3bd96282123f25479811c46d8fd58575a9`.

Held-out outcomes had not been computed or inspected when this protocol,
fixture content, hashes, features, hyperparameters, baselines, tie rule,
failure thresholds, bootstrap method, and evidence decision criteria were
frozen. Final evidence is reported below only after the one-time held-out run.

The first post-hoc evaluation artifact contained all oracle values in its raw
prediction rows but omitted an explicit top-level oracle baseline summary and
pooled the layer-23 failure block across budgets. This was a reporting-only
omission found during result audit: it did not affect fixtures, features,
weights, predictions, outcomes, regret values, costs, or the frozen decision.
A versioned second analysis artifact adds those summaries from the same
immutable held-out outcome file. The model forward and retrieval matrix were
not rerun.

### Held-out execution and results

The one-time held-out matrix produced 3,072 unique observations per budget and
61,440 retrieval records including exact Top-K and the 100% control. All
192/192 independent full-attention reconstructions passed, and all 15,360
full-budget per-head invariants passed. The immutable held-out feature and
outcome SHA-256 values are respectively
`d09b06bbe225bdfda0c1fb9aa794ba0e3fe999028630695639045b5d4f302067`
and `af8233d85e4679264ce5c16973f614f250bfa17a042ffc5bc37751467013e67a`.
The complete versioned evaluation artifact SHA-256 is
`5df19fc6330631bafa9ee054414f70d31a01ae2dd3dca301d4f678c8694e91ad`.

The development-selected layer choices were p64/p16/p64 for layers 0/12/23 at
12.5%; M4/C8, p16, and p64 at 25%; and the same M4/C8, p16, and p64 choices at
50%. The layer/head lookup selected all four configurations: across the 48
layer/head identities its p16/p64/M2C4/M4C8 counts were `6/29/1/12`,
`6/20/2/20`, and `8/9/2/29` at the three budgets.

Held-out mean metrics show the complete adaptivity chain. Error is per-head
relative attention-output error; mass/error regret use the retrospective
maximum-mass/minimum-error oracles respectively. Gap recovery uses group means
relative to the development-selected global fixed baseline.

| Budget | Selector | Attention mass | Output error | Mass regret | Error regret | Fixed-to-mass-oracle gap recovered |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 12.5% | global p64 | 0.5785 | 0.7192 | 0.1559 | 0.3714 | 0.0% |
| 12.5% | layer fixed | 0.5989 | 0.6643 | 0.1355 | 0.3165 | 13.1% |
| 12.5% | layer/head fixed | 0.6337 | 0.6298 | 0.1007 | 0.2820 | 35.4% |
| 12.5% | learned cheap features | 0.5897 | 0.7584 | 0.1447 | 0.4106 | 7.2% |
| 12.5% | retrospective mass oracle | 0.7344 | 0.4137 | 0 | 0.0659 | 100% |
| 12.5% | retrospective error oracle | 0.7129 | 0.3478 | 0.0215 | 0 | 86.2% |
| 25% | global p64 | 0.6977 | 0.4608 | 0.1483 | 0.2742 | 0.0% |
| 25% | layer fixed | 0.7299 | 0.4456 | 0.1161 | 0.2591 | 21.7% |
| 25% | layer/head fixed | 0.7606 | 0.3961 | 0.0854 | 0.2096 | 42.4% |
| 25% | learned cheap features | 0.7157 | 0.5376 | 0.1303 | 0.3510 | 12.1% |
| 25% | retrospective mass oracle | 0.8460 | 0.2203 | 0 | 0.0338 | 100% |
| 25% | retrospective error oracle | 0.8279 | 0.1865 | 0.0181 | 0 | 87.8% |
| 50% | global p64 | 0.8122 | 0.2638 | 0.1354 | 0.2020 | 0.0% |
| 50% | layer fixed | 0.8536 | 0.2328 | 0.0939 | 0.1710 | 30.6% |
| 50% | layer/head fixed | 0.8885 | 0.1896 | 0.0591 | 0.1278 | 56.4% |
| 50% | learned cheap features | 0.8387 | 0.3363 | 0.1089 | 0.2746 | 19.6% |
| 50% | retrospective mass oracle | 0.9476 | 0.0716 | 0 | 0.0098 | 100% |
| 50% | retrospective error oracle | 0.9370 | 0.0618 | 0.0105 | 0 | 92.2% |

Distributions remain highly skewed. Learned-policy mass-regret
mean/median/p90 was `0.1447/0.000002/0.6601`,
`0.1303/0/0.5849`, and `0.1089/0/0.5362` as budget increased. Error-regret
mean/median/p90 was `0.4106/~0/1.1861`, `0.3510/0.00318/0.8783`, and
`0.2746/0.000118/0.5357`. Thus many wrong configuration labels were harmless,
but a smaller set of near-total misses dominated the means.

Fixture-cluster bootstrap 95% intervals for learned mean attention mass were
`[0.5571, 0.6260]`, `[0.6947, 0.7335]`, and `[0.8221, 0.8540]`; intervals for
mean mass regret were `[0.1252, 0.1639]`, `[0.1207, 0.1401]`, and
`[0.0935, 0.1237]`. More importantly, learned-minus-layer/head mean mass was
`-0.0440`, `-0.0449`, and `-0.0498`, with fixture-cluster 95% intervals
`[-0.0531,-0.0347]`, `[-0.0534,-0.0381]`, and `[-0.0681,-0.0290]`.
Cheap query-dependent features therefore failed to beat head identity at every
budget; they were materially worse.

Layer 23 remained the central failure. Learned mass/error and mass/error regret
were `0.4246/1.5256` and `0.2855/0.9822` at 12.5%, `0.5640/1.0545` and
`0.2741/0.7837` at 25%, and `0.6789/0.7499` and `0.2738/0.6837` at 50%.
For the three difficult heads selected using development data only (heads 15,
1, and 4), learned mass was `0.2804/0.5465/0.7018` and mean mass regret was
`0.3412/0.2358/0.2340`. Applying the development-frozen normalized-entropy
threshold post hoc, low-entropy layer-23 mean mass regret remained
`0.2934/0.2760/0.2712`; exact entropy never entered prediction.

Across all budgets, the learned model disagreed with the retrospective mass
oracle in 4,603/9,216 observations. Of those, 1,588 had mass regret at most
0.01, while 1,941 total predictions lost at least 0.10 mass. Failures were
distributed across every fixture and position, but mean regret was highest for
heads 10, 8, and 4 (`0.192/0.182/0.181`), for the bilingual glossary and
mathematical derivation fixtures (`0.144/0.141`), and at 50%/25% query position
(`0.134/0.133`). This is not a single-text or final-position artifact.

### Cost and index coexistence

On this unoptimized CPU/Python reference path, maintained-state query feature
extraction for all 16 heads had median 386.6 microseconds. Logistic prediction
had median about 3.7 microseconds per head (about 59 microseconds for 16 heads),
for an estimated combined 446 microseconds per layer/query batch. Median
candidate retrieval was 282.6 microseconds, so feature plus prediction cost was
158% of measured retrieval time and failed the frozen 10% overhead criterion.
The `O(H*S*D)` prefix-statistic reconstruction used only to measure the
experiment is not an inference requirement; the maintained state costs 280
bytes/head, 4,480 bytes/layer, or 107,520 bytes for all 24 layers.

Reference index bytes, excluding shared full-precision KV, were:

| Context/model scope | Quest p16+p64 | PQ M2/C4+M4/C8 actual / logical packed | All four actual / logical packed |
| --- | ---: | ---: | ---: |
| 512, one layer/16 heads | 327,680 | 442,368 / 65,536 | 770,048 / 393,216 |
| 512, 24 layers | 7,864,320 | 10,616,832 / 1,572,864 | 18,481,152 / 9,437,184 |
| 2,048, one layer/16 heads | 1,310,720 | 1,622,016 / 114,688 | 2,932,736 / 1,425,408 |
| 2,048, 24 layers | 31,457,280 | 38,928,384 / 2,752,512 | 70,385,664 / 34,209,792 |

Model A, which can choose all four configurations, requires both Quest indexes
and both PQ indexes resident. Model B can retain only the Quest pair, only the
PQ pair, or a smaller individual subset at the corresponding table cost, but
cannot make the omitted choices. Actual reference PQ codes are int64; logical
figures assume packed 2-bit M2/C4 and 3-bit M4/C8 codes. No index lifecycle or
switching infrastructure was implemented; dispatch alone is included in
prediction timing.

### Evidence decision and next experiment

The fixed-to-oracle mass opportunity remained material at
`0.1559/0.1483/0.1354`. Layer/head identity recovered only
`35.4%/42.4%/56.4%`, below the preregistered 75% static-policy criterion, while
the learned features recovered only `7.2%/12.1%/19.6%`, lost to layer/head at
every budget, and exceeded the overhead limit. The frozen classification is
therefore **D — ORACLE GAP NOT PREDICTABLE WITH CHEAP FEATURES**.

No adaptive planner is justified now. No shared KVWeave abstraction, Quest/PQ
implementation, or public/core policy type changed.

The exact proposed next experiment is one final preregistered metadata-only
feasibility check on another newly locked eight-fixture test set: retain this
same candidate matrix and general features, add only precomputed resident-index
summaries that require no query retrieval (Quest min/max-range moments at p16
and p64; PQ centroid occupancy and build-time reconstruction-error summaries at
M2/C4 and M4/C8), impose a hard 10% retrieval-latency feature-cost ceiling, and
train/select on development plus the now-spent held-out set while evaluating
once on the new lock. If that test does not beat the layer/head lookup by at
least 0.01 mean mass with a positive fixture-cluster 95% lower bound at two
budgets, stop query-adaptive policy work and retain fixed/static research
baselines only. This next experiment has not been started.

## Phase 2 research and implementation: PQCache

### Sources and attribution

- **Paper:** Hailin Zhang, Xiaodong Ji, Yilin Chen, Fangcheng Fu, Xupeng Miao,
  Xiaonan Nie, Weipeng Chen, and Bin Cui, "PQCache: Product
  Quantization-based KVCache for Long Context LLM Inference," *Proceedings of
  the ACM on Management of Data* 3(3), Article 201, SIGMOD 2025.
  [DOI 10.1145/3725338](https://doi.org/10.1145/3725338),
  [arXiv 2407.12820v2](https://arxiv.org/abs/2407.12820v2).
- **Official implementation:**
  [HugoZHL/PQCache](https://github.com/HugoZHL/PQCache).
- **Repository revision inspected:**
  [`0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828`](https://github.com/HugoZHL/PQCache/tree/0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828)
  (current `master` when inspected on 2026-08-26).

The 30-page arXiv v2/SIGMOD paper was reviewed in full. Reported model-quality
and latency results belong to the authors' model, hardware, workload, and
runtime configuration. KVWeave has not reproduced them and does not use them as a
performance claim.

### Standard product-quantization concepts

Standard product quantization (PQ), attributed by the PQCache paper to Jégou,
Douze, and Schmid (2011), divides a vector of dimension `D` into `M` disjoint,
equal-dimensional subspaces. Each subspace has its own independently learned
K-means codebook. A database vector is represented by one nearest-centroid ID
per subspace. Its reconstructed vector is the concatenation of the selected
centroids.

For approximate raw inner-product search, a query is partitioned identically.
For each subspace, a lookup table contains the query dot product with every
centroid. The approximate score of an encoded database vector is the sum of
the `M` table entries selected by that vector's codes. Ranking those approximate
scores produces token candidates without reconstructing every vector. PQ does
not intrinsically require an inverted index, a cache policy, CPU offload,
GQA-specific aggregation, or an attention implementation.

### Algorithm and policies described in the PQCache paper

PQCache applies PQ independently to the keys of each transformer layer and KV
head. For keys with per-head dimension `D`, the paper uses `M` subspaces of
dimension `D / M`, `2**b` centroids per subspace, centroid tensor shape
`[M, 2**b, D / M]`, and code shape `[S, M]` after omitting batch/head dimensions.
At decode time it computes query-to-centroid inner products, gathers through
the codes, sums subspace contributions, approximately ranks middle-context
tokens, fetches the selected full-precision keys and values, and performs
ordinary attention over the fetched set.

The complete PQCache system adds runtime and inference policies that are not
part of standard PQ:

- full-precision initial (sink) and recent/local tokens are always included;
- only middle-context tokens participate in approximate PQ Top-K retrieval;
- newly generated tokens stay local, then receive codes when evicted from the
  local window;
- prefill KV offload and per-layer/per-head/per-subspace CPU clustering overlap
  model computation;
- an adaptive, hardware-profiled iteration cap attempts to hide clustering
  behind prefill computation;
- centroids remain on GPU while codes are prefetched layer by layer;
- fetched full-precision KV can be served from a block-level LFU/LRU GPU cache;
  and
- GQA requires a policy for combining query-head evidence into a KV-head token
  selection.

Those policies materially affect the paper's end-to-end semantics and latency,
but they are intentionally outside KVWeave's Phase 2 reference-PQ experiment.

### Behavior observed in the official repository

The following observations describe revision `0b74e125...`; they are not a
specification for KVWeave and no source was copied:

- The main runtime defaults to Euclidean K-means for each head/subspace and
  uses centroid dot-product lookup tables for approximate token scoring. A
  separate experimental inner-product mode uses a maximum-inner-product to
  L2 augmentation.
- The clustering workers use scikit-learn K-means with one initialization,
  sampled input rows as initial centroids, a fixed environment-controlled seed
  (default `4321`), Lloyd iterations, and an adaptive iteration limit clipped
  to `[3, 300]` when the user does not provide one.
- The adaptive runtime currently asserts batch size one. It accepts subspace
  counts from `{1, 2, 4, 8, 16}` and requires GQA in its decode entry point.
- In the Euclidean/GQA path, each query head produces approximate token logits;
  the implementation applies softmax per query head, sums probabilities across
  query heads sharing a KV head, and selects Top-K middle tokens per KV head.
  Sink, recent, and current tokens are then concatenated outside that ranking.
- The repository stores code tensors as `int64` in shared CPU/GPU buffers even
  though the paper's memory analysis assumes logically packed `b`-bit codes.
- Codebooks reserve additional capacity for generated tokens. Tokens receive
  nearest-centroid codes when they leave the local window; the initial
  codebooks are not retrained during ordinary short-output decoding.
- The implementation includes CUDA/FlashAttention integration, multiprocessing
  CPU clustering, cache management, model patches, GQA handling, dataset
  evaluation, and timing overlap. None is needed to test KVWeave's index/storage
  boundary.

### Repository licensing and provenance boundary

Revision `0b74e125...` has **no top-level `LICENSE`, `COPYING`, or `NOTICE`
file**, and GitHub reports no detected repository license. Publication of
source code alone does not grant KVWeave permission to copy, modify, or
redistribute it. The paper's ACM publication notice is a publication license,
not a software license for the repository.

The upstream README also says code was borrowed from LongBench, H2O, InfLLM,
SPARQ, and Hetu. The snapshot contains an embedded InfLLM tree with its own MIT
license, a modified `sparq_official` tree with Graphcore copyright notices and
some Transformers-derived files carrying Apache-2.0 notices, H2O/model-derived
files without a uniform top-level notice, and a shared-memory helper explicitly
marked as copied from an external gist. These file-level origins must not be
collapsed into a single assumed license.

Accordingly, KVWeave will not copy or adapt any upstream PQCache source, including
its PQ search/compressor, initialization details, multiprocessing code,
GPU-cache manager, model patches, attention kernels, evaluation code, or
third-party subtrees. Any future source reuse would require an explicit license
from the relevant copyright holder plus a file-by-file provenance and notice
audit. Phase 2 uses only independently written code based on the paper's
mathematical description and standard PQ concepts.

### KVWeave independent reference implementation

KVWeave now has a deterministic, readable PyTorch reference with equal contiguous
subspaces, bounded Lloyd-style K-means, explicit farthest-error reinitialization
for empty clusters, nearest-centroid encoding, and raw-dot-product lookup-table
scoring. It returns the existing token-level `Selection`, fetches through
`TensorStorage`, and uses the existing reference attention. No upstream
PQCache source was copied or adapted.

The reference codebooks have shape `[B, Hkv, M, C, D / M]`, codes have shape
`[B, Hkv, S, M]`, lookup tables have shape `[B, Hkv, M, C]`, and approximate
token scores have shape `[B, Hkv, S]`. Codes use `int64` for readable PyTorch
gather operations; benchmark output distinguishes those actual tensor bytes
from a logical packed-bit estimate. Full-budget search returns every token
exactly once and recovers full attention through the common storage path.

It intentionally does not reproduce adaptive iteration scheduling, CPU/GPU
offload, packed codes, the initial/local-token policy, incremental decode
updates, GQA aggregation, GPU caching, FlashAttention, model integration,
multiple processes, or the PQCache evaluation runtime. Its reconstruction and
synthetic recall/error measurements are diagnostics for the KVWeave architecture
hypothesis, not a reproduction of PQCache's quality or performance results.

## Phase 0/Phase 1 research and implementation: Quest

### Sources and attribution

- **Paper:** Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci,
  and Song Han, "QUEST: Query-Aware Sparsity for Efficient Long-Context LLM
  Inference," ICML 2024, PMLR 235:47901-47911.
  [PMLR record](https://proceedings.mlr.press/v235/tang24l.html),
  [arXiv 2406.10774](https://arxiv.org/abs/2406.10774).
- **Official implementation:** MIT HAN Lab,
  [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest).
- **Repository revision inspected:**
  [`01c1623bf9395009520874e989e29f683203b357`](https://github.com/mit-han-lab/Quest/tree/01c1623bf9395009520874e989e29f683203b357)
  (current `main` when inspected on 2026-08-21).
- **Upstream implementation license:** MIT License, copyright 2024 MIT HAN Lab;
  see the revision's
  [`LICENSE`](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/LICENSE).

The paper's camera-ready PDF and the upstream README report up to 7.03x
self-attention speedup and 2.23x end-to-end inference speedup under their tested
conditions. The PMLR HTML abstract appears to swap those two quantities. KVWeave
has not reproduced either result and makes no performance claim from them.

### Ideas described in the paper

Quest targets autoregressive decode, where reading the full KV cache makes
self-attention memory-bound at long context lengths. Its central observation is
that token importance changes with the current query, so permanently evicting
tokens based on historical importance can remove tokens needed later.

For each KV head, Quest:

1. partitions the sequence dimension of the key cache into fixed-size pages;
2. stores the element-wise minimum and maximum key values for every page;
3. scores every page for the current query using a cheap upper bound;
4. selects the highest-scoring pages; and
5. performs ordinary attention using the keys and values in those pages.

For query component `q[d]` and page bounds `minimum[d]` and `maximum[d]`, the
per-dimension contribution is:

```text
max(q[d] * minimum[d], q[d] * maximum[d])
```

The page score is the sum of those contributions over `D`. It is an upper bound
on the dot product between the query and every key in the page. The bound can be
loose because the independently chosen extrema may come from different tokens,
but it is inexpensive and is intended to avoid missing pages that may contain a
high-attention token.

The paper defines the token budget as the number of tokens in the selected
pages. It evaluates page size 16 in its main kernel analysis, keeps the first two
transformer layers dense in model experiments because they exhibited much less
sparsity, and applies sparse selection during decode rather than optimizing
prefill. These model-policy choices are not intrinsic to the page estimator.

### Behavior observed in the official repository

The following details are observations of revision `01c1623...`, not additions
to the paper's algorithm and not commitments for KVWeave:

- The optimized path updates min/max metadata incrementally as keys are appended
  to paged storage. RoPE is applied before keys are appended, so metadata is
  computed from the same position-encoded keys used by attention.
- Decode selection is per attention head. The newest, possibly partial page is
  excluded from Top-K ranking and always included in sparse attention. The page
  budget includes that newest page.
- The optimized controller converts token budget to page budget with integer
  division. If the cache fits within the page budget, it takes the dense path.
- The optimized integration uses dense causal attention for prefill and changes
  the page budget to keep the first two transformer layers dense.
- The pure-PyTorch evaluation path expresses the same sign-aware bound by
  multiplying keys by the query sign, taking a per-page maximum, and multiplying
  by the absolute query. Its helper selects at least three pages even when the
  requested budget is smaller; that minimum is an upstream evaluation policy,
  not part of Algorithm 1 in the paper.
- Partial pages are padded with the lowest finite value before page maxima are
  computed in the evaluation path. The optimized cache instead tracks the
  actual tail-page length.
- Upstream examples commonly use page size 16. Its CUDA tests also exercise
  other page sizes, so 16 is an experimental setting rather than an algorithmic
  requirement.
- GQA support is not a safe Phase 1 assumption. The README says GQA models are
  supported and the evaluation code can repeat KV heads, while the inspected
  optimized estimator test explicitly requires equal query and KV head counts.
  KVWeave should initially test one query per KV head and revisit GQA at the model
  integration boundary.

These observations are useful for designing boundary-case tests, especially for
partial pages, budget conversion, per-head selection, position-encoded keys, and
the always-included tail page. They will not be reproduced mechanically.

### Licensing boundary

The current official Quest repository is permissively licensed under MIT, which
requires retaining its copyright and permission notice in copies or substantial
portions. However, the repository also:

- contains files with FlashInfer Apache-2.0 headers;
- includes FlashInfer, GoogleTest, NVBench, pybind11, and RAFT as submodules; and
- states that it adapts snippets from H2O, StreamingLLM, and Punica.

The top-level MIT license must not be treated as replacing those third-party
licenses or notices. KVWeave will not copy the upstream CUDA kernels, model forks,
cache manager, evaluation helpers, or third-party-derived snippets. Any future
proposal to incorporate upstream source must first trace that file's provenance,
verify all applicable licenses, preserve required notices, and document the
derivation.

### KVWeave implementation status

KVWeave now has an independent, readable PyTorch Quest-style reference index based
on the paper's mathematical description. No upstream Quest source code, CUDA
kernels, model forks, cache-management code, or evaluation helpers were copied.
Quest remains attributed to Tang et al.; the official upstream implementation
is MIT licensed as recorded above. KVWeave's code is an independent implementation,
not a port or claim of algorithmic originality.

Implemented and model-download-free validated behavior includes:

- batch-aware page min/max metadata `[B, Hkv, P, D]`, including partial tails;
- the paper's sign-aware upper-bound page score and a per-token invariant test;
- positive token budgets rounded up with `ceil(budget / page_size)`;
- deterministic per-batch/per-KV-head page selection and valid token expansion;
- candidate recall against exact raw-dot-product Top-K; and
- ordinary selected-token attention compared with full synthetic attention.

The reference tie policy is descending page score, then ascending page ID, with
ascending token IDs within each ranked page. This is a KVWeave reproducibility
choice; upstream tie compatibility has not been established.

KVWeave's paper-level index differs deliberately from observed upstream runtime
policies. It does not force-include the newest page, keep early transformer
layers dense, decide where RoPE or incremental metadata updates occur, impose an
upstream evaluation minimum page count, or aggregate GQA query heads. Those are
future integration/runtime choices. Phase 1 accepts exactly one query per KV
head with query shape `[B, Hkv, D]`.

Synthetic candidate recall and attention-output error validate implementation
behavior only. They are not evidence of model quality, end-to-end inference
speed, or reproduction of the paper's reported performance.

### Independent reference plan status

1. **Complete:** Add a page-partition helper for canonical keys `[B, Hkv, S, D]`, including a
   final partial page without synthetic tokens affecting its extrema.
2. **Complete:** Add `QuestMetadata` containing `minimum` and `maximum` tensors with shape
   `[B, Hkv, P, D]`, plus `page_size` and the original sequence length.
3. **Complete:** Build metadata with readable PyTorch reductions only. Validate it against a
   slow loop oracle on small tensors and test partial pages explicitly.
4. **Complete:** Accept decode queries `[B, Hkv, D]` and calculate the paper's sign-aware page
   upper bound independently. Test the bound against every exact token score in
   each page.
5. **Complete:** Convert a token budget into pages with the reviewed Phase 1
   policy: require a positive budget, select `ceil(budget / page_size)` pages,
   clamp at the number of pages, and report all tokens in selected pages. This
   makes page granularity explicit instead of silently promising an exact token
   count.
6. **Complete:** Select pages independently for every batch item and KV head, expand page IDs
   to valid token IDs, remove indices beyond `S` from the partial page, and
   return the existing token-level `Selection` representation.
7. **Complete:** Keep tail-page inclusion, dense early layers, RoPE placement, and GQA outside
   the core estimator initially. Add tail-page inclusion later as an explicit
   retrieval policy if model-level experiments show it is needed.
8. **Complete:** Add correctness tests for shapes, signs, ties, partial pages, budgets,
   per-head independence, upper-bound validity, and full-budget recovery.
9. **Complete:** Compare selected-token recall against `BruteForceIndex`, then compare sparse
   attention output against full attention. Verify that increasing budgets
   reaches full-attention behavior at the full-page budget.
10. **Complete:** Add a full-vs-Quest synthetic benchmark reporting build time, metadata size,
    retrieval latency, selected-token/page recall, attention-output error, and
    relevant tensor/hardware configuration. Do not optimize until those results
    are reproducible and reviewed.

### Reviewed Phase 1 decisions

- The newest/partial page is ranked normally; forced tail inclusion remains a
  separately measurable future decode policy.
- Non-page-aligned positive token budgets round up to pages, and actual valid
  candidate counts are reported separately.
- GQA remains out of scope. Future model-level evidence must determine whether
  query heads select independently or share/merge selection within KV groups.
