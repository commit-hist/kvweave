# Benchmarks

Benchmarks are first-class research code. These synthetic reference scripts
validate measurement and correctness behavior; their numbers are not model
quality evidence or performance claims.

From the repository root, run `mise install` once to install the locked Python
and Pants launcher. The benchmark tasks below delegate execution to Pants, which
builds each benchmark's isolated environment from the repository dependency
lockfile.

Run the exact brute-force retrieval benchmark with:

```bash
mise run bench:brute-force
```

The script uses deterministic synthetic inputs, runs warmups, and reports the
median retrieval latency across repeated searches. Each row records:

- context length;
- exact token retrieval budget;
- retrieval latency in milliseconds;
- device; and
- dtype.

It also records the synthetic workload identity, PyTorch version, hardware,
batch size, KV-head count, head dimension, score/selection policy, estimated
working tensor bytes, generated-token count, git commit, seed, warmup count, and
measured repetitions. The memory figure is a transparent tensor-size estimate,
not allocator or process peak memory. Use command-line flags to change workload
values; do not compare runs unless all configuration and hardware details match.
Pass benchmark flags after `--`, for example:

```bash
mise run bench:brute-force -- --context-lengths 128 512 --budget 64
```

Run the Phase 1 Quest reference benchmark with:

```bash
mise run bench:quest
```

It compares Quest page candidates with exact raw-dot-product Top-K retrieval and
compares ordinary attention over those candidates with full attention on the
same deterministic synthetic Q/K/V tensors. The default matrix covers context
lengths 512, 2,048, 8,192, and 32,768; page sizes 16 and 64; and requested token
budgets of 100%, 50%, 25%, and 12.5%.

Each result separates the requested token budget, page count implied by ceil
rounding, actual valid candidate-count range/mean, and exact comparison K.
Candidate recall means the fraction of exact Top-K token IDs contained anywhere
in Quest's page-expanded candidates; it does not compare equal-sized candidate
sets when page rounding selects extra tokens. Attention-output relative error is
a synthetic tensor metric and must not be interpreted as generation quality.
Timings cover readable PyTorch reference operations and do not justify speedup
or backend-optimization claims.

The Quest script also emits one fixed ragged-boundary regression for sequence
length 65, page size 8, and requested budget 64. Its deterministic construction
makes some batch/head rows select the one-token tail page and other rows exclude
it. The result separately reports Quest search, `TensorStorage.fetch`, and
mask-aware selected-attention latency. This preserves visibility into the
accepted mask boundary; it is not an optimization benchmark.

Run the Phase 2 PQ and equal-requested-budget Quest comparison with:

```bash
mise run bench:pq
```

The default deterministic synthetic matrix covers context lengths 64, 256, and
1,024; requested token fractions 25%, 50%, and 100%; PQ subspace counts 2 and
4; and centroid counts 4 and 8. Quest uses page size 8 by default. Each row
reports requested and actual token counts, selected percentage, index build,
retrieval, storage-fetch, and attention timings, candidate recall against exact
raw-dot-product Top-K, and relative error against full attention. PQ rows also
report key-reconstruction relative error as a diagnostic. Strategy parameters
are recorded explicitly.

PQ code storage is reported twice: actual bytes for the reference `int64` code
tensor and a logical packed-code estimate using the minimum whole-bit width for
the configured centroid count. Codebook storage reports actual tensor bytes.
The reference does not implement bit packing. Quest/PQ Python timings describe
different readable algorithms and must not be used to claim that either
strategy is faster or better.

Run the opt-in Phase 3A Pythia real-activation experiment with:

```bash
mise run bench:real-model -- \
  --output benchmarks/results/pythia-410m-phase3a-reference.json
```

This command uses the optional pinned Transformers dependency and downloads the
pinned `EleutherAI/pythia-410m` model/tokenizer revision on first use. Ordinary
tests and synthetic benchmarks do not download it. The default matrix captures
post-RoPE Q/K and unchanged V for layers 0, 12, and 23 at exact sequence lengths
256, 512, 1,024, and 2,048. It evaluates all 16 heads, four budgets, two Quest
page sizes, and two small PQ configurations through the common index,
selection, storage, retrieved-KV, and reference-attention path.

The script refuses to evaluate approximate retrieval unless independent full
causal attention reconstruction matches the model. Output is structured JSON
with per-head records, mean/median/min/max aggregates, attention-mass capture,
full-budget invariants, correlations, and controlled configuration comparisons.
Its timing fields are single diagnostic observations, not optimized latency
measurements or speed claims. The experiment evaluates internal activations
only; it makes no generation, perplexity, downstream-quality, or end-to-end
inference claim.

The model-download validation test is separately opt-in:

```bash
mise exec -- pants test tests/integration/test_pythia_real_model.py -- \
  -m model_download
```

Run the Phase 3A structural-replication matrix with:

```bash
mise run bench:phase3a-replication
```

The default command writes the gitignored structured artifact
`benchmarks/results/pythia-410m-phase3a-replication.json`. It retains the same
pinned Pythia model, Transformers version, extraction semantics, retrieval
rankings, and full-attention reconstruction as the accepted reference run. It
expands the activation sample to eight locally authored text structures, exact
lengths 512 and 2,048, four deterministic query positions, layers 0/12/23, all
16 heads, the existing four partial/full budgets, Quest pages 16/64, and PQ
M2/C4 and M4/C8 with eight iterations and seed zero.

The result includes exact-attention entropy/support and Top-N mass, Quest page
bound looseness, PQ score approximation error, per-head full-budget invariants,
controlled configuration comparisons, descriptive correlations stratified by
layer/budget, and a retrospective four-configuration oracle. The oracle is a
diagnostic upper bound, not an adaptive policy. Candidate order is retained for
retrieval diagnostics; selected token IDs are sorted into causal order before
the mathematically permutation-invariant attention reduction so float32
reduction order cannot create a false full-budget failure. The benchmark still
makes no generation, downstream-quality, optimized-latency, or speed claim.

## Phase 3A policy-feasibility validation

This is an experimental, non-public benchmark workflow. It does not add a
planner, policy, router, or adaptive-index API. The accepted eight replication
fixtures remain development data. Eight separately authored fixtures and their
pinned-tokenizer hashes are locked in `benchmarks/policy_feasibility.py` before
held-out retrieval outcomes are generated.

The staged commands are intentionally separate:

```bash
mise run bench:phase3a-policy-development-features
mise run bench:phase3a-policy-freeze
mise run bench:phase3a-policy-heldout-features
mise run bench:phase3a-policy-heldout-outcomes
mise run bench:phase3a-policy-evaluate
```

The freeze step joins legal pre-retrieval features to the accepted development
outcomes, selects logistic-regression regularization by leave-one-development-
fixture-out attention mass, and serializes weights, standardization values,
fixed baselines, hashes, and tie rules. Held-out feature, outcome, freeze, and
evaluation commands refuse to overwrite their artifacts. A rerun after a real
evaluation-invalidating bug therefore requires an explicit artifact disposition
and written explanation rather than silently replacing evidence.

The legal feature set contains layer/head identity, causal length, normalized
query position, five query-vector summaries, six incrementally maintainable key
summaries, and the cosine between the query and maintained mean-key vector. The
extractor cannot accept exact token scores, exact Top-K, attention weights,
retrieval selections, recall, attention mass, or output error. Those values are
joined only after prediction for labels and analysis.

All artifacts are written under the gitignored `benchmarks/results/` directory.
The workflow retains the exact pinned Pythia/Transformers extraction, RoPE,
causal slicing, attention reconstruction, Quest, and PQ setup from structural
replication. It reports policy regret and cost; it makes no decode, generation,
downstream-quality, optimized-latency, or production-performance claim.

## Phase 3B stateful decode validation

Run the pinned Phase 3B matrix with:

```bash
mise run bench:phase3b-decode
```

The default command uses the existing narrative prose, technical exposition,
code-like, and list/table development fixtures at prompt lengths 256, 512, and
1,024. It generates 32 greedy tokens with the pinned Pythia-410M model and
Transformers revision. The first token comes from dense-prefill logits; the
remaining 31 positions execute explicit stateful decode retrieval. Quest p64
and PQ M4/C8 are evaluated at 25%, 50%, and 100% in both teacher-forced and
free-running modes.

Quest rebuilds page metadata after each causal KV append. PQ trains codebooks
on prefill keys, freezes them, and assigns appended keys to those existing
codebooks. The integration force-includes the newest token when absent, then
sorts selected token IDs into causal order before fetch/attention. This policy
is outside both index rankings.

The command refuses to continue a case if custom dense decode differs from
Hugging Face or if either 100% path fails full selection or numerical equality.
Its JSON result contains per-step logit/generation metrics, per-layer arrays of
per-head mass/error/counts, residual-stream errors, timing, memory, provenance,
and update policies. A sidecar PyTorch artifact stores exact dense logits,
per-layer attention outputs, residual streams, and cache lengths. Both outputs
are gitignored. The static layer/head table is intentionally deferred because
heterogeneous per-head Quest/PQ assembly would add a separate integration
problem; the failed learned policy is not used.

The model-dependent decode tests remain opt-in:

```bash
mise exec -- pants test tests/integration/test_pythia_decode.py -- \
  -m model_download
```

Ordinary pytest remains offline. Reference CPU timings identify possible future
profiling targets only; they must not be interpreted as speedups or production
memory savings.
