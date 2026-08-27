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
