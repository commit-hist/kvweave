# Benchmarks

Benchmarks are first-class research code. These synthetic reference scripts
validate measurement and correctness behavior; their numbers are not model
quality evidence or performance claims.

Run the exact brute-force retrieval benchmark with:

```bash
python benchmarks/scripts/brute_force.py
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

Run the Phase 1 Quest reference benchmark with:

```bash
python benchmarks/scripts/quest_reference.py
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
