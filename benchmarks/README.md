# Benchmarks

Benchmarks are first-class research code, but the Phase 0 script is only a smoke
test for installation, tensor generation, timing, and result reporting. Its
numbers are not performance claims.

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
