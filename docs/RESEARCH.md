# Research Notes

This document records the research provenance used by KVDB. Algorithmic ideas,
observations from upstream implementations, and KVDB-authored code are kept
separate so that attribution and licensing remain explicit.

## PQCache

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
runtime configuration. KVDB has not reproduced them and does not use them as a
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
but they are intentionally outside KVDB's Phase 2 reference-PQ experiment.

### Behavior observed in the official repository

The following observations describe revision `0b74e125...`; they are not a
specification for KVDB and no source was copied:

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
  evaluation, and timing overlap. None is needed to test KVDB's index/storage
  boundary.

### Repository licensing and provenance boundary

Revision `0b74e125...` has **no top-level `LICENSE`, `COPYING`, or `NOTICE`
file**, and GitHub reports no detected repository license. Publication of
source code alone does not grant KVDB permission to copy, modify, or
redistribute it. The paper's ACM publication notice is a publication license,
not a software license for the repository.

The upstream README also says code was borrowed from LongBench, H2O, InfLLM,
SPARQ, and Hetu. The snapshot contains an embedded InfLLM tree with its own MIT
license, a modified `sparq_official` tree with Graphcore copyright notices and
some Transformers-derived files carrying Apache-2.0 notices, H2O/model-derived
files without a uniform top-level notice, and a shared-memory helper explicitly
marked as copied from an external gist. These file-level origins must not be
collapsed into a single assumed license.

Accordingly, KVDB will not copy or adapt any upstream PQCache source, including
its PQ search/compressor, initialization details, multiprocessing code,
GPU-cache manager, model patches, attention kernels, evaluation code, or
third-party subtrees. Any future source reuse would require an explicit license
from the relevant copyright holder plus a file-by-file provenance and notice
audit. Phase 2 uses only independently written code based on the paper's
mathematical description and standard PQ concepts.

### KVDB independent reference implementation

KVDB now has a deterministic, readable PyTorch reference with equal contiguous
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
synthetic recall/error measurements are diagnostics for the KVDB architecture
hypothesis, not a reproduction of PQCache's quality or performance results.

## Quest

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
conditions. The PMLR HTML abstract appears to swap those two quantities. KVDB
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
to the paper's algorithm and not commitments for KVDB:

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
  KVDB should initially test one query per KV head and revisit GQA at the model
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
licenses or notices. KVDB will not copy the upstream CUDA kernels, model forks,
cache manager, evaluation helpers, or third-party-derived snippets. Any future
proposal to incorporate upstream source must first trace that file's provenance,
verify all applicable licenses, preserve required notices, and document the
derivation.

### KVDB implementation status

KVDB now has an independent, readable PyTorch Quest-style reference index based
on the paper's mathematical description. No upstream Quest source code, CUDA
kernels, model forks, cache-management code, or evaluation helpers were copied.
Quest remains attributed to Tang et al.; the official upstream implementation
is MIT licensed as recorded above. KVDB's code is an independent implementation,
not a port or claim of algorithmic originality.

Implemented and model-download-free validated behavior includes:

- batch-aware page min/max metadata `[B, Hkv, P, D]`, including partial tails;
- the paper's sign-aware upper-bound page score and a per-token invariant test;
- positive token budgets rounded up with `ceil(budget / page_size)`;
- deterministic per-batch/per-KV-head page selection and valid token expansion;
- candidate recall against exact raw-dot-product Top-K; and
- ordinary selected-token attention compared with full synthetic attention.

The reference tie policy is descending page score, then ascending page ID, with
ascending token IDs within each ranked page. This is a KVDB reproducibility
choice; upstream tie compatibility has not been established.

KVDB's paper-level index differs deliberately from observed upstream runtime
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
