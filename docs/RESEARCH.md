# Research Notes

This document records the research provenance used by KVDB. Algorithmic ideas,
observations from upstream implementations, and KVDB-authored code are kept
separate so that attribution and licensing remain explicit.

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
