# Research Notes

This document records the research provenance used by KVDB. Algorithmic ideas,
observations from upstream implementations, and KVDB-authored code are kept
separate so that attribution and licensing remain explicit.

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
  and its Model Details section. This is a license for the model release; KVDB
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

KVDB does not patch the model's attention path. For selected layers, a forward
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
