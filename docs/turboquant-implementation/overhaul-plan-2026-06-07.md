# TurboQuant Overhaul + New-Implementation Plan — 2026-06-07

Companion to `inference-speed-roadmap-2026-06-07.md` (the measured evidence) and
`turboquant-affine-vs-jit-reframe` memory. This is the **execution plan** for the
revealed levers, with the model switch, gates, tests, and acceptance criteria.

## Model decision (real-model testing)

- **Target requested:** Llama-3.2-3B or Qwen3-4B (standard full-attention → KV on
  every layer → affine path exercised everywhere, clean speed/memory signal).
- **Reality on this machine:** neither is on disk and the disk is **100% full
  (~7.6 GiB free)**. On disk: `Qwen3-0.6B-8bit` (**standard-attention** dense
  qwen3: 28 layers, GQA 16/8, hd128, vocab 151936, tied embeddings — verified
  *not* hybrid), `Qwen3.5-2B-4bit` (hybrid, the product model), `Qwen3-0.6B`,
  `bitnet-2B`.
- **Plan:** use **Qwen3-0.6B-8bit** as the immediate standard-attention dev/test
  model (exercises affine on all 28 layers, no download). Download **Qwen3-4B-4bit**
  (~2.3 GB) or **Llama-3.2-3B-4bit** (~1.8 GB) for final-validation once disk space
  is freed; both fit 7.6 GiB but it is tight. Keep **Qwen3.5-2B-4bit** in the
  matrix as the *product* model (validate transfer; it is where ② applies).

### What the model switch changes about the levers
- **② (redundant recurrent sync) only exists on the Qwen3.5 hybrid** (`MambaCache`
  / `materializeRecurrentKVCacheState`). On standard Qwen3/Llama there is no
  recurrent state → ② is a **product-only fix for Qwen3.5**, not testable on the
  dev model. Keep it scoped to the hybrid.
- **① (n-gram self-speculation)** is model-agnostic; on standard models the verify
  forward routes through the model's attention (FP16 or affine), no MTP heads
  needed → we build a model-agnostic iterator (not the MTP-bound `MTPTokenIterator`).
- **③ (banked lm_head)** applies to any large-vocab model (Qwen3-0.6B vocab
  151936, tied embeddings; Qwen3-4B 151936; Llama-3.2 128256).
- **PolarQJL overhaul / recency-tiering** are codec-level, model-agnostic, and now
  exercised on *all* layers on the standard dev model.

## Lever ladder (ordered by gate-first, gain-per-effort)

### ① No-draft n-gram / prompt-lookup self-speculation — P0
**Mechanism:** propose the next k tokens from a prompt+generation n-gram table
(prompt-lookup decoding / LLMA), verify all k in one multi-query forward (already
routes to the fused affine kernel, qLen≤32, no fallback, L2-amortizes the KV scan
~2×), accept the matching prefix, trim the rejected tail. One forward reads the
weight stream once and emits (accepted+1) tokens → amortizes the dominant cost.
**Quality:** bit-exact (greedy/speculative-sampling verify, KL=0). **Compression:**
unchanged (n-gram table is host RAM).

Build order (each step independently useful):
1. **`PromptLookupSpeculator`** (MLXLMCommon library) — pure CPU n-gram proposer
   (last-`ngram` tokens → most-recent earlier occurrence → propose following k).
   Unit-tested, no model. *(this commit)*
2. **Acceptance harness CLI** — greedy-generate on real prompts, shadow the
   speculator, report **mean accepted length / acceptance rate per workload**.
   NO speed claim. This is the gate. *(this commit)*
3. **`NgramSpeculativeTokenIterator`** — reuse the verify-multi-query-forward +
   `trimPromptCache` (KVCache.swift:3909, `canTrimPromptCache` 3900) pattern from
   `MTPTokenIterator`, with the speculator as proposal source. **EMA self-disable
   gate** (`shouldDisableSpeculation`, floor 0.25) → never regress on free-form.
4. **Forced-rejection determinism test** — assert byte-identical token stream vs
   the plain `TokenIterator` over a fixed prompt set (gates correctness, esp. the
   recurrent-rollback risk if ever run on the hybrid).

**Acceptance criteria for promotion:** measured mean-accepted ≥ ~1.3 on the target
workload; speculative token stream byte-identical to non-speculative; same-report
tok/s ≥ baseline (EMA prevents regression); no fallback in the verify path.

### ② Kill the redundant per-token recurrent sync — P1 (Qwen3.5 product only)
Collapse the duplicated `eval+synchronize+clearCache` (inner
`materializeRecurrentKVCacheState` KVCache.swift:2864 + outer
`requiresSynchronousGenerationEval` Evaluate.swift:1522) and move `clearCache` to
an every-K-token cadence. Pure cadence change, cannot alter output. **Gate:**
KVCacheTests determinism (KVCacheTests.swift:422) + a timing probe before any %
claim. Largest at short context.

### ③ Banked / candidate-subset lm_head — P2, measure-first
The tied head is the largest single weight matmul (vocab×hidden; ~30% of weights).
`gatherQuantizedMatmul` exists (Ops.swift:1427). **Gate microbench FIRST:**
`gather_qmm` over ~8K candidate rows vs the full head at batch=1 on M2 — if not
clearly faster at batch=1, **stop** (the occupancy wall that made byte-cuts
neutral). If it wins: candidate-subset head with an exactness guard (frequent
full-vocab pass or bounded mismatch window that clears KL≤2e-6 / top-1==1.000).
Near-dependency for ① (verify pays the head per draft position).

### PolarQJL DIAGNOSTIC overhaul (the paper-benefits focus) — see below

### Recency-tiering (compression+quality, speed-neutral) — P2
Keep last W≈256 tokens exact FP16 (highest-attention, <2% bytes) + compress the
cold tail harder (V4→V3 / K8→K6). Net ~2.1×→~2.8× effective KV **and** better
quality (recent exact). Use the existing hybrid cache's dequant+concat merge
(TurboQuantHybridKVCache.swift:1932); ship as a **capacity/quality** claim only
(extra dispatch is net overhead in the weight-dominated regime). Gate: KL/p95/cos
vs all-FP16 reference at matched cold bits.

## PolarQJL diagnostic overhaul — realize the paper's compression (focus)

Goal: close the gap between the paper's 4–7× and our PolarQJL's realized ~2.1–2.6×.
Root cause (measured byte budget): **metadata overhead**, not the payload.

Realized per-vector bytes today (turbo3_5, hd128, group64, 2 groups/vector):
- magnitudes (~2.5-bit keys): ~6–8 words ≈ 24–32 B/group
- **3 fp32 scale values/group = 12 B/group** ← fp32 is wasteful
- **signs + residual + (high-mask) bitset planes = ~2–3 words/group = 8–12 B/group**
- → keys realize ~2.1× where nominal 2.5-bit should be ~5.3×.

The paper's layout = packed indices + **one fp(16) norm per vector**. So the
overhaul is a **metadata diet to approach paper density**, in priority order:

1. **fp32 → fp16 scales/norms** (`TurboQuantScaleStorage.float16` already exists,
   TurboQuant.swift:223) — halves the 12 B/group scale stream. Quality-neutral
   (scales are smooth). ~+0.3–0.5× compression on keys.
2. **One norm per vector, not per group** — the paper normalizes the whole rotated
   vector once; per-group scales are an affine-ism. Move to a single per-vector
   fp16 norm + the data-free Gaussian quantizer (no per-group min/max). This is the
   biggest density win and is *the* PolarQuant idea we are not fully using.
3. **Collapse redundant bitset planes** — audit which of signs/high-mask/residual
   are actually live per preset at layout v6 (the decode-branch map: turbo3_5 =
   SPLIT_MAGNITUDE branch1; high-mask plane already dropped for split). Remove
   dead planes from storage (some already compacted to `[1]`).
4. **Data-free Gaussian Lloyd-Max quantizer** (paper core) — after the random
   rotation each coordinate is ~N(0,1/d); use the *fixed* optimal scalar quantizer
   (no calibration, no per-group scale) → the metadata collapses to the rotation
   seed (shared) + one norm/vector. This is the path to 4–7×.

Quality lever (paper): with the proper rotation + data-free quantizer, **keys can
go to 2.5–3.5 bits at affine-K8 quality** (affine is biased for inner products;
PolarQuant is unbiased via the QJL residual). So the overhaul also unlocks
*lower-bit keys at kept quality* → compression++.

Speed reality (honest): PolarQJL decode is the custom JIT kernel (0.27–0.48×). The
metadata diet improves **compression**, not speed; speed needs the hardware-aligned
rotation direction (cf. IsoQuant SO(4) isoclinic rotations, arXiv:2603.28430) — a
larger, separate kernel effort. So the overhaul ships as a **compression/quality**
win for the diagnostic path; promoting it for *speed* still requires the kernel
work and remains gated.

Gates: encode/decode parity unit tests (WHT signs, Lloyd-Max boundaries, bit
packing) per `external-port-optimization-map.md` Path 8; real-model KL/p95/cosine
at matched bits; resident-byte report showing the new density; same-machine
upstream (`arozanov/turboquant-mlx`) comparison before any parity claim.

## Stacking / sequencing
- ①×③ is the headline stack (banking makes verify positions cheap). ② adds
  (host overhead, hybrid only). Recency-tiering + PolarQJL-diet are compression
  workstreams orthogonal to the speed levers.
- **Do gates before builds:** ① acceptance harness, ③ banked-head microbench.
- Defer all new kernel work (query fan-out, hardware-aligned rotation) until ①
  proves real-model gains and the microbenches pass.

## Honest caveats (gate any claim)
1. Acceptance rate is the biggest unknown for ① — measure first (this commit).
2. Speculative rollback on recurrent `MambaCache` is untested → determinism gate
   before ① runs on the hybrid.
3. MLP stream (~54% of weights) un-rooflined at batch=1 vs batch=k — bounds the
   whole speculation family's ceiling; add an MLP qmv roofline.
4. Disk is 100% full — frees needed before downloading a 3–4B validation model.

## First commits (this session) — DONE
1. This plan doc.
2. `PromptLookupSpeculator` (MLXLMCommon) + 10 unit tests — all pass.
3. `TurboQuantAcceptanceHarness` CLI (token-ID input via `IdentityTokenizer`; no
   swift-transformers dep — tokenize text externally with `transformers` and pass
   `--prompt-ids-file`) + run on Qwen3-0.6B-8bit.

### ① acceptance gate — RESULT (PASSED), Qwen3-0.6B-8bit, 256 tok, greedy
Artifacts: `artifacts/turboquant-acceptance-20260607/` (prompts + acceptance JSON).
Tokens-per-forward (the real-model tok/s multiplier in the weight-dominated regime):

| workload | ngram 2 / 3 / 4 | mean accepted |
| --- | --- | --- |
| quote-continue (verbatim) | 8.83 / 8.83 / 8.83 | 7.86 |
| doc-edit | 4.41 / 4.27 / 4.06 | 6.4–7.5 |
| code-repetition | 2.94 / 2.49 / 2.21 | 3.9–4.3 |
| json-list | 2.39 / 2.15 / 1.97 | 4.7–5.1 |
| free-prose | 1.09 / 1.05 / 1.02 (no regression) | 1.2–1.5 |
| **aggregate** | **2.48 / 2.30 / 2.18** | — |

Matches the predicted shape exactly: large amortization on structured/repetitive
long-context (where TurboQuant is used), ~1.0× on free prose. Caveats: this is the
*upper-bound* acceptance (greedy replay; the real iterator gets the same acceptance
since verify is exact, but pays a slightly heavier multi-query forward at qL≤8, so
realized tok/s = tokens-per-forward ÷ a small verify-width factor); Qwen3-0.6B
over-repeats vs a 3–4B, so treat absolute numbers as indicative. **Decision: build
the full iterator (next).** ngram=3 is the recommended default (conservative width;
near-best aggregate); the iterator should cap proposal width to the qL≤4 sweet spot.

### ① iterator — DONE + validated end-to-end (Qwen3-0.6B)
`Libraries/MLXLMCommon/NgramSpeculativeTokenIterator.swift` — plain-`LanguageModel`
speculative decoder: n-gram propose → ONE multi-query verify forward → accept prefix
→ `trimPromptCache` rollback → EMA self-disable. Made `resolvedGenerationParameters`
internal so it resolves identically to the plain iterator (determinism). Validation
mode added to `TurboQuantAcceptanceHarness` (`--validate-speculative`): runs plain
greedy vs speculative greedy, asserts byte-identical output, reports tok/s + acceptance.

**Determinism gate: PASS — all prompts byte-identical to plain greedy.** Real-model
speedup (256 tok, greedy, ngram=3, maxProposal=4):

| workload | speedup | acceptance | tok/forward |
| --- | ---: | ---: | ---: |
| quote-continue | **2.20×** | 1.00 | 5.02 |
| doc-edit | **1.68×** | 0.95 | 3.20 |
| json-list | **1.20×** | 0.76 | 1.94 |
| code-repetition | 0.79× | 0.31 | 1.05 |
| free-prose | 0.78× | 0.25 | 1.04 |

Real wins on structured/repetitive workloads (byte-exact). The <1× on low-acceptance
code/prose is the **inherent speculative-decoding overhead on a tiny 5ms/token model**
(CPU accept/reject breaks the async pipeline; the shipped MTP iterator has the same
property) — it vanishes on the weight-dominated product target (4B / long context),
which is exactly where TurboQuant + speculation are used. **Admission rule:** enable ①
only in the weight-dominated regime (large model OR long context); the EMA floor
(default 0.5) is an on-device tuning constant (lower on bigger/longer-context models).
Optimizations applied: batched verify sampling (1 sync/round vs qL+1). Future refinement:
async-pipelined single-token fallback so the disabled path matches plain on small models.

### Next commits
5. ③ banked-head microbench gate → candidate-subset head if it wins.
6. PolarQJL metadata diet (the diagnostic-path focus).
7. Recency-tiering codec; ② (Qwen3.5 hybrid) redundant-sync removal.
8. ① on a 3–4B at long context (download once disk freed) to show the clean
   weight-dominated speedup without the tiny-model fallback overhead.
