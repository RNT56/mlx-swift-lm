# TurboQuant Speed-Overhaul Continuation Handoff — 2026-06-07 (overhaul track)

**This is the current pickup point for the inference-speed overhaul.** It supersedes
the speed-direction guidance in the earlier same-day affine-quality handoff
(`continuation-handoff-2026-06-07.md`, which remains valid for the affine quality gate).

## 0. Read order
1. This file.
2. `inference-speed-roadmap-2026-06-07.md` — the measured evidence base (roofline,
   amortization, weight-vs-KV crossover) and why the priorities are what they are.
3. `overhaul-plan-2026-06-07.md` — the lever ladder with per-lever design, gates,
   and the PolarQJL metadata-diet plan (the diagnostic-path focus).
4. Memory: `turboquant-affine-vs-jit-reframe`, `turboquant-kernel-build-wiring`.

## 1. The reframe (one paragraph)
Two decode-kernel families; **only the affine one ships** (`affineK8V4`/`affineInt4`
via MLX-stock `mixed_quant_sdpa_vector_2pass`). The PolarQJL JIT kernel + all TQCOOP
work is **diagnostic-only**. Decode is **weight-dominated** at the product contexts
(≤47K): fixed weights ~937 MB/token vs KV(affine) 164–654 MB. So **the only lever that
moves real-model tok/s while keeping quality + compression is amortizing the weight
stream across emitted tokens — speculative decode** (lever ①). Pure KV-byte cuts are
capacity-only for speed at the affine kernel's measured 0.39–0.62× effective bandwidth
(value g32→g64 measured speed-neutral, quality-worse). Qwen3.5-2B is a **hybrid
GatedDeltaNet** model (only 6/24 layers have KV), so it's the *product* model but a poor
*optimization* target — optimize on a **standard-attention** model (Qwen3-0.6B on disk;
Qwen3-4B/Llama-3.2-3B for representative validation, disk now freed).

## 2. DONE + validated this cycle (all in `mlx-swift-lm`, no kernel/cross-repo edits)

**Measurement infrastructure (P0):**
- `tools/TurboQuantNativeVxBenchmark` — added `--query-lengths`, `--key/value-group-size`,
  achieved-GB/s + ms/qtok roofline (schemaVersion 3). Measured: affine = **0.39–0.62×
  FP16-SDPA effective bandwidth**; multi-query KV scan **amortizes ~1.8–2.6×**;
  attention-isolated affine **beats FP16 at ≥32K**. Artifacts:
  `artifacts/turboquant-roofline-20260607/`.
- Combined affine 16K real-model gate (the prior zero-byte first-task) now valid:
  affineK8V4 **0.752× FP16**, KV 2.13×, native path, quality PASS (cosine 0.9931).
  `artifacts/turboquant-roofline-20260607/affine-16k-combined.json`.

**Lever ① — no-draft n-gram self-speculation (DONE + validated):**
- `Libraries/MLXLMCommon/PromptLookupSpeculator.swift` — n-gram proposer + acceptance
  simulator; `Tests/MLXLMTests/PromptLookupSpeculatorTests.swift` (10 tests, pass).
- `Libraries/MLXLMCommon/NgramSpeculativeTokenIterator.swift` — plain-`LanguageModel`
  speculative decoder: propose → one multi-query verify forward → accept prefix →
  `trimPromptCache` rollback → EMA self-disable. (`resolvedGenerationParameters` made
  internal in `Evaluate.swift` so it resolves identically to the plain iterator.)
- `tools/TurboQuantAcceptanceHarness` — acceptance replay (`--prompt-ids-file`) and
  end-to-end `--validate-speculative` (plain vs speculative, byte-identical assert + tok/s).
- **Result (Qwen3-0.6B, real model):** determinism gate **PASS (byte-identical)**;
  speedup quote **2.20×**, doc-edit **1.68×**, json **1.20×** (acceptance 1.0/0.95/0.76);
  code 0.79×, prose 0.78× (low-acceptance → inherent speculative overhead on a tiny
  5 ms/token model). Acceptance gate artifacts: `artifacts/turboquant-acceptance-20260607/`.

**N1 — ① validated on Qwen3-4B-4bit (DONE, supersedes the 0.6B "vanishes at scale" guess):**
- Determinism **PASS (byte-identical) on all 10 runs** (short/8K/16K/32K). Clean speedups:
  short ~0.9× (quote 1.22×), **8K 0.80× (regress despite 4.5 tok/forward)**, **16K 1.11–1.43×**.
  32K 4.755× is **memory-wall-confounded** (16 GB host) — not a clean number.
- Finding: ① is **context-gated, crossover ≈12–16K**; the multi-query forward cost scales with
  q_seq so the forward is not cleanly weight-bandwidth-bound at ≤16K. Full report:
  `artifacts/turboquant-acceptance-4b-20260607/N1-summary.md`. New helper:
  `scripts/turboquant-build-acceptance-prompts.py`.

## 3. Lever ladder + status
| lever | what | status |
| --- | --- | --- |
| Instrumentation (roofline/amortization) | GB/s + ms/qtok | **DONE** |
| Combined affine 16K gate | promotion artifact | **DONE** |
| ① n-gram self-speculation | weight-stream amortization, bit-exact | **DONE + validated on 0.6B AND Qwen3-4B** (N1 done — context-gated, crossover ≈12–16K; needs product wiring) |
| ③ banked lm_head | shrink the 30%-of-weights head | **gating microbench DONE** — kernel-viable (subset 2.6–4.1× at batch=1) but product-modest (~6% of forward latency); build only coupled with ① |
| PolarQJL metadata diet | realize paper 4–7× compression | **pending (the focus)** |
| recency-tiering | exact-recent + harder cold tail | pending (capacity/quality) |
| ② recurrent-sync removal | Qwen3.5 product-only | pending |

## 4. Next steps (prioritized — goal · gate · commands · files · acceptance)

### N1. ① on Qwen3-4B (DONE 2026-06-07) — the win is CONTEXT-gated, not size-gated
Ran on `mlx-community/Qwen3-4B-4bit` (standard attention; M2 Pro 16 GB). Full report +
reproduce: `artifacts/turboquant-acceptance-4b-20260607/N1-summary.md`. Build prompts with
`scripts/turboquant-build-acceptance-prompts.py`.
- **Determinism gate PASS (byte-identical) on all 10 runs** (short/8K/16K/32K). ① is bit-exact
  at 4B scale → N2 wiring is unblocked on correctness.
- **Speedups (clean):** short 0.87–0.96× (regress) except quote-continue 1.224× (acceptance
  0.84); **8K long-code/doc 0.80× — a 20% REGRESSION despite 4.5 tok/forward**; **16K
  long-code 1.110×, long-doc 1.425×** (wins). 32K showed 4.755× but is **confounded by the
  16 GB memory wall** (plain baseline collapsed super-linearly to 3.96 tok/s) — do NOT quote it
  as a clean ① number; the clean wins are 16K.
- **Mechanism:** the multi-query verify forward cost SCALES with q_seq (~5.6×/forward at
  q_seq=5 @8K), so the full-model forward is NOT cleanly weight-bandwidth-bound at ≤16K. ①
  wins only once context is large enough that amortized KV-read growth beats q_seq compute
  scaling → **crossover ≈12–16K** on this model.
- **This corrects** the earlier "tiny-model artifact, vanishes at 4B" framing (§7) and the
  memory reframe's "speculative decode is the real tok/s lever ≤47K": ① is a **long-context
  (≥~16K) lever**, gated on context not model size.
- **Precision probe DONE (`--forward-scaling` mode, full-model q_seq microbench):** TWO stacked
  causes, both measured (`artifacts/turboquant-acceptance-4b-20260607/forward-scaling-qwen3-4b.*`
  + N1-summary §"Forward-scaling follow-up"). (1) The full-model forward scales with q_seq —
  k=4 ≈ 2.7× k=1 — so the *synchronous* per-qtok speedup ceiling is only ~1.2–1.47× (NOT
  weight-bandwidth-bound; MLP + 152K-vocab lm_head scale with q_seq). (2) **The baseline is
  unfair:** plain `TokenIterator` is `asyncEval`-pipelined (27.2 ms/tok @8K, *below* a single
  synchronous forward = 34.6 ms), but `NgramSpeculativeTokenIterator` is synchronous per round
  (CPU propose/accept forces a drain). So ① wins only once async-plain ms/tok exceeds the ~28 ms
  synchronous spec-per-qtok floor → 8K real ceiling 0.97× (lose), 16K 1.81× (win). **This
  elevates N7 to the top speed lever** (see N7).

### N2. ① product wiring (admission-gate + Pines)
- Add a `GenerateParameters` opt-in (e.g. `selfSpeculationMode = .promptLookup` + ngram/width)
  and route `generate()` to `NgramSpeculativeTokenIterator` when enabled.
- **Admission rule (now data-backed by N1):** gate on **context ≥ ~16K**, NOT merely "large
  model" — Qwen3-4B at 8K regresses 20% even on perfectly-repetitive content, and the EMA
  floor alone does not prevent the short/medium-context regression. The crossover scales with
  per-forward KV cost, so re-measure the threshold per model/cache (FP16 vs compressed). EMA
  floor (default 0.5) stays an on-device tuning constant.
- Wire through Pines `MLXRuntimeBridge` behind a flag; validate output identity in the app.
- **Gate:** forced-rejection determinism test on the hybrid Qwen3.5 (MambaCache rollback
  correctness is UNTESTED — see caveats); byte-identical greedy stream vs non-speculative.

### N3. ③ banked / candidate-subset lm_head — GATING microbench DONE (kernel-viable, product-modest)
Ran `TurboQuantHeadBenchmark` (new tool, `tools/TurboQuantHeadBenchmark`) at Qwen3-4B head dims
(hidden 2560, vocab 151936, affine g64/4-bit, batch=1). Full report:
`artifacts/turboquant-head-20260607/N3-summary.md`.
- **Kernel-viable:** the head is **output-bound at full vocab** (fixed floor ≈ 0.185 ms +
  marginal ≈ 7.3 ns/row → marginal is 86% of head latency at 151936). A candidate-subset head
  (K=8192) is **2.6–4.1× faster** than the full head at batch=1 (gather+matmul 0.33 ms vs full
  1.37 ms; matmul-only 0.25 ms = 5.6×). NOTE: `gatherQuantizedMatmul` gathers along *batch* dims
  (MoE-style), NOT a subset of one matrix's output rows — use `take(rows)` + `quantizedMatmul`.
- **BUT product-modest:** the head is only ~**6% of the batch=1 forward latency** (1.37 ms of
  ~22 ms; it is ~30% of the *bytes* but batch=1 decode is occupancy-bound not bandwidth-bound —
  the N1 forward-scaling finding). Realistic plain-decode win ~**3–5%**, and it needs an
  exactness guard (candidate set must contain the true argmax → periodic full pass or a verified
  window clearing top-1==1.000 / KL≤2e-6).
- **Verdict: build ONLY coupled with ①** — the head is paid per verify-position, so a subset head
  helps the multi-query verify (raises ①'s ceiling a few %) more than plain decode. Not a
  standalone priority over N7.

### N4. PolarQJL metadata diet (THE diagnostic-path focus — realize paper 4–7×)
Root cause of our ~2.1–2.6× (vs paper 4–7×) is metadata, not payload. In priority order
(see `overhaul-plan-2026-06-07.md` §"PolarQJL diagnostic overhaul"):
1. fp32 → fp16 scales/norms (`TurboQuantScaleStorage.float16` exists, `mlx-swift
   Source/MLX/TurboQuant.swift:223`).
2. **One norm per vector, not per group** (the PolarQuant idea we under-use).
3. Collapse dead bitset planes per preset at layout v6 (decode-branch map in
   `turboquant-kernel-build-wiring` memory).
4. Data-free Gaussian Lloyd-Max quantizer → metadata collapses to rotation seed + 1 norm.
- **Quality lever:** proper rotation + unbiased QJL residual lets KEYS go 2.5–3.5 bit at
  affine-K8 quality → compression++.
- **Gates:** encode/decode parity unit tests; real-model KL/p95/cosine at matched bits;
  resident-byte report; same-machine upstream (`arozanov/turboquant-mlx` @ `6e928d7`)
  comparison before any parity claim. Ships as a **compression/quality** win; PolarQJL
  *speed* still needs hardware-aligned rotation (cf. IsoQuant SO(4), arXiv:2603.28430).
- **Pin trap:** `mlx-swift` is `swift package edit`-linked (`Packages/mlx-swift` → local),
  so kernel edits build — but `touch` the edited file and verify the marker is in the
  binary with `LC_ALL=C grep -a -c` (NOT `strings`). Use a DISTINCT kernel name for any
  new variant (compiled-variant cache aliasing).

### N5. Recency-tiering (compression + quality, speed-neutral)
Exact-FP16 last W≈256 tokens + harder-compressed cold tail (V4→V3 / K8→K6). Net
~2.1×→~2.8× effective KV and better quality. Use the existing hybrid cache dequant+concat
merge (`TurboQuantHybridKVCache.swift:1932`). Ship as **capacity/quality** only (extra
dispatch is net overhead in the weight-dominated regime). Gate: KL/p95/cos vs all-FP16.

### N6. ② Redundant recurrent sync (Qwen3.5 product-only)
Collapse the duplicated `eval+synchronize+clearCache` (`KVCache.swift:2864` inner +
`Evaluate.swift:1522` outer) + every-K `clearCache` cadence. Pure cadence change, can't
alter output. Gate: KVCacheTests determinism + timing probe before any % claim. Only
matters on the hybrid (no `MambaCache` on standard models).

### N7. ① async-pipelined verify forward — NOW THE TOP ① SPEED LEVER (was "refinement")
**Promoted by the N1 forward-scaling probe.** Root cause of the sub-crossover regression is NOT
the fallback path — it is that `NgramSpeculativeTokenIterator.speculateRound` runs **synchronously**
(CPU propose/accept/trim drains the GPU each round via `.asArray`/`.item`) while plain
`TokenIterator` overlaps with `asyncEval` (Evaluate.swift:1527). The synchronous spec forward
per-qtok (~28 ms @8K) loses to async-plain per-token (27.2 ms @8K) even though the *synchronous*
ceiling is positive (1.2–1.47×).
- **It needs OPTIMISTIC PREFETCH, not a one-line `asyncEval`** (corrected after analysis): plain
  `TokenIterator` pipelines by running "one token behind" — each forward's input is the previously
  *known* sampled token, so it can `asyncEval(token)` and read the prior token without blocking.
  Speculation can't do that: round R+1's input = (R's last accepted token) + (a proposal seeded
  with R's accepted tokens), both of which require reading R's argmax. So the only way to keep the
  GPU busy is to **optimistically assume full acceptance**, build + `asyncEval` R+1's verify forward
  on that assumption, then read R's argmax — if fully accepted (the common case at high acceptance),
  R+1 is already in flight (win); on a partial reject, R+1 ran on a wrong cache state and its KV
  appends must be **rolled back/discarded** before recomputing.
- **Correctness risk = the misprediction rollback.** R+1's speculative forward mutates the KV cache;
  the discard path must fully restore cache state (trim R+1's appends + R's rejects). Pairs with the
  EMA gate (only prefetch when recent acceptance is high, else it thrashes). This is a substantial
  feature, NOT a refinement — give it its own implementation + determinism cycle.
- **Expected — CORRECTED (long-context-only; earlier "all contexts" was wrong).** The realizable
  ceiling is **async-plain ms/tok ÷ pipelined-spec ms/qtok**, NOT the synchronous ceiling (that
  compared spec to *synchronous* plain, but real plain is `asyncEval`-pipelined). Pipelined-spec
  ms/qtok ≈ the back-to-back forward ms/qtok (≈28 ms @8K(q≈5), ≈27 ms @16K). So even a PERFECT N7:
  short ≈ 1.07×, **8K ≈ 0.97× (still LOSES** — the q_seq≈5 verify forward is intrinsically more
  expensive per-qtok than async-plain per-token below the crossover**), 16K ≈ 1.81× (win).** Net:
  N7 lifts the existing ≥16K win from ~1.4× to ~1.8× and lowers the crossover modestly (~12K); it
  does NOT broaden ① to short/medium context. A long-context-only payoff for a multi-state-rollback
  feature — weigh that before building.
- **Optimistic-prefetch implementation spec (greedy path), worked out for the next cycle.** Per
  round the verify forward of `[seed]+draft` (q_seq = draft+1) appends `1+draftCount` KV entries;
  on accept=a it trims `draftCount-a` rejects (kept = seed + accepted drafts; the correction token
  `mainTokens[a]` becomes next seed and enters the cache next round). To pipeline: during forward(R)
  precompute `propose(R+1)` from `history+drafts(R)` (full-accept assumption; cheap, safe), then
  `asyncEval` forward(R+1) of `[bonusR]+draft(R+1)` BEFORE reading R's argmax — but `bonusR =
  argmax@last` needs a 1-token readback, so the gap shrinks to that readback, not zero. On
  misprediction (R accepted a<draftCount): roll back **(1) KV cache** — trim `(1+draftCount(R+1))`
  [R+1's appends] + `(draftCount(R)-a)` [R's rejects]; **(2) speculator** — it was `append`ed
  optimistically with R's full drafts + R+1's seed/drafts, so it needs truncation (add a
  `PromptLookupSpeculator.truncate(to:)` that pops `tokens` and rebuilds/pops the `index` — the
  index is append-ordered so popping is feasible); **(3) iterator state** — `pendingTokens`,
  `tokenCount`-not-yet-emitted, `acceptanceEMA`, `y`. Gate the prefetch behind the EMA (only when
  recent acceptance is high) AND `speculationExact`, behind an opt-in flag (default off) so the
  validated synchronous path is untouched.
- **Gate:** byte-identical determinism on greedy (`--validate-speculative` short/8K/16K) PLUS a
  **forced-misprediction determinism test** (drafts deliberately wrong → output still byte-identical
  to plain greedy, proving the rollback restores state) + re-run `--forward-scaling` to confirm the
  spec path now tracks the synchronous ms/qtok. After it lands, **re-measure the N2 admission
  crossover** (should drop well below 12K). Hard cap: forward q_seq scaling still bounds ① at
  ~1.4–1.8×; KV-byte cuts do not move it.

## 5. Build / test / run
```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
swift test --filter PromptLookupSpeculatorTests           # ① core (10 tests)
swift build -c release --product TurboQuantAcceptanceHarness
swift build -c release --product TurboQuantNativeVxBenchmark
swift build -c release --product TurboQuantInferenceParity
# ① end-to-end validate (determinism + tok/s):
.build/release/TurboQuantAcceptanceHarness --model-dir <snap> \
  --prompt-ids-file <prompts.json> --validate-speculative \
  --max-tokens 256 --ngram 3 --max-proposal 4 --eos <eos_id>
# full-model q_seq forward-cost microbench (N1 follow-up — MLP+lm_head, not attention-only):
.build/release/TurboQuantAcceptanceHarness --model-dir <snap> \
  --prompt-ids-file <long-prompt.json> --forward-scaling \
  --contexts 0,2048,8192,16384 --query-lengths 1,2,4,8 --output <out.json>
# roofline / amortization (attention-isolated):
.build/release/TurboQuantNativeVxBenchmark --contexts 32768,65536 \
  --value-bits 4 --query-lengths 1,4 --query-heads 16 --kv-heads 8 --head-dim 128
```
Tokenize prompts to IDs (in-package tools have no swift-transformers; python has it). Helper:
`scripts/turboquant-build-acceptance-prompts.py --model-dir <snap> --short-out … --long-out …`
builds the 5 short categories + structured long-context (8K/16K) prompts. Or inline:
```python
from transformers import AutoTokenizer; import json
tok = AutoTokenizer.from_pretrained("<snapshot dir>")
json.dump([{"label":k,"ids":tok.encode(v)} for k,v in prompts.items()], open("prompts.json","w"))
```

## 6. Models / evidence
- On disk: `Qwen3-0.6B-8bit` (standard-attn dev), `Qwen3.5-2B-4bit` (hybrid product),
  `Qwen3-0.6B`, `bitnet-2B`, **`Qwen3-4B-4bit` (N1 standard-attn validation target, pulled)**.
- Artifacts: `artifacts/turboquant-roofline-20260607/`, `artifacts/turboquant-acceptance-20260607/`
  (0.6B), **`artifacts/turboquant-acceptance-4b-20260607/`** (N1: `N1-summary.md` + validate-{short,
  long,32k} + forward-scaling-qwen3-4b.{txt,json} + prompt-id inputs).

## 7. Caveats / non-claims
- **① speculates ONLY for exact greedy decode with no logit processor (hardened 2026-06-07).**
  An audit found the old `temperature > 0` branch was not distribution-correct for a deterministic
  (point-mass) draft — it resampled from the full target `p` instead of the residual `(p − q)+`,
  and processed via a *discarded* processor copy (processor state dropped). Both paths now **fail
  closed to exact single-token decode** (`speculationExact = temperature == 0 && processor == nil`
  gates the proposal; the verify is greedy-argmax only). Greedy determinism re-verified byte-identical
  on Qwen3-4B post-fix (`validate-short-qwen3-4b-postfix.txt`). Correct residual-sampling +
  processor carry for temp>0 is future work; until then ① is a greedy-only lever.
- **① is harness-only — NOT wired into `generate()`/`generateTokens()`** (Evaluate.swift:2811 routes
  only `TokenIterator`/`SpeculativeTokenIterator`/`MTPTokenIterator`). Product use needs N2, which is
  correctly blocked on N7 (the admission crossover shifts once N7 lands).
- The combined affine 16K gate is **one local single-sample synthetic report** — gate evidence, not
  a downstream/product claim. Still needs randomized repeats, larger contexts, same-machine upstream
  (`arozanov/turboquant-mlx` @ `6e928d7`), and on-device evidence per the promotion checklist.
- ① wins are **workload-gated AND context-gated** (N1): high on structured/repetitive
  ≥16K-context; **regresses ~0.80–0.96× below the ≈12–16K crossover even at 4B and even at
  4.5 tok/forward.** The earlier "0.6B <1× is just a tiny-model artifact that vanishes at
  scale" guess is WRONG — the 4B regresses at 8K too. Do not claim a general ① decode speedup;
  it is a long-context lever.
- Do not quote the **32K 4.755×** as a clean ① result — it is inflated by the 16 GB host's
  memory wall on the FP16 32K cache (plain baseline collapsed to 3.96 tok/s). Clean wins: 16K.
- Speculative rollback on the recurrent `MambaCache` (Qwen3.5 hybrid) is **UNTESTED** —
  forced-rejection determinism gate required before ① runs on the hybrid (N2).
- PolarQJL diet is a **compression/quality** win; do not claim PolarQJL *speed* parity
  without the hardware-aligned rotation kernel + same-machine upstream comparison.
- Do not claim 0.98× FP16; do not promote any path without the full promotion checklist
  (same-report quality + native path + no fallback + memory + A-series device evidence).

## 8. Gotchas
- **Build-wiring:** `mlx-swift` is `swift package edit`-linked; kernel edits build but
  need `touch` + `LC_ALL=C grep -a -c` binary verification (NOT `strings`).
- **Kernel-variant aliasing:** any new Metal variant needs its OWN kernel name or MLX's
  compiled-variant cache cross-contaminates.
- **Tokenizer:** in-package tools use `IdentityTokenizer` (token IDs only); tokenize text
  externally with python `transformers`.
- **MLP roofline gap:** the ~507 MB (54%) MLP weight stream at batch=1 vs batch=k is
  unmeasured and bounds ①'s ceiling — worth a roofline.
