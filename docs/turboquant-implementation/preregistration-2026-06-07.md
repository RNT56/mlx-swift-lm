# TurboQuant pre-registration — gates, canonical models, evidence rigor (2026-06-07)

Pre-registered BEFORE the LM-V/LM-K kernels and §5a exist, so pass/fail thresholds
cannot be rationalized post-hoc. Any promotion claim must cite this file's thresholds
and the canonical models. (Applies the measure-first rule to the items the corrected
plan just promoted.)

## Canonical models (pin for commensurable numbers across the whole program)
- **Dense / optimization target:** `mlx-community/Qwen3-4B-4bit` (standard attention,
  hidden 2560, intermediate 9728, vocab 151936, 36 layers). KV on every layer →
  clean affine/codec/speculation signal.
- **Hybrid / product target:** `mlx-community/Qwen3.5-2B-4bit` (GatedDeltaNet, 6/24
  attention layers + MambaCache). Used for §4 (① on hybrids) and product transfer.
- **Small dev (fast iteration only, NOT a promotion model):** `Qwen3-0.6B-8bit`.
- Report the model + its exact dims with every number. Do NOT mix the 937 MB
  (Qwen3.5-2B) weight-byte split with Qwen3-4B forward measurements — that drift
  (286 MB vs 194 MB tied head) is what made the §1 micro↔macro reconciliation
  necessary.

## Gate A — §2 codec-in-SDPA, Metal↔CPU bit-consistency (PRIMARY, catches the silent bugs)
Run the new polarLM SDPA variant against the CPU reference codec on random fixtures
at N ∈ {256, 2048, 16384}, head_dim ∈ {128, 256}, GQA factors {1,2,4}:
- **cosine(out_metal, out_cpu) ≥ 0.9999 AND max-abs ≤ 3e-3** at every N.
- This single gate catches: the Lloyd-Max table mismatch (§2-PRE), the inverse-WHT
  butterfly sign/order, the pass-2 deferred-rotation placement, and the polarLM
  bias-axis removal (the affine `bias·Σq` accumulator MUST compile out under the
  codec function-constant, else it injects spurious bias).
- **Fail-closed:** any combination below threshold blocks the variant; no "averaged"
  pass.

## Gate B — §2 real-model quality at matched bits (staged, keys are the risk)
Same-report quality on Qwen3-4B (and re-run on the hybrid before product):
- **Stage 1 — LM-V3/4 under affine-K8** (values are averaging-tolerant): top-1 == 1.000,
  KL ≤ 2e-6, p95 max-logit abs ≤ 1.2, cosine ≥ 0.993, AND a **long-context retrieval
  probe** (a needle-in-16K-haystack exact-recall task) within 1 pt of FP16. Target
  byte ratio ≈ 2.75×.
- **Stage 2 — LM-K5** (keys feed the softmax exponent → stricter): same thresholds
  AND the retrieval probe degradation ≤ 0.5 pt vs Stage 1, gated separately. Target
  ≈ 3.9×. Do NOT ship Stage 2 if Stage 1's retrieval probe is already at the margin.
- No promotion without: selected-native-path diagnostics, peak/steady memory, zero
  raw/decoded fallback, same-machine upstream (`arozanov/turboquant-mlx` @ `6e928d7`),
  on-device A-series.

## Gate §5a — norm-pruned argmax lm_head (the §5a keystone, BEFORE writing Metal)
§5a is an unmeasured distributional bet. Half-day offline gate FIRST:
1. Dump the Qwen3-4B tied-head per-row **dequantized** L2-norm histogram.
2. Replay real final-layer hidden states (captured from a real forward) offline;
   measure the **realized Cauchy-Schwarz prune rate** (fraction of the 194 MB head's
   banks skipped) — INCLUDING its degradation at q_seq = k (a bank survives if it can
   beat ANY of the k running bests).
- **PASS to build Metal:** realized prune rate ≥ 50% at q_seq=1 AND ≥ 30% at q_seq=4.
- **If weak (clustered norms):** pivot to **cluster bounds** (k-means centroids,
  bound = ⟨h,c⟩ + ‖h‖·r — still exact, tighter) and re-gate, before any kernel.
- **Bit-exactness (when built):** argmax index == full-head argMax for **100%** of
  positions over a 2K-token run (strict `<` on the bound, exact row-norm, greedy only).
  Note: greedy decode needs only argmax too → §5a, if it passes, accelerates EVERY
  token, not just verify rounds.

## Gate §1 — batched-qmv (lm_head-localized, conditional on the reconciliation)
- Only build if the #1 reconciliation says the lm_head/matmul fall-off is the real
  (non-drain) cost. Then: route verify-width m ∈ [3, vector_limit) for HUGE-N matmuls
  (lm_head) onto the batched-qmv; MLP untouched.
- **PASS:** batched-qmv lm_head at m=4 ≤ ~1.5× m=1 (vs current 2.50×) AND end-to-end
  forward-scaling improves, re-measured with `--forward-scaling`. Distinct kernel name
  (the existing `batched` flag means batched-WEIGHTS — do not reuse it).

## Gate §4 — SSM per-position rollback (the hybrid correctness proof)
- **cacheStateHash covers BOTH MambaCache slots** (conv state + fp32 SSM recurrent
  state) + offsets.
- Differential fuzz over accept lengths {0, partial, full, full+bonus} and adversarial
  proposers {always-wrong, always-right, boundary, misprediction storm}: assert
  **byte-identical output AND byte-identical cacheStateHash** after every round vs the
  plain-greedy oracle, plus: full-reject ⇒ slots byte-identical to the pre-verify
  snapshot; j-accept ⇒ slots byte-identical to j sequential single-steps.
- Per-step states MUST be fp32. Bound the q_seq × per-position state memory before
  assuming it is free on the hybrid; `isTrimmable=true` armed only around the verify.
- N7 prefetch stays OFF for hybrids this cycle (optimistic R+1 advances recurrent
  state past unverified R).

## §2-PRE — fix the Lloyd-Max table mismatch STRUCTURALLY
The N4 codec table (`TurboQuant.swift:7599`) and the JIT `tq_codebook_unit`
(`turbo_quant_attention_jit.h:93`) diverge ~0.6% — a silent-bias landmine. Fix:
**emit both the Swift codec table and the Metal kernel table from ONE generator**
(single source of centroids) with a **golden parity unit test** asserting bit-equality,
rather than hand-syncing two literals (which already diverged once).

## Evidence rigor (fold into the probe harness now; full §5b machinery later)
- **Interleaved A/B repeats** (plain vs candidate, order-randomized) — not back-to-back
  blocks — to cancel thermal drift.
- **`ProcessInfo.thermalState` logged per trial**; discard / flag trials at
  `.serious`/`.critical`.
- **Bootstrap CIs** on the tok/s ratio; report the CI, not a point estimate.
- Stay **≤16K resident on the 16 GB M2 Pro** (the 32K 4.755× was a memory-wall
  artifact; such rows are non-promotable). Log peak active memory.
- Microbenchmarks (`--qmm-scaling`, `--forward-scaling`, ghost-SDPA, native Vx) are
  **kernel regression tests, not product claims**.

## Measured this cycle (Qwen3-4B-4bit, M2 Pro 16 GB; artifacts/turboquant-reconcile-20260607/)

**#1 micro↔macro reconciliation (the keystone gate) — DONE.** Three-way run, all
determinism-PASS (byte-identical):
- Synchronous full-forward q_seq scaling: **T(4)/T(1) ≈ 3.0–3.1×** (8K/16K) → naive
  specCeiling ≈ **1.29–1.32×**.
- ① N7-OFF (synchronous spec) @16K: long-code **1.627×** (accept 0.95), long-doc
  **1.313×** (0.88); @8K 1.16–1.26×.
- ① N7-ON (prefetch) @16K: long-code 1.558×, long-doc 1.472×; @8K 1.07–1.17×.

**Verdict:** ① N7-off **already exceeds** the synchronous specCeiling (1.63× > 1.32×).
That is only possible because real plain decode (48.2 ms/tok @16K) is *slower* than a
bare synchronous forward (35.9 ms) — there is ~12 ms/tok of **per-token loop overhead**
(sampling, cache-quantize, graph-build) that ① amortizes across accepted tokens on top
of the forward. So most of the "3× forward scaling" is **recoverable overhead, not
weight re-streaming** — and ① recovers it. N7 prefetch is **marginal/mixed** (helps
long-doc, neutral-to-slightly-negative on code/8K; does **NOT** reproduce the prior
1.76× claim). The residual cost ① cannot amortize is genuine *forward* work (lm_head
fall-off + attention), which N7 (a pipeliner) cannot reduce. **Consequence: §5a stays
INCREMENTAL** (it attacks the lm_head verify cost, the one matmul cost ① doesn't
amortize); it must beat ① end-to-end under A/B+CIs to justify the Metal build.

**Adaptive-k (free lever) — byte-exact, ≈NEUTRAL, NOT promoted.** Determinism PASS in
all configs. Short prompts: ≈ fixed-k (0.88–1.15×; short-context regression is
structural pipeline drain that width cannot fix — and it is below the ≥8192 production
admission floor anyway). Long prompts (production envelope): high acceptance → full
width → ≈ fixed-k by construction (long-code-16K 1.651× vs fixed 1.627×); an apparent
long-doc-16K gain (1.313→1.568×) is **within single-sample noise** — the same run's
long-doc-8K returned 0.678× at full width (a clear thermal outlier), which *proves* the
A/B+bootstrap-CI requirement below. Kept opt-in/default-off; not a claimed win.

## The live baseline (must be beaten to justify §5a / §1-batched-qmv)
**Adaptive-k** (landed, opt-in, bit-exact): per-round proposal width tracks the
acceptance EMA, defaulting to the cheap width (≤2, below the lm_head verify cliff)
and widening only when acceptance is high. Zero new kernels. Any kernel lever (§5a,
§1) must beat adaptive-① end-to-end (interleaved A/B, the rigor above) to justify its
build + maintenance cost.
