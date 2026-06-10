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

**#2 §5a prune-rate keystone (BEFORE writing Metal) — FAILED for plain norm-pruning.**
Dequantized the Qwen3-4B tied head (`model.embed_tokens`, 4-bit g64, 151936×2560) and
measured the row-norm spread: **std/mean 0.156, p50 1.127, p95 1.294, p99 1.353** —
tightly clustered. Cauchy-Schwarz prune fraction = P(‖w_i‖ < ‖w*‖·cos*): **0% at
cos*≤0.3, 3% @0.5, 5% @0.7, 30% @0.95, 50% only @cos*=1.0**. LM-head top-1 cosines are
low (~0.1–0.4: the hidden state is not aligned to a single embedding row), so realized
pruning ≈ **0–5%** — far below the pre-registered ≥50%@k1. **Plain norm-pruning is dead
before any kernel** (the exact "clustered norms ⇒ prune≈0" failure mode). Pivot per the
gate = **cluster bounds** (k-means centroids, bound `⟨h,c⟩ + ‖h‖·r` — uses direction, not
just magnitude), but that is a larger build with its own realized-prune-rate gate, and
#1 already demoted §5a to incremental. **Verdict: §5a deprioritized this cycle** (both
gates point away); revisit only as cluster-bounds with a fresh gate if the lm_head verify
cost becomes the binding constraint after §2/§3.

**#3 §3a ghost-kernel attribution (BEFORE §2 codec-in-SDPA) — DONE: affine decode is
launch/ALU-bound, NOT bandwidth-bound ⇒ §2 is NO-GO.** Built a `ghost_mode` function
constant (id 40) in the affine SDPA kernel (`mlx-swift` submodule
`backend/metal/{kernels/sdpa_vector.h, scaled_dot_product_attention.cpp}`), env-driven
(`TURBOQUANT_GHOST_SDPA_MODE`), default 0 = production byte-identical (distinct pipeline
via the `_ghost` hash). Mode 2 = math-only (hold the per-key K/V code/scale/bias pointers
⇒ loads collapse to L1, full dequant/dot/accumulate runs every iter ⇒ isolates ALU+launch
from DRAM streaming). Measured (Qwen3-4B affine K8/V4, hd128 16Q/8KV; artifacts/
turboquant-ghost-sdpa-20260607/, gitignored): **removing DRAM streaming saves only ~12%
@16K, ~8% @32K, ~0% @65K/131K.** Corroborated: affine is *slower* than FP16 @16K (fp16x
0.874) despite moving 2× fewer bytes. **Verdict: the affine decode is launch/occupancy/ALU-
bound, not bandwidth-bound** — the "0.39–0.62× effective bandwidth" is a red herring; bytes
move in the shadow of the launch/compute cost. **⇒ §2 (port polarLM LUT-gather + d·log d
WHT butterfly INTO this kernel) is NO-GO**: it adds heavy per-element math to a kernel with
no compute headroom and would regress (precisely why PolarQJL lost round one — the codec
*math*, not just coalescing). **§3b (inline metadata) and §3c (DRAM prefetch) are also low-
value** (only ~12% of time is DRAM to recover). The N4 Gaussian capacity codec must live on
a decode path / accept the slow diagnostic path — it cannot be made fast by porting into the
affine kernel. The launch/occupancy-bound component is the only structural lever left
(more SIMD-groups per threadgroup; PERF_AUDIT #1) — a separate investigation.
*Instrument status:* the two submodule edits are default-off/production-safe and left in the
working tree (uncommitted) to avoid an uncoordinated mlx-swift pin bump; include them in the
next coordinated mlx-swift commit. Reproduce: `TURBOQUANT_GHOST_SDPA_MODE=2
.build/release/TurboQuantNativeVxBenchmark --value-bits 4 --query-lengths 1 --head-dim 128
--query-heads 16 --kv-heads 8 --contexts 16384,32768`.

**#3 §3a second-stage (launch vs ALU vs LSU) — DONE: affine decode is LAUNCH/MERGE-bound,
and the block-count ladder OVER-SPLITS batch=1 decode.** Added ghost_mode==3 (launch-only:
skip all per-key work, keep grid/ladder + pass-2 merge). 3-mode decomposition (Qwen3-4B
affine K8/V4 hd128; artifacts/turboquant-ghost-sdpa-20260607/): **launch/dispatch/merge is
~45–63% of the kernel** (0.82/1.55/1.55/1.70 ms @16/32/65/131K), tracking the block-count
ladder (256/512/512/1024) almost exactly — confirming the N-trend fingerprint (the growing
cost is block count, i.e. dispatch + pass-2 merge, NOT per-element ALU/bytes). **Block-count
sweep (`TURBOQUANT_SDPA_DECODE_BLOCKS`, env override, no new kernel): 256 beats the default
512/1024 by ~10–15% at ≥32K** (16K 256=1.63 vs 512=2.35; 32K 256=2.01 vs 512=2.32; 65K
256=2.38 vs 512=2.62; b1024 worst everywhere). **Lever = a decode-specific ladder cap (~256)
in `select_sdpa_blocks`** — tiny, env-validated, no codec change. CAVEATS: single-sample
(needs the A/B+CI rigor below before changing the ladder); attention is a minority of
weight-dominated decode (≤47K) so the end-to-end win is a few %; **M2-Pro-measured** — the
optimal block count is device-specific, re-tune on A-series; production change is
coordination-gated (mlx-swift pin bump). PERF_AUDIT #1 is re-pointed at the **launch/merge
path** (ladder cap + cheaper pass-2 merge), not simdgroups-per-threadgroup.
NOTE the instrument can't yet split launch- vs ALU- vs LSU-bound below the launch floor
(ghost keeps load instructions, cache-hitting) — a further ALU-vs-LSU stub (codec math → 1
FMA, loads kept) is the next decomposition if the ladder cap doesn't fully close the gap.

**Repricing the §2 / §5a / §1 NO-GOs (annotation, honest record):**
- §2 is "compute-bound kernel + a *nonzero* per-element delta (the deferred-WHT V path is ONE
  threadgroup LUT gather; rotation/residual were algebraically removed; the butterfly is
  per-query, amortized to ~0) + *modest* marginal capacity (the big jump needs LM-K5, which
  DOES touch the hot K loop) ⇒ **not worth it NOW**" — a **priced contingency**, NOT "any codec
  math regresses heavily." Revive if the device cycle changes the boundedness.
- **The capacity story is NOT closed** — it moved to the lever that survives a compute-bound
  kernel **for free: affine-V3** (already native in the value-bits gate, same FMA decode) **+
  N5 recency tiering** (wide protected-edge). If the device cycle shows RAM pressure, that is
  the play, **no new kernels**.
- **Every NO-GO above is M2-Pro-measured.** A-series has less bandwidth per ALU + different
  launch/thermal behaviour, so the boundedness verdicts may shift on-device. The ghost_mode
  and the block-count env overrides are function-constant / env — they **travel to the device
  for free**; carry them in the device cycle (one extra env var per run settles transfer).

## The live baseline (must be beaten to justify §5a / §1-batched-qmv)
**Adaptive-k** (landed, opt-in, bit-exact): per-round proposal width tracks the
acceptance EMA, defaulting to the cheap width (≤2, below the lm_head verify cliff)
and widening only when acceptance is high. Zero new kernels. Any kernel lever (§5a,
§1) must beat adaptive-① end-to-end (interleaved A/B, the rigor above) to justify its
build + maintenance cost.
