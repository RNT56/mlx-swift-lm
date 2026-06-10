# A-series device cycle — pre-registered measurement campaign (2026-06-07)

The one remaining unmeasured assumption is **the thesis**: does compressed long-context
inference actually work on an A-series die? It has repricing power over the whole program
(incl. §4 priority and the parked capacity contingency), and ① runs **today** on the dense
model, so the cycle needs **nothing built first**. Scope = a measurement **campaign, not a
ship**. All M2-Pro verdicts (launch/compute-bound, the block-count ladder, the §2/§5a NO-GOs)
are tagged device-conditional; the ghost_mode + block-count env overrides travel for free and
settle transfer with one extra env var per run.

Run on the physical device (the held A-series cycle; GBU-12 per the N2 Pines notes). Canonical
models: **Qwen3-4B-4bit (dense)** + **Qwen3.5-2B-4bit (hybrid)**. Use the on-device harness
(`TurboQuantBench` / `run-ios-turboquant-bench.sh`) + `xcodebuild test`.

## Pre-registered gates (pass/fail fixed BEFORE the run)

### G1 — RAM fit (the binary product question)
Target model + affine KV at the target context fits in the device budget **with OS headroom**
(no jetsam). Report: resident model + resident KV bytes + peak/steady active + the jetsam
limit for the device class. PASS = fits at the target context with ≥ a safe headroom margin;
record the **largest context that fits** for dense and for hybrid. (This is what decides §4
priority: if the dense model fits the target contexts on the device's RAM class, §4 is
nice-to-have; if only the hybrid fits, §4 becomes critical.)

### G2 — sustained vs burst (thermal)
tok/s at a **10-minute thermal soak**, not first-minute. Log `ProcessInfo.thermalState`
throughout; report first-minute tok/s, 10-min-sustained tok/s, and the % drop. PASS =
sustained tok/s within an acceptable fraction of burst AND thermalState does not reach
`.critical` during the soak (flag/non-promote any run that does).

### G3 — ① on-device (re-earn the bit-exactness + the speedup)
- **Byte-identity gate RE-VERIFIED on-device** (the M2-Pro proof is same-device internal
  consistency — re-earn it on A-series, do not assume transfer): `--validate-speculative`
  greedy output **byte-identical** to plain greedy on short/8K/16K. FAIL-CLOSED on any mismatch.
- **Realized ① speedup** at adaptive-k and fixed-k: report tok/s vs plain at 8K/16K/(32K if it
  fits G1), the **on-device crossover context** (where ① ≥ 1.0×), and acceptance. PASS = a
  clear ≥1.0× win at the device's long-context regime with determinism intact.

### G4 — FP16↔affine crossover + the carried ghost/block configs (does the M2 verdict transfer?)
- **FP16↔compressed crossover on A-series:** the context where affine is faster than (or the
  only thing that fits vs) FP16 — re-measure (A-series bandwidth/ALU ratio differs).
- **Ghost decomposition on-device** (one extra env var per run): `TURBOQUANT_GHOST_SDPA_MODE`
  ∈ {0,2,3} at 16K/32K → does the **launch/merge-bound** verdict (~45–63% launch on M2) hold,
  or is A-series bandwidth-bound (less BW per ALU)? If A-series is bandwidth-bound, §2/§3b/§3c
  revive; if still launch-bound, the **block-count ladder cap** is the lever there too.
- **Block-count sweep on-device:** `TURBOQUANT_SDPA_DECODE_BLOCKS` ∈ {64,128,256,512} at
  16K/32K → the device-optimal decode block count (the M2 optimum 256 is device-specific;
  re-tune). If a cap beats the default with A/B+CI, that is a tiny, low-risk `select_sdpa_blocks`
  change to land in the next coordinated mlx-swift bump.

## Evidence rigor (mandatory, per the pre-registration)
- **Interleaved A/B repeats** (plain vs candidate, order-randomized), NOT back-to-back blocks.
- **Bootstrap CIs** on every tok/s ratio; report the CI, not a point estimate.
- `thermalState` logged per trial; flag `.serious`/`.critical` runs non-promotable.
- Emit a machine-readable artifact: `{schemaVersion, device, repoCommits{mlx,mlx-c,mlx-swift,
  mlx-swift-lm,pines}, model, contexts, gates{G1..G4}, arms[{name, msSamples, medianMs, ci95}],
  thermalStates}` under `artifacts/`. Assemble `compatibility-pair.json` from this REAL
  evidence (status stays `failed`/`unverified` until compressed throughput meets the bar —
  hand-editing it is fabrication, guardrail-blocked).

## Decision logic the cycle feeds
- G1 outcome sets **§4 priority** (dense-fits ⇒ §4 nice-to-have; hybrid-only ⇒ §4 critical).
- G1 RAM pressure **revives the parked capacity contingency** (affine-V3 + N5 recency, no new
  kernels).
- G4 ghost/block results decide whether the M2 launch-bound verdict (and the ladder-cap lever)
  **transfer**, or whether A-series is bandwidth-bound (reviving §2/§3).
- ① ships **default-off** until G3 passes on-device; the working tree is behavior-inert until
  enabled + device-validated (N2 Pines wiring is applied-but-held for exactly this).

## §4 in parallel (CPU-verifiable, kernel-free, can't be invalidated by device data — only repriced)
SSM per-position rollback (turn ① on for the Qwen3.5 hybrid). Build alongside the device cycle
if agent bandwidth allows; else it follows. Gate: byte-identical output AND byte-identical
cacheStateHash (both MambaCache slots) under the differential fuzz {accept 0/partial/full/
full+bonus, adversarial proposers}. Per-step states fp32; `isTrimmable` armed only around the
verify; N7 prefetch OFF for hybrids this cycle.
