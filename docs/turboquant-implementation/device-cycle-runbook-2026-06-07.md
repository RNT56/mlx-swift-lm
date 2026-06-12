# A-series device cycle — pre-registered measurement campaign (2026-06-07)

> **RESULTS (2026-06-13):** first physical A-series run landed — GBU-12 (iPhone 15 Pro Max /
> A17 Pro), full 8K→131K × turbo8/turbo4v2/turbo3_5 matrix, **15/15 ok**, via the Pines DEBUG
> app host (`PINES_TURBOQUANT_BENCH=1`). Quality holds (cosine 0.9976–0.99999), 2.21× KV memory,
> compressed/FP16 ratio 0.22 @8K → 0.47 @64K (never parity ⇒ dynamic FP16-when-it-fits validated;
> ~2–3× better ratio than M2 Pro). Synthetic-attention scope — promotes nothing; G2/G3 not covered.
> Evidence + provenance: `evidence/device-cycle-20260613/README.md`.

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

## Turnkey invocations (the benchmark is refined — 2026-06-11)

Two measurement vehicles; **which gate runs where is fixed by where the code can run**. A
SwiftPM executable runs only on macOS; only an `xcodebuild test` bundle runs on the A-series die.

### Vehicle A — `TurboQuantAcceptanceHarness` CLI (macOS; full model; G3 + G2-soak methodology)
Refined to emit the pre-registered rigor. `--ab-repeats N` interleaves plain/spec trials with the
arm order **alternated each trial** (cancels order/thermal drift), reports **bootstrap 95% CIs** on
both arms and on the speedup ratio, logs `thermalState` per trial, captures MLX active/peak memory,
and writes the machine-readable G1–G4 artifact. `--soak-seconds` is the G2 sustained-vs-burst loop.

```bash
cd mlx-swift-lm
swift build -c release --product TurboQuantAcceptanceHarness
# tokenize prompts externally (token IDs only — the in-package tool has no swift-transformers):
python3 - <<'PY'  # writes prompts-*.json = [{"label","ids":[Int]}], order them short→long
from transformers import AutoTokenizer; import json
# ... encode short / 8K / 16K / 32K contexts ...
PY
# G3 determinism + ① speedup with CIs (order prompts short→long so crossoverPrompt is meaningful):
TQ_DEVICE="<host>" TQ_COMMIT_MLX_SWIFT_LM=<sha> TQ_COMMIT_MLX_SWIFT=<sha> \
  .build/release/TurboQuantAcceptanceHarness --model-dir <dir> --prompt-ids-file prompts.json \
    --validate-speculative --ab-repeats 8 --bootstrap 2000 --ngram 3 --max-proposal 8 \
    --max-tokens 96 --output artifacts/<run>/validate-spec.json
# G2 soak (sustained vs burst), e.g. 600s = the mandated 10-minute thermal soak:
.build/release/TurboQuantAcceptanceHarness --model-dir <dir> --prompt-ids-file prompts.json \
    --soak-seconds 600 --max-tokens 128 --output artifacts/<run>/soak.json
# G4 ghost/block decomposition travels for FREE — one extra env var per run, re-run --validate-speculative:
TURBOQUANT_GHOST_SDPA_MODE=2 ...   # {0,2,3}; echoed into artifact.config.ghostSdpaMode
TURBOQUANT_SDPA_DECODE_BLOCKS=256 ...   # {64,128,256,512}; echoed into artifact.config.sdpaDecodeBlocks
```

Mac same-machine reference (2026-06-11, Qwen3-4B-4bit, ace7d5c/4a83f63, 3 repeats, 48 tok) PROVING
the rigor emits a valid artifact: `artifacts/device-cycle-prep-20260611/mac-ref-validate-spec.json`
— determinism PASS byte-identical; short 1.323× [1.323,1.417], ctx≈11K 1.777× **[0.969,2.033]** (the
wide CI at 3 repeats is the point — use **≥8 repeats** on device). NOTE: tok/s here includes the
one-time prefill amortized over the generated tokens (both arms equally), so the **ratio** is the
gate metric, not the absolute tok/s; raise `--max-tokens` to dilute prefill. `availableMemoryBytes`
is 0 on macOS (jetsam headroom is iOS-only) — the G1 RAM number is only trustworthy on device.

### Vehicle B — `MLXLMTests/TurboQuantBenchSuite` (A-series via xcodebuild; kernels; G1 + G2-trajectory + G4)
The on-device attention-throughput / quality / KV-bytes sweep. Refined (2026-06-11) to snapshot a
`TurboQuantBenchEnvironment` (thermalState + MLX active/peak/cache + iOS jetsam headroom) **before and
after each context's sub-sweep**, so the emitted `{environments, results}` JSON carries a G1 memory +
G2 thermal **trajectory** across the multi-minute full matrix. A `.critical` thermalState fails the run.

```bash
xcodebuild test -scheme mlx-swift-lm-Package \
  -destination 'platform=iOS,name=<device>' \
  -only-testing:MLXLMTests/TurboQuantBenchSuite TQ_BENCH=1
# G4 on-device: prefix the test action env with TURBOQUANT_GHOST_SDPA_MODE / TURBOQUANT_SDPA_DECODE_BLOCKS.
```

### Gate → vehicle map (honest)
- **G1 RAM fit** — Vehicle B `environments[].{activeMemoryBytes,peakMemoryBytes,availableMemoryBytes}`
  (jetsam headroom is real only here) + `results[].{compressedKVBytes,plainKVBytes,memoryReductionRatio}`.
- **G2 thermal** — Vehicle A `--soak-seconds` for the true 10-min sustained-vs-burst tok/s on the full
  model; Vehicle B's per-context environment trajectory for the kernel-sweep thermal drift. Both fail on `.critical`.
- **G3 ① byte-identity + speedup** — Vehicle A `--validate-speculative`. **GAP:** Vehicle A is macOS-only,
  so a literal *on-device* G3 needs a small full-model ①-A/B test added to the iOS test bundle (next
  increment); the M2-Pro byte-identity proof is same-device-internal and conservative to re-earn. Until
  then, run G3 on the Mac and treat the on-device ① speedup as carried by Vehicle B-adjacent evidence + Pines.
- **G4 ghost/block** — either vehicle via the two `TURBOQUANT_*` env vars; both echo them into the artifact.

## Decision logic the cycle feeds
- G1 outcome sets **§4 priority** (dense-fits ⇒ §4 nice-to-have; hybrid-only ⇒ §4 critical).
- G1 RAM pressure **revives the parked capacity contingency** (affine-V3 + N5 recency, no new
  kernels).
- G4 ghost/block results decide whether the M2 launch-bound verdict (and the ladder-cap lever)
  **transfer**, or whether A-series is bandwidth-bound (reviving §2/§3).
- ① ships **default-off** until G3 passes on-device; the working tree is behavior-inert until
  enabled + device-validated (N2 Pines wiring is applied-but-held for exactly this).

## §4 SSM per-position rollback (turn ① on for the Qwen3.5 hybrid) — DE-RISKED + foundation landed

**KEY FINDING: §4 needs NO Metal kernel change.** The GatedDeltaNet scan is a Metal kernel
(`GatedDelta.swift` `gatedDeltaKernel`) that loops T tokens and emits only the *final* state,
PLUS a Swift single-step reference (`GatedDelta.swift:180-213`). So during a q_seq=k verify the
scan can run **step-by-step via the existing single-step path** to capture per-position
recurrent states, while the projections (conv/in/out) stay **batched at m=k** (weight-
amortization preserved). The recurrent scan is O(k) sequential either way and the state is
small ⇒ negligible cost, verify-only.

**Foundation LANDED + verified (this cycle):** `MambaCache` rollback API
(`KVCache.swift`): `beginVerify()` snapshots pre-verify state+offset and arms recording;
`recordVerifyStep()` appends the (conv,ssm) state after each consumed verify token;
`selectAcceptedStep(consumed:)` commits `verifyStepStates[consumed-1]` and sets
`offset = preOffset + consumed` (the off-by-one); `discardVerify()` restores pre-state;
`supportsSpeculativeRollback = true`. 6 CPU unit tests pin the index mapping
(`MambaCacheRollbackTests.swift`). MambaCache stays NON-trimmable so standard speculation
admission still excludes it — the hybrid routes through `selectAcceptedStep`, not `trim`.

**Remaining wiring (next increment, gated by the fuzz determinism test):**
1. **Forward verify-mode:** when a layer's `MambaCache.inVerify` and S>1, run
   `Qwen35GatedDeltaNet` (Qwen35.swift:258-297) scan step-by-step over the S tokens via the
   single-step reference, calling `recordVerifyStep()` after each (conv state from the
   step's tail, ssm state from the step's `gatedDeltaUpdate`). Keep conv1d/projections batched.
2. **Iterator/admission:** add `canSpeculate(cache)` = each layer `isTrimmable` (attention) OR
   `supportsSpeculativeRollback` (recurrent); `makeGenerationIterator` admits the hybrid under
   it. In `NgramSpeculativeTokenIterator`: before the verify, `beginVerify()` on each MambaCache;
   after accept-j, `selectAcceptedStep(consumed: accepted+1)` on MambaCaches AND
   `trimPromptCache(k-accepted)` on attention caches (committed tokens this round = accepted
   drafts + 1 bonus). N7 prefetch stays OFF for hybrids (optimistic R+1 advances recurrent
   state past unverified R).
3. **CONVENTION (pin before wiring):** `consumed = accepted + 1` (greedy round commits ≥1);
   `verifyStepStates[i]` = state after consuming verify-input token i (the verify feeds
   `[seed]+draft`, so index alignment must match the iterator's cache-lag — verify against the
   fuzz oracle, do not assume).
4. **GATE (the proof obligation):** differential fuzz on the real Qwen3.5-2B hybrid —
   adversarial proposers {always-wrong, always-right, boundary, misprediction storm} × accept
   lengths {0, partial, full, full+bonus}, asserting **byte-identical output AND byte-identical
   cacheStateHash** (BOTH MambaCache slots: conv + fp32 ssm) after EVERY round vs the
   plain-greedy oracle; plus full-reject ⇒ slots byte-identical to the pre-verify snapshot,
   j-accept ⇒ slots byte-identical to j sequential single-steps. Per-step states MUST be fp32.
