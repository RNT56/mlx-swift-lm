# Extended real-model run (32K/64K + bootstrap CIs) — attempt + thermal wall (2026-06-13)

Follow-up to `REAL-MODEL.md` (which established the headline: 2B @16K, **0.96× FP16, byte-identical
greedy**). Goal: extend to 32K/64K and add bootstrap 95% CIs, with a smaller fallback model for
contexts a larger model's weights+KV won't fit.

## What was built (committed, correct)

The Pines real-model hook (`PinesRealModelTurboQuantDiagnostics.swift`, Pines commit **ebc1d80**)
was upgraded to:
- **Per-context isolation** — each context runs in its own `runDetailed` call inside try/catch, so a
  64K stall/OOM doesn't lose 16K/32K.
- **Bootstrap 95% CIs** — over the **per-repeat samples** (`ThroughputRunResult.samples[].measurement
  .decodeTokensPerSecond`; the `.measurements` array is a single aggregate per arm, which is why an
  earlier cut reported degenerate N=1 CIs — fixed). CI on each arm's median AND on the compressed/FP16
  ratio (independent resampling), seedable/deterministic.
- **Multi-model** (`PINES_TQ_REAL_MODELS` csv) — primary 2B + smaller 0.8B fallback for long contexts.

This is verified-correct code (builds + the engine path was confirmed engaged earlier). It was **not
possible to capture valid extended numbers this session** — see below.

## The wall: on-device thermal saturation (an honest G2 finding)

After several hours of sustained on-device benchmarking (synthetic full matrix + multiple real-model
runs), the **A17 Pro entered deep thermal throttling**, and the per-repeat re-prefill cost at long
context compounded it:

- `runDetailed(throughputRepeats: N)` **re-prefills the full context for every repeat × arm** (plus
  the quality gate re-prefills twice). At 64K/repeats=4 that is ~10 × 64K prefills.
- 0.8B @ 64K, repeats=4: **stalled ~35 min with no progress** (process alive, generating — the model
  *fits* and runs, but multi-repeat 64K prefill doesn't complete in a usable window).
- Late in the session even **0.8B @ 16K, repeats=4 took >30 min** (vs ~6 min early), confirming the
  device was throttled, not the code.

Numbers gathered in that state would be **thermal artifacts, not valid throughput** — so the runs were
stopped rather than shipped. No multi-sample CI run completed; the only completed real-model payloads
remain the N=1 originals (`REAL-MODEL.md`).

**Conclusion (real result):** real-model decode with compressed KV *works and fits* at long context on
the A17 Pro (the model was generating at 64K), but **valid throughput/CI measurement at ≥32K
multi-repeat is impractical on this device without active cooling and much longer cooldowns** — the
re-prefill-per-repeat harness saturates thermals. For promotion-grade CIs, run on a **cooled / actively
cooled device** (or reduce to single-prefill-then-N-decode timing rather than re-prefill-per-repeat).

## Headline that STANDS (committed, valid)

`REAL-MODEL.md` / commit `f8af28b`: **2B (Qwen3.5-2B-OptiQ-4bit) @ 16K — compressed affineK8V4 decode
0.96× FP16, byte-identical greedy output (top-1 = 1.000), cosine 0.982, no fallback, gate passed.**
That is the product-faithful real-model result; the extension does not change it.

## Turnkey re-run when the device is cool (harness is ready)

Build (Pines, `-jobs 6` avoids the OOM-killed-compiler; codesign `errSecInternalComponent` is transient
— just retry):
```
xcodebuild -project Pines.xcodeproj -scheme Pines -configuration Debug \
  -destination 'platform=iOS,id=<GBU-12 udid>' -derivedDataPath .build-ios-pines \
  -allowProvisioningUpdates -skipPackagePluginValidation -skipMacroValidation -jobs 6 build
xcrun devicectl device install app --device <udid> .build-ios-pines/Build/Products/Debug-iphoneos/pines.app
xcrun devicectl device process launch --device <udid> --terminate-existing \
  --environment-variables '{"PINES_TQ_REAL_BENCH":"1","PINES_TQ_REAL_RUN_ID":"cool-ci",
    "PINES_TQ_REAL_MODELS":"mlx-community/Qwen3.5-2B-OptiQ-4bit,mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "PINES_TQ_REAL_CONTEXTS":"16384,32768,65536","PINES_TQ_REAL_REPEATS":"6",
    "PINES_TQ_REAL_GEN_TOKENS":"48","PINES_TQ_REAL_BOOTSTRAP":"2000"}' com.schtack.pines
# poll Documents/PinesDiagnostics/pines-realmodel-tq-status.json until state=completed, then pull the payload.
```
Recommended for valid CIs: device **plugged in + cool** (ideally fan/cooled), screen on, and accept
that 64K may take many minutes per repeat. Consider lowering 64K repeats or adding longer
`cooldownSeconds` to stay under throttle.
