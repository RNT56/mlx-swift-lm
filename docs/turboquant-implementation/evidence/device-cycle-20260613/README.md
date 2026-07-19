# A-series device cycle — first on-device evidence (2026-06-13)

First **physical A-series** run of the TurboQuant compressed-KV attention kernels. Executed on
the held device **GBU-12 = iPhone 15 Pro Max (`iPhone16,2`, A17 Pro), iOS 26.5 (23F77)**.

## How it was run (reproducible)

The SwiftPM unit-test bundle (`MLXLMTests/TurboQuantBenchSuite`) **cannot** run on a physical
device — `xcodebuild` rejects it with *"Tool-hosted testing is unavailable on device
destinations."* The on-device vehicle is the purpose-built **Pines DEBUG app host**:
`Pines/App/PinesTurboQuantBenchmarkDiagnostics.swift`, gated by `PINES_TURBOQUANT_BENCH=1`,
invoked from `PinesRootView` at launch, writing JSON to `Documents/PinesDiagnostics/`. The bench
drives the **same public Metal kernels** the production cache uses on **synthetic** K/V/Q at the
Qwen3.5-2B geometry (kv=4, q=16, head_dim=256), so no model weights are needed on device.

```bash
# build (signing worked with the user's Xcode-managed WLDMWQ963U profiles; the headless
# Metal-plugin trust gate needs the two skip flags — NOT a signing issue):
xcodebuild -project Pines.xcodeproj -scheme Pines -configuration Debug \
  -destination 'platform=iOS,id=00008130-00041C6E2EB8001C' \
  -derivedDataPath .build-ios-pines -allowProvisioningUpdates \
  -skipPackagePluginValidation -skipMacroValidation build
xcrun devicectl device install app --device <id> .build-ios-pines/Build/Products/Debug-iphoneos/pines.app
xcrun devicectl device process launch --device <id> --terminate-existing \
  --environment-variables '{"PINES_TURBOQUANT_BENCH":"1","PINES_TQ_BENCH_FULL":"1","PINES_TQ_BENCH_RUN_ID":"device-full-20260613",...}' \
  com.schtack.pines
xcrun devicectl device copy from --device <id> --domain-type appDataContainer \
  --domain-identifier com.schtack.pines \
  --source Documents/PinesDiagnostics/pines-turboquant-bench-device-full-20260613.json --destination .
```

Pins: **mlx-swift `6bfa04e` + mlx-swift-lm `295e66b`** (Pines' pinned revs; recorded in the payload
`appHost.mlxPinPair`). Full matrix: 12 iterations / 3 warmup per cell. Payloads in this dir.

## Full matrix — `polar_qjl` codec, 15/15 ok, 0 failed, 0 skipped

```
scheme    ctx    comp tok/s  plain tok/s  ratio   mlx/swift×  nativeGate  cosine     mem×
turbo8    8K        100.4      463.4       0.22      2.39        -         0.998471   2.21
turbo4v2  8K        100.8      459.1       0.22      2.32        -         0.999992   2.21
turbo3_5  8K        102.0      473.9       0.22      2.37        -         0.997620   2.21
turbo8    16K        77.0      306.8       0.25      1.81        -         0.999953   2.21
turbo4v2  16K        71.1      338.3       0.21      1.68        -         0.999993   2.21
turbo3_5  16K        70.8      303.0       0.23      1.66        -         0.999833   2.21
turbo8    32K        78.5      178.3       0.44      1.84        fail      0.998999   2.21
turbo4v2  32K        74.4      195.3       0.38      1.73        fail      0.999504   2.21
turbo3_5  32K        72.7      176.0       0.41      1.73        fail      0.999263   2.21
turbo8    64K        45.2       96.5       0.47      1.60        fail      0.999818   2.21
turbo4v2  64K        44.3       98.5       0.45      1.59        fail      0.999569   2.21
turbo3_5  64K        45.4      100.2       0.45      1.64        fail      0.999737   2.21
turbo8    128K       19.9       49.7       0.40      1.73        fail      0.999696   2.21
turbo4v2  128K       21.8       49.5       0.44      1.48        fail      0.999953   2.21
turbo3_5  128K       21.9       49.4       0.44      1.45        fail      0.999802   2.21
```

## What it shows

- **Quality holds on device:** cosine vs FP16 is 0.9976–0.99999 at every context incl. 131K — no
  codec regression on A-series.
- **2.21× KV memory reduction**, context-independent (the context-unlock metric).
- **Compressed never reaches FP16 parity** at any context (max ratio ≈0.47 at 64K) → on device,
  **FP16 is faster whenever it fits**; compressed's value is reaching contexts FP16 can't. This
  **validates the dynamic FP16↔compressed strategy on real silicon.**
- **Ratio improves with context** (0.22 @8K → 0.47 @64K) then dips at 131K (≈0.42) where absolute
  decode is slow (~20 tok/s compressed). The compressed path is **relatively more competitive on
  A17 Pro than on M2 Pro**: M2-Pro baseline ratios were ≈0.07–0.08 @8K → ≈0.14–0.18 @128K; A17 Pro
  is ≈0.22 @8K → ≈0.42 @128K — roughly **2–3× better ratio per context** (different bandwidth/ALU
  balance), but the M2 conclusion (FP16 wins when it fits) **transfers**.
- **Native perf gate `fail` at ≥32K** is the pre-registered native-vs-Swift-Metal **2.0×** bar:
  native MLX is 1.45–1.84× faster than the Swift-Metal compressed fallback — a real win, but below
  the 2.0× promotion bar on this device.

## Nonclaims / scope (honest)

- This is **synthetic-attention-shape** throughput (the payload `comparisonBasis` says so);
  release/product promotion still requires **real-model-inference-v1**. This run characterizes the
  **kernels on real A-series silicon**; it promotes nothing.
- **G2 (thermal soak)** and **G3 (① on-device byte-identity + speedup)** are NOT covered by this
  vehicle. The Pines bench at pin `295e66b` predates `TurboQuantBenchEnvironment`/`captureEnvironment`
  (mlx-swift-lm `0be78ef`), so there is **no per-context thermalState / memory trajectory** in this
  payload — only host model + KV bytes. Capturing G2 here needs a Pines pin bump to pick up the env
  capture (held). G3 needs a full-model ①-A/B path in an app host (next increment).
- `mem×` = 2.21 here (uniform across schemes) reflects the `295e66b` KV-byte accounting; earlier M2
  harness notes cited 3.05× for turbo4v2 at a different pin. Reported as measured.
