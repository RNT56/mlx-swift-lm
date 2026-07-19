# TurboQuant Continuation Handoff - 2026-06-07

This is the current pickup point for agents continuing the upstream-equivalent
TurboQuant work in `mlx-swift-lm` and the paired native `mlx-swift` kernels.

## Current Status

- The practical upstream-comparable route is still affine K8/V4: affine/Q8
  keys with 4-bit affine values. Upstream's visible README K8+V4 row at commit
  `6e928d7` is not the PolarWHT-V route; it is the mixed affine route using
  MLX quantized matmul and mixed quantized SDPA.
- Full `polarWHTV3` and value-only `hybridK8PolarWHTV3/V4` are implemented as
  experimental diagnostics, not default promotion candidates. The hybrid path
  can pass bounded logit quality but is far slower than affine K8/V4 locally.
- Promotion now fails closed unless the throughput row selects the expected
  native compressed path and the quality row, when required, also selects the
  expected native compressed path.
- Affine dynamic conversion now triggers at `offset >= quantizedKVStart`.
  Default affine K8/V4 starts at `16_384`, so 4K runs correctly report raw
  residency and no compressed native path.
- Dynamic conversion is materialized immediately after conversion: converted
  cache state is evaluated, the GPU stream is synchronized, and cache pressure
  is cleared. Conversion cost is reported under
  `dynamicCacheQuantizationSeconds`, usually in the prompt-prefill timing
  block rather than the first decode token.

## Upstream PR Branch Queue - 2026-06-25

The authoritative local delivery contract for the upstream queue, fork-stack
staging branches, dependency order, validation gates, and product evidence is
[Fork Stack Delivery Roadmap - 2026-06-25](fork-stack-delivery-roadmap-2026-06-25.md).
The local preparation packet for future upstream PRs is
[Upstream PR Launch Packets - 2026-06-25](upstream-pr-launch-packets-2026-06-25.md).
The focused support plan for active upstream `ml-explore/mlx#3026` is
[MLX #3026 Support Roadmap - 2026-06-26](mlx-3026-support-roadmap-2026-06-26.md).

- `mlx-swift-lm` upstream maintenance queue:
  - `upstream-pr/vlm-processor-completions` is open as
    `ml-explore/mlx-swift-lm#301`, approved, and waiting on GitHub-side
    `mac_build_and_test` recovery/rerun.
  - `upstream-pr/model-compatibility-docs` is open as
    `ml-explore/mlx-swift-lm#303` and was marked ready after docs validation.
  - `upstream-pr/rope-config-validation` is open as
    `ml-explore/mlx-swift-lm#371` and replaces the useful narrow slice of the
    former broad runtime-hardening draft.
  - `upstream-pr/runtime-stop-strings` is open as
    `ml-explore/mlx-swift-lm#372`. It was split from the broader runtime
    parity branch as a single stop-string handling commit and deliberately
    excludes DFlash/MTP/TurboQuant stack changes.
  - Drafts `#302` and `#304` were closed because they were too broad for
    reviewable upstream PRs.
- `mlx-swift` upstream maintenance queue:
  - `upstream-pr/swiftpm-metal-library-resource` is open as draft
    `ml-explore/mlx-swift#430`.
  - The former MLX automatic SwiftPM bundle lookup route is closed:
    `ml-explore/mlx#3767` duplicated the approach rejected in `#3562` and must
    not be reopened.
  - The accepted SwiftPM Metal foundation is the explicit path chain:
    `mlx#3597` merged C++ `set_metallib_path`,
    `mlx-c#117` draft C wrapper,
    `mlx-swift#416` draft `GPU.setMetallibPath`, then draft `mlx-swift#430`.
  - Local proof branches validate the corrected route without making `#430`
    upstream-ready yet:
    `mlx-c/upstream-pr/metallib-path-c-api` at `6173b85` and
    `mlx-swift/prep/swiftpm-metallib-via-path-api` at `57449af`.
  - `upstream-pr/linalg-norm-kind-nuc` is already merged upstream and is
    archival only.
- `mlx` quantized SDPA queue:
  - Do not open a competing main-based PR for the current quantized SDPA stack.
    Upstream already has active `ml-explore/mlx#3026` for the core API/kernels.
  - Keep future `mlx/upstream-pr/quantized-sdpa-api-tests` and
    `mlx/upstream-pr/quantized-sdpa-metal-kernels` local-only until `#3026`
    lands or maintainers request a stacked split.
  - Local `pr/quantized-sdpa-followups` is a one-commit follow-up on top of
    `#3026` (`6 files changed, 184 insertions, 11 deletions`) but is still a
    large stale diff when compared to upstream `main`. Rebase it only after
    `#3026` lands or if maintainers ask for a follow-up against that PR branch.
  - The next inspected candidates are not upstream-ready as small PRs:
    `pr/runtime-parity-dflash-mtp` depends on broader lazy-load/MTP runtime
    work, and `mlx-swift` runtime/SSD commits are submodule or generated-runtime
    changes tied to larger native stacks.
  - The broader `codex/*`, `pr/turboquant-*`, and `integration/*` branches are
    staging branches, not upstream-ready PR branches.

## Key Local Files

- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Libraries/IntegrationTestHelpers/InferenceParityBenchmark.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Libraries/MLXLMCommon/TurboQuantTiming.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/tools/TurboQuantInferenceParity/main.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Tests/MLXLMTests/TurboQuantInferenceParitySparseVTests.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift`

## Verified Commands

These passed after the latest implementation changes:

```bash
swift test --filter TurboQuantInferenceParitySparseVTests
swift test --filter KVCacheTests/testAdaptiveTurboQuantStartsRawAndConvertsAtThreshold
swift test --filter KVCacheTests
python3 scripts/turboquant-polarwht-acceptance-gate.py --self-test
swift build -c release --product TurboQuantInferenceParity
```

SwiftPM Metal explicit-path preparation validated on 2026-06-25:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/.upstream-pr-worktrees/mlx-c-metallib-path-api
cmake --build build/metallib-path-api
ctest --test-dir build/metallib-path-api --output-on-failure
git diff --check upstream/main...HEAD

cd /Users/mt/Programming/Schtack/mlx-forks/.pr-worktrees/mlx-swift-metal-resource-clean
swift build --target MLX
swift build --target MLXNN
bash -n tools/build-swiftpm-metallib.sh
SDK_NAME=macosx tools/build-swiftpm-metallib.sh /tmp/mlx-swift-default-macos-pr430.metallib
git diff --check upstream/main...HEAD

cd /Users/mt/Programming/Schtack/mlx-forks/.pr-worktrees/mlx-swift-metal-resource-via-path-api
swift build --target MLX
swift build --target MLXNN
swift test --filter SwiftPMMetallibResourceTests
swift test --filter StreamTests/testDeviceType
bash -n tools/build-swiftpm-metallib.sh
SDK_NAME=macosx tools/build-swiftpm-metallib.sh /tmp/mlx-swift-default-macos.metallib
git diff --check upstream/main...HEAD
```

The `ctest` command above found no tests in the configured `mlx-c` tree. Treat
that as no CTest coverage, not a failing test.

Native K8/V4 microbenchmark command:

```bash
.build/release/TurboQuantNativeVxBenchmark \
  --contexts 16384,32768 \
  --value-bits 4 \
  --iterations 5 \
  --warmup 1 \
  --cooldown-ms 25 \
  --query-heads 16 \
  --kv-heads 8 \
  --head-dim 128 \
  --output artifacts/turboquant-hybrid-smoke-20260607/native-vx-qwen06b-k8v4.json
```

It reported K8/V4 native attention at about `0.94x` FP16 for 16K and `1.43x`
FP16 for 32K in the operator-only harness. Treat this as kernel regression
evidence, not real-model promotion evidence.

## Latest Artifacts

Artifact root:

```text
/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/artifacts/turboquant-hybrid-smoke-20260607
```

Important rows:

- `diagnostics-4k-g32-fp16-affinek8v4-fixed.json`: 4K affine K8/V4 correctly
  remains raw because the default quantized start is 16K. It reports `1.00x`
  resident KV compression and blocks promotion with
  `affine compressed native path was not selected`.
- `diagnostics-16k-quality-affinek8v4-threshold-fixed.json`: 16K affine K8/V4
  quality passes with top-1 `1.000`, KL `1.1265e-7`, p95 max-logit error
  `1.0859`, and cosine `0.9942`. This artifact predates the newest quality
  path fields, so do not use it to verify `qualitySelectedAttentionPaths`.
- `diagnostics-16k-g1-affinek8v4-quality-path-fields.json`: early schema smoke
  from the quality path field work. Superseded by the combined rerun below.
- `diagnostics-16k-g32-fp16-affinek8v4-materialized-conversion.json`: 16K
  throughput after materialized dynamic conversion. FP16 measured `12.23`
  tok/s and affine K8/V4 measured `44.37` tok/s with
  `selectedAttentionPaths=["affineK8V4Native"]`,
  `residentKVCompressionRatio=2.13`, and prompt
  `dynamicCacheQuantizationSeconds=2.455`. This is a single ordered sample and
  did not run quality in the same report, so it is not a promotion result.
- `diagnostics-16k-g32-fp16-affinek8v4-quality-materialized-conversion.jsonl`:
  zero-byte failed attempt. Superseded by the rerun artifact below.
- `diagnostics-16k-g32-fp16-affinek8v4-quality-materialized-conversion-rerun.json`:
  combined 16K quality plus 32-token throughput report from 2026-06-25. FP16
  measured `8.55` decode tok/s; affine K8/V4 measured `42.84` decode tok/s
  (`5.01x` FP16), selected `affineK8V4Native` on all 28 layers, reported
  `residentKVCompressionRatio=2.13`, allocated no raw or decoded fallback, and
  passed quality with top-1 `1.000`, KL `1.1265e-7`, p95 max-logit error
  `1.0859`, and cosine `0.9942`. This is usable single-run product evidence,
  but still not a product-readiness result without repeated randomized runs and
  physical-device/compatibility-pair evidence.

## Completed 16K Rerun

The combined 16K quality plus 32-token throughput report completed on
2026-06-25:

```bash
swift build -c release --product TurboQuantInferenceParity

TQ_QUALITY_PRINT_CACHE_DIAGNOSTICS=1 \
.build/release/TurboQuantInferenceParity \
  --model-dir /Users/mt/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --contexts 16384 \
  --generate-tokens 32 \
  --throughput-repeats 1 \
  --throughput-cooldown 0.25 \
  --configs fp16,affineK8V4 \
  --quality-gates \
  --quality-contexts 16384 \
  --quality-cooldown 0.5 \
  --turboquant-timing \
  --diagnostics-output artifacts/turboquant-hybrid-smoke-20260607/diagnostics-16k-g32-fp16-affinek8v4-quality-materialized-conversion-rerun.json
```

Observed pass conditions:

- `throughput[].label=="affineK8V4"` has
  `selectedAttentionPaths=["affineK8V4Native"]` and
  `codecCounts.affineK8V4Native=28`.
- `throughput[].promotionGate.qualitySelectedAttentionPaths` includes
  `affineK8V4Native`.
- `throughput[].residentKVCompressionRatio == 2.1333333333333333`.
- `throughput[].promotionBlockReasons` is empty and
  `promotionEligible == true`.
- raw fallback and decoded fallback are both false.
- `quality[].label=="affineK8V4"` includes `selectedAttentionPaths` and
  `codecCounts`, and `passed == true`.
- `promptPrefillTiming.dynamicCacheQuantizationCalls == 1` for affine K8/V4 and
  generation dynamic quantization is zero.

## Next Development Work

1. Keep `mlx-swift#430` draft until `mlx-c#117` and `mlx-swift#416` are
   available through upstream-owned commits/submodule pins.
2. Run randomized repeated 16K and 32K affine K8/V4 reports:

```bash
.build/release/TurboQuantInferenceParity \
  --model-dir /path/to/mlx-model \
  --contexts 16384,32768 \
  --generate-tokens 32 \
  --throughput-repeats 3 \
  --throughput-cooldown 0.25 \
  --randomize-throughput-order \
  --configs fp16,affineK8V4 \
  --quality-gates \
  --quality-contexts 16384,32768 \
  --quality-cooldown 0.5 \
  --turboquant-timing \
  --diagnostics-output artifacts/turboquant-current/affine-k8v4-16k-32k.json
```

3. Build the same-machine upstream reproduction harness and stop using
   `TQ_POLARWHT_SKIP_UPSTREAM=1` for promotion runs.
4. Compare local affine K8/V4 against reproduced upstream K8+V4, current
   `affineInt4`, and FP16 at 32K, then 64K/128K where FP16 fits.
5. Keep PolarWHT hybrid promotion disabled by default. Only set
   `TURBOQUANT_ENABLE_POLARWHT_HYBRID_PROMOTION=1` for deliberate experiments
   that report the experimental blocker and compare against dense affine K8/V4.

## Nonclaims

- Do not claim `0.98x` FP16 for this branch. Upstream's visible pinned README
  K8+V4 row is about `0.72x` FP16.
- Do not claim the single 16K `42.84` tok/s combined quality/throughput rerun as
  product-ready. It is useful evidence that affine K8/V4 selects the native path
  and passes quality in one report, but it still lacks repeated randomized
  samples and physical-device/compatibility-pair validation.
- Do not promote full PolarWHT K/V or hybrid K8+PolarWHT-V. They remain
  diagnostic until native path, quality, resident memory, fallback, and
  same-machine upstream comparison gates all pass.
