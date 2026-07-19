# Fork Stack Delivery Roadmap - 2026-06-25

This is the local delivery contract for the MLX fork stack. It separates
reviewable upstream PRs from staging branches, records the current live PR
state, and defines the end-to-end path from upstream maintenance patches to a
validated downstream TurboQuant compatibility pair.

The goal is to be able to answer:

```text
Yes, finished - here is the reviewed upstream queue, here are the fork-stack
branches, here are the validation gates, and here is the product evidence.
```

## Current Verdict

The independent maintenance PR queue is prepared and open where it can be
opened safely. The TurboQuant upstream stack is not ready to open as a broad
PR: its first core dependency is upstream `ml-explore/mlx#3026`, and the local
follow-up branch should not compete with that PR against `main`.

Current practical state:

- Open and scoped upstream PRs exist for the reviewable maintenance slices.
- Broad draft PRs `ml-explore/mlx-swift-lm#302` and `#304` were retired instead
  of promoted.
- The former automatic MLX SwiftPM bundle lookup route is closed. `mlx#3767`
  was superseded by the explicit metallib path API route:
  `mlx#3597 -> mlx-c#117 -> mlx-swift#416 -> mlx-swift#430`.
- No RNT56-authored `ml-explore/mlx` PR is open; this is intentional while
  `ml-explore/mlx#3026` owns the core quantized SDPA API/kernel discussion.
- The downstream TurboQuant product path now has one same-report 16K real-model
  quality plus throughput artifact for affine K8/V4. It still needs repeated
  randomized reports, physical-device evidence, and a green compatibility-pair
  proof before product promotion.

## Live Upstream PR Board

Checked on 2026-06-25.

| Repo | PR | Branch | Head | State | Checks | Local action |
| --- | --- | --- | --- | --- | --- | --- |
| `mlx-swift-lm` | [`#301`](https://github.com/ml-explore/mlx-swift-lm/pull/301) Complete VLM processor TODOs | `upstream-pr/vlm-processor-completions` | `301233ad41afcbd5c95abca0bdcfd85026787faf` | Open, approved, merge blocked | `lint` passed; `mac_build_and_test` marked failed after the upstream self-hosted macOS runner lost communication following successful build/docs/tests | Wait for maintainer rerun/recovery. Do not change code unless a real test/build failure appears. |
| `mlx-swift-lm` | [`#303`](https://github.com/ml-explore/mlx-swift-lm/pull/303) Document model compatibility requirements | `upstream-pr/model-compatibility-docs` | `56df900cb0beb07aa4bcc18d5d1b24c928c88082` | Open, review required | No checks reported | Wait for maintainer review. Maintainers may still decide docs belong elsewhere. |
| `mlx-swift-lm` | [`#371`](https://github.com/ml-explore/mlx-swift-lm/pull/371) Validate RoPE model configurations | `upstream-pr/rope-config-validation` | `88a44a36f7a62c763cef508cfb7022bfbc4d61ab` | Open, review required | No checks reported | Wait for maintainer review/check execution. |
| `mlx-swift-lm` | [`#372`](https://github.com/ml-explore/mlx-swift-lm/pull/372) Add runtime stop string handling | `upstream-pr/runtime-stop-strings` | `dd5accfcfa523a2ee490a52d3c3feabe37109302` | Open, review required | No checks reported | Wait for maintainer review/check execution. |
| `mlx` | [`#3597`](https://github.com/ml-explore/mlx/pull/3597) Add `metal::set_metallib_path()` | `metallib-path` | merge `51b2768da7e1897d3c4258f7ddbb47083d1eef01` | Merged | Upstream checks passed before merge | Use as the accepted MLX-side foundation for SwiftPM Metal loading. |
| `mlx-c` | [`#117`](https://github.com/ml-explore/mlx-c/pull/117) Add `mlx_metal_set_metallib_path()` C API | `metallib-path` | `71371f2ffd6036b210f782abe4f4cf32d3e6299a` | Draft/open | No checks reported | Coordinate readiness. Local fallback branch `RNT56/mlx-c:upstream-pr/metallib-path-c-api` at `6173b85` validates the current-main dependency shape. |
| `mlx-swift` | [`#416`](https://github.com/ml-explore/mlx-swift/pull/416) Add `GPU.setMetallibPath()` | `metallib-path` | `800f9ae91d81e387a7e1febe5a9ab93ff452c7c3` | Draft/open | No checks reported | Wait for or coordinate after `mlx-c#117`; do not mix SwiftPM resource generation into this PR. |
| `mlx-swift` | [`#430`](https://github.com/ml-explore/mlx-swift/pull/430) Build SwiftPM default Metal library resource | `upstream-pr/swiftpm-metal-library-resource` | `2b9dda12f9c1a9681e54b38aa718a9437eb1e13e` | Draft/open | No checks reported | Keep draft until rebuilt on upstream-owned `#117`/`#416` commits. Local proof branch `prep/swiftpm-metallib-via-path-api` at `57449af` validates runtime loading. |
| `mlx` | none by RNT56 | n/a | n/a | n/a | n/a | Do not open a competing quantized SDPA PR while `ml-explore/mlx#3026` is active. |

Closed or archival upstream PRs:

| Repo | PR/Branch | Decision |
| --- | --- | --- |
| `mlx-swift-lm` | `#302` / `upstream-pr/model-config-runtime-hardening` | Closed and superseded by narrow `rope-config-validation`. Keep as staging history only. |
| `mlx-swift-lm` | `#304` / `pr/turboquant-kv-cache` | Closed as too broad for upstream. Keep as staging history only. |
| `mlx-swift` | `upstream-pr/linalg-norm-kind-nuc` | Already merged upstream as `ml-explore/mlx-swift#411`. Archival only. |
| `mlx` | `#3767` / `upstream-pr/swiftpm-metallib-bundle-lookup` | Closed on 2026-06-25 as a duplicate of the automatic lookup approach rejected in `#3562`. Do not reopen. |

## Branch Readiness Classes

Use these classes before opening or updating upstream PRs.

### Ready Upstream Branches

These branches are already clean, narrow, and open as upstream PRs:

```text
mlx-swift-lm/upstream-pr/vlm-processor-completions
mlx-swift-lm/upstream-pr/model-compatibility-docs
mlx-swift-lm/upstream-pr/rope-config-validation
mlx-swift-lm/upstream-pr/runtime-stop-strings
```

These are the only branches that should be discussed as ready upstream PR
branches right now.

The local future-PR launch packet is
[Upstream PR Launch Packets - 2026-06-25](upstream-pr-launch-packets-2026-06-25.md).
It is preparation only and does not authorize opening more PRs.

### Staging Branches Only

These branches contain useful work but are not reviewable upstream PR branches:

```text
mlx/pr/quantized-sdpa-followups
mlx/upstream-pr/swiftpm-metallib-bundle-lookup
mlx-c/upstream-pr/metallib-path-c-api
mlx-c/codex/mlx-c-quantized-sdpa-parity
mlx-swift/prep/swiftpm-metallib-via-path-api
mlx-swift/upstream-pr/swiftpm-metal-library-resource
mlx-swift/codex/update-mlx-c-quantized-sdpa
mlx-swift/pr/turboquant-swift-support
mlx-swift/pr/turboquant-linear-conversion
mlx-swift/pr/turboquant-metal-runtime-completion
mlx-swift/pr/runtime-0-31-2-support
mlx-swift-lm/pr/turboquant-kv-cache
mlx-swift-lm/pr/turboquant-profile-routing
mlx-swift-lm/pr/turboquant-model-integration
mlx-swift-lm/pr/turboquant-compressed-attention
mlx-swift-lm/pr/turboquant-rotating-cache
mlx-swift-lm/pr/turboquant-profiles
mlx-swift-lm/pr/turboquant-shared-latent-attention
```

Reasons:

- they are stacked on fork-only runtime pins or generated/submodule changes;
- they mix API, runtime, model integration, diagnostics, and benchmarks;
- several depend on the final shape of upstream `mlx#3026`;
- they need to be rebuilt into dependency-ordered `upstream-pr/*` branches.

### Never Open Directly

Do not open these as upstream PRs:

```text
codex/*
integration/*
tq/*
wave0-worktrees/*
.codex-worktrees/*
```

They are continuation, integration, or worker branches. They can be used as
source material, not as upstream review branches.

## Upstream Delivery Order

### Stage 1: Maintenance PR Queue

Deliver the already-open independent upstream PRs:

1. `mlx-swift-lm#301`: VLM processor completions.
2. `mlx-swift-lm#303`: model compatibility docs.
3. `mlx-swift-lm#371`: RoPE config validation.
4. `mlx-swift-lm#372`: runtime stop-string handling.
5. Explicit Metal path chain for SwiftPM resources:
   - `mlx#3597`: merged C++ `set_metallib_path` foundation.
   - `mlx-c#117`: draft C wrapper.
   - `mlx-swift#416`: draft Swift `GPU.setMetallibPath`.
   - `mlx-swift#430`: draft SwiftPM `default.metallib` resource packaging.

Acceptance for this stage:

- each PR is rebased on current upstream if maintainers request it;
- each PR has passing upstream checks or a maintainer-accepted CI exception;
- each PR remains single-purpose;
- no TurboQuant fork-only code is folded into these PRs.
- `mlx-swift#430` remains draft until it uses upstream-owned `mlx-c#117` and
  `mlx-swift#416` commits/submodule pins, even though the local proof branch
  validates the design.

### Stage 2: Core Quantized SDPA Dependency

External dependency:

```text
ml-explore/mlx#3026
```

Focused local support plan:

```text
mlx-3026-support-roadmap-2026-06-26.md
```

Current upstream state:

- PR is open, non-draft, review required.
- It is the owner of the core quantized SDPA API/kernel discussion.
- Its latest checked head was `bfed86bfc2e84821dfc768465d13aaf26147d888`.

Local follow-up source:

```text
/Users/mt/Programming/Schtack/mlx-forks/.pr-worktrees/mlx-quantized-sdpa-followups
branch: pr/quantized-sdpa-followups
head: 627040cd
```

Local follow-up contents:

- one commit on top of the `#3026` branch;
- `6 files changed, 184 insertions, 11 deletions` relative to its immediate
  parent;
- large stale diff against upstream `main` because it includes the whole active
  `#3026` stack.

Do not open this against `main` now. After `#3026` lands, rebase the one local
follow-up commit and, if still useful, open:

```text
mlx/upstream-pr/quantized-sdpa-verifier-batches
```

Scope:

- verifier batch support;
- focused tests;
- no broad kernel rewrite;
- no benchmark/report payload beyond what is needed to prove the API behavior.

### Stage 3: C API and Swift Bindings

After the core `mlx` API is accepted or stable enough for a maintainer-requested
stack, split from the current fork-only branches in this order:

1. `mlx-c/upstream-pr/quantized-sdpa-c-api`
   - Source: `mlx-c/codex/mlx-c-quantized-sdpa-parity`.
   - Scope: expose only the accepted `mlx` quantized SDPA ABI surface.
   - Exclude: segmented attention experiments, distributed runtime parity, and
     fork-only diagnostics unless maintainers request them.

2. `mlx-swift/upstream-pr/quantized-sdpa-bindings`
   - Source: `mlx-swift/codex/update-mlx-c-quantized-sdpa` plus the smallest
     needed slices from `pr/turboquant-swift-support`.
   - Scope: Swift binding, capability probe, small tests.
   - Exclude: LM cache policy, model integration, benchmark reports, and
     runtime-generated submodule churn.

Acceptance:

- `mlx-c` builds and tests against the accepted `mlx` API.
- `mlx-swift` builds with the intended C/core pins.
- capability probing fails closed when the native API is missing.

### Stage 4: LM TurboQuant Stack

Only after Swift bindings are accepted or stable, rebuild `mlx-swift-lm` into
small upstream PRs:

1. `mlx-swift-lm/upstream-pr/turboquant-cache-core`
   - Source: narrow slice from `pr/turboquant-kv-cache`,
     `pr/turboquant-rotating-cache`, and cache lifecycle work.
   - Scope: minimal cache strategy surface and tests.
   - Exclude: profile routing, model-family integration, benchmark reports.

2. `mlx-swift-lm/upstream-pr/turboquant-profile-routing`
   - Source: `pr/turboquant-profile-routing` and relevant profile validation
     files.
   - Scope: profile selection, diagnostics, fail-closed behavior.
   - Exclude: model-family wiring and product activation.

3. `mlx-swift-lm/upstream-pr/turboquant-model-integration`
   - Source: `pr/turboquant-model-integration`,
     `pr/turboquant-compressed-attention`, and supported-model slices only.
   - Scope: model-family integration for the accepted compressed attention path.
   - Exclude: experimental PolarWHT promotion and sparse auto policy.

4. `mlx-swift-lm/upstream-pr/turboquant-benchmarks-and-reports`
   - Source: benchmark/report docs and tools after the runtime path is accepted.
   - Scope: methodology, quality gates, compact diagnostics.
   - Decision: keep fork-only if maintainers do not want benchmark/report
     payloads upstream.

Acceptance:

- each PR compiles and tests independently on top of its declared dependency;
- no PR mixes runtime API, C/Swift binding, LM integration, and reports;
- every fail-closed branch reports a reason instead of silently falling back;
- exact prefill remains exact.

### Stage 5: Downstream Compatibility Pair

After the upstream-stack or fork-stack equivalent is stable, validate the exact
downstream pair from bottom to top:

```text
mlx -> mlx-c -> mlx-swift -> mlx-swift-lm -> Pines
```

Pines docs are the downstream release-train owner:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation
```

Do not update Pines pins from this roadmap alone. A pin update requires:

- exact commit pair recorded;
- `project.yml`, `Package.resolved`, and generated Xcode project synchronized;
- real-device evidence attached;
- compatibility-pair JSON updated;
- local and device gates passing.

The current Pines working tree was observed dirty on 2026-06-25, so this
roadmap intentionally does not edit Pines docs or pins.

## Validation Gates

### `mlx-swift-lm` Maintenance PRs

Use the relevant subset for each branch:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
swift build --target MLXLMCommon
swift build --target MLXLLM
swift build --target MLXVLM
swift test --filter StopStringTests
swift test --filter ModelConfigurationTests
scripts/verify-docs.sh
pre-commit run --all-files
git diff --check
```

### `mlx-swift`

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift
git submodule status --recursive
swift build --target MLX
swift build --target MLXNN
bash -n tools/build-swiftpm-metallib.sh
SDK_NAME=macosx tools/build-swiftpm-metallib.sh /tmp/mlx-swift-default-macos.metallib
pre-commit run --all-files
git diff --check
```

For the local SwiftPM Metal proof branch:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/.pr-worktrees/mlx-swift-metal-resource-via-path-api
swift build --target MLX
swift build --target MLXNN
swift test --filter SwiftPMMetallibResourceTests
swift test --filter StreamTests/testDeviceType
bash -n tools/build-swiftpm-metallib.sh
SDK_NAME=macosx tools/build-swiftpm-metallib.sh /tmp/mlx-swift-default-macos.metallib
git diff --check upstream/main...HEAD
```

### `mlx`

After `mlx#3026` lands or a maintainer asks for a stacked follow-up:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/.pr-worktrees/mlx-quantized-sdpa-followups
python -m pytest python/tests/test_quantized.py
cmake --build build --target quantized_sdpa_tests
python benchmarks/python/quantized_sdpa_bench.py
git diff --check
```

Use the exact local build/test names that exist after rebasing; the commands
above are the intended gate shape, not a substitute for inspecting the current
build tree.

### `mlx-c`

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-c
cmake --build build/metallib-path-api
ctest --test-dir build/metallib-path-api --output-on-failure
git diff --check
```

Run this only after aligning the `mlx` dependency to the accepted core API.

### TurboQuant Product Evidence

The first combined real-model rerun completed on 2026-06-25:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
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

Observed result:

- quality and throughput are in the same report at
  `artifacts/turboquant-hybrid-smoke-20260607/diagnostics-16k-g32-fp16-affinek8v4-quality-materialized-conversion-rerun.json`;
- affine K8/V4 selects `affineK8V4Native` for all 28 layers in throughput and
  quality;
- no raw fallback or decoded fallback is allocated;
- resident KV compression is `2.13x`;
- FP16 decode is `8.55` tok/s and affine K8/V4 decode is `42.84` tok/s
  (`5.01x` FP16) in this single ordered sample;
- quality passes with top-1 `1.000`, KL `1.1265e-7`, p95 max-logit error
  `1.0859`, and cosine `0.9942`;
- the report records exact commits and artifact paths.

## What We Can Say Now

Safe maintainer-facing status:

```text
We split the upstream work into narrow PRs. The VLM processor PR is approved
and waiting on the failing macOS CI job/rerun. The model compatibility docs,
RoPE validation, runtime stop-string handling, and SwiftPM Metal resource
packaging PRs are open as separate changes. The old automatic MLX SwiftPM
lookup route is closed; the SwiftPM Metal path now follows the accepted explicit
metallib path chain. For TurboQuant proper, we are waiting on the upstream core
quantized SDPA PR before proposing any dependent mlx-c, Swift, or LM slices.
```

Safe product-facing status:

```text
The fork stack has staging branches for the full TurboQuant path, but product
promotion is not complete. We have one same-report 16K affine K8/V4
quality/throughput artifact with the native compressed path selected. The next
proof is repeated randomized 16K/32K evidence, followed by downstream
compatibility-pair and physical-device validation.
```

Do not claim:

- that `#301` is fully mergeable while `mac_build_and_test` is failing;
- that newly opened PRs have passed upstream CI when no upstream check data is
  currently reported;
- that local `mlx/pr/quantized-sdpa-followups` is a standalone upstream PR
  against `main`;
- that TurboQuant is product-ready before repeated randomized reports,
  compatibility-pair evidence, and physical-device evidence exist.

## Next Operator Checklist

1. Watch or request rerun for `mlx-swift-lm#301` if maintainers indicate the
   failing macOS job was infrastructure-side.
2. Monitor review on `#303`, `#371`, and `#372`.
3. Coordinate `mlx-c#117`, then `mlx-swift#416`, then rebuild `mlx-swift#430`
   from upstream-owned commits/submodule pins.
4. Monitor `ml-explore/mlx#3026`.
5. When `#3026` lands, rebase `mlx/pr/quantized-sdpa-followups` and decide
   whether `mlx/upstream-pr/quantized-sdpa-verifier-batches` is still needed.
6. Split `mlx-c`, `mlx-swift`, and `mlx-swift-lm` TurboQuant branches only after
   the lower dependency is accepted or stable enough to test.
7. Run randomized repeated 16K and 32K affine K8/V4 real-model
   quality/throughput reports.
8. Only after the fork stack is green, update Pines pins and compatibility-pair
   evidence through the Pines release-train docs.
