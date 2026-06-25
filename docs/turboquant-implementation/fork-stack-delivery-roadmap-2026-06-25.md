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
- No RNT56-authored `ml-explore/mlx` PR is open; this is intentional while
  `ml-explore/mlx#3026` owns the core quantized SDPA API/kernel discussion.
- The downstream TurboQuant product path still needs same-report real-model
  quality and throughput evidence, physical-device evidence, and a green
  compatibility-pair proof before product promotion.

## Live Upstream PR Board

Checked on 2026-06-25.

| Repo | PR | Branch | Head | State | Checks | Local action |
| --- | --- | --- | --- | --- | --- | --- |
| `mlx-swift-lm` | [`#301`](https://github.com/ml-explore/mlx-swift-lm/pull/301) Complete VLM processor TODOs | `upstream-pr/vlm-processor-completions` | `301233ad41afcbd5c95abca0bdcfd85026787faf` | Open, approved, merge blocked | `lint` passed; `mac_build_and_test` marked failed after the upstream self-hosted macOS runner lost communication following successful build/docs/tests | Wait for maintainer rerun/recovery. Do not change code unless a real test/build failure appears. |
| `mlx-swift-lm` | [`#303`](https://github.com/ml-explore/mlx-swift-lm/pull/303) Document model compatibility requirements | `upstream-pr/model-compatibility-docs` | `56df900cb0beb07aa4bcc18d5d1b24c928c88082` | Open, ready for review | Workflow `28139307769` is `action_required` with zero jobs | Wait for maintainer workflow approval/review. Maintainers may still decide docs belong elsewhere. |
| `mlx-swift-lm` | [`#371`](https://github.com/ml-explore/mlx-swift-lm/pull/371) Validate RoPE model configurations | `upstream-pr/rope-config-validation` | `4a05447ba8b8d2c0e430d49657e24db175d8b1bb` | Open, review required | Workflow `28140304842` is `action_required` with zero jobs | Wait for maintainer workflow approval/review. |
| `mlx-swift-lm` | [`#372`](https://github.com/ml-explore/mlx-swift-lm/pull/372) Add runtime stop string handling | `upstream-pr/runtime-stop-strings` | `2aeeee6fee40af3557f8d590d7a9d85bb931cc83` | Open, review required | Workflow `28151349347` is `action_required` with zero jobs | Wait for maintainer workflow approval/review. |
| `mlx-swift` | [`#430`](https://github.com/ml-explore/mlx-swift/pull/430) Build SwiftPM default Metal library resource | `upstream-pr/swiftpm-metal-library-resource` | `ca2924ff26bf4e45f5bbdf7dca72496220cbd0cf` | Open, review required | Workflow `28140304642` is `action_required` with zero jobs | Wait for maintainer workflow approval/review. |
| `mlx` | none by RNT56 | n/a | n/a | n/a | n/a | Do not open a competing quantized SDPA PR while `ml-explore/mlx#3026` is active. |

Closed or archival upstream PRs:

| Repo | PR/Branch | Decision |
| --- | --- | --- |
| `mlx-swift-lm` | `#302` / `upstream-pr/model-config-runtime-hardening` | Closed and superseded by narrow `rope-config-validation`. Keep as staging history only. |
| `mlx-swift-lm` | `#304` / `pr/turboquant-kv-cache` | Closed as too broad for upstream. Keep as staging history only. |
| `mlx-swift` | `upstream-pr/linalg-norm-kind-nuc` | Already merged upstream as `ml-explore/mlx-swift#411`. Archival only. |

## Branch Readiness Classes

Use these classes before opening or updating upstream PRs.

### Ready Upstream Branches

These branches are already clean, narrow, and open as upstream PRs:

```text
mlx-swift-lm/upstream-pr/vlm-processor-completions
mlx-swift-lm/upstream-pr/model-compatibility-docs
mlx-swift-lm/upstream-pr/rope-config-validation
mlx-swift-lm/upstream-pr/runtime-stop-strings
mlx-swift/upstream-pr/swiftpm-metal-library-resource
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
mlx-c/codex/mlx-c-quantized-sdpa-parity
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
5. `mlx-swift#430`: SwiftPM default Metal library resource.

Acceptance for this stage:

- each PR is rebased on current upstream if maintainers request it;
- each PR has passing upstream checks or a maintainer-accepted CI exception;
- each PR remains single-purpose;
- no TurboQuant fork-only code is folded into these PRs.

### Stage 2: Core Quantized SDPA Dependency

External dependency:

```text
ml-explore/mlx#3026
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
pre-commit run --all-files
git diff --check
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
cmake --build build
ctest --test-dir build --output-on-failure
git diff --check
```

Run this only after aligning the `mlx` dependency to the accepted core API.

### TurboQuant Product Evidence

The first required real-model rerun remains:

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

Pass conditions:

- quality and throughput are in the same report;
- affine K8/V4 selects `affineK8V4Native` for throughput;
- quality selected path includes `affineK8V4Native`;
- no raw fallback or decoded fallback is allocated;
- resident KV compression is greater than `1.0x`;
- active memory is reported;
- sparse was not requested-but-inactive;
- the report records exact commits and artifact paths.

## What We Can Say Now

Safe maintainer-facing status:

```text
We split the upstream work into narrow PRs. The VLM processor PR is approved
and waiting on the failing macOS CI job/rerun. The model compatibility docs,
RoPE validation, runtime stop-string handling, and SwiftPM Metal resource
packaging PRs are open as separate reviewable changes. We closed the overly
broad runtime/TurboQuant drafts instead of asking maintainers to review them as
large mixed patches. For TurboQuant proper, we are waiting on the upstream core
quantized SDPA PR before proposing any dependent mlx-c, Swift, or LM slices.
```

Safe product-facing status:

```text
The fork stack has staging branches for the full TurboQuant path, but product
promotion is not complete. The next proof is a same-report real-model quality
and throughput run for affine K8/V4, followed by downstream compatibility-pair
and physical-device validation.
```

Do not claim:

- that `#301` is fully mergeable while `mac_build_and_test` is failing;
- that newly opened PRs have passed upstream CI while their workflows are still
  `action_required` with zero jobs;
- that local `mlx/pr/quantized-sdpa-followups` is a standalone upstream PR
  against `main`;
- that TurboQuant is product-ready before the combined real-model and device
  evidence exists.

## Next Operator Checklist

1. Watch or request rerun for `mlx-swift-lm#301` if maintainers indicate the
   failing macOS job was infrastructure-side.
2. Monitor review on `#303`, `#371`, `#372`, and `mlx-swift#430`.
3. Monitor `ml-explore/mlx#3026`.
4. When `#3026` lands, rebase `mlx/pr/quantized-sdpa-followups` and decide
   whether `mlx/upstream-pr/quantized-sdpa-verifier-batches` is still needed.
5. Split `mlx-c`, `mlx-swift`, and `mlx-swift-lm` TurboQuant branches only after
   the lower dependency is accepted or stable enough to test.
6. Rerun the combined affine K8/V4 real-model quality/throughput gate.
7. Only after the fork stack is green, update Pines pins and compatibility-pair
   evidence through the Pines release-train docs.
