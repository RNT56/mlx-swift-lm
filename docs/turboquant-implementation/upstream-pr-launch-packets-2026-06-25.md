# Upstream PR Launch Packets - 2026-06-25

This file is the local preparation board for future upstream PRs. It exists so
we can keep preparing without prematurely pushing or opening more PRs.

## Decision

Do not push or open additional upstream PRs right now.

Reasons:

- The current upstream queue is already five open PRs across two repos.
- `mlx-swift-lm#301` is approved but blocked by an upstream self-hosted macOS
  runner communication failure after all job steps succeeded.
- `mlx-swift-lm#303`, `#371`, `#372`, and `mlx-swift#430` have GitHub Actions
  runs in `action_required` state with zero jobs. They need maintainer approval
  or rerun before they can satisfy the upstream "passing tests" expectation.
- The next TurboQuant PRs depend on upstream `ml-explore/mlx#3026`; opening a
  competing or stacked PR against `main` would be noisy and harder to review.

Allowed work now:

- prepare local-only launch packets;
- keep source branches and worktrees categorized;
- refine validation commands;
- pre-write PR descriptions and checklists;
- inspect diffs and blockers.

Disallowed work now:

- do not open more upstream PRs;
- do not push placeholder upstream branches;
- do not comment on upstream PRs unless the user explicitly asks;
- do not update Pines pins from this preparation alone.

## Current Open PR Gate

Checked on 2026-06-25.

| PR | Branch | Current gate | What must happen before merge |
| --- | --- | --- | --- |
| `mlx-swift-lm#301` | `upstream-pr/vlm-processor-completions` | Approved; `lint` passed; `mac_build_and_test` failed because the self-hosted macOS runner lost communication after successful build/docs/tests | Maintainer reruns failed macOS job or accepts it as runner infrastructure. No code change unless a real test/build failure appears. |
| `mlx-swift-lm#303` | `upstream-pr/model-compatibility-docs` | Workflow `28139307769` is `action_required` with zero jobs | Maintainer approves/runs CI, then review. Also confirm maintainers want this docs payload upstream. |
| `mlx-swift-lm#371` | `upstream-pr/rope-config-validation` | Workflow `28140304842` is `action_required` with zero jobs | Maintainer approves/runs CI, then review. |
| `mlx-swift-lm#372` | `upstream-pr/runtime-stop-strings` | Workflow `28151349347` is `action_required` with zero jobs | Maintainer approves/runs CI, then review. |
| `mlx-swift#430` | `upstream-pr/swiftpm-metal-library-resource` | Workflow `28140304642` is `action_required` with zero jobs | Maintainer approves/runs CI, then review. |

Maintainer-facing short status:

```text
We should not add more PRs to the queue yet. The current small PRs are open and
scoped, but most need maintainer CI approval, and #301 needs the failed
self-hosted macOS job rerun. We can keep preparing the dependent TurboQuant
stack locally, but should wait to publish it until the current queue and the
upstream quantized SDPA dependency are clearer.
```

## Launch Rule

A future upstream PR can be opened only when all of these are true:

- it has one purpose and one repo ownership boundary;
- it is based on current upstream `main` or the accepted dependency branch;
- its lower dependency is merged, accepted, or explicitly requested by
  maintainers as a stack;
- it has tests for code changes;
- it has docs for API changes;
- it has `pre-commit run --all-files` or a documented upstream-equivalent
  formatter/check;
- it has `git diff --check`;
- any efficiency-impacting `mlx` change has before/after benchmark evidence;
- its PR body includes the upstream checklist substance.

Do not open a branch just to reserve a name.

## Prepared Future Packets

### Packet A: `mlx/upstream-pr/quantized-sdpa-verifier-batches`

Status: blocked; do not push.

Upstream dependency:

```text
ml-explore/mlx#3026
```

Source branch:

```text
/Users/mt/Programming/Schtack/mlx-forks/.pr-worktrees/mlx-quantized-sdpa-followups
branch: pr/quantized-sdpa-followups
head: 627040cd
```

Intended scope:

- verifier batch support;
- focused Python/C++ tests;
- minimal benchmark or smoke evidence only if needed for behavior validation.

Exclude:

- full quantized SDPA API/kernel payload already covered by `mlx#3026`;
- broad Metal rewrites;
- Swift/C/LM bindings;
- product docs.

Open only when:

- `mlx#3026` lands, or maintainers ask for a stacked follow-up;
- the one local follow-up commit rebases cleanly on the accepted core API;
- the diff against upstream `main` no longer includes the whole `#3026` stack.

Validation packet:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/.pr-worktrees/mlx-quantized-sdpa-followups
python -m pytest python/tests/test_quantized.py
cmake --build build --target quantized_sdpa_tests
python benchmarks/python/quantized_sdpa_bench.py
pre-commit run --all-files
git diff --check
```

PR description draft:

```markdown
## Proposed changes

Adds verifier batch coverage for the accepted quantized SDPA API. This is a
small follow-up to the upstream quantized SDPA implementation and does not
change the public API beyond the accepted surface.

## Validation

- `python -m pytest python/tests/test_quantized.py`
- `cmake --build build --target quantized_sdpa_tests`
- `python benchmarks/python/quantized_sdpa_bench.py`
- `pre-commit run --all-files`
- `git diff --check`

## Checklist

- [x] I have read the CONTRIBUTING document
- [x] I have run pre-commit
- [x] I have added tests for the behavior
- [ ] I have updated documentation, if needed
```

### Packet B: `mlx-c/upstream-pr/quantized-sdpa-c-api`

Status: blocked; do not push.

Source branch:

```text
mlx-c/codex/mlx-c-quantized-sdpa-parity
```

Intended scope:

- expose only the accepted `mlx` quantized SDPA C ABI;
- keep C names and signatures aligned to upstream `mlx-c` style;
- add minimal C API tests or compile smoke.

Exclude:

- segmented attention experiments;
- distributed runtime parity;
- fork-only diagnostics;
- Swift bindings.

Open only when:

- `mlx` core quantized SDPA API is accepted or stable;
- ABI shape is no longer moving;
- source branch is split down to the accepted API only.

Validation packet:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-c
cmake --build build
ctest --test-dir build --output-on-failure
pre-commit run --all-files
git diff --check
```

### Packet C: `mlx-swift/upstream-pr/quantized-sdpa-bindings`

Status: blocked; do not push.

Source branches:

```text
mlx-swift/codex/update-mlx-c-quantized-sdpa
mlx-swift/pr/turboquant-swift-support
```

Intended scope:

- Swift binding for accepted quantized SDPA API;
- capability probe;
- tests or compile smoke for missing-native fail-closed behavior.

Exclude:

- MLX core kernel changes;
- C ABI changes;
- LM cache policy;
- model integration;
- benchmark reports.

Open only when:

- `mlx` and `mlx-c` dependencies are accepted or maintainer-requested as a
  stack;
- the branch no longer requires fork-only generated/submodule churn;
- `mlx-swift#430` is resolved or maintainers say the Metal resource work should
  be independent.

Validation packet:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift
git submodule status --recursive
swift build --target MLX
swift build --target MLXNN
pre-commit run --all-files
git diff --check
```

### Packet D: `mlx-swift-lm/upstream-pr/turboquant-cache-core`

Status: blocked; do not push.

Source branches:

```text
mlx-swift-lm/pr/turboquant-kv-cache
mlx-swift-lm/pr/turboquant-rotating-cache
```

Intended scope:

- minimal cache strategy surface;
- compressed-cache lifecycle basics;
- focused tests for exact prefill and fail-closed behavior.

Exclude:

- profile routing;
- model-family integration;
- benchmark reports;
- PolarWHT promotion;
- sparse auto policy.

Open only when:

- Swift quantized SDPA bindings are accepted or stable;
- the cache API can compile against upstream dependencies without fork-only
  pins;
- the PR can be tested independently.

Validation packet:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
swift build --target MLXLMCommon
swift test --filter KVCacheTests
pre-commit run --all-files
git diff --check
```

### Packet E: `mlx-swift-lm/upstream-pr/turboquant-profile-routing`

Status: blocked; do not push.

Source branch:

```text
mlx-swift-lm/pr/turboquant-profile-routing
```

Intended scope:

- profile selection;
- diagnostics;
- mismatch reasons;
- fail-closed routing behavior.

Exclude:

- cache implementation;
- model-family wiring;
- product activation;
- benchmark payloads.

Open only when:

- `turboquant-cache-core` is accepted or stable;
- profile routing has no dependency on fork-only Swift/Core APIs outside the
  accepted stack.

Validation packet:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
swift test --filter TurboQuantProfileTests
pre-commit run --all-files
git diff --check
```

### Packet F: `mlx-swift-lm/upstream-pr/turboquant-model-integration`

Status: blocked; do not push.

Source branches:

```text
mlx-swift-lm/pr/turboquant-model-integration
mlx-swift-lm/pr/turboquant-compressed-attention
mlx-swift-lm/pr/turboquant-shared-latent-attention
```

Intended scope:

- wire accepted compressed attention path into supported model families only;
- keep exact prefill exact;
- prove missing/unsupported native paths fail closed.

Exclude:

- core kernels;
- Swift bindings;
- cache core;
- profile routing;
- experimental PolarWHT/Sparse-V promotion;
- broad benchmark reports.

Open only when:

- profile routing is accepted or stable;
- the accepted native path is available in upstream dependencies;
- same-report quality/throughput evidence exists for any performance claim.

Validation packet:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
swift build -c release --product TurboQuantInferenceParity
swift test --filter KVCacheTests
swift test --filter TurboQuantProfileTests
pre-commit run --all-files
git diff --check
```

### Packet G: `mlx-swift-lm/upstream-pr/turboquant-benchmarks-and-reports`

Status: optional; likely fork-only unless maintainers ask.

Intended scope:

- benchmark methodology;
- compact diagnostics;
- quality gate report format.

Open only when:

- maintainers explicitly want benchmark/report payloads upstream;
- model integration is accepted or stable;
- reports are small and reproducible.

Validation packet:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
swift build -c release --product TurboQuantInferenceParity
scripts/verify-docs.sh
pre-commit run --all-files
git diff --check
```

## Current PR Body Polish Queue

These are local notes only. Do not edit upstream PR descriptions unless the user
asks.

### `mlx-swift-lm#303`

Risk:

- docs-only PR has no maintainer signal yet that the payload belongs upstream;
- current validation text is weaker than the final local preparation state.

Preferred body if updating later:

```markdown
## Proposed changes

Documents model compatibility requirements for model assets, tokenizer and
processor files, configuration expectations, and runtime support notes.

## Validation

- `scripts/verify-docs.sh`
- `pre-commit run --all-files`
- `git diff --check`

## Checklist

- [x] I have read the CONTRIBUTING document
- [x] I have run pre-commit
- [ ] I have added tests that prove my fix is effective or that my feature works
- [x] I have updated the necessary documentation
```

### `mlx-swift-lm#371`, `mlx-swift-lm#372`, and `mlx-swift#430`

Risk:

- PR bodies are already clear, but GitHub Actions has not produced jobs.

Preferred next action:

- wait for maintainer workflow approval;
- if maintainers request local proof, paste exact local validation logs instead
  of adding code.

## Operator Checklist

Before opening any further PR:

1. Confirm current open PR gate status.
2. Confirm no maintainer requested a different scope.
3. Confirm lower dependency branch is accepted or stable.
4. Create the local branch from the correct upstream base.
5. Cherry-pick or manually port only the minimal source slice.
6. Run the packet validation.
7. Run `git diff --check`.
8. Fill the upstream PR checklist substance.
9. Push/open only after explicit user approval.
