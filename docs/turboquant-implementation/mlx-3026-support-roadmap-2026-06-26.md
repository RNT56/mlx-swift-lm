# MLX #3026 Support Roadmap - 2026-06-26

This roadmap turns the live read of `ml-explore/mlx#3026` into the local
support plan for the RNT56 MLX fork stack. It is scoped to helping the upstream
quantized SDPA dependency land cleanly, then turning the accepted shape into
small follow-up PRs, Swift bindings, LM integration, and downstream evidence.

## Current State

Checked on 2026-06-26:

- Upstream PR: `ml-explore/mlx#3026` (`Quantized SDPA`).
- State: open, non-draft, review required.
- Head: `bfed86bfc2e84821dfc768465d13aaf26147d888`.
- GitHub checks: no check runs or commit statuses reported for the head.
- Mergeability: conflicting/dirty against current upstream `main`.
- Known conflict area: `mlx/backend/metal/scaled_dot_product_attention.cpp`.
- Important upstream-main overlap: current `main` has `MLX_SDPA_BLOCKS` for
  2-pass vector block-count override; preserve it in any rebase.

`#3026` is the upstream owner for the core quantized SDPA API and Metal kernels.
Do not open a competing `mlx` quantized SDPA PR against `main` while it is
active.

## What #3026 Gives Us

Accepted locally as the dependency target, subject to upstream review:

- Public `mx.fast.quantized_scaled_dot_product_attention(...)`.
- Packed `uint32` quantized K/V.
- Modes: `mxfp4`, `mxfp8`, `nvfp4`, and `affine`.
- Affine support: bits `{4, 6, 8}`, group size `{32, 64}`.
- Fused Metal decode/vector path for:
  - `q_seq_len <= 8`;
  - `q_seq_len <= key_seq_len`;
  - `head_dim in {64, 128, 256, 512}`;
  - `q_seq_len * gqa_factor <= 32`.
- Fallback for unsupported cases via
  `quantized_matmul + softmax + quantized_matmul`.
- The packed-value kernel-name issue is fixed in the current PR head by using
  the logical query head dimension for quantized dispatch naming.
- `head_dim=512` support is included using `BD=8` to keep per-lane state bounded.

## Nonclaims

Do not claim any of these from `#3026` alone:

- Product-ready TurboQuant.
- Swift or LM integration readiness.
- Pines compatibility-pair readiness.
- Multi-row speculative verifier support.
- Speedup on all model families or all head dimensions.
- Real memory savings without resident-memory evidence.
- Promotion from PR charts, synthetic operator timings, or single ordered runs.

## Roadmap Lanes

### Lane 0: Live PR Monitoring

Owner: whoever picks up upstream status work.

Cadence:

- Recheck `#3026` before any local follow-up work.
- Recheck after upstream pushes, maintainer comments, or mergeability changes.

Commands:

```bash
gh pr view 3026 --repo ml-explore/mlx \
  --json state,isDraft,mergeable,reviewDecision,headRefOid,updatedAt,statusCheckRollup

gh api repos/ml-explore/mlx/issues/3026/comments --paginate
gh api repos/ml-explore/mlx/pulls/3026/comments --paginate
gh api repos/ml-explore/mlx/pulls/3026/reviews --paginate
```

Acceptance:

- Report separates PR metadata, code state, checks, review state, and local
  implications.
- No local branch is pushed or opened from monitoring alone.

### Lane 1: Upstream Support Without Noise

Status: allowed only when user asks or maintainers request input.

Useful support:

- Validate a maintainer-requested benchmark or failure on local Apple Silicon.
- Provide a small focused reproduction if `#3026` hits dispatch, mask, sink,
  GQA, or D=512 issues.
- Confirm conflict resolution around `MLX_SDPA_BLOCKS` if upstream asks.

Disallowed by default:

- Do not comment on `#3026` just to summarize our internal roadmap.
- Do not push a placeholder branch.
- Do not open a competing PR against `main`.
- Do not attach TurboQuant product claims to upstream review.

Acceptance:

- Any upstream comment is short, evidence-backed, and scoped to the specific
  maintainer ask.
- Evidence includes commit, device, OS, exact command, model/config shape, and
  whether the fused kernel path or fallback path ran.

### Lane 2: MLX Follow-Up After #3026

Primary future branch:

```text
mlx/upstream-pr/quantized-sdpa-verifier-batches
```

Source material:

```text
mlx/pr/quantized-sdpa-followups
.pr-worktrees/mlx-quantized-sdpa-followups
```

Dependency:

- Open only after `#3026` lands, or if maintainers explicitly request a stacked
  follow-up against the PR branch.

Scope:

- Multi-row quantized SDPA dispatch for verifier batches.
- Move `qsl` out of threadgroup `z` and into grid `z`, or preserve the accepted
  upstream equivalent if maintainers choose another layout.
- Allow verifier-friendly `qsl` up to 32 without falling back solely because
  `qsl * gqa_factor > 32`.
- Focused tests for `qsl in {2, 4, 8, 16, 32}` across representative
  `head_dim` and dtype cases.
- Optional debug path logging only if it is accepted as small and off by default.

Exclude:

- Full `#3026` API/kernel payload.
- C API, Swift bindings, LM policy, Pines pins.
- Broad Metal retuning unrelated to verifier-batch dispatch.

Rebase notes:

- Preserve `MLX_SDPA_BLOCKS` from upstream `main`.
- Keep `#3026` D=512 logic and packed logical-dim naming intact unless upstream
  changed the accepted implementation.
- Make the diff against `upstream/main` show only the follow-up.

Validation gate:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx
python -m pytest python/tests/test_quantized.py
pre-commit run --all-files
git diff --check upstream/main...HEAD
```

Add C++/Metal-specific build or symbol checks if the rebased branch includes a
C++ test target.

### Lane 3: Fused-Path Diagnostics

Dependency:

- Can be prepared locally before `#3026` lands, but should not be pushed
  upstream unless requested or included in a focused follow-up.

Goal:

- Make tests and LM reports distinguish fused quantized SDPA from fallback.

Deliverables:

- A small MLX-level test or debug hook proving the selected path for supported
  shapes.
- LM-side diagnostics fields for selected quantized attention path, unsupported
  shape, fallback reason, and fallback allocation.
- Same naming across MLX, Swift, and LM layers where practical.

Acceptance:

- A verifier-batch test can fail specifically because it fell back.
- A product evidence report can show native compressed path selection for both
  throughput and quality rows.

### Lane 4: C API And Swift Bindings

Dependency:

- Wait for the accepted `mlx` API shape from `#3026` or a maintainer-approved
  stack.

Order:

1. `mlx-c/upstream-pr/quantized-sdpa-c-api`
   - expose only the accepted MLX ABI;
   - add compile or minimal C API test;
   - exclude LM policy and diagnostics.
2. `mlx-swift/upstream-pr/quantized-sdpa-bindings`
   - bind the accepted C/MLX API;
   - add capability probe and small Swift tests;
   - exclude model integration and benchmark reports.

Acceptance:

- Swift can call the accepted quantized SDPA surface without fork-only shim
  assumptions.
- Capability checks fail closed when the native kernel is absent or unsupported.
- Submodule and package pins are deliberate and documented.

### Lane 5: LM Integration

Dependency:

- Swift binding is available locally and capability-probed.

Scope:

- Route affine K8/V4 cache attention through the accepted native path.
- Keep exact prefill exact.
- Keep `quantizedKVStart` behavior and materialized conversion semantics.
- Ensure fallback allocation obeys policy and is visible in diagnostics.
- Keep PolarWHT/PolarQJL diagnostic-only unless a separate gate explicitly
  revives them.

Acceptance:

- `swift test --filter KVCacheTests`
- `swift test --filter TurboQuantProfileTests`
- `swift build -c release --product TurboQuantInferenceParity`
- Same-report quality plus throughput report where both rows select the expected
  native compressed path.

### Lane 6: Evidence And Pines Promotion

Dependency:

- LM integration has native-path diagnostics and repeated local evidence.

Required evidence:

- Same-report quality plus throughput for target model/context.
- Repeated randomized throughput order.
- Native compressed path selected in throughput and quality.
- No raw fallback allocation.
- No decoded fallback allocation.
- Resident KV compression greater than `1.0x` from effective cache policy.
- Peak and steady active memory.
- Physical-device Pines app-host run.
- Green compatibility pair.

Acceptance:

- Product-readiness docs can say exactly which repo commits, model, device,
  context length, policy, selected path, quality result, memory result, and
  fallback state were used.
- Any non-green condition remains explicit in the compatibility pair.

## Immediate Next Actions

1. Keep `#3026` as the active upstream dependency and do not open a competing
   MLX PR.
2. Refresh the local `mlx/pr/quantized-sdpa-followups` branch against the final
   accepted `#3026` shape only after `#3026` lands or maintainers ask for a
   stack.
3. Prepare, but do not publish, fused-path diagnostics that make verifier-batch
   fallback impossible to miss.
4. Keep C/Swift/LM work dependency-ordered: MLX first, then C API, then Swift,
   then LM, then Pines.
5. Treat benchmark charts as kernel evidence only; do not promote TurboQuant
   until the full LM/Pines evidence gate passes.

## Open Questions

- Will maintainers merge `#3026` as-is, split it, or ask for a smaller first
  mode such as one FP format plus affine?
- Will upstream accept debug path logging, or should path selection remain a
  local LM/Swift diagnostic only?
- Should group-size 64 affine get an explicit upstream Python test before merge,
  or should that be a follow-up if maintainers request more coverage?
- Does the accepted `#3026` shape keep the current fallback gate, or does it
  change enough that our verifier-batch follow-up needs to be redesigned?
