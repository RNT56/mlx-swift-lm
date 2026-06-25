# TurboQuant LM Implementation Packet

This folder contains the `mlx-swift-lm` side of the TurboQuant implementation train. `mlx-swift-lm` is the model/cache integration layer: it owns typed attention failures, exact prefill, compressed cache commits, cache lifecycle, fallback policy, profile validation, model benchmark output, snapshot export/import, and speculative target verification.

Cross-repo release-train docs live in:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation
```

The executable launch order is owned by:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation/14-worker-launch-schedule.md
```

The PR and merge train is owned by:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation/15-pr-merge-plan.md
```

Current status:

- The current local delivery contract for upstream PRs, staging branches,
  dependency order, and product gates is
  [Fork Stack Delivery Roadmap - 2026-06-25](fork-stack-delivery-roadmap-2026-06-25.md).
- The active Pines compatibility pair is non-green. Local TurboQuant gates and
  synthetic app-host smoke have passed in recent runs, but product promotion
  still requires same-report real-model quality/throughput evidence, physical
  device evidence, and compatibility-pair proof.
- The current continuation anchor for the **inference-speed overhaul** is
  [Speed-Overhaul Continuation Handoff - 2026-06-07](continuation-handoff-2026-06-07-overhaul.md)
  (with [inference-speed-roadmap](inference-speed-roadmap-2026-06-07.md) for the
  measured evidence and [overhaul-plan](overhaul-plan-2026-06-07.md) for the lever
  ladder). Start there for speed/throughput work. Lever ① (no-draft n-gram
  self-speculation) is implemented and validated (byte-exact; 1.2–2.2× on
  structured/long-context workloads).
- The earlier same-day anchor
  [TurboQuant Continuation Handoff - 2026-06-07](continuation-handoff-2026-06-07.md)
  remains valid for the affine quality-gate work (state, artifacts, nonclaims).
- For a full, self-contained, externally-auditable description of the whole design
  (architecture, every codec's math, how the decode kernels are built at the Metal
  level, the speculation algorithms, the measured evidence, the non-claims, and an
  explicit "what to scrutinize" list), see
  [TurboQuant Architecture Report](turboquant-architecture-2026-06-07.md).
- The practical upstream-comparable route is affine K8/V4. Upstream's pinned
  README K8+V4 row is the mixed affine route, not the PolarWHT-V scaffold.
  Full `polarWHTV3` and `hybridK8PolarWHTV3/V4` remain experimental diagnostics
  and are blocked from default promotion.
- Native Sparse-V threshold, top-k, cumulative-mass, hybrid, pageTopK, and
  CandidateSparse diagnostics are wired, but the measured sparse family is
  rejected for promotion. Sparse-V auto policy now resolves to off; only
  explicit proof/debug requests can activate sparse modes.
- Snapshot and speculative contracts are implemented, but product activation
  remains controlled by Pines admission, compatibility, quality, fallback,
  memory, and real-device evidence gates.

For this repo, the launch order is:

1. Wave 0: W4 runtime failures/no zero/no fatal.
2. Wave 1: W5 cache lifecycle/runtime snapshot.
3. Wave 3: W6 model profile v2 and W22 quality outputs.
4. Wave 4: W14A KV snapshot export/import.
5. Wave 6: W15A speculative verifier.

W4 can start immediately in parallel with MLX Swift W1 and Pines W7/W24. W14A and W15A must not product-activate before lifecycle, evidence, and rollback prerequisites are proven.

Worker PRs for the completed train targeted:

```text
tq/wave7-lm-platform
```

Final merge to the repo default branch should preserve the cross-repo
compatibility pair and Pines production pin gate recorded in the Pines packet.

## LM responsibilities

`mlx-swift-lm` owns:

- app-safe typed runtime failures;
- throwing product attention path;
- no zero/guessed tensor fallback;
- exact prefill invariant;
- compressed cache lifecycle;
- budgeted fallback policy;
- runtime cache snapshot;
- model profile schema v2;
- profile mismatch reasons;
- model benchmark JSON;
- quality gate outputs;
- KV snapshot export/import;
- speculative target verifier and rollback.

## Required reading

1. [Fork Stack Delivery Roadmap - 2026-06-25](fork-stack-delivery-roadmap-2026-06-25.md)
2. [LM Worker Cards](worker-cards.md)
3. [Runtime Failures](runtime-failures.md)
4. [Cache Lifecycle](cache-lifecycle.md)
5. [Fallback Policy](fallback-policy.md)
6. [Model Profile v2 and Quality](model-profile-v2-quality.md)
7. [Current Paths and Benchmarks](current-paths-and-benchmarks.md)
8. [External Port Optimization Map](external-port-optimization-map.md)
9. [TurboQuant Continuation Handoff - 2026-06-07](continuation-handoff-2026-06-07.md)
10. [KV Snapshots and Speculative Verifier](kv-snapshots-speculative.md)

## Non-negotiables

1. No product path returns an all-zero or guessed attention output.
2. No product path calls `fatalError`.
3. Non-throwing wrappers are debug-only/deprecated and not used by Pines-facing generation.
4. Prefill logits remain exact.
5. Compressed cache commits only after successful eval.
6. Fallback allocation obeys budget/policy.
7. Cache lifecycle is visible to Pines.
8. Snapshot import validates before cache use.
