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

For this repo, the launch order is:

1. Wave 0: W4 runtime failures/no zero/no fatal.
2. Wave 1: W5 cache lifecycle/runtime snapshot.
3. Wave 3: W6 model profile v2 and W22 quality outputs.
4. Wave 4: W14A KV snapshot export/import.
5. Wave 6: W15A speculative verifier.

W4 can start immediately in parallel with MLX Swift W1 and Pines W7/W24. W14A and W15A must not product-activate before lifecycle, evidence, and rollback prerequisites are proven.

Worker PRs in this repo target:

```text
codex/turboquant-completion-hardening
```

Final merge to the repo default branch waits for the cross-repo compatibility pair and Pines production pin gate.

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

1. [LM Worker Cards](worker-cards.md)
2. [Runtime Failures](runtime-failures.md)
3. [Cache Lifecycle](cache-lifecycle.md)
4. [Fallback Policy](fallback-policy.md)
5. [Model Profile v2 and Quality](model-profile-v2-quality.md)
6. [KV Snapshots and Speculative Verifier](kv-snapshots-speculative.md)

## Non-negotiables

1. No product path returns an all-zero or guessed attention output.
2. No product path calls `fatalError`.
3. Non-throwing wrappers are debug-only/deprecated and not used by Pines-facing generation.
4. Prefill logits remain exact.
5. Compressed cache commits only after successful eval.
6. Fallback allocation obeys budget/policy.
7. Cache lifecycle is visible to Pines.
8. Snapshot import validates before cache use.
