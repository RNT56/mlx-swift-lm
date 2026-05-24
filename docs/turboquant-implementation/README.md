# TurboQuant LM Implementation Packet

This folder contains the `mlx-swift-lm` side of the TurboQuant implementation train. `mlx-swift-lm` is the model/cache integration layer: it owns typed attention failures, exact prefill, compressed cache commits, cache lifecycle, fallback policy, profile validation, model benchmark output, snapshot export/import, and speculative target verification.

Cross-repo release-train docs live in:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation
```

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

1. [Runtime Failures](runtime-failures.md)
2. [Cache Lifecycle](cache-lifecycle.md)
3. [Fallback Policy](fallback-policy.md)
4. [Model Profile v2 and Quality](model-profile-v2-quality.md)
5. [KV Snapshots and Speculative Verifier](kv-snapshots-speculative.md)
6. [LM Worker Cards](worker-cards.md)

## Non-negotiables

1. No product path returns an all-zero or guessed attention output.
2. No product path calls `fatalError`.
3. Non-throwing wrappers are debug-only/deprecated and not used by Pines-facing generation.
4. Prefill logits remain exact.
5. Compressed cache commits only after successful eval.
6. Fallback allocation obeys budget/policy.
7. Cache lifecycle is visible to Pines.
8. Snapshot import validates before cache use.
