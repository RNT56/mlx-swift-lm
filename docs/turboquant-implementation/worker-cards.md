# LM Worker Cards

This file preserves the `mlx-swift-lm` worker scope from the cross-repo plan.

Use the Pines [Worker Launch Schedule](/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation/14-worker-launch-schedule.md) for execution order. For `mlx-swift-lm`, the executable order is:

| Wave | Worker | Can run when |
| --- | --- | --- |
| Wave 0 | W4 typed errors/no zero/no fatal | immediately |
| Wave 1 | W5 cache lifecycle | W4 product path behavior is safe enough to record state |
| Wave 3 | W6 profile v2 and W22 quality outputs | contracts and evidence schemas are stable |
| Wave 4 | W14A KV snapshot export/import | W5 lifecycle/runtime snapshot exists |
| Wave 6 | W15A speculative verifier | rollback-safe compressed cache prerequisites exist |

W4 is the first LM implementation branch. W14A and W15A may be designed earlier, but activation waits for their wave gates.

## Wave 0 - W4 typed errors and no fatal/zero

Branch: `tq/lm-typed-errors-no-zero`

Owned files:

- `Libraries/MLXLMCommon/AttentionUtils.swift`
- `Libraries/MLXLMCommon/TurboQuantRuntimeFailure.swift`
- tests

Tasks:

- remove zero fallback;
- remove product-path fatal;
- add `TurboQuantRuntimeFailure`;
- make app-facing path throwing;
- deprecate unsafe non-throwing wrapper;
- preserve fatal only under debug/test flag;
- add forced failure tests.

Acceptance:

- no product path returns zero tensor;
- no product path fatal-errors;
- Pines can catch typed error.

## Wave 1 - W5 cache lifecycle

Branch: `tq/lm-cache-lifecycle`

Owned files:

- `Libraries/MLXLMCommon/TurboQuantKVCache.swift`
- `Libraries/MLXLMCommon/TurboQuantCacheRuntimeSnapshot.swift`
- tests

Tasks:

- add `TurboQuantCacheLifecycle`;
- add `runtimeSnapshot()`;
- track logical length;
- track capacity;
- track ring offset;
- track pinned prefix;
- track key/value bytes;
- track raw shadow;
- track packed fallback;
- track last path/failure;
- add invalid-transition asserts.

Acceptance:

- decode never reads compressing cache;
- snapshot stable after expansion;
- Pines can query lifecycle.

## Wave 3 - W6 profile schema v2

Branch: `tq/lm-profile-v2`

Owned files:

- `TurboQuantProfiles/*.json`;
- `Libraries/MLXLMCommon/TurboQuantProfiles.swift`;
- documentation;
- tests.

Tasks:

- add `schema_version`;
- add architecture requirements;
- add hidden/layer/head/KV/headDim fields;
- add RoPE fields;
- add TurboQuant layout fields;
- validate model config;
- reject name-only match;
- surface mismatch reason.

Acceptance:

- profile mismatch disables TurboQuant safely;
- every profile has test coverage.

## Wave 4 - W14A LM KV snapshot export/import

Branch: `tq/lm-kv-snapshots`

Tasks:

- export compressed cache arrays;
- import compressed cache arrays;
- include layout version;
- include logical length;
- include ring offset;
- include pinned prefix;
- validate before import;
- roundtrip next-token logits.

Acceptance:

- valid snapshot restores;
- invalid snapshot fails before use.

## Wave 6 - W15A LM speculative verifier

Branch: `tq/lm-speculative`

Tasks:

- add target verifier;
- tentative append;
- rollback rejected tokens;
- preserve compressed cache;
- track acceptance rate;
- expose metrics.

Acceptance:

- accepted tokens match target;
- rejected tokens do not corrupt cache.

## LM backlog

| ID | Task |
| --- | --- |
| LM-001 | Typed errors |
| LM-002 | Throwing product path |
| LM-003 | Forced-failure tests |
| LM-004 | Cache lifecycle |
| LM-005 | Runtime snapshot |
| LM-006 | Budgeted fallback |
| LM-007 | Layer-local fallback |
| LM-008 | Exact prefill invariant |
| LM-009 | Profile schema v2 |
| LM-010 | Profile mismatch reasons |
| LM-011 | Model benchmark JSON |
| LM-012 | Quality gate outputs |
| LM-013 | Cache expansion tests |
| LM-014 | Ring offset tests |
| LM-015 | Pinned prefix tests |
| LM-016 | GQA/MQA tests |
| LM-017 | Unsupported mask tests |
| LM-018 | Snapshot export |
| LM-019 | Snapshot import |
| LM-020 | Snapshot roundtrip logits |
| LM-021 | Speculative verifier |
| LM-022 | Rollback append |
| LM-023 | Acceptance metrics |
