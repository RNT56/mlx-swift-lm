# Runtime Failures

W4 owns the P0 safety blocker: no product-path zero output and no product-path fatal on TurboQuant attention failure.

## Worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W4 | `tq/lm-typed-errors-no-zero` | MVP 0 | P0 blocker |

## Owned files

Primary:

- `Libraries/MLXLMCommon/AttentionUtils.swift`
- `Libraries/MLXLMCommon/TurboQuantRuntimeFailure.swift`
- attention fallback tests

Do not touch:

- cache lifecycle internals unless required to record failure;
- profile registry;
- benchmark executable.

## Current observed state

The local branch already has a throwing attention path and no obvious all-zero fallback in the inspected path. Deprecated non-throwing wrappers still contain `fatalError` paths. The P0 target is therefore:

- product call sites must use throwing APIs;
- non-throwing wrappers must be debug-only/deprecated and not Pines-facing;
- every TurboQuant failure must throw typed error or use budgeted semantically correct fallback;
- no product path returns zeros.

## Required error type

```swift
public enum TurboQuantRuntimeFailure: Error, Codable, Sendable {
    case compressedAttentionUnavailable(String)
    case packedFallbackUnavailable(String)
    case decodedFallbackUnavailable(String)
    case unsupportedAttentionShape(String)
    case unsupportedAttentionMask(String)
    case unsupportedTensorDType(String)
    case cacheLayoutInvalid(String)
    case cacheLifecycleInvalid(String)
    case noBudgetedFallback(String)
    case fallbackBudgetExceeded(String)
    case modelProfileMismatch(String)
}
```

If existing error names remain, they must map losslessly into this product-safe set for Pines.

## Throwing product path

Required:

```swift
public func attentionWithKVStateThrowing(...) throws -> MLXArray
```

Rules:

- tries compressed attention path;
- tries only semantically correct fallback;
- respects fallback policy;
- records fallback reason;
- throws typed failure if no allowed fallback exists;
- never returns zero as a substitute for failed attention.

## Non-throwing wrappers

Rules:

- mark deprecated;
- restrict to debug/test use;
- do not call from Pines-facing generation;
- if retained, do not silently return invalid output.

Debug fatal flag:

```text
TURBOQUANT_FATAL_FALLBACK=1
```

May exist only for explicit debug/test runs.

## Required forced-failure tests

Test cases:

- compressed fused failure -> fallback or typed error;
- packed fallback unavailable -> typed error or decoded fallback if allowed;
- decoded fallback unavailable -> typed error;
- unsupported mask -> fallback/error;
- unsupported head dimension -> fallback/error;
- fallback budget exceeded -> typed error;
- non-throwing wrapper is not used by product call path;
- no failure path returns all-zero tensor.

## Pines mapping requirements

Pines must be able to map LM errors to:

- `turboQuantPathUnavailable`;
- `turboQuantFallbackUnavailable`;
- `fallbackBudgetExceeded`;
- `unsupportedAttentionShape`;
- `unsupportedAttentionMask`;
- `unsupportedTensorDType`;
- `cacheLayoutInvalid`;
- `cacheLifecycleInvalid`;
- `modelProfileMismatch`;
- `mlxRuntimeFailure`.

## Acceptance

W4 is complete when:

- product attention path is throwing and typed-error safe;
- no product path fatal-errors;
- no product path returns zero/guessed output;
- forced-failure tests pass;
- Pines can catch and map errors.
