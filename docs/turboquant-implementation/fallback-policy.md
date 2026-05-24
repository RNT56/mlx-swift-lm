# Fallback Policy

Fallbacks are correctness and memory contracts. A fallback may only run if it is semantically correct and budgeted by Pines admission.

## Worker

Fallback work is shared by:

- W4 for typed failures and no-zero behavior;
- W5 for lifecycle state;
- LM-006/LM-007 for budgeted and layer-local fallback.

## Fallback policy enum

```swift
public enum TurboQuantFallbackPolicy: Codable, Sendable {
    case exactRequired
    case packedAllowed
    case compressedDecodeAllowed
    case fatalOnFailure
}
```

If more granularity is required, it must map back to the Pines `LocalFallbackContract`.

## Pines contract mapping

Pines controls:

- allow packed fallback;
- allow decoded layer-local fallback;
- allow full decoded fallback;
- allow shorter context retry;
- fail if compressed path unavailable;
- reserve bytes.

LM must not allocate fallback forms that admission did not allow.

## Allowed ladder

Preferred ladder:

```text
compressed fused
  -> two-stage compressed QK/AV
  -> packed quantized fallback if allowed and allocated/budgeted
  -> decoded layer-local fallback if allowed and budgeted
  -> typed error
```

Full all-layer decoded fallback is disabled by default and must not happen in product paths unless explicitly admitted.

## Required failure reasons

Fallback reasons should distinguish:

- fused unsupported;
- tiled unsupported;
- QK unavailable;
- AV unavailable;
- unsupported mask;
- unsupported head dimension;
- packed fallback unavailable;
- decoded fallback disabled;
- decoded fallback would exceed budget;
- cache layout invalid;
- lifecycle invalid.

## Runtime snapshot fields

Snapshot must expose:

- raw shadow allocated;
- packed fallback allocated;
- last attention path;
- last failure;
- key bytes;
- value bytes.

## Tests

Required:

- fused fails -> two-stage;
- two-stage fails -> packed fallback if allowed;
- packed unavailable -> decoded fallback if allowed;
- decoded fallback disabled -> typed error;
- full decoded fallback never allocates without explicit budget;
- fallback reason recorded;
- lifecycle records degraded fallback state;
- RunDecision can identify fallback used.

## Acceptance

Fallback policy is complete when no fallback allocates unbudgeted all-layer decoded KV, and every fallback failure reaches Pines as a typed error with reason.
