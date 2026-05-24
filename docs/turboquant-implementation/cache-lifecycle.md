# Cache Lifecycle

W5 owns the compressed-cache lifecycle and runtime snapshot. Pines needs these facts to record RunDecision, calibrate memory, debug failures, and validate snapshots.

## Worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W5 | `tq/lm-cache-lifecycle` | MVP 1 | P1 |

## Owned files

Primary:

- `Libraries/MLXLMCommon/TurboQuantKVCache.swift`
- `Libraries/MLXLMCommon/TurboQuantCacheRuntimeSnapshot.swift`
- cache lifecycle tests

Do not touch:

- fallback routing except to record snapshot state after W4;
- profile registry;
- benchmark executable.

## Lifecycle enum

```swift
public enum TurboQuantCacheLifecycle: Codable, Sendable {
    case empty
    case rawPrefillChunkOpen
    case compressingChunk(start: Int, count: Int)
    case compressedCommitted(logicalLength: Int, capacity: Int)
    case decodeCompressed
    case degradedPackedFallback(reason: String)
    case degradedDecodedFallback(reason: String)
    case failed(reason: String)
}
```

Rules:

- decode cannot read `compressingChunk`;
- compressed commit happens only after successful eval;
- fallback states record reason;
- failure state records reason;
- invalid transitions assert in debug and produce typed runtime failure in product.

## Runtime snapshot

```swift
public struct TurboQuantCacheRuntimeSnapshot: Hashable, Codable, Sendable {
    public var schemaVersion: Int
    public var lifecycleDescription: String
    public var logicalLength: Int
    public var capacity: Int
    public var pinnedPrefixLength: Int
    public var ringOffset: Int
    public var keyBytes: Int
    public var valueBytes: Int
    public var rawShadowAllocated: Bool
    public var packedFallbackAllocated: Bool
    public var lastAttentionPath: String?
    public var lastFailure: String?
}
```

Required API:

```swift
public func runtimeSnapshot() -> TurboQuantCacheRuntimeSnapshot
```

## Transitions

Expected transitions:

```text
empty
  -> rawPrefillChunkOpen
  -> compressingChunk
  -> compressedCommitted
  -> decodeCompressed
  -> compressedCommitted
```

Fallback transitions:

```text
decodeCompressed
  -> degradedPackedFallback(reason)
  -> compressedCommitted

decodeCompressed
  -> degradedDecodedFallback(reason)
  -> compressedCommitted
```

Failure:

```text
any committed/decode state -> failed(reason)
```

## Exact prefill invariant

Prefill logits come from the exact path unless a future certified profile explicitly allows approximate prefill.

Required behavior:

- compress prefill K/V as side effect after logits;
- commit compressed cache only after eval;
- release raw chunk after commit according to fallback policy;
- partial compressed writes are not visible to decode.

## Tests

Required:

- chunked prefill transitions;
- decode append transitions;
- cache expansion;
- ring offset wrap;
- pinned prefix;
- GQA/MQA mapping;
- batch > 1 if supported;
- decode cannot read compressing chunk;
- failure records reason;
- snapshot stable after expansion.

## Acceptance

W5 is complete when Pines can query lifecycle and storage bytes after prefill and decode, and lifecycle prevents decode from observing incomplete compressed state.
