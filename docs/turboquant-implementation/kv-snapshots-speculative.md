# KV Snapshots and Speculative Verifier

This document covers the later LM work for compressed KV snapshot export/import and speculative target verification. Both are gated behind earlier safety, lifecycle, and evidence work.

Launch waves: W14A snapshots are Wave 4; W15A speculative verifier is Wave 6. Earlier design is allowed behind disabled flags, but activation waits for lifecycle, evidence, and rollback gates.

## Snapshot worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W14A | `tq/lm-kv-snapshots` | MVP 3 | P2 |

## Snapshot payload

```swift
public struct TurboQuantKVSnapshotPayload: Sendable {
    public var manifest: TurboQuantKVSnapshotManifest
    public var compressedArrays: [String: MLXArray]
}
```

LM may define a local manifest mirror, but it must map to Pines `KVSnapshotManifest.v1` fields.

## Export requirements

Export:

- compressed key arrays/pages;
- compressed value arrays/pages;
- layout version;
- logical length;
- capacity;
- ring offset;
- pinned prefix length;
- role metadata;
- dtype/shape metadata needed for validation.

Export must not require raw KV.

## Import requirements

Import:

1. validate manifest schema/version;
2. validate layout version;
3. validate role/shape/dtype;
4. validate logical length/capacity/ring offset/pinned prefix;
5. rebuild compressed cache state;
6. expose runtime snapshot;
7. fail before cache use on invalid data.

## Snapshot acceptance

- export/import roundtrip works;
- restored next-token logits match within tolerance;
- invalid manifest fails before use;
- ring offset and pinned prefix survive roundtrip;
- no raw KV required.

## Speculative worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W15A | `tq/lm-speculative` | MVP 5 | P3 |

## Speculative requirements

Speculative TurboQuant decode requires:

- target verifier API;
- tentative compressed cache append;
- rollback rejected tokens;
- acceptance metrics;
- tokenizer compatibility surfaced to Pines;
- no cache corruption on rejection.

## Speculative flow

```text
draft model proposes tokens
  -> target model verifies against compressed KV
  -> accepted tokens commit
  -> rejected tokens rollback
  -> acceptance stats update
```

## Speculative acceptance

- accepted sequence equals target baseline;
- rejected draft tokens do not corrupt cache;
- rollback restores logical length/ring offset/pinned prefix;
- acceptance rate is exposed;
- poor acceptance can disable speculation in Pines.

## Tests

Snapshot:

- valid export/import;
- invalid layout rejected;
- prefix mismatch rejected by Pines manifest validation;
- next-token logits within tolerance.

Speculative:

- accept all draft tokens;
- reject partial span;
- reject first token;
- rollback after ring wrap;
- rollback with pinned prefix;
- acceptance metrics recorded.
