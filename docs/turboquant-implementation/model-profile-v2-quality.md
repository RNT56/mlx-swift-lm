# Model Profile v2 and Quality Output

W6 owns model profile schema v2. W22 owns quality gate output. Together they ensure TurboQuant activation is based on model structure and measured quality, not model names alone.

## W6 worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W6 | `tq/lm-profile-v2` | MVP 1.5 | P1 |

Owned files:

- `TurboQuantProfiles/*.json`;
- `Libraries/MLXLMCommon/TurboQuantProfiles.swift`;
- `Libraries/MLXLMCommon/Documentation.docc/turboquant-profiles.md`;
- profile tests.

## Profile schema v2

```json
{
  "schema_version": 2,
  "architecture": "",
  "requires": {
    "hidden_size": 0,
    "num_hidden_layers": 0,
    "num_attention_heads": 0,
    "num_key_value_heads": 0,
    "head_dim": 0,
    "rope_type": ""
  },
  "turboquant": {
    "layout_version": 4,
    "key_preset": "turbo3_5",
    "value_bits": 4,
    "group_size": 64,
    "preferred_paths": ["onlineFused", "twoStageCompressed", "mlxPackedFallback"]
  }
}
```

## Validation requirements

Reject or disable TurboQuant when:

- architecture mismatch;
- hidden size mismatch;
- layer count mismatch;
- attention head count mismatch;
- KV head count mismatch;
- head dimension mismatch;
- unsupported RoPE config;
- unsupported sliding-window config;
- layout version unsupported;
- profile schema newer and fail-closed.

No model-name-only activation.

## Mismatch output

Pines needs mismatch reasons:

```swift
public struct TurboQuantProfileMismatch: Codable, Sendable {
    public var field: String
    public var expected: String
    public var actual: String
    public var disablesTurboQuant: Bool
}
```

## Quality output

LM benchmark output should include:

- deterministic top-1 match;
- KL divergence;
- p95 max logit abs error;
- no NaN/Inf;
- fallback equivalence;
- prefill exactness;
- snapshot roundtrip equivalence when applicable.

Quality object is defined by Pines `QualityGate.v1`, but LM benchmark output should be able to populate it.

## Tests

Profile tests:

- every profile JSON decodes;
- v1 compatibility or migration is explicit if retained;
- model config mismatch disables TurboQuant;
- mismatch reasons are surfaced;
- supported config enables profile;
- schema newer fails closed when required.

Quality tests:

- prefill exactness comparison;
- fallback equivalence;
- no NaN/Inf;
- quality block appears in benchmark JSON.

## Acceptance

W6/W22 are complete when model profile activation fails closed on mismatch and benchmark reports can produce quality data consumed by Pines evidence import.
