# Model Profile v2 and Quality Output

W6 owns model profile schema v2. W22 owns quality gate output. Together they ensure TurboQuant activation is based on model structure and measured quality, not model names alone.

Launch wave: Wave 3. Profile v2 and quality output become product-relevant once bridge integration and evidence schemas exist.

Status: implemented on the LM side for W6 and W22 quality-output support. This does not activate Verified product claims, snapshots, speculative decode, Pines bridge wiring, project pins, or generated project changes.

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

LM now exposes `TurboQuantProfileMismatch` with these fields. Manifest validation exposes `mismatches`, and profile selection diagnostics also carry structured mismatches while preserving existing human-readable rejection reasons.

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

`TurboQuantModelBenchmark` now emits:

- top-level `qualityGate`;
- per-result `quality`;
- `schemaVersion = 1` and `gateVersion = 1`;
- `benchmarkSuiteID`;
- `deterministicTop1MatchRate`;
- `logitKLDivergenceMean`;
- `logitMaxAbsErrorP95`;
- optional `attentionOutputCosineMean`;
- `noNaNOrInf`;
- `fallbackEquivalent`;
- `prefillExact`;
- optional `snapshotRoundtripEquivalent`;
- `gateReason`;
- `passed`.

The benchmark computes the synthetic reference by decoding the compressed cache state and running exact SDPA over that decoded state, then compares compressed attention output against that reference. That populates the fallback-equivalence suite and intentionally reports `prefillExact = false`, so the quality gate fails closed until a real prefill-logit exactness suite is run. Snapshot roundtrip remains optional and unpopulated here because W14 snapshot activation is out of scope.

## Tests

Profile tests:

- every profile JSON decodes;
- v1 compatibility or migration is explicit if retained;
- model config mismatch disables TurboQuant;
- mismatch reasons are surfaced;
- supported config enables profile;
- schema newer fails closed when required.

Quality tests:

- prefill exactness comparison fails closed when not measured;
- fallback equivalence;
- no NaN/Inf;
- quality block appears in benchmark JSON.

Implemented coverage:

- schema v1 and future schema profiles fail closed during selection;
- unsupported layout version fails closed during selection;
- manifest mismatch DTO exports stable `field`, `expected`, `actual`, and `disablesTurboQuant`;
- golden QualityGate-shaped benchmark JSON decodes with aggregate `qualityGate` and per-result `quality`;
- quality gate fails closed when NaN/Inf is reported;
- quality gate fails closed when prefill exactness is unmeasured.

## Acceptance

W6/W22 are complete when model profile activation fails closed on mismatch and benchmark reports can produce quality data consumed by Pines evidence import.

Current validation:

```sh
swift build --target MLXLMCommon
swift build --target TurboQuantModelBenchmark
swift test --filter TurboQuantProfileTests
swift run TurboQuantModelBenchmark --iterations 1 --head-dims 64 --contexts 1 --query-lengths 1
```

All four commands passed locally on 2026-05-25. The benchmark run emitted aggregate and per-result quality blocks; it is still evidence input, not a product compatibility claim.

## Qwen Larger-Context Proof Rows

`TurboQuantQwenProof` keeps `--contexts` as the production strict-gate matrix.
Use `--experimental-contexts 65536,131072,262144` for 64K/128K/256K stress
experiments that should be reported but not production-certified. Experimental
rows are emitted with `gateScope = "largeContextExperiment"`,
`strictGateRequired = false`, and
`certificationStatus = "experiment-only-not-production-certified"`. They are
excluded from `summary.strictPassed` unless `--require-experimental-gates` is
specified. `--warmup` controls unmeasured per-case warmup iterations so Metal
compile and first-use cache effects stay out of reported p50/p95 timing. These
fields are part of `TurboQuantQwenProof` report schema v2.

With `mlx-swift` `cff5d0ad87f79585ac778224c21a5278d25a4e79`, the Mac proof
uses the grouped-query block fused kernel for Qwen decode. Local results keep
32K production proof rows above the 20 tok/s p95 gate for Turbo8, Turbo4V2, and
Turbo3.5. The 64K row remains experiment-only: Turbo4V2 p50 was around 20 tok/s,
but p95 was still below the promotion floor during the sustained pinned-package
run, so 64K is not production-certified.
