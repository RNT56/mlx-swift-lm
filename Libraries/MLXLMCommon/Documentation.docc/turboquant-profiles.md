# TurboQuant Profiles

TurboQuant profiles describe a safe runtime KV-cache policy for ordinary MLX model
weights. They are not calibrated model files and they do not replace Hugging Face
or MLX weight artifacts.

Use a profile when an application wants a model-aware default for
``GenerateParameters``:

```swift
let parameters = GenerateParameters(
    turboQuantModelID: "mlx-community/Qwen3-4B-4bit",
    keyHeadDimension: 128,
    valueHeadDimension: 128
)
```

If a bundled profile matches, the initializer enables
``KVCacheStrategy/turboQuant`` and fills in the preset, value bits, group size,
backend, and optimization policy. If no profile matches, the supplied base
parameters are returned unchanged.

The root `TurboQuantProfiles/` directory contains JSON copies of the bundled
profiles for downstream apps, release tooling, and manual inspection. The Swift
registry lives in ``TurboQuantProfileRegistry`` so library consumers do not need
package resources at runtime.

## Safety Contract

Profiles are advisory. The runtime still enforces:

- Metal availability and bundled metallib loading.
- TurboQuant runtime self-tests.
- Query/key/value shape compatibility.
- Mask support for the selected attention path.
- Fallback to MLX packed quantization when the Metal path is unavailable.

Measured fields such as perplexity delta, token throughput, and memory footprint
remain optional and should stay empty until reproduced on the target hardware and
model revision.

## Converted Weights

TurboQuant profile selection is independent of converted model weights. For
weight compression, use `swift run TurboQuantConverter` from the `mlx-swift`
fork. Converted safetensors carry `quant_method=turboquant` and
`linear_class=TurboQuantLinear` metadata; `loadWeights` uses that metadata to
replace matching linear layers with `TurboQuantLinear` before applying the packed
weight, scale, and bias tensors.

## Scheme Aliases

``TurboQuantScheme/turbo4v2`` maps to the `.turbo4v2` runtime preset with
4-bit keys and values. It is the balanced default profile scheme. ``TurboQuantScheme/turbo3``
maps to the more memory-oriented `.turbo2_5` runtime preset and should be reserved
for memory-pressure profiles.
