// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if let turboQuantOptions = turboQuantLinearLoadOptions(
        metadata: metadata,
        quantization: quantization,
        perLayerQuantization: perLayerQuantization
    ) {
        turboQuantize(
            model: model,
            preset: turboQuantOptions.preset,
            groupSize: turboQuantOptions.groupSize,
            mode: turboQuantOptions.mode,
            backend: .mlxPacked,
            seed: turboQuantOptions.seed,
            valueBits: turboQuantOptions.valueBits
        ) { path, _ in
            weights["\(path).scales"] != nil
        }
    } else if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, _ in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}

private struct TurboQuantLinearLoadOptions {
    var preset: TurboQuantPreset
    var groupSize: Int
    var mode: QuantizationMode
    var seed: UInt64
    var valueBits: Int?
}

private func turboQuantLinearLoadOptions(
    metadata: [String: String],
    quantization: BaseConfiguration.Quantization?,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) -> TurboQuantLinearLoadOptions? {
    let method = metadata["quant_method"]?.lowercased()
    let linearClass = metadata["linear_class"]?.lowercased()
    guard method == "turboquant" || linearClass == "turboquantlinear" else {
        return nil
    }

    let preset =
        metadata["turboquant_preset"]
        .flatMap(TurboQuantPreset.init(rawValue:))
        ?? ((metadata["turboquant_bits"].flatMap(Int.init) ?? quantization?.bits ?? 4) <= 2
            ? .turbo2_5 : .turbo4v2)
    let groupSize =
        metadata["turboquant_group_size"].flatMap(Int.init)
        ?? perLayerQuantization?.quantization?.groupSize
        ?? quantization?.groupSize
        ?? 64
    let mode =
        metadata["turboquant_mode"]
        .flatMap(QuantizationMode.init(rawValue:))
        ?? perLayerQuantization?.quantization?.mode
        ?? quantization?.mode
        ?? .affine
    let seed =
        metadata["turboquant_seed"].flatMap(UInt64.init)
        ?? 0x9E37_79B9_7F4A_7C15
    let valueBits = metadata["turboquant_value_bits"].flatMap(Int.init)

    return TurboQuantLinearLoadOptions(
        preset: preset,
        groupSize: groupSize,
        mode: mode,
        seed: seed,
        valueBits: valueBits
    )
}
