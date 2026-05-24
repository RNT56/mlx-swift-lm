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
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    lazyLoad: Bool = false
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

    if ExpertStreamingConfig.shared.isEnabled {
        ExpertStreamerManager.shared = ExpertStreamerManager(modelDirectory: modelDirectory)
    }

    if TurboQuantCheckpointMetadataValidator.isTurboQuantCheckpoint(metadata) {
        try TurboQuantCheckpointMetadataValidator.validate(metadata)
    }

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

    assignExpertStreamingMetadata(model: model, weights: weights)

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    let verification: Module.VerifyUpdate =
        ExpertStreamingConfig.shared.isEnabled
        ? .noUnusedKeys
        : .all
    try model.update(parameters: parameters, verify: verification)

    if !lazyLoad {
        eval(model)
    }
}

private func assignExpertStreamingMetadata(
    model: BaseLanguageModel,
    weights: [String: MLXArray]
) {
    guard ExpertStreamingConfig.shared.isEnabled,
        let manager = ExpertStreamerManager.shared
    else {
        return
    }

    let knownPrefixes = ["", "model.", "language_model.", "model.language_model."]
    let scaleTensors = weights.filter {
        $0.key.contains(".switch_mlp.") && $0.key.hasSuffix(".weight_scale_inv")
    }

    for (path, module) in model.namedModules() {
        guard let switchLinear = module as? ExpertStreamingSwitchLinear else {
            continue
        }

        if let scale = scaleTensors["\(path).weight_scale_inv"] {
            switchLinear.weightScaleInv = scale
        }

        let bareName = "\(path).weight"
        let strippedBareName = stripCommonPrefixes(from: bareName)
        let strippedMTPName = strippedBareName.replacingOccurrences(of: ".mtp.0.", with: ".mtp.")
        let stackedCandidates =
            [bareName, strippedBareName, strippedMTPName]
            + knownPrefixes.map { $0 + strippedBareName }
            + knownPrefixes.map { $0 + strippedMTPName }

        if let originalKey = stackedCandidates.first(where: { manager.getFile(for: $0) != nil }) {
            switchLinear.tensorName = originalKey
            continue
        }

        guard bareName.contains(".switch_mlp.") else {
            continue
        }

        let expert0Name =
            bareName
            .replacingOccurrences(of: ".switch_mlp.", with: ".experts.")
            .replacingOccurrences(of: ".experts.", with: ".experts.0.")
        let strippedExpert0Name = stripCommonPrefixes(from: expert0Name)
        let strippedMTPExpert0Name = strippedExpert0Name.replacingOccurrences(
            of: ".mtp.0.", with: ".mtp.")
        let unstackedCandidates =
            [expert0Name, strippedExpert0Name, strippedMTPExpert0Name]
            + knownPrefixes.map { $0 + strippedExpert0Name }
            + knownPrefixes.map { $0 + strippedMTPExpert0Name }

        guard
            let matchedExpert0 = unstackedCandidates.first(where: {
                manager.getFile(for: $0) != nil
            })
        else {
            continue
        }

        var map = [Int: (path: String, tensorName: String)]()
        for expertIndex in 0 ..< switchLinear.expertCount {
            let tensorName = matchedExpert0.replacingOccurrences(
                of: ".experts.0.", with: ".experts.\(expertIndex).")
            if let file = manager.getFile(for: tensorName) {
                map[expertIndex] = (
                    manager.modelDirectory.appendingPathComponent(file).path,
                    tensorName
                )
            }
        }
        if !map.isEmpty {
            switchLinear.unstackedSSDMap = map
        }
    }
}

private func stripCommonPrefixes(from name: String) -> String {
    var stripped = name
    for prefix in ["language_model.model.", "language_model.", "model."] {
        if stripped.hasPrefix(prefix) {
            stripped.removeFirst(prefix.count)
            break
        }
    }
    return stripped
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
