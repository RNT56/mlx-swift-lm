// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

private final class TinyTurboQuantLoadModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "linear") var linear: Linear

    override init() {
        self._linear.wrappedValue = Linear(64, 2, bias: false)
    }
}

public class BaseConfigurationTests: XCTestCase {

    func testQuantization() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 128,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"), .init(groupSize: 128, bits: 4))
    }

    func testHeterogenousQuantization() throws {
        // from https://huggingface.co/mlx-community/Qwen3-1.7B-4bit-AWQ/blob/main/config.json#L20
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.0.self_attn.q_norm": false,
                    "true_layer": true
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        // a random layer -- no specific configuration gets default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))

        // layer with an override
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 32, bits: 4))

        // layer with an override -- not quant
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))

        // layer with an override -- true, use the default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "true_layer"),
            .init(groupSize: 64, bits: 4))
    }

    func testTurboQuantQuantizationConfigIgnoresMetadataKeys() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "quant_method": "turboquant",
                    "linear_class": "TurboQuantLinear",
                    "turboquant_format": "mlx_packed",
                    "preset": "turbo4v2",
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "seed": "11400714819323198485",
                    "value_bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        let quantization = try XCTUnwrap(config.perLayerQuantization?.quantization(layer: "linear"))
        XCTAssertEqual(quantization.groupSize, 64)
        XCTAssertEqual(quantization.bits, 4)
        XCTAssertEqual(quantization.mode, .affine)
    }

    func testLoadWeightsUsesTurboQuantLinearForConvertedCheckpoints() throws {
        let temporaryPath = FileManager.default.temporaryDirectory.appending(
            path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: temporaryPath, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryPath) }

        let converted = try turboQuantConvertedArrays(
            ["linear.weight": MLXArray.ones([2, 64], dtype: .float32)],
            metadata: ["format": "mlx"],
            options: TurboQuantCheckpointConversionOptions(groupSize: 64)
        )
        try save(
            arrays: converted.arrays,
            metadata: converted.metadata,
            url: temporaryPath.appending(path: "model.safetensors")
        )

        let model = TinyTurboQuantLoadModel()
        try loadWeights(
            modelDirectory: temporaryPath,
            model: model
        )

        let linear = try XCTUnwrap(model.linear as? TurboQuantLinear)
        XCTAssertEqual(linear.preset, .turbo4v2)
        XCTAssertEqual(linear.activeBackend, .mlxPacked)
        XCTAssertEqual(linear(MLXArray.ones([1, 64], dtype: .float32)).shape, [1, 2])
    }

}
