import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite(.serialized)
struct SwitchLayersStreamingTests {
    @Test func quantizedSwitchLinearMmapFallbackMatchesEager() throws {
        MLXRandom.seed(7)
        let weight = MLXRandom.normal([2, 3, 32])
        let base = SwitchLinear(inputDims: 32, outputDims: 3, numExperts: 2, weight: weight)
        let layer = QuantizedSwitchLinear(base, groupSize: 32, bits: 4, mode: .affine)
        let x = MLXRandom.normal([3, 32])

        let ranges = [
            ExpertRange(id: 0, start: 0, end: 1),
            ExpertRange(id: 1, start: 1, end: 2),
            ExpertRange(id: 0, start: 2, end: 3),
        ]
        let eager = layer.computeExperts(x, buffers: nil, ranges: ranges)

        let directory = try temporaryDirectory()
        ExpertStreamingConfig.shared.activate(modelDirectory: directory, useDirectIO: false)
        defer { ExpertStreamingConfig.shared.deactivate() }
        let streamed = layer.computeStreamedExperts(x, ranges: ranges)

        assertClose(streamed, eager)
    }

    @Test func switchGLUStreamingFallbackMatchesEagerWithScatter() throws {
        MLXRandom.seed(42)
        let layer = SwitchGLU(
            inputDims: 3,
            hiddenDims: 4,
            numExperts: 2,
            activation: { $0 },
            bias: true)
        let x = MLXArray(
            [
                1.0, 0.5, -1.0,
                -0.25, 0.75, 1.25,
                0.5, -1.5, 0.25,
                1.5, 0.25, -0.75,
            ] as [Float]
        ).reshaped([4, 3])
        let indices = MLXArray([0, 1, 0, 1] as [UInt32]).reshaped([4, 1])

        ExpertStreamingConfig.shared.deactivate()
        let eager = layer(x, indices)

        let directory = try temporaryDirectory()
        ExpertStreamingConfig.shared.activate(modelDirectory: directory, useDirectIO: false)
        defer { ExpertStreamingConfig.shared.deactivate() }
        let streamed = layer(x, indices)

        assertClose(streamed, eager)
    }

    @Test func directResolverUsesUnstackedExpertMetadata() throws {
        let layer = SwitchLinear(inputDims: 2, outputDims: 3, numExperts: 4, bias: false)
        let directory = try temporaryDirectory()
        ExpertStreamingConfig.shared.activate(modelDirectory: directory, useDirectIO: true)
        defer { ExpertStreamingConfig.shared.deactivate() }

        layer.unstackedSSDMap = [
            2: (
                path: directory.appendingPathComponent("expert-2.safetensors").path,
                tensorName: "model.layers.0.experts.2.down_proj.weight"
            )
        ]

        #if os(macOS)
            let info = layer.resolveSSDInfo(expertIndex: 2)
            #expect(info?.path.hasSuffix("expert-2.safetensors") == true)
            #expect(info?.tensorName == "model.layers.0.experts.2.down_proj.weight")
            #expect(info?.readIndex == 0)
        #else
            #expect(layer.resolveSSDInfo(expertIndex: 2) == nil)
        #endif
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private func assertClose(
    _ actual: MLXArray,
    _ expected: MLXArray,
    tolerance: Float = 1e-4,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    MLX.eval([actual, expected])
    #expect(actual.shape == expected.shape, sourceLocation: sourceLocation)

    let actualValues = actual.asArray(Float.self)
    let expectedValues = expected.asArray(Float.self)
    #expect(actualValues.count == expectedValues.count, sourceLocation: sourceLocation)

    for (actual, expected) in zip(actualValues, expectedValues) {
        #expect(abs(actual - expected) <= tolerance, sourceLocation: sourceLocation)
    }
}
