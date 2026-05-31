import Foundation
import Testing

@testable import MLXLMCommon

@Suite("TurboQuant Wave 1 runtime policy")
struct TurboQuantRuntimePolicyTests {
    @Test func runtimeModeAndPrecisionPolicyRoundTripCodable() throws {
        let policy = TurboQuantKVPrecisionPolicy(
            key: .fp16OrQ8,
            value: .turbo4v2,
            boundary: .protectedEdges(first: 1, last: 2)
        )
        let parameters = GenerateParameters(
            kvCacheStrategy: .turboQuant,
            turboQuantRuntimeMode: .throughputTurboQuant,
            turboQuantPrecisionPolicy: policy
        )
        let decodedMode = try JSONDecoder().decode(
            TurboQuantRuntimeMode.self,
            from: try JSONEncoder().encode(TurboQuantRuntimeMode.throughputTurboQuant)
        )

        let decodedPolicy = try JSONDecoder().decode(
            TurboQuantKVPrecisionPolicy.self,
            from: try JSONEncoder().encode(policy)
        )

        #expect(decodedMode == .throughputTurboQuant)
        #expect(decodedPolicy == policy)
        #expect(parameters.turboQuantRuntimeMode == .throughputTurboQuant)
        #expect(parameters.effectiveTurboQuantPrecisionPolicy == policy)
    }

    @Test func legacyPresetAndValueBitsStillResolvePolicy() {
        let parameters = GenerateParameters(
            kvCacheStrategy: .turboQuant,
            turboQuantPreset: .turbo3_5,
            turboQuantValueBits: 2
        )

        let policy = parameters.effectiveTurboQuantPrecisionPolicy

        #expect(policy.key == .turbo3_5)
        #expect(policy.value == .turbo2_5)
        #expect(policy.compressedKeyPreset == .turbo3_5)
        #expect(policy.resolvedValueBits == 2)
    }

    @Test func sparseValuePolicyGatesToCapacityAtLongContext() throws {
        let policy = TurboQuantSparseValuePolicy.auto(threshold: 1e-6)
        let decoded = try JSONDecoder().decode(
            TurboQuantSparseValuePolicy.self,
            from: try JSONEncoder().encode(policy)
        )

        #expect(decoded == policy)
        #expect(
            policy.resolvedThreshold(runtimeMode: .capacityTurboQuant, contextLength: 16_384)
                == 1e-6
        )
        #expect(
            policy.resolvedThreshold(runtimeMode: .throughputTurboQuant, contextLength: 65_536)
                == nil
        )
        #expect(
            policy.resolvedThreshold(runtimeMode: .capacityTurboQuant, contextLength: 4096)
                == nil
        )
        #expect(
            TurboQuantSparseValuePolicy.force(threshold: 1e-5)
                .resolvedThreshold(runtimeMode: .capacityTurboQuant, contextLength: 4096)
                == 1e-5
        )
    }

    @Test func boundaryPolicyProtectsLeadingAndTrailingLayersWhenRequired() {
        let policy = TurboQuantKVPrecisionPolicy(
            key: .turbo4v2,
            value: .turbo4v2,
            boundary: .profileDefault
        )

        #expect(policy.requiresRawBoundaryProtection)
        #expect(policy.protectedBoundaryLayerIndexes(layerCount: 8) == Set([0, 1, 6, 7]))
        #expect(policy.protectsBoundaryLayer(layerIndex: 0, layerCount: 8))
        #expect(!policy.protectsBoundaryLayer(layerIndex: 3, layerCount: 8))
    }

    @Test func boundaryPolicyCreatesRawCachesForEdges() {
        let parameters = GenerateParameters(
            maxKVSize: 16,
            kvCacheStrategy: .turboQuant,
            turboQuantRuntimeMode: .capacityTurboQuant,
            turboQuantPrecisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .turbo4v2,
                value: .turbo4v2,
                boundary: .profileDefault
            )
        )

        let caches = makePromptCacheWithLayerCount(numLayers: 8, parameters: parameters)

        #expect(caches[0] is RotatingKVCache)
        #expect(caches[1] is RotatingKVCache)
        #expect(caches[2] is RotatingTurboQuantKVCache)
        #expect(caches[5] is RotatingTurboQuantKVCache)
        #expect(caches[6] is RotatingKVCache)
        #expect(caches[7] is RotatingKVCache)
    }
}
