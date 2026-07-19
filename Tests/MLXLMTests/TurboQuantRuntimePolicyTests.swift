import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("TurboQuant Wave 1 runtime policy")
struct TurboQuantRuntimePolicyTests {
    @Test func runtimeModeAndPrecisionPolicyRoundTripCodable() throws {
        let policy = TurboQuantKVPrecisionPolicy(
            key: .fp16OrQ8,
            value: .turbo4v2,
            boundary: .protectedEdges(first: 1, last: 2),
            boundaryCachePrecision: .affineK8V4
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

    @Test func compressedValuePrecisionKeepsThreeBitValuesDistinct() {
        let policy = TurboQuantKVPrecisionPolicy(
            key: .affineQ8,
            value: .compressed(bits: 3)
        )

        #expect(policy.value == .turbo3_5)
        #expect(policy.resolvedValueBits == 3)
        #expect(policy.requiresBoundaryProtection)
        #expect(policy.protectedBoundaryLayerIndexes(layerCount: 8) == Set([0, 1, 6, 7]))
    }

    @Test func sparseValuePolicyAutoStaysOffUntilExplicitForce() throws {
        let policy = TurboQuantSparseValuePolicy.auto(threshold: 1e-6)
        let decoded = try JSONDecoder().decode(
            TurboQuantSparseValuePolicy.self,
            from: try JSONEncoder().encode(policy)
        )

        #expect(decoded == policy)
        #expect(
            policy.resolvedThreshold(runtimeMode: .capacityTurboQuant, contextLength: 16_384)
                == nil
        )
        #expect(
            policy.resolvedThreshold(runtimeMode: .throughputTurboQuant, contextLength: 65_536)
                == nil
        )
        #expect(
            policy.resolvedThreshold(runtimeMode: .capacityTurboQuant, contextLength: 4096)
                == nil
        )
        #expect(TurboQuantSparseValuePolicy.profileDefault == .off)
        #expect(TurboQuantSparseValueSelection.thresholdPolicy(policy) == .off)
        #expect(
            TurboQuantSparseValuePolicy.force(threshold: 1e-5)
                .resolvedThreshold(runtimeMode: .capacityTurboQuant, contextLength: 4096)
                == 1e-5
        )
    }

    @Test func sparseValuePolicyDefaultsOffUntilExplicitlyRequested() {
        let parameters = GenerateParameters(
            maxKVSize: 16_384,
            kvCacheStrategy: .turboQuant,
            turboQuantRuntimeMode: .capacityTurboQuant
        )

        #expect(parameters.turboQuantSparseValuePolicy == .off)
        #expect(parameters.effectiveTurboQuantSparseValuePolicy == .off)
        #expect(
            parameters.effectiveTurboQuantSparseValuePolicy.resolvedThreshold(
                runtimeMode: .capacityTurboQuant,
                contextLength: 16_384
            ) == nil
        )
    }

    @Test func boundaryPolicyProtectsLeadingAndTrailingLayersWhenRequired() {
        let policy = TurboQuantKVPrecisionPolicy(
            key: .turbo4v2,
            value: .turbo4v2,
            boundary: .profileDefault
        )

        #expect(policy.requiresBoundaryProtection)
        #expect(policy.boundaryCachePrecision == .affineK8V4)
        #expect(policy.protectedBoundaryLayerIndexes(layerCount: 8) == Set([0, 1, 6, 7]))
        #expect(policy.protectsBoundaryLayer(layerIndex: 0, layerCount: 8))
        #expect(!policy.protectsBoundaryLayer(layerIndex: 3, layerCount: 8))
    }

    @Test func pureTurbo8DoesNotProtectBoundaryLayersByDefault() {
        let policy = TurboQuantKVPrecisionPolicy.legacy(
            preset: .turbo8,
            valueBits: 8
        )

        #expect(!policy.requiresBoundaryProtection)
        #expect(policy.protectedBoundaryLayerIndexes(layerCount: 8).isEmpty)
    }

    @Test func boundaryPolicyCreatesK8V4CachesForEdges() throws {
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

        #expect(caches[0] is AffineK8V4KVCache)
        #expect(caches[1] is AffineK8V4KVCache)
        #expect(caches[2] is RotatingTurboQuantKVCache)
        #expect(caches[5] is RotatingTurboQuantKVCache)
        #expect(caches[6] is AffineK8V4KVCache)
        #expect(caches[7] is AffineK8V4KVCache)

        let first = try #require(caches[0] as? AffineK8V4KVCache)
        #expect(first.attentionDiagnostics.layerIndex == 0)
        #expect(first.attentionDiagnostics.boundaryProtectedLayerCount == 4)
        #expect(first.attentionDiagnostics.activeAttentionPath == .affineK8V4Native)
    }

    @Test func deferredAffineK8VxConversionPreservesLayerDiagnostics() throws {
        try Device.withDefaultDevice(.cpu) {
            var caches: [KVCache] = (0 ..< 4).map { _ in KVCacheSimple() }
            let keys = MLXArray.ones([1, 1, 4, 64], dtype: .float32)
            let values = MLXArray.ones([1, 1, 4, 64], dtype: .float32)
            for index in caches.indices {
                _ = caches[index].update(keys: keys, values: values)
            }

            maybeQuantizeKVCache(
                cache: &caches,
                kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
                kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
                quantizedKVStart: 0,
                kvCacheStrategy: .affineK8Vx,
                kvCodec: .affineK8Vx,
                turboQuantValueBits: 3,
                turboQuantValueGroupSize: 64,
                kvLayerPolicy: .affineK8VxProtectedEdges(
                    layerCount: 4,
                    valueBits: 3,
                    first: 1,
                    last: 1
                )
            )

            let edge = try #require(caches[0] as? AffineK8V4KVCache)
            let middle = try #require(caches[1] as? AffineK8V4KVCache)
            #expect(edge.attentionDiagnostics.layerIndex == 0)
            #expect(edge.attentionDiagnostics.activeAttentionPath == .affineK8V4Native)
            #expect(edge.attentionDiagnostics.boundaryProtectedLayerCount == 2)
            #expect(middle.attentionDiagnostics.layerIndex == 1)
            #expect(middle.attentionDiagnostics.activeAttentionPath == .affineK8VxNative)
            #expect(middle.attentionDiagnostics.boundaryProtectedLayerCount == 2)
            #expect(middle.valueBits == 3)
            #expect(middle.valueGroupSize == 64)
        }
    }

    @Test func rawBoundaryCachesRemainExplicitCompatibilityOverride() {
        let parameters = GenerateParameters(
            maxKVSize: 16,
            kvCacheStrategy: .turboQuant,
            turboQuantRuntimeMode: .capacityTurboQuant,
            turboQuantPrecisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .turbo4v2,
                value: .turbo4v2,
                boundary: .profileDefault,
                boundaryCachePrecision: .raw
            )
        )

        let caches = makePromptCacheWithLayerCount(numLayers: 8, parameters: parameters)

        #expect(caches[0] is RotatingKVCache)
        #expect(caches[1] is RotatingKVCache)
        #expect(caches[2] is RotatingTurboQuantKVCache)
        #expect(caches[6] is RotatingKVCache)
        #expect(caches[7] is RotatingKVCache)
    }

    @Test func boundaryOrdinalsCanIgnoreNonAttentionOrSlidingLayers() {
        let parameters = GenerateParameters(
            maxKVSize: 16,
            kvCacheStrategy: .turboQuant,
            turboQuantRuntimeMode: .capacityTurboQuant,
            turboQuantPrecisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .turbo4v2,
                value: .turbo4v2,
                boundary: .protectedEdges(first: 1, last: 1)
            )
        )

        let firstFullAttention = makeAttentionKVCache(
            parameters: parameters,
            layerIndex: 4,
            layerCount: 12,
            boundaryLayerIndex: 0,
            boundaryLayerCount: 3
        )
        let slidingLayer = makeAttentionKVCache(
            parameters: parameters,
            layerIndex: 0,
            layerCount: 12,
            boundaryLayerIndex: -1,
            boundaryLayerCount: 3
        )
        let middleFullAttention = makeAttentionKVCache(
            parameters: parameters,
            layerIndex: 8,
            layerCount: 12,
            boundaryLayerIndex: 1,
            boundaryLayerCount: 3
        )

        #expect(firstFullAttention is AffineK8V4KVCache)
        #expect(slidingLayer is RotatingTurboQuantKVCache)
        #expect(middleFullAttention is RotatingTurboQuantKVCache)
    }
}
