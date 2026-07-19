import Foundation
import Testing

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {

    @Suite
    struct TurboQuantAdmissionPlannerTests {

        @Test func testAdmitsRequestedContextWithEnoughMemory() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 24)

            let admission = planner.admit(
                profile: profile,
                requestedContextLength: 8192,
                promptTokenCount: 128,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo3_5,
                memorySample: Self.sample(availableGiB: 12, modelGiB: 2)
            )

            #expect(admission.admitted)
            #expect(admission.admittedContextLength == 8192)
            #expect(admission.selectedMode == .balanced)
            #expect(admission.downgradeReasons.isEmpty)
            #expect(admission.memoryPlan.runtimeZones.compressedKVBytes > 0)
            #expect(!admission.memoryPlan.usesRawShadow)
            #expect(admission.memoryPlan.boundaryLayerCount == 4)
            #expect(admission.memoryPlan.boundaryCachePrecision == .affineK8V4)
            #expect(admission.memoryPlan.boundaryKVBytes > 0)
            #expect(admission.memoryPlan.rawBoundaryLayerCount == 0)
            #expect(admission.memoryPlan.rawBoundaryKVBytes == 0)
            #expect(admission.memoryPlan.runtimeZones.rawShadowBytes == 0)
            #expect(admission.userMessage.contains("8192 tokens"))
        }

        @Test func testFastestAdmissionDoesNotReserveRawShadow() {
            let planner = Self.planner()
            let admission = planner.admit(
                profile: Self.profile(weightGiB: 2, layers: 24),
                requestedContextLength: 8192,
                userMode: .fastest,
                fallbackPolicy: .packedAllowed,
                preset: .turbo8,
                valueBits: 8,
                memorySample: Self.sample(availableGiB: 12, modelGiB: 2)
            )

            #expect(admission.admitted)
            #expect(admission.selectedMode == .fastest)
            #expect(!admission.memoryPlan.usesRawShadow)
            #expect(admission.memoryPlan.boundaryLayerCount == 0)
            #expect(admission.memoryPlan.boundaryKVBytes == 0)
            #expect(admission.memoryPlan.runtimeZones.rawShadowBytes
                == admission.memoryPlan.rawBoundaryKVBytes)
        }

        @Test func testLowMemoryReducesAdmittedContextBeforeGeneration() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 24)

            let admission = planner.admit(
                profile: profile,
                requestedContextLength: 65_536,
                userMode: .maxContext,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo3_5,
                memorySample: Self.sample(availableGiB: 3, modelGiB: 2)
            )

            #expect(admission.admitted)
            #expect(admission.admittedContextLength < 65_536)
            #expect(admission.admittedContextLength >= 512)
            #expect(admission.primaryDowngradeReason == .reducedContext)
            #expect(
                admission.downgradeReasons.contains { $0.reason == .reducedContext }
            )
        }

        @Test func testFallbackReserveCanBeDisabledByPolicy() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 16)
            let sample = Self.sample(availableGiB: 16, modelGiB: 2)

            let packed = planner.admit(
                profile: profile,
                requestedContextLength: 4096,
                userMode: .balanced,
                fallbackPolicy: .packedAllowed,
                preset: .turbo3_5,
                memorySample: sample
            )
            let fatal = planner.admit(
                profile: profile,
                requestedContextLength: 4096,
                userMode: .balanced,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo3_5,
                memorySample: sample
            )

            #expect(packed.admitted)
            #expect(fatal.admitted)
            #expect(packed.memoryPlan.runtimeZones.fallbackReserveBytes > 0)
            #expect(fatal.memoryPlan.runtimeZones.fallbackReserveBytes == 0)
            #expect(packed.memoryPlan.packedFallbackEnabled)
            #expect(!fatal.memoryPlan.packedFallbackEnabled)
        }

        @Test func testDecodedFallbackReserveCoversFullContextPerDecodedLayer() {
            let planner = TurboQuantAdmissionPlanner(
                options: TurboQuantAdmissionPlanner.Options(
                    defaultTokenizerBytes: 0,
                    defaultUIReserveBytes: 0,
                    defaultScratchBytes: 0,
                    exactFallbackDecodeLayerCount: 2
                )
            )
            let profile = Self.profile(weightGiB: 2, layers: 16)
            let contextLength = 4096
            let admission = planner.admit(
                profile: profile,
                requestedContextLength: contextLength,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo3_5,
                memorySample: Self.sample(availableGiB: 16, modelGiB: 2)
            )

            let rawBytesPerTokenPerLayer =
                admission.memoryPlan.rawBytesPerToken / max(1, profile.layerCount)
            #expect(admission.admitted)
            #expect(
                admission.memoryPlan.runtimeZones.fallbackReserveBytes
                    == rawBytesPerTokenPerLayer * 2 * admission.admittedContextLength
            )
        }

        @Test func testBalancedDowngradesToMaxContextBeforeReducingContext() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 32)
            let available = Self.gib(4)

            let admission = planner.admit(
                profile: profile,
                requestedContextLength: 32_768,
                userMode: .balanced,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo3_5,
                precisionPolicy: .legacy(
                    preset: .turbo3_5,
                    valueBits: nil,
                    boundary: .disabled
                ),
                memorySample: TurboQuantRuntimeMemorySample(
                    availableAppMemoryBytes: available,
                    modelResidentBytes: Self.gib(2),
                    tokenizerBytes: 0,
                    promptBytes: 0,
                    uiReserveBytes: 0,
                    thermalState: .nominal
                )
            )

            #expect(admission.admitted)
            #expect(admission.admittedContextLength == 32_768)
            #expect(admission.selectedMode == .maxContext)
            #expect(
                admission.downgradeReasons.contains {
                    $0.reason == .movedBalancedToMaxContext
                }
            )
            #expect(admission.memoryPlan.valueBits == 2)
            #expect(admission.memoryPlan.preset == .turbo2_5)
        }

        @Test func testValueBitsDowngradeHappensBeforeKeyPresetDowngrade() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 24)

            let admission = planner.admit(
                profile: profile,
                requestedContextLength: 65_536,
                userMode: .balanced,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo8,
                valueBits: 8,
                precisionPolicy: .legacy(
                    preset: .turbo8,
                    valueBits: 8,
                    boundary: .disabled
                ),
                memorySample: TurboQuantRuntimeMemorySample(
                    availableAppMemoryBytes: Self.gib(6),
                    modelResidentBytes: Self.gib(2),
                    tokenizerBytes: 0,
                    promptBytes: 0,
                    uiReserveBytes: 0,
                    thermalState: .nominal
                )
            )

            #expect(admission.admitted)
            #expect(admission.memoryPlan.preset == .turbo8)
            #expect(admission.memoryPlan.valueBits == 2)
            #expect(
                admission.downgradeReasons.contains {
                    $0.reason == .loweredValueBits
                }
            )
            #expect(
                !admission.downgradeReasons.contains {
                    $0.reason == .movedBalancedToMaxContext
                }
            )
        }

        @Test func testAdmissionAccountsForK8V4BoundaryLayerMemory() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 8)
            let contextLength = 4096
            let sample = Self.sample(availableGiB: 16, modelGiB: 2)

            let boundary = planner.admit(
                profile: profile,
                requestedContextLength: contextLength,
                userMode: .maxContext,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo3_5,
                precisionPolicy: .legacy(
                    preset: .turbo3_5,
                    valueBits: nil,
                    boundary: .profileDefault
                ),
                memorySample: sample
            )
            let rawBoundary = planner.admit(
                profile: profile,
                requestedContextLength: contextLength,
                userMode: .maxContext,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo3_5,
                precisionPolicy: .legacy(
                    preset: .turbo3_5,
                    valueBits: nil,
                    boundary: .profileDefault,
                    boundaryCachePrecision: .raw
                ),
                memorySample: sample
            )
            let disabled = planner.admit(
                profile: profile,
                requestedContextLength: contextLength,
                userMode: .maxContext,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo3_5,
                precisionPolicy: .legacy(
                    preset: .turbo3_5,
                    valueBits: nil,
                    boundary: .disabled
                ),
                memorySample: sample
            )

            let rawBytesPerTokenPerLayer =
                boundary.memoryPlan.rawBytesPerToken / max(1, profile.layerCount)
            let k8v4BytesPerTokenPerLayer = profile.affineK8V4BytesPerTokenPerLayer()
            #expect(boundary.admitted)
            #expect(rawBoundary.admitted)
            #expect(disabled.admitted)
            #expect(boundary.memoryPlan.boundaryLayerCount == 4)
            #expect(boundary.memoryPlan.boundaryCachePrecision == .affineK8V4)
            #expect(
                boundary.memoryPlan.boundaryKVBytes
                    == k8v4BytesPerTokenPerLayer * 4 * contextLength
            )
            #expect(boundary.memoryPlan.rawBoundaryLayerCount == 0)
            #expect(boundary.memoryPlan.rawBoundaryKVBytes == 0)
            #expect(
                boundary.memoryPlan.runtimeZones.rawShadowBytes
                    == boundary.memoryPlan.rawBoundaryKVBytes
            )
            #expect(
                boundary.memoryPlan.runtimeZones.compressedKVBytes
                    == boundary.memoryPlan.layerFootprint.bytesPerTokenPerLayer
                        * (profile.layerCount - 4)
                        * contextLength
                        + boundary.memoryPlan.boundaryKVBytes
            )
            #expect(rawBoundary.memoryPlan.boundaryCachePrecision == .raw)
            #expect(rawBoundary.memoryPlan.rawBoundaryLayerCount == 4)
            #expect(
                rawBoundary.memoryPlan.rawBoundaryKVBytes
                    == rawBytesPerTokenPerLayer * 4 * contextLength
            )
            #expect(boundary.memoryPlan.boundaryKVBytes < rawBoundary.memoryPlan.rawBoundaryKVBytes)
            #expect(disabled.memoryPlan.rawBoundaryKVBytes == 0)
            #expect(disabled.memoryPlan.boundaryKVBytes == 0)
        }

        @Test func testAffineK8VxAdmissionMemoryUsesLowerValueBitsAndK8V4Boundaries() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 24)
            let contextLength = 4096
            let policy = TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .protectedEdges(first: 2, last: 2),
                boundaryCachePrecision: .affineK8V4
            )

            let admission = planner.admit(
                profile: profile,
                requestedContextLength: contextLength,
                userMode: .maxContext,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo8,
                valueBits: 3,
                precisionPolicy: policy,
                memorySample: Self.sample(availableGiB: 16, modelGiB: 2)
            )

            let middle = profile.affineK8VxKeyValueBytesPerTokenPerLayer(valueBits: 3)
            let boundary = profile.affineK8VxKeyValueBytesPerTokenPerLayer(valueBits: 4)
            let middleLayerCount = profile.layerCount - 4
            #expect(admission.admitted)
            #expect(admission.memoryPlan.valueBits == 3)
            #expect(admission.memoryPlan.precisionPolicy?.key == .affineQ8)
            #expect(admission.memoryPlan.boundaryLayerCount == 4)
            #expect(admission.memoryPlan.boundaryCachePrecision == .affineK8V4)
            #expect(
                admission.memoryPlan.runtimeZones.compressedKVBytes
                    == (middle.total * middleLayerCount + boundary.total * 4) * contextLength
            )
            #expect(
                admission.memoryPlan.compressedKeyBytes
                    == (middle.key * middleLayerCount + boundary.key * 4) * contextLength
            )
            #expect(
                admission.memoryPlan.compressedValueBytes
                    == (middle.value * middleLayerCount + boundary.value * 4) * contextLength
            )
        }

        @Test func testKVLayerPolicyAdmissionMemoryCountsRawAndAffineK8VxLayers() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 8)
            let contextLength = 1024
            let policy = KVLayerPolicy.affineK8VxProtectedEdges(
                layerCount: profile.layerCount,
                valueBits: 2,
                boundaryCachePrecision: .raw,
                first: 1,
                last: 1
            )

            let admission = planner.admit(
                profile: profile,
                requestedContextLength: contextLength,
                userMode: .maxContext,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo8,
                valueBits: 2,
                precisionPolicy: TurboQuantKVPrecisionPolicy(
                    key: .affineQ8,
                    value: .compressed(bits: 2),
                    boundary: .protectedEdges(first: 1, last: 1),
                    boundaryCachePrecision: .raw
                ),
                kvLayerPolicy: policy,
                memorySample: Self.sample(availableGiB: 16, modelGiB: 2)
            )

            let middle = profile.affineK8VxKeyValueBytesPerTokenPerLayer(valueBits: 2)
            let rawPerLayer = admission.memoryPlan.rawBytesPerToken / profile.layerCount
            #expect(admission.admitted)
            #expect(admission.memoryPlan.rawBoundaryLayerCount == 2)
            #expect(admission.memoryPlan.rawBoundaryKVBytes == rawPerLayer * 2 * contextLength)
            #expect(
                admission.memoryPlan.runtimeZones.compressedKVBytes
                    == middle.total * 6 * contextLength
            )
            #expect(
                admission.memoryPlan.runtimeZones.rawShadowBytes
                    == admission.memoryPlan.rawBoundaryKVBytes
            )
        }

        @Test func testRefusesWhenModelCannotFitMinimumPlan() {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 24)

            let admission = planner.admit(
                profile: profile,
                requestedContextLength: 4096,
                userMode: .maxContext,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo3_5,
                memorySample: Self.sample(availableGiB: 1, modelGiB: 2)
            )

            #expect(!admission.admitted)
            #expect(admission.admittedContextLength == 0)
            #expect(
                admission.downgradeReasons.contains {
                    $0.reason == .refusedInsufficientMemory
                }
            )
            #expect(!admission.rejectedPaths.isEmpty)
            #expect(admission.userMessage.contains("cannot safely run"))
        }

        @Test func testLongContextUsesActualLayoutBytesPerToken() {
            let profile = Self.profile(weightGiB: 2, layers: 32)
            let footprint = profile.turboQuantLayerCacheFootprint(
                preset: .turbo3_5,
                valueBits: 4,
                groupSize: 64
            )

            #expect(footprint.keyMagnitudeWordsPerGroup == 5)
            #expect(footprint.valueMagnitudeWordsPerGroup == 8)
            #expect(footprint.bytesPerTokenAllLayers == 49_152)
            #expect(
                profile.turboQuantCompressedKVBytes(
                    contextLength: 65_536,
                    preset: .turbo3_5,
                    valueBits: 4,
                    groupSize: 64
                ) == 49_152 * 65_536
            )
        }

        @Test func testPolarWHTFootprintUsesPackedIndicesAndNorms() {
            let profile = Self.profile(weightGiB: 2, layers: 32)
            let generic = profile.turboQuantLayerCacheFootprint(
                preset: .turbo4v2,
                valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                groupSize: 64
            )
            let polarKeyBytes = profile.polarWHTKeyBytesPerTokenPerLayer(preset: .turbo4v2)
            let polarValueBytes = profile.polarWHTValueBytesPerTokenPerLayer()
            let polarLayerBytes = profile.polarWHTLayerCacheBytesPerTokenPerLayer(
                preset: .turbo4v2,
                valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                groupSize: 64
            )

            #expect(polarKeyBytes == 544)
            #expect(polarValueBytes == 416)
            #expect(
                polarLayerBytes
                    == generic.keyBytesPerTokenPerLayer + polarKeyBytes + polarValueBytes
            )
            #expect(polarKeyBytes + polarValueBytes < generic.bytesPerTokenPerLayer)
        }

        @Test func testPolarWHTAdmissionUsesPackedKeyAndValueFootprint() {
            let profile = Self.profile(weightGiB: 2, layers: 32)
            let valueBits = TurboQuantKVCodec.polarWHTDefaultValueBits
            let expectedKeyBytes = profile.turboQuantLayerCacheFootprint(
                preset: .turbo4v2,
                valueBits: valueBits,
                groupSize: 64
            ).keyBytesPerTokenPerLayer
                + profile.polarWHTKeyBytesPerTokenPerLayer(preset: .turbo4v2)
            let expectedValueBytes = profile.polarWHTValueBytesPerTokenPerLayer(
                valueBits: valueBits
            )
            let precisionPolicy = TurboQuantKVPrecisionPolicy.legacy(
                preset: .turbo4v2,
                valueBits: valueBits,
                boundary: .disabled
            )
            let expectedLayerBytes = profile.polarWHTLayerCacheBytesPerTokenPerLayer(
                preset: .turbo4v2,
                valueBits: valueBits,
                groupSize: 64
            )

            let admission = Self.planner().admit(
                profile: profile,
                requestedContextLength: 8192,
                userMode: .fastest,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo4v2,
                valueBits: valueBits,
                precisionPolicy: precisionPolicy,
                kvCodec: .polarWHT,
                turboQuantBackend: .polarWHTReference,
                memorySample: Self.sample(availableGiB: 3, modelGiB: 2)
            )

            #expect(admission.admitted)
            #expect(!admission.recommendsPlainKVCache)
            #expect(admission.memoryPlan.boundaryLayerCount == 0)
            #expect(admission.memoryPlan.compressedBytesPerToken == expectedLayerBytes * 32)
            #expect(
                admission.memoryPlan.compressedKeyBytes
                    == expectedKeyBytes
                        * 32
                        * admission.admittedContextLength
            )
            #expect(
                admission.memoryPlan.compressedValueBytes
                    == expectedValueBytes
                        * 32
                        * admission.admittedContextLength
            )
        }

        @Test func testPolarWHTLayerPolicyUsesBackendSpecificFootprint() {
            let profile = Self.profile(weightGiB: 2, layers: 32)
            let valueBits = TurboQuantKVCodec.polarWHTDefaultValueBits
            let expectedKeyBytes = profile.turboQuantLayerCacheFootprint(
                preset: .turbo4v2,
                valueBits: valueBits,
                groupSize: 64
            ).keyBytesPerTokenPerLayer
                + profile.polarWHTKeyBytesPerTokenPerLayer(preset: .turbo4v2)
            let expectedValueBytes = profile.polarWHTValueBytesPerTokenPerLayer(
                valueBits: valueBits
            )
            let policy = KVLayerPolicy(
                defaultCodec: .turboQuant(
                    preset: .turbo4v2,
                    valueBits: valueBits,
                    groupSize: 64,
                    backend: .metalPolarWHT
                )
            )
            let precisionPolicy = TurboQuantKVPrecisionPolicy.legacy(
                preset: .turbo4v2,
                valueBits: valueBits,
                boundary: .disabled
            )
            let expectedLayerBytes = profile.polarWHTLayerCacheBytesPerTokenPerLayer(
                preset: .turbo4v2,
                valueBits: valueBits,
                groupSize: 64
            )

            let admission = Self.planner().admit(
                profile: profile,
                requestedContextLength: 8192,
                userMode: .fastest,
                fallbackPolicy: .fatalOnFailure,
                preset: .turbo4v2,
                valueBits: valueBits,
                precisionPolicy: precisionPolicy,
                kvLayerPolicy: policy,
                memorySample: Self.sample(availableGiB: 3, modelGiB: 2)
            )

            #expect(admission.admitted)
            #expect(admission.memoryPlan.compressedBytesPerToken == expectedLayerBytes * 32)
            #expect(
                admission.memoryPlan.compressedKeyBytes
                    == expectedKeyBytes
                        * 32
                        * admission.admittedContextLength
            )
            #expect(
                admission.memoryPlan.compressedValueBytes
                    == expectedValueBytes
                        * 32
                        * admission.admittedContextLength
            )
        }

        @Test func testRequiredAdmissionRejectsMissingPlanBeforeGeneration() {
            let parameters = GenerateParameters(
                kvCacheStrategy: .turboQuant,
                turboQuantAdmissionPolicy: .required
            )

            #expect(throws: TurboQuantGenerationError.admissionRequired) {
                _ = try parameters.resolvedForTurboQuantRuntime(layerCount: 32)
            }
        }

        @Test func testAdmissionPlanResolvesGenerationCacheBudget() throws {
            let planner = Self.planner()
            let admission = planner.admit(
                profile: Self.profile(weightGiB: 2, layers: 32),
                requestedContextLength: 8192,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo3_5,
                memorySample: Self.sample(availableGiB: 12, modelGiB: 2)
            )
            let parameters = GenerateParameters(
                kvCacheStrategy: .turboQuant,
                turboQuantRuntimeMode: .capacityTurboQuant,
                turboQuantAdmission: admission
            )

            let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 32)

            #expect(resolved.maxKVSize == admission.admittedContextLength)
            #expect(resolved.turboQuantPreset == admission.memoryPlan.preset)
            #expect(resolved.turboQuantValueBits == admission.memoryPlan.valueBits)
            #expect((resolved.turboQuantPerCacheResidentBudgetBytes ?? 0) > 0)
        }

        @Test func testAdaptiveTurboQuantUsesRawSDPAWhenShortContextFits() throws {
            let planner = Self.planner()
            let admission = planner.admit(
                profile: Self.profile(weightGiB: 2, layers: 32),
                requestedContextLength: 8192,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo3_5,
                memorySample: Self.sample(availableGiB: 12, modelGiB: 2)
            )
            let parameters = GenerateParameters(
                kvCacheStrategy: .adaptiveTurboQuant,
                turboQuantAdmission: admission,
                turboQuantRawSDPAThreshold: 16_384
            )

            let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 32)

            #expect(resolved.kvCacheStrategy == .mlxAffine)
            #expect(resolved.turboQuantResolvedRuntimeMode == .rawPreferred)
            #expect(resolved.kvBits == nil)
            #expect(resolved.turboQuantPerCacheResidentBudgetBytes == nil)
        }

        @Test func testAdaptiveTurboQuantUsesCompressedRouteWhenShortRawSDPADoesNotFit() throws {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 32)
            let admission = planner.admit(
                profile: profile,
                requestedContextLength: 8192,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo3_5,
                memorySample: TurboQuantRuntimeMemorySample(
                    availableAppMemoryBytes: Self.gib(3) + Self.mib(200),
                    modelResidentBytes: Self.gib(2),
                    tokenizerBytes: 0,
                    promptBytes: 0,
                    uiReserveBytes: 0,
                    thermalState: .nominal
                )
            )
            #expect(admission.admitted)
            #expect(!admission.memoryPlan.rawSDPAFits(contextLength: 8192))
            let parameters = GenerateParameters(
                kvCacheStrategy: .adaptiveTurboQuant,
                turboQuantAdmission: admission,
                turboQuantRawSDPAThreshold: 16_384
            )

            let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 32)

            #expect(resolved.kvCacheStrategy == .turboQuant)
            #expect(resolved.turboQuantResolvedRuntimeMode == .capacityTurboQuant)
            #expect(resolved.turboQuantRuntimeFallbackReason?.contains("decoded active KV") == true)
            #expect(resolved.quantizedKVStart == 0)
            #expect((resolved.turboQuantPerCacheResidentBudgetBytes ?? 0) > 0)
        }

        @Test func testAdaptiveTurboQuantExactPrefillDelaysCompressedCacheConstruction() throws {
            let planner = Self.planner()
            let profile = Self.profile(weightGiB: 2, layers: 32)
            let admission = planner.admit(
                profile: profile,
                requestedContextLength: 8192,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo8,
                memorySample: TurboQuantRuntimeMemorySample(
                    availableAppMemoryBytes: Self.gib(3) + Self.mib(200),
                    modelResidentBytes: Self.gib(2),
                    tokenizerBytes: 0,
                    promptBytes: 0,
                    uiReserveBytes: 0,
                    thermalState: .nominal
                )
            )
            #expect(admission.admitted)
            #expect(!admission.memoryPlan.rawSDPAFits(contextLength: 8192))
            let parameters = GenerateParameters(
                kvCacheStrategy: .adaptiveTurboQuant,
                turboQuantBackend: .metalPolarWHT,
                turboQuantAdmission: admission,
                turboQuantRawSDPAThreshold: 16_384,
                turboQuantExactPrefill: true
            )

            let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 32)

            #expect(resolved.kvCacheStrategy == .adaptiveTurboQuant)
            #expect(resolved.turboQuantResolvedRuntimeMode == .capacityTurboQuant)
            #expect(resolved.quantizedKVStart == 0)
            #expect(resolved.turboQuantExactPrefill)
            #expect((resolved.turboQuantPerCacheResidentBudgetBytes ?? 0) > 0)
        }

        @Test func testAutoTurboQuantUsesCapacityWhenThroughputResidencyDoesNotFit() throws {
            let planner = Self.planner()
            let admission = planner.admit(
                profile: Self.profile(weightGiB: 2, layers: 32),
                requestedContextLength: 65_536,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo3_5,
                memorySample: Self.sample(availableGiB: 12, modelGiB: 2)
            )
            let parameters = GenerateParameters(
                kvCacheStrategy: .adaptiveTurboQuant,
                turboQuantAdmission: admission,
                turboQuantRawSDPAThreshold: 16_384,
                turboQuantPromptTokenCount: 128
            )

            let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 32)

            #expect(resolved.kvCacheStrategy == .turboQuant)
            #expect(resolved.turboQuantResolvedRuntimeMode == .capacityTurboQuant)
            #expect(resolved.quantizedKVStart == 0)
            #expect(resolved.maxKVSize == admission.admittedContextLength)
        }

        @Test func testAdaptiveTurboQuantStartsCompressedWhenPromptAlreadyExceedsRawWindow() throws {
            let planner = Self.planner()
            let admission = planner.admit(
                profile: Self.profile(weightGiB: 2, layers: 32),
                requestedContextLength: 65_536,
                promptTokenCount: 32_768,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo3_5,
                memorySample: Self.sample(availableGiB: 12, modelGiB: 2)
            )
            let parameters = GenerateParameters(
                kvCacheStrategy: .adaptiveTurboQuant,
                turboQuantAdmission: admission,
                turboQuantRawSDPAThreshold: 16_384,
                turboQuantPromptTokenCount: 32_768
            )

            let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 32)

            #expect(resolved.kvCacheStrategy == .turboQuant)
            #expect(resolved.turboQuantResolvedRuntimeMode == .capacityTurboQuant)
            #expect(resolved.quantizedKVStart == 0)
        }

        @Test func testAutoTurboQuantUsesThroughputWhenDecodedBackingFits() throws {
            let planner = Self.planner()
            let admission = planner.admit(
                profile: Self.profile(weightGiB: 2, layers: 24),
                requestedContextLength: 32_768,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo8,
                valueBits: 4,
                memorySample: Self.sample(availableGiB: 16, modelGiB: 2)
            )
            let parameters = GenerateParameters(
                kvCacheStrategy: .adaptiveTurboQuant,
                turboQuantRuntimeMode: .auto,
                turboQuantPrecisionPolicy: .qwenQ4Default,
                turboQuantAdmission: admission,
                turboQuantRawSDPAThreshold: 16_384
            )

            let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 24)

            #expect(resolved.kvCacheStrategy == .affineK8V4)
            #expect(resolved.kvCodec == .affineK8V4)
            #expect(resolved.kvBits == TurboQuantKVCodec.affineK8V4KeyBits)
            #expect(resolved.kvGroupSize == TurboQuantKVCodec.affineK8V4KeyGroupSize)
            #expect(resolved.quantizedKVStart == 16_384)
            #expect(resolved.turboQuantResolvedRuntimeMode == .throughputTurboQuant)
            #expect(resolved.turboQuantResolvedPrecisionPolicy == .qwenQ4Default)
            #expect(resolved.turboQuantValueBits == TurboQuantKVCodec.affineK8V4ValueBits)
            #expect((resolved.turboQuantAdmission?.memoryPlan.decodedActiveKVBytes ?? 0) > 0)
        }

        @Test func testThroughputModeRequiredRejectsWhenDecodedBackingDoesNotFit() {
            let admission = Self.planner().admit(
                profile: Self.profile(weightGiB: 2, layers: 24),
                requestedContextLength: 65_536,
                userMode: .balanced,
                fallbackPolicy: .compressedDecodeAllowed,
                preset: .turbo8,
                valueBits: 4,
                memorySample: Self.sample(availableGiB: 8, modelGiB: 2)
            )
            let parameters = GenerateParameters(
                kvCacheStrategy: .turboQuant,
                turboQuantRuntimeMode: .throughputTurboQuant,
                turboQuantAdmissionPolicy: .required,
                turboQuantAdmission: admission
            )

            #expect(throws: TurboQuantGenerationError.self) {
                _ = try parameters.resolvedForTurboQuantRuntime(layerCount: 24)
            }
        }

        @Test func testCacheFactoryRoutesThroughputAndCapacityCaches() {
            let throughput = makeAttentionKVCache(
                parameters: GenerateParameters(
                    maxKVSize: 128,
                    kvCacheStrategy: .turboQuant,
                    turboQuantRuntimeMode: .auto,
                    turboQuantResolvedRuntimeMode: .throughputTurboQuant,
                    turboQuantPrecisionPolicy: .qwenQ4Default
                )
            )
            let capacity = makeAttentionKVCache(
                parameters: GenerateParameters(
                    maxKVSize: 128,
                    kvCacheStrategy: .turboQuant,
                    turboQuantRuntimeMode: .capacityTurboQuant,
                    turboQuantResolvedRuntimeMode: .capacityTurboQuant,
                    turboQuantPrecisionPolicy: .qwenQ4Default
                )
            )
            let raw = makeAttentionKVCache(
                parameters: GenerateParameters(
                    maxKVSize: 128,
                    kvCacheStrategy: .mlxAffine
                )
            )

            #expect(throughput is ThroughputTurboQuantKVCache)
            #expect(capacity is RotatingTurboQuantKVCache)
            #expect(raw is RotatingKVCache)
        }

        @Test func testEnvironmentCanForceAffineK8VxValueBitsWithProtectedEdges() throws {
            setenv("TURBOQUANT_FORCE_AFFINE_K8VX", "1", 1)
            setenv("TURBOQUANT_AFFINE_VALUE_BITS", "3", 1)
            defer {
                unsetenv("TURBOQUANT_FORCE_AFFINE_K8VX")
                unsetenv("TURBOQUANT_AFFINE_VALUE_BITS")
            }

            let resolved = try GenerateParameters().resolvedForTurboQuantRuntime(layerCount: 8)

            #expect(resolved.kvCacheStrategy == .affineK8Vx)
            #expect(resolved.kvCodec == .affineK8Vx)
            #expect(resolved.turboQuantValueBits == 3)
            #expect(resolved.turboQuantPrecisionPolicy?.key == .affineQ8)
            #expect(resolved.turboQuantPrecisionPolicy?.value == .turbo3_5)
            #expect(resolved.kvLayerPolicy?.defaultCodec == .affineK8Vx(valueBits: 3))
            #expect(resolved.kvLayerPolicy?.codec(forLayerIndex: 0) == .affineK8V4)
            #expect(resolved.kvLayerPolicy?.codec(forLayerIndex: 3) == .affineK8Vx(valueBits: 3))
        }

        private static func planner() -> TurboQuantAdmissionPlanner {
            TurboQuantAdmissionPlanner(
                options: TurboQuantAdmissionPlanner.Options(
                    defaultTokenizerBytes: 0,
                    defaultUIReserveBytes: 0,
                    defaultScratchBytes: 0
                )
            )
        }

        @Test func testPrefersPlainFP16WhenItFits() {
            // 8K FP16 KV (~0.8 GB @24 layers) + 2 GB weights fits 12 GB easily → plain SDPA.
            let admission = Self.planner().admit(
                profile: Self.profile(weightGiB: 2, layers: 24),
                requestedContextLength: 8192,
                userMode: .balanced,
                memorySample: Self.sample(availableGiB: 12, modelGiB: 2)
            )
            #expect(admission.admitted)
            #expect(admission.recommendsPlainKVCache)  // FP16 fits → faster full-precision path
            #expect(admission.admittedContextLength == 8192)
        }

        @Test func testFallsToTurboQuantWhenFP16TooLarge() {
            // 64K FP16 KV (~6.4 GB @24 layers) + 2 GB weights exceeds 8 GB; compressed fits.
            let admission = Self.planner().admit(
                profile: Self.profile(weightGiB: 2, layers: 24),
                requestedContextLength: 65536,
                userMode: .balanced,
                memorySample: Self.sample(availableGiB: 8, modelGiB: 2)
            )
            #expect(admission.admitted)
            #expect(!admission.recommendsPlainKVCache)  // FP16 won't fit → compressed
        }

        @Test func testMaxContextSkipsPlainEvenWhenFP16Fits() {
            // FP16 would fit, but .maxContext deliberately uses compression to reach further.
            let admission = Self.planner().admit(
                profile: Self.profile(weightGiB: 2, layers: 24),
                requestedContextLength: 8192,
                userMode: .maxContext,
                memorySample: Self.sample(availableGiB: 12, modelGiB: 2)
            )
            #expect(admission.admitted)
            #expect(!admission.recommendsPlainKVCache)
        }

        private static func profile(weightGiB: Int, layers: Int) -> ModelMemoryProfile {
            ModelMemoryProfile(
                modelID: "synthetic-worker6-\(weightGiB)gib",
                modelType: "llama",
                layerCount: layers,
                hiddenSize: 4096,
                attentionHeadCount: 32,
                kvHeadCount: 8,
                headDimension: 128,
                intermediateSize: 16_384,
                vocabularySize: 32_000,
                quantizationBits: 4,
                weightBytes: gib(weightGiB)
            )
        }

        private static func sample(availableGiB: Int, modelGiB: Int)
            -> TurboQuantRuntimeMemorySample
        {
            TurboQuantRuntimeMemorySample(
                availableAppMemoryBytes: gib(availableGiB),
                modelResidentBytes: gib(modelGiB),
                tokenizerBytes: 0,
                promptBytes: 0,
                uiReserveBytes: 0,
                thermalState: .nominal
            )
        }

        private static func gib(_ value: Int) -> Int {
            value * 1024 * 1024 * 1024
        }

        private static func mib(_ value: Int) -> Int {
            value * 1024 * 1024
        }
    }
}
