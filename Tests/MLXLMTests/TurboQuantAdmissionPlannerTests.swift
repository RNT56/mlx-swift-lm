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
            #expect(admission.memoryPlan.runtimeZones.rawShadowBytes == 0)
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
                turboQuantAdmission: admission
            )

            let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 32)

            #expect(resolved.maxKVSize == admission.admittedContextLength)
            #expect(resolved.turboQuantPreset == admission.memoryPlan.preset)
            #expect(resolved.turboQuantValueBits == admission.memoryPlan.valueBits)
            #expect((resolved.turboQuantPerCacheResidentBudgetBytes ?? 0) > 0)
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
    }
}
