import Foundation
import Testing

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {

    @Suite
    struct ModelFitPlannerTests {

        @Test func testClassifiesFullGPUFit() {
            let planner = ModelFitPlanner()
            let profile = Self.denseProfile(weightGiB: 6, layers: 32)

            let plan = planner.plan(
                profile: profile,
                contextLength: 4096,
                systemMemoryBytes: Self.gib(32)
            )

            #expect(plan.strategy == ModelFitStrategy.fullGPU)
            #expect(plan.fitsInMemory)
            #expect(plan.recommendedGPULayerCount == profile.layerCount)
            #expect(plan.recommendedMaxKVSize == 4096)
            #expect(!plan.expertStreamingEligible)
        }

        @Test func testClassifiesDenseOvercommitAsLayerPartitioned() {
            let planner = ModelFitPlanner()
            let profile = Self.denseProfile(weightGiB: 80, layers: 40)

            let plan = planner.plan(
                profile: profile,
                contextLength: 4096,
                systemMemoryBytes: Self.gib(32)
            )

            #expect(plan.strategy == ModelFitStrategy.layerPartitioned)
            #expect(plan.recommendedGPULayerCount > 0)
            #expect(plan.recommendedGPULayerCount < profile.layerCount)
            #expect(plan.layerPartitionPlan.cpuLayerCount > 0)
            #expect(plan.recommendedCacheLimitBytes == 2 * 1024 * 1024)
            #expect(plan.warnings.contains { $0.contains("Layer partitioning") })
        }

        @Test func testClassifiesEligibleMoEAsStreamAssisted() {
            let planner = ModelFitPlanner()
            let profile = ModelMemoryProfile(
                modelID: "synthetic-moe-q4",
                modelType: "qwen3_moe",
                layerCount: 48,
                hiddenSize: 4096,
                attentionHeadCount: 32,
                kvHeadCount: 8,
                headDimension: 128,
                intermediateSize: 14336,
                vocabularySize: 151936,
                quantizationBits: 4,
                isMixtureOfExperts: true,
                expertCount: 16,
                activeExpertCount: 2,
                weightBytes: Self.gib(120)
            )

            let plan = planner.plan(
                profile: profile,
                contextLength: 4096,
                systemMemoryBytes: Self.gib(32)
            )

            #expect(plan.strategy == ModelFitStrategy.streamAssisted)
            #expect(plan.recommendsExpertStreaming)
            #expect(plan.expertStreamingEligible)
            #expect(plan.expertStreamingWorkingSetBytes != nil)
            #expect(plan.recommendedGPULayerCount == profile.layerCount)
            #expect(plan.recommendedCacheLimitBytes == ModelFitPlanner.ssdStreamingCacheBudget(totalMemoryBytes: Self.gib(32)))
            #expect(plan.warnings.contains { $0.contains("MoE expert streaming") })
        }

        @Test func testClassifiesVeryLargeDenseModelAsTooLarge() {
            let planner = ModelFitPlanner()
            let profile = Self.denseProfile(weightGiB: 500, layers: 80)

            let plan = planner.plan(
                profile: profile,
                contextLength: 4096,
                systemMemoryBytes: Self.gib(32)
            )

            #expect(plan.strategy == ModelFitStrategy.tooLarge)
            #expect(!plan.fitsInMemory)
            #expect(plan.recommendedGPULayerCount < profile.layerCount)
            #expect(plan.warnings.contains { $0.contains("smaller quantization") })
        }

        @Test func testRecommendsLowerMaxKVSizeUnderContextPressure() {
            let planner = ModelFitPlanner()
            let profile = Self.denseProfile(
                weightGiB: 8,
                layers: 32,
                hiddenSize: 4096,
                attentionHeads: 32,
                kvHeads: 32,
                headDimension: 128
            )

            let plan = planner.plan(
                profile: profile,
                contextLength: 32768,
                systemMemoryBytes: Self.gib(16)
            )

            #expect(plan.recommendedMaxKVSize < 32768)
            #expect(plan.recommendedMaxKVSize >= 512)
        }

        @Test func testProfilesModelConfigAndWeightFiles() throws {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("ModelFitPlannerTests-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer {
                try? FileManager.default.removeItem(at: directory)
            }

            let config = """
                {
                  "model_type": "qwen3_moe",
                  "num_hidden_layers": 12,
                  "hidden_size": 1024,
                  "num_attention_heads": 16,
                  "num_key_value_heads": 4,
                  "head_dim": 64,
                  "intermediate_size": 4096,
                  "vocab_size": 32000,
                  "num_local_experts": 8,
                  "num_experts_per_tok": 2,
                  "quantization_config": { "bits": 4 }
                }
                """
            try config.data(using: .utf8)?.write(to: directory.appendingPathComponent("config.json"))
            let weights = Data(repeating: 0, count: 4096)
            try weights.write(to: directory.appendingPathComponent("model.safetensors"))

            let profile = try ModelMemoryProfile.profile(modelDirectory: directory, modelID: "test/qwen3-moe-q4")

            #expect(profile.modelType == "qwen3_moe")
            #expect(profile.layerCount == 12)
            #expect(profile.kvHeadCount == 4)
            #expect(profile.quantizationBits == 4)
            #expect(profile.expertStreamingEligible)
            #expect(profile.weightBytes == 4096)
        }

        private static func denseProfile(
            weightGiB: Int,
            layers: Int,
            hiddenSize: Int = 4096,
            attentionHeads: Int = 32,
            kvHeads: Int = 8,
            headDimension: Int = 128
        ) -> ModelMemoryProfile {
            ModelMemoryProfile(
                modelID: "synthetic-dense-\(weightGiB)gib",
                modelType: "llama",
                layerCount: layers,
                hiddenSize: hiddenSize,
                attentionHeadCount: attentionHeads,
                kvHeadCount: kvHeads,
                headDimension: headDimension,
                intermediateSize: hiddenSize * 4,
                vocabularySize: 32000,
                quantizationBits: 4,
                weightBytes: gib(weightGiB)
            )
        }

        private static func gib(_ value: Int) -> Int {
            value * 1024 * 1024 * 1024
        }
    }
}
