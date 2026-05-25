import Foundation

public enum ModelFitStrategy: String, Codable, Sendable, CaseIterable {
    case fullGPU = "full_gpu"
    case layerPartitioned = "layer_partitioned"
    case streamAssisted = "stream_assisted"
    case tooLarge = "too_large"
}

public struct ModelMemoryProfile: Codable, Equatable, Sendable {
    public var modelID: String
    public var modelType: String
    public var layerCount: Int
    public var hiddenSize: Int
    public var attentionHeadCount: Int
    public var kvHeadCount: Int
    public var headDimension: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var quantizationBits: Int
    public var isMixtureOfExperts: Bool
    public var expertCount: Int?
    public var activeExpertCount: Int?
    public var weightBytes: Int?

    public init(
        modelID: String,
        modelType: String = "unknown",
        layerCount: Int,
        hiddenSize: Int,
        attentionHeadCount: Int,
        kvHeadCount: Int? = nil,
        headDimension: Int? = nil,
        intermediateSize: Int? = nil,
        vocabularySize: Int = 32000,
        quantizationBits: Int = 16,
        isMixtureOfExperts: Bool = false,
        expertCount: Int? = nil,
        activeExpertCount: Int? = nil,
        weightBytes: Int? = nil
    ) {
        self.modelID = modelID
        self.modelType = modelType
        self.layerCount = max(0, layerCount)
        self.hiddenSize = max(0, hiddenSize)
        self.attentionHeadCount = max(1, attentionHeadCount)
        self.kvHeadCount = max(1, kvHeadCount ?? attentionHeadCount)
        self.headDimension = max(
            1, headDimension ?? max(1, hiddenSize / max(1, attentionHeadCount)))
        self.intermediateSize = max(0, intermediateSize ?? hiddenSize * 4)
        self.vocabularySize = max(0, vocabularySize)
        self.quantizationBits = max(1, quantizationBits)
        self.isMixtureOfExperts = isMixtureOfExperts
        self.expertCount = expertCount
        self.activeExpertCount = activeExpertCount
        self.weightBytes = weightBytes.map { max(0, $0) }
    }

    public var estimatedParameterCount: Double {
        let hidden = Double(hiddenSize)
        let embedding = Double(vocabularySize) * hidden * 2.0
        let densePerLayer = hidden * hidden * 12.0

        guard isMixtureOfExperts, let expertCount, expertCount > 1 else {
            return densePerLayer * Double(layerCount) + embedding
        }

        let ffnPerLayer = hidden * Double(intermediateSize) * 3.0
        let sharedPerLayer = max(0.0, densePerLayer - ffnPerLayer)
        return (sharedPerLayer + ffnPerLayer * Double(expertCount)) * Double(layerCount) + embedding
    }

    public var estimatedParameterBillions: Double {
        estimatedParameterCount / 1_000_000_000.0
    }

    public var estimatedWeightBytes: Int {
        let bytes = estimatedParameterCount * (Double(quantizationBits) / 8.0)
        return ModelFitPlanner.clampedInt(bytes)
    }

    public var resolvedWeightBytes: Int {
        weightBytes ?? estimatedWeightBytes
    }

    public func kvCacheBytes(contextLength: Int, bytesPerElement: Int = 2) -> Int {
        guard contextLength > 0, layerCount > 0 else { return 0 }
        let bytes =
            Double(2 * max(1, bytesPerElement))
            * Double(layerCount)
            * Double(kvHeadCount)
            * Double(headDimension)
            * Double(contextLength)
        return ModelFitPlanner.clampedInt(bytes)
    }

    public func totalMemoryBytes(
        contextLength: Int,
        draftWeightBytes: Int = 0,
        overheadMultiplier: Double = ModelFitPlanner.defaultOptions.overheadMultiplier
    ) -> Int {
        let bytes =
            Double(resolvedWeightBytes + max(0, draftWeightBytes)) * overheadMultiplier
            + Double(kvCacheBytes(contextLength: contextLength))
        return ModelFitPlanner.clampedInt(bytes)
    }

    public var perLayerWeightBytes: Int {
        guard layerCount > 0 else { return 0 }
        return resolvedWeightBytes / layerCount
    }

    public var expertStreamingEligible: Bool {
        guard isMixtureOfExperts,
            let expertCount,
            let activeExpertCount,
            expertCount > 1,
            activeExpertCount > 0,
            activeExpertCount < expertCount
        else {
            return false
        }
        return true
    }

    public var estimatedStreamingResidentWeightBytes: Int? {
        guard expertStreamingEligible, let expertCount, let activeExpertCount else {
            return nil
        }

        let hidden = Double(hiddenSize)
        let densePerLayer = hidden * hidden * 12.0
        let ffnPerLayer = hidden * Double(intermediateSize) * 3.0
        let sharedPerLayer = max(0.0, densePerLayer - ffnPerLayer)
        let allExpertPerLayer = sharedPerLayer + ffnPerLayer * Double(expertCount)
        let activeExpertPerLayer = sharedPerLayer + ffnPerLayer * Double(activeExpertCount)

        guard allExpertPerLayer > 0 else {
            return resolvedWeightBytes
        }

        let activeRatio = min(1.0, max(0.0, activeExpertPerLayer / allExpertPerLayer))
        let bytes = Double(resolvedWeightBytes) * activeRatio
        return ModelFitPlanner.clampedInt(bytes)
    }

    public static func detectQuantizationBits(modelID: String) -> Int {
        let lower = modelID.lowercased()
        if lower.contains("4bit") || lower.contains("4-bit") || lower.contains("q4")
            || lower.contains("int4") || lower.contains("mxfp4") || lower.contains("nf4")
        { return 4 }
        if lower.contains("8bit") || lower.contains("8-bit") || lower.contains("q8")
            || lower.contains("int8") || lower.contains("mxfp8")
        { return 8 }
        if lower.contains("3bit") || lower.contains("3-bit") || lower.contains("q3") { return 3 }
        if lower.contains("2bit") || lower.contains("2-bit") || lower.contains("q2")
            || lower.contains("int2")
        { return 2 }
        if lower.contains("bf16") || lower.contains("fp16") { return 16 }
        if lower.contains("fp32") || lower.contains("f32") { return 32 }
        return 16
    }

    public static func profile(modelDirectory: URL, modelID: String) throws -> ModelMemoryProfile {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(ModelConfig.self, from: configData)

        let layerCount = config.numHiddenLayers ?? config.textConfig?.numHiddenLayers ?? 32
        let hiddenSize = config.hiddenSize ?? config.textConfig?.hiddenSize ?? 4096
        let attentionHeadCount =
            config.numAttentionHeads ?? config.textConfig?.numAttentionHeads ?? 32
        let kvHeadCount =
            config.numKeyValueHeads ?? config.textConfig?.numKeyValueHeads ?? attentionHeadCount
        let headDimension =
            config.headDim ?? config.textConfig?.headDim
            ?? (hiddenSize / max(1, attentionHeadCount))
        let intermediateSize =
            config.intermediateSize ?? config.textConfig?.intermediateSize ?? (hiddenSize * 4)
        let vocabularySize = config.vocabSize ?? config.textConfig?.vocabSize ?? 32000
        let quantizationBits =
            config.quantizationConfig?.bits ?? detectQuantizationBits(modelID: modelID)
        let expertCount = config.numExperts ?? config.textConfig?.numExperts
        let activeExpertCount = config.numExpertsPerToken ?? config.textConfig?.numExpertsPerToken
        let isMixtureOfExperts = (expertCount ?? 0) > 1
        let weightBytes = measureWeightFiles(directory: modelDirectory)

        return ModelMemoryProfile(
            modelID: modelID,
            modelType: config.modelType ?? config.textConfig?.modelType ?? "unknown",
            layerCount: layerCount,
            hiddenSize: hiddenSize,
            attentionHeadCount: attentionHeadCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            intermediateSize: intermediateSize,
            vocabularySize: vocabularySize,
            quantizationBits: quantizationBits,
            isMixtureOfExperts: isMixtureOfExperts,
            expertCount: expertCount,
            activeExpertCount: activeExpertCount,
            weightBytes: weightBytes == 0 ? nil : weightBytes
        )
    }

    private static func measureWeightFiles(directory: URL) -> Int {
        let fileManager = FileManager.default
        guard
            let enumerator = fileManager.enumerator(
                at: directory,
                includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return 0
        }

        var total = 0
        for case let fileURL as URL in enumerator {
            let name = fileURL.lastPathComponent
            guard
                name.hasSuffix(".safetensors") || name.hasSuffix(".bin") || name.hasSuffix(".gguf")
            else {
                continue
            }
            let resolvedURL = fileURL.resolvingSymlinksInPath()
            if let attributes = try? fileManager.attributesOfItem(atPath: resolvedURL.path),
                let size = attributes[.size] as? NSNumber
            {
                total += size.intValue
            }
        }
        return total
    }
}

public struct LayerPartitionPlan: Codable, Equatable, Sendable {
    public var totalLayerCount: Int
    public var gpuLayerCount: Int
    public var cpuLayerCount: Int
    public var perLayerResidentBytes: Int
    public var gpuResidentBytes: Int
    public var cpuResidentBytes: Int

    public init(
        totalLayerCount: Int,
        gpuLayerCount: Int,
        perLayerResidentBytes: Int
    ) {
        self.totalLayerCount = max(0, totalLayerCount)
        self.gpuLayerCount = min(max(0, gpuLayerCount), max(0, totalLayerCount))
        self.cpuLayerCount = max(0, totalLayerCount - self.gpuLayerCount)
        self.perLayerResidentBytes = max(0, perLayerResidentBytes)
        self.gpuResidentBytes = self.gpuLayerCount * self.perLayerResidentBytes
        self.cpuResidentBytes = self.cpuLayerCount * self.perLayerResidentBytes
    }
}

public struct ModelFitPlan: Codable, Equatable, Sendable {
    public var strategy: ModelFitStrategy
    public var profile: ModelMemoryProfile
    public var contextLength: Int
    public var weightBytes: Int
    public var draftWeightBytes: Int
    public var kvCacheBytes: Int
    public var totalRequiredBytes: Int
    public var systemMemoryBytes: Int
    public var availableMemoryBytes: Int
    public var recommendedWorkingSetBytes: Int
    public var overcommitRatio: Double
    public var recommendedGPULayerCount: Int
    public var recommendedMaxKVSize: Int
    public var recommendedMemoryLimitBytes: Int
    public var recommendedCacheLimitBytes: Int
    public var estimatedTokensPerSecond: Double
    public var expertStreamingEligible: Bool
    public var expertStreamingWorkingSetBytes: Int?
    public var layerPartitionPlan: LayerPartitionPlan
    public var cacheLimitCandidatesBytes: [Int]
    public var warnings: [String]

    public var fitsInMemory: Bool {
        strategy == .fullGPU
    }

    public var recommendsExpertStreaming: Bool {
        strategy == .streamAssisted
    }
}

public struct ModelFitPlanner: Sendable {
    public struct Options: Codable, Equatable, Sendable {
        public var osReservedBytes: Int
        public var usableMemoryFraction: Double
        public var fullGPUComfortFraction: Double
        public var partitionMaximumOvercommit: Double
        public var overheadMultiplier: Double
        public var streamMemoryLimitBytes: Int
        public var minimumRecommendedKVSize: Int

        public init(
            osReservedBytes: Int = 4 * 1024 * 1024 * 1024,
            usableMemoryFraction: Double = 0.85,
            fullGPUComfortFraction: Double = 0.85,
            partitionMaximumOvercommit: Double = 4.0,
            overheadMultiplier: Double = 1.2,
            streamMemoryLimitBytes: Int = 200 * 1024 * 1024 * 1024,
            minimumRecommendedKVSize: Int = 512
        ) {
            self.osReservedBytes = max(0, osReservedBytes)
            self.usableMemoryFraction = usableMemoryFraction
            self.fullGPUComfortFraction = fullGPUComfortFraction
            self.partitionMaximumOvercommit = partitionMaximumOvercommit
            self.overheadMultiplier = overheadMultiplier
            self.streamMemoryLimitBytes = max(0, streamMemoryLimitBytes)
            self.minimumRecommendedKVSize = max(0, minimumRecommendedKVSize)
        }
    }

    public static let defaultOptions = Options()

    public var options: Options

    public init(options: Options = ModelFitPlanner.defaultOptions) {
        self.options = options
    }

    public func plan(
        profile: ModelMemoryProfile,
        contextLength: Int = 4096,
        systemMemoryBytes: Int = ModelFitPlanner.currentSystemMemoryBytes(),
        recommendedWorkingSetBytes: Int? = nil,
        draftWeightBytes: Int = 0
    ) -> ModelFitPlan {
        let normalizedContextLength = max(0, contextLength)
        let normalizedSystemMemoryBytes = max(0, systemMemoryBytes)
        let normalizedDraftWeightBytes = max(0, draftWeightBytes)
        let availableMemoryBytes = max(0, normalizedSystemMemoryBytes - options.osReservedBytes)
        let workingSetBytes =
            recommendedWorkingSetBytes
            ?? max(
                0,
                Int(Double(normalizedSystemMemoryBytes) * options.usableMemoryFraction)
                    - options.osReservedBytes)
        let weightBytes = profile.resolvedWeightBytes
        let kvBytes = profile.kvCacheBytes(contextLength: normalizedContextLength)
        let totalRequiredBytes = totalBytes(
            residentWeightBytes: weightBytes,
            kvCacheBytes: kvBytes,
            draftWeightBytes: normalizedDraftWeightBytes
        )
        let overcommitRatio = ratio(totalRequiredBytes, availableMemoryBytes)
        let expertStreamingWorkingSetBytes = streamingWorkingSetBytes(
            profile: profile,
            kvCacheBytes: kvBytes,
            draftWeightBytes: normalizedDraftWeightBytes
        )

        var warnings: [String] = []
        let strategy = chooseStrategy(
            profile: profile,
            totalRequiredBytes: totalRequiredBytes,
            availableMemoryBytes: availableMemoryBytes,
            expertStreamingWorkingSetBytes: expertStreamingWorkingSetBytes,
            warnings: &warnings
        )

        appendProfileWarnings(
            profile: profile,
            strategy: strategy,
            totalRequiredBytes: totalRequiredBytes,
            availableMemoryBytes: availableMemoryBytes,
            expertStreamingWorkingSetBytes: expertStreamingWorkingSetBytes,
            draftWeightBytes: normalizedDraftWeightBytes,
            warnings: &warnings
        )

        let residentWeightForKV =
            strategy == .streamAssisted
            ? (profile.estimatedStreamingResidentWeightBytes ?? weightBytes)
            : weightBytes
        let recommendedMaxKVSize = recommendMaxKVSize(
            requestedContextLength: normalizedContextLength,
            residentWeightBytes: residentWeightForKV,
            draftWeightBytes: normalizedDraftWeightBytes,
            profile: profile,
            availableMemoryBytes: availableMemoryBytes
        )

        let partitionPlan = layerPartitionPlan(
            profile: profile,
            kvCacheBytes: min(kvBytes, profile.kvCacheBytes(contextLength: recommendedMaxKVSize)),
            availableMemoryBytes: availableMemoryBytes
        )
        let recommendedGPULayerCount =
            strategy == .streamAssisted
            ? profile.layerCount
            : partitionPlan.gpuLayerCount
        let memoryLimitBytes = recommendedMemoryLimitBytes(
            strategy: strategy,
            availableMemoryBytes: availableMemoryBytes,
            totalSystemMemoryBytes: normalizedSystemMemoryBytes,
            mainWeightBytes: weightBytes,
            draftWeightBytes: normalizedDraftWeightBytes
        )
        let cacheLimitBytes = recommendedCacheLimitBytes(
            strategy: strategy,
            workingSetBytes: workingSetBytes,
            totalSystemMemoryBytes: normalizedSystemMemoryBytes,
            draftWeightBytes: normalizedDraftWeightBytes,
            availableMemoryBytes: availableMemoryBytes
        )

        return ModelFitPlan(
            strategy: strategy,
            profile: profile,
            contextLength: normalizedContextLength,
            weightBytes: weightBytes,
            draftWeightBytes: normalizedDraftWeightBytes,
            kvCacheBytes: kvBytes,
            totalRequiredBytes: totalRequiredBytes,
            systemMemoryBytes: normalizedSystemMemoryBytes,
            availableMemoryBytes: availableMemoryBytes,
            recommendedWorkingSetBytes: workingSetBytes,
            overcommitRatio: overcommitRatio,
            recommendedGPULayerCount: recommendedGPULayerCount,
            recommendedMaxKVSize: recommendedMaxKVSize,
            recommendedMemoryLimitBytes: memoryLimitBytes,
            recommendedCacheLimitBytes: cacheLimitBytes,
            estimatedTokensPerSecond: estimateTokensPerSecond(
                strategy: strategy,
                weightBytes: weightBytes,
                overcommitRatio: overcommitRatio,
                gpuLayerCount: recommendedGPULayerCount,
                totalLayerCount: profile.layerCount
            ),
            expertStreamingEligible: profile.expertStreamingEligible,
            expertStreamingWorkingSetBytes: expertStreamingWorkingSetBytes,
            layerPartitionPlan: partitionPlan,
            cacheLimitCandidatesBytes: calibrationCacheLimitCandidates(
                modelWeightBytes: weightBytes,
                systemMemoryBytes: normalizedSystemMemoryBytes
            ),
            warnings: warnings
        )
    }

    public static func currentSystemMemoryBytes() -> Int {
        clampedInt(Double(ProcessInfo.processInfo.physicalMemory))
    }

    public static func ssdStreamingCacheBudget(
        totalMemoryBytes: Int,
        draftWeightBytes: Int = 0,
        osReservedBytes: Int = defaultOptions.osReservedBytes,
        usableMemoryFraction: Double = defaultOptions.usableMemoryFraction
    ) -> Int {
        let raw =
            Int(Double(max(0, totalMemoryBytes)) * usableMemoryFraction)
            - max(0, osReservedBytes)
            - max(0, draftWeightBytes)
        return max(raw, 2 * 1024 * 1024 * 1024)
    }

    public func calibrationCacheLimitCandidates(
        modelWeightBytes: Int,
        systemMemoryBytes: Int
    ) -> [Int] {
        let modelWeightBytes = max(0, modelWeightBytes)
        let freeBytes = max(0, systemMemoryBytes - modelWeightBytes)
        return [
            modelWeightBytes + modelWeightBytes / 5,
            modelWeightBytes + freeBytes / 4,
            modelWeightBytes + freeBytes / 2,
            0,
        ]
    }

    fileprivate static func clampedInt(_ value: Double) -> Int {
        guard value.isFinite else { return Int.max }
        if value <= 0 { return 0 }
        if value >= Double(Int.max) { return Int.max }
        return Int(value.rounded())
    }

    private func chooseStrategy(
        profile: ModelMemoryProfile,
        totalRequiredBytes: Int,
        availableMemoryBytes: Int,
        expertStreamingWorkingSetBytes: Int?,
        warnings: inout [String]
    ) -> ModelFitStrategy {
        let available = Double(max(1, availableMemoryBytes))
        let required = Double(totalRequiredBytes)

        if required <= available * options.fullGPUComfortFraction {
            return .fullGPU
        }
        if required <= available {
            warnings.append(
                "Model fits available memory but leaves limited headroom for concurrent work.")
            return .fullGPU
        }

        if let expertStreamingWorkingSetBytes,
            Double(expertStreamingWorkingSetBytes) <= available
        {
            return .streamAssisted
        }

        if required <= available * options.partitionMaximumOvercommit {
            return .layerPartitioned
        }

        if profile.expertStreamingEligible, expertStreamingWorkingSetBytes != nil {
            warnings.append(
                "Expert streaming is available, but the estimated active working set still exceeds memory."
            )
        }
        return .tooLarge
    }

    private func appendProfileWarnings(
        profile: ModelMemoryProfile,
        strategy: ModelFitStrategy,
        totalRequiredBytes: Int,
        availableMemoryBytes: Int,
        expertStreamingWorkingSetBytes: Int?,
        draftWeightBytes: Int,
        warnings: inout [String]
    ) {
        let overcommit = ratio(totalRequiredBytes, availableMemoryBytes)
        if overcommit > 1.0, strategy != .streamAssisted {
            warnings.append(
                "Estimated memory requirement is \(formatted(overcommit))x available memory."
            )
        }

        if strategy == .layerPartitioned {
            warnings.append(
                "Layer partitioning is recommended; throughput depends on model support for GPU/CPU layer placement."
            )
        }

        if strategy == .streamAssisted,
            let expertStreamingWorkingSetBytes
        {
            warnings.append(
                "MoE expert streaming can keep the active working set near \(formatGB(expertStreamingWorkingSetBytes)) instead of loading all expert weights."
            )
            if draftWeightBytes > 0 {
                warnings.append(
                    "Draft models increase page-cache pressure during expert streaming; use a small draft model or one draft token per round."
                )
            }
        }

        if profile.expertStreamingEligible,
            let expertCount = profile.expertCount,
            let activeExpertCount = profile.activeExpertCount
        {
            warnings.append(
                "MoE model activates \(activeExpertCount) of \(expertCount) experts per token.")
        }

        if strategy == .tooLarge {
            warnings.append(
                "Use a smaller quantization, lower context length, expert streaming for supported MoE models, or a machine with more memory."
            )
        }
    }

    private func streamingWorkingSetBytes(
        profile: ModelMemoryProfile,
        kvCacheBytes: Int,
        draftWeightBytes: Int
    ) -> Int? {
        guard let streamingResidentWeightBytes = profile.estimatedStreamingResidentWeightBytes
        else {
            return nil
        }
        return totalBytes(
            residentWeightBytes: streamingResidentWeightBytes,
            kvCacheBytes: kvCacheBytes,
            draftWeightBytes: draftWeightBytes
        )
    }

    private func totalBytes(
        residentWeightBytes: Int,
        kvCacheBytes: Int,
        draftWeightBytes: Int
    ) -> Int {
        let bytes =
            Double(max(0, residentWeightBytes) + max(0, draftWeightBytes))
            * options.overheadMultiplier
            + Double(max(0, kvCacheBytes))
        return Self.clampedInt(bytes)
    }

    private func layerPartitionPlan(
        profile: ModelMemoryProfile,
        kvCacheBytes: Int,
        availableMemoryBytes: Int
    ) -> LayerPartitionPlan {
        guard profile.layerCount > 0 else {
            return LayerPartitionPlan(
                totalLayerCount: 0, gpuLayerCount: 0, perLayerResidentBytes: 0)
        }

        let perLayerWeight = Double(profile.resolvedWeightBytes) / Double(profile.layerCount)
        let perLayerKV = Double(kvCacheBytes) / Double(profile.layerCount)
        let perLayerResident = Self.clampedInt(
            (perLayerWeight + perLayerKV) * options.overheadMultiplier)
        let gpuLayers =
            perLayerResident > 0
            ? Int(Double(max(0, availableMemoryBytes)) / Double(perLayerResident))
            : profile.layerCount

        return LayerPartitionPlan(
            totalLayerCount: profile.layerCount,
            gpuLayerCount: min(profile.layerCount, max(0, gpuLayers)),
            perLayerResidentBytes: perLayerResident
        )
    }

    private func recommendMaxKVSize(
        requestedContextLength: Int,
        residentWeightBytes: Int,
        draftWeightBytes: Int,
        profile: ModelMemoryProfile,
        availableMemoryBytes: Int
    ) -> Int {
        guard requestedContextLength > 0 else { return 0 }
        let bytesPerToken = profile.kvCacheBytes(contextLength: 1)
        guard bytesPerToken > 0 else { return requestedContextLength }

        let residentBytes =
            Double(max(0, residentWeightBytes) + max(0, draftWeightBytes))
            * options.overheadMultiplier
        let kvBudget = max(0.0, Double(max(0, availableMemoryBytes)) - residentBytes)
        let maxTokens = Int(kvBudget / Double(bytesPerToken))
        let bounded = min(requestedContextLength, maxTokens)
        return max(min(requestedContextLength, options.minimumRecommendedKVSize), bounded)
    }

    private func recommendedMemoryLimitBytes(
        strategy: ModelFitStrategy,
        availableMemoryBytes: Int,
        totalSystemMemoryBytes: Int,
        mainWeightBytes: Int,
        draftWeightBytes: Int
    ) -> Int {
        switch strategy {
        case .fullGPU:
            return Self.clampedInt(Double(max(0, availableMemoryBytes)) * 1.5)
        case .streamAssisted:
            let combinedWeightBytes = max(0, mainWeightBytes) + max(0, draftWeightBytes)
            if draftWeightBytes > 0,
                combinedWeightBytes > Int(Double(max(0, totalSystemMemoryBytes)) * 0.70)
            {
                return Self.clampedInt(Double(max(0, totalSystemMemoryBytes)) * 1.1)
            }
            return options.streamMemoryLimitBytes
        case .layerPartitioned, .tooLarge:
            return Self.clampedInt(
                Double(max(0, availableMemoryBytes)) * options.fullGPUComfortFraction)
        }
    }

    private func recommendedCacheLimitBytes(
        strategy: ModelFitStrategy,
        workingSetBytes: Int,
        totalSystemMemoryBytes: Int,
        draftWeightBytes: Int,
        availableMemoryBytes: Int
    ) -> Int {
        switch strategy {
        case .fullGPU:
            return max(0, workingSetBytes)
        case .streamAssisted:
            return Self.ssdStreamingCacheBudget(
                totalMemoryBytes: totalSystemMemoryBytes,
                draftWeightBytes: draftWeightBytes,
                osReservedBytes: options.osReservedBytes,
                usableMemoryFraction: options.usableMemoryFraction
            )
        case .layerPartitioned:
            return 2 * 1024 * 1024
        case .tooLarge:
            return Self.clampedInt(
                Double(max(0, availableMemoryBytes)) * options.fullGPUComfortFraction)
        }
    }

    private func estimateTokensPerSecond(
        strategy: ModelFitStrategy,
        weightBytes: Int,
        overcommitRatio: Double,
        gpuLayerCount: Int,
        totalLayerCount: Int
    ) -> Double {
        let weightGB = max(0.1, Double(weightBytes) / 1_000_000_000.0)
        let baseSpeed = max(5.0, 100.0 / max(1.0, weightGB / 2.0))

        switch strategy {
        case .fullGPU:
            return baseSpeed
        case .streamAssisted:
            return baseSpeed / 2.5
        case .layerPartitioned:
            guard totalLayerCount > 0 else { return baseSpeed / 6.0 }
            let gpuFraction = Double(max(0, gpuLayerCount)) / Double(totalLayerCount)
            let cpuFraction = max(0.0, 1.0 - gpuFraction)
            return baseSpeed * gpuFraction + (baseSpeed / 6.0) * cpuFraction
        case .tooLarge:
            return min(1.0, baseSpeed / max(4.0, overcommitRatio))
        }
    }

    private func ratio(_ numerator: Int, _ denominator: Int) -> Double {
        guard denominator > 0 else { return Double.infinity }
        return Double(numerator) / Double(denominator)
    }

    private func formatted(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    private func formatGB(_ bytes: Int) -> String {
        String(format: "%.1f GB", Double(bytes) / 1_000_000_000.0)
    }
}

private struct ModelConfig: Decodable {
    var modelType: String?
    var numHiddenLayers: Int?
    var hiddenSize: Int?
    var numAttentionHeads: Int?
    var numKeyValueHeads: Int?
    var headDim: Int?
    var intermediateSize: Int?
    var vocabSize: Int?
    var numExperts: Int?
    var numExpertsPerToken: Int?
    var quantizationConfig: QuantizationConfig?
    var textConfig: TextConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numHiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case numExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case quantizationConfig = "quantization_config"
        case textConfig = "text_config"
    }
}

private struct TextConfig: Decodable {
    var modelType: String?
    var numHiddenLayers: Int?
    var hiddenSize: Int?
    var numAttentionHeads: Int?
    var numKeyValueHeads: Int?
    var headDim: Int?
    var intermediateSize: Int?
    var vocabSize: Int?
    var numExperts: Int?
    var numExpertsPerToken: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numHiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case numExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
    }
}

private struct QuantizationConfig: Decodable {
    var bits: Int?
}
