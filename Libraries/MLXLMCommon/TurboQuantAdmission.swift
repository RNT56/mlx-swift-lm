// Copyright (c) 2026 RNT56.

import Foundation
import MLX

#if os(iOS) || os(tvOS) || os(visionOS)
    import Darwin
#endif

private func turboQuantClampedInt(_ value: Double) -> Int {
    guard value.isFinite else { return Int.max }
    if value <= 0 { return 0 }
    if value >= Double(Int.max) { return Int.max }
    return Int(value.rounded())
}

public enum TurboQuantUserMode: String, Codable, Sendable, CaseIterable, Equatable {
    case fastest
    case balanced
    case maxContext
    case batterySaver
}

public enum TurboQuantFallbackPolicy: Codable, Sendable, CaseIterable, Equatable {
    case exactRequired
    case packedAllowed
    case compressedDecodeAllowed
    case fatalOnFailure
}

public enum TurboQuantCacheLifecycle: Codable, Sendable, Equatable {
    case empty
    case rawPrefillChunkOpen
    case compressingChunk(start: Int, count: Int)
    case compressedCommitted(logicalLength: Int, capacity: Int)
    case decodeCompressed
    case degradedPackedFallback(reason: String)
    case degradedDecodedFallback(reason: String)
    case failed(reason: String)
}

public struct RejectedPath: Codable, Equatable, Sendable {
    public var path: String
    public var reason: String

    public init(path: String, reason: String) {
        self.path = path
        self.reason = reason
    }
}

public struct TurboQuantFallbackResult: Codable, Equatable, Sendable {
    public var fromPath: TurboQuantAttentionPath
    public var toPath: TurboQuantAttentionPath?
    public var policy: TurboQuantFallbackPolicy
    public var reason: String
    public var isSemanticallyExact: Bool

    public init(
        fromPath: TurboQuantAttentionPath,
        toPath: TurboQuantAttentionPath?,
        policy: TurboQuantFallbackPolicy,
        reason: String,
        isSemanticallyExact: Bool
    ) {
        self.fromPath = fromPath
        self.toPath = toPath
        self.policy = policy
        self.reason = reason
        self.isSemanticallyExact = isSemanticallyExact
    }
}

public struct TurboQuantAttentionDecision: Codable, Equatable, Sendable {
    public var selectedPath: TurboQuantAttentionPath?
    public var fallbackPolicy: TurboQuantFallbackPolicy
    public var rejectedPaths: [RejectedPath]
    public var reason: String?

    public init(
        selectedPath: TurboQuantAttentionPath?,
        fallbackPolicy: TurboQuantFallbackPolicy,
        rejectedPaths: [RejectedPath] = [],
        reason: String? = nil
    ) {
        self.selectedPath = selectedPath
        self.fallbackPolicy = fallbackPolicy
        self.rejectedPaths = rejectedPaths
        self.reason = reason
    }
}

public struct TurboQuantDiagnosticEvent: Codable, Equatable, Sendable {
    public var category: String
    public var message: String
    public var metadata: [String: String]

    public init(
        category: String,
        message: String,
        metadata: [String: String] = [:]
    ) {
        self.category = category
        self.message = message
        self.metadata = metadata
    }
}

public struct TurboQuantLayerCacheFootprint: Codable, Equatable, Sendable {
    public var layerCount: Int
    public var kvHeadCount: Int
    public var headDimension: Int
    public var groupSize: Int
    public var preset: TurboQuantPreset
    public var valueBits: Int
    public var groupsPerVector: Int
    public var bitsetWordsPerGroup: Int
    public var keyMagnitudeWordsPerGroup: Int
    public var valueMagnitudeWordsPerGroup: Int
    public var keyBytesPerTokenPerLayer: Int
    public var valueBytesPerTokenPerLayer: Int
    public var bytesPerTokenPerLayer: Int
    public var bytesPerTokenAllLayers: Int
    public var actualBitsPerValue: Double

    public init(
        layerCount: Int,
        kvHeadCount: Int,
        headDimension: Int,
        groupSize: Int,
        preset: TurboQuantPreset,
        valueBits: Int
    ) {
        let layerCount = max(0, layerCount)
        let kvHeadCount = max(1, kvHeadCount)
        let headDimension = max(1, headDimension)
        let groupSize = max(1, groupSize)
        let valueBits = min(8, max(2, valueBits))
        let groupsPerVector = (headDimension + groupSize - 1) / groupSize
        let bitsetWordsPerGroup = (groupSize + 31) / 32
        let keyMagnitudeWordsPerGroup = Self.metalMagnitudeWordsPerGroup(
            groupSize: groupSize,
            preset: preset,
            role: .key,
            valueBits: valueBits
        )
        let valueMagnitudeWordsPerGroup = Self.metalMagnitudeWordsPerGroup(
            groupSize: groupSize,
            preset: preset,
            role: .value,
            valueBits: valueBits
        )

        let keyBytesPerHead =
            groupsPerVector
            * (keyMagnitudeWordsPerGroup * MemoryLayout<UInt32>.stride
                + bitsetWordsPerGroup * MemoryLayout<UInt32>.stride * 3
                + 3 * MemoryLayout<Float>.stride)
        let valueBytesPerHead =
            groupsPerVector
            * (valueMagnitudeWordsPerGroup * MemoryLayout<UInt32>.stride
                + 2 * MemoryLayout<Float>.stride)
        let keyBytesPerTokenPerLayer = keyBytesPerHead * kvHeadCount
        let valueBytesPerTokenPerLayer = valueBytesPerHead * kvHeadCount
        let bytesPerTokenPerLayer = keyBytesPerTokenPerLayer + valueBytesPerTokenPerLayer
        let bytesPerTokenAllLayers = bytesPerTokenPerLayer * layerCount
        let representedValues = max(1, kvHeadCount * headDimension * 2)

        self.layerCount = layerCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.groupSize = groupSize
        self.preset = preset
        self.valueBits = valueBits
        self.groupsPerVector = groupsPerVector
        self.bitsetWordsPerGroup = bitsetWordsPerGroup
        self.keyMagnitudeWordsPerGroup = keyMagnitudeWordsPerGroup
        self.valueMagnitudeWordsPerGroup = valueMagnitudeWordsPerGroup
        self.keyBytesPerTokenPerLayer = keyBytesPerTokenPerLayer
        self.valueBytesPerTokenPerLayer = valueBytesPerTokenPerLayer
        self.bytesPerTokenPerLayer = bytesPerTokenPerLayer
        self.bytesPerTokenAllLayers = bytesPerTokenAllLayers
        self.actualBitsPerValue = Double(bytesPerTokenPerLayer * 8) / Double(representedValues)
    }

    private static func metalMagnitudeWordsPerGroup(
        groupSize: Int,
        preset: TurboQuantPreset,
        role: TurboQuantTensorRole,
        valueBits: Int
    ) -> Int {
        if role == .value {
            return (groupSize * valueBits + 31) / 32
        }

        let baseBits = max(1, preset.baseMagnitudeBits - 1)
        let highBits = max(baseBits, preset.highMagnitudeBits - 1)
        let targetBits = max(1, preset.targetMagnitudeBits - 1)
        let highCount: Int
        if highBits > baseBits {
            let fraction = (targetBits - Float(baseBits)) / Float(highBits - baseBits)
            let clampedFraction = max(0, min(1, fraction))
            highCount = Int((Float(groupSize) * clampedFraction).rounded())
        } else {
            highCount = 0
        }
        let bitCount = groupSize * baseBits + highCount * (highBits - baseBits)
        return (bitCount + 31) / 32
    }
}

extension ModelMemoryProfile {
    public func turboQuantLayerCacheFootprint(
        preset: TurboQuantPreset = .turbo3_5,
        valueBits: Int? = nil,
        groupSize: Int = 64
    ) -> TurboQuantLayerCacheFootprint {
        TurboQuantLayerCacheFootprint(
            layerCount: layerCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            groupSize: groupSize,
            preset: preset,
            valueBits: valueBits ?? preset.defaultValueBits
        )
    }

    public func turboQuantCompressedKVBytes(
        contextLength: Int,
        preset: TurboQuantPreset = .turbo3_5,
        valueBits: Int? = nil,
        groupSize: Int = 64
    ) -> Int {
        let footprint = turboQuantLayerCacheFootprint(
            preset: preset,
            valueBits: valueBits,
            groupSize: groupSize
        )
        return turboQuantClampedInt(
            Double(max(0, contextLength)) * Double(footprint.bytesPerTokenAllLayers)
        )
    }
}

public enum TurboQuantThermalState: String, Codable, Sendable, CaseIterable, Equatable {
    case nominal
    case fair
    case serious
    case critical
    case unknown
}

public struct TurboQuantRuntimeMemorySample: Codable, Equatable, Sendable {
    public var availableAppMemoryBytes: Int
    public var mlxActiveBytes: Int
    public var mlxCacheBytes: Int
    public var modelResidentBytes: Int?
    public var tokenizerBytes: Int
    public var promptBytes: Int
    public var uiReserveBytes: Int
    public var thermalState: TurboQuantThermalState
    public var lowPowerModeEnabled: Bool

    public init(
        availableAppMemoryBytes: Int,
        mlxActiveBytes: Int = 0,
        mlxCacheBytes: Int = 0,
        modelResidentBytes: Int? = nil,
        tokenizerBytes: Int = 0,
        promptBytes: Int = 0,
        uiReserveBytes: Int = 0,
        thermalState: TurboQuantThermalState = .unknown,
        lowPowerModeEnabled: Bool = false
    ) {
        self.availableAppMemoryBytes = max(0, availableAppMemoryBytes)
        self.mlxActiveBytes = max(0, mlxActiveBytes)
        self.mlxCacheBytes = max(0, mlxCacheBytes)
        self.modelResidentBytes = modelResidentBytes.map { max(0, $0) }
        self.tokenizerBytes = max(0, tokenizerBytes)
        self.promptBytes = max(0, promptBytes)
        self.uiReserveBytes = max(0, uiReserveBytes)
        self.thermalState = thermalState
        self.lowPowerModeEnabled = lowPowerModeEnabled
    }
}

public struct TurboQuantRuntimeMemoryZones: Codable, Equatable, Sendable {
    public var availableAppMemoryBytes: Int
    public var runtimeBudgetBytes: Int
    public var mlxActiveBytes: Int
    public var mlxCacheBytes: Int
    public var modelResidentBytes: Int
    public var compressedKVBytes: Int
    public var rawShadowBytes: Int
    public var fallbackReserveBytes: Int
    public var scratchBytes: Int
    public var promptAndTokenizerBytes: Int
    public var uiReserveBytes: Int
    public var safetyReserveBytes: Int
    public var rollingSummaryBytes: Int
    public var totalRuntimeBytes: Int
    public var headroomBytes: Int

    public init(
        availableAppMemoryBytes: Int,
        runtimeBudgetBytes: Int,
        mlxActiveBytes: Int,
        mlxCacheBytes: Int,
        modelResidentBytes: Int,
        compressedKVBytes: Int,
        rawShadowBytes: Int,
        fallbackReserveBytes: Int,
        scratchBytes: Int,
        promptAndTokenizerBytes: Int,
        uiReserveBytes: Int,
        safetyReserveBytes: Int,
        rollingSummaryBytes: Int = 0
    ) {
        self.availableAppMemoryBytes = max(0, availableAppMemoryBytes)
        self.runtimeBudgetBytes = max(0, runtimeBudgetBytes)
        self.mlxActiveBytes = max(0, mlxActiveBytes)
        self.mlxCacheBytes = max(0, mlxCacheBytes)
        self.modelResidentBytes = max(0, modelResidentBytes)
        self.compressedKVBytes = max(0, compressedKVBytes)
        self.rawShadowBytes = max(0, rawShadowBytes)
        self.fallbackReserveBytes = max(0, fallbackReserveBytes)
        self.scratchBytes = max(0, scratchBytes)
        self.promptAndTokenizerBytes = max(0, promptAndTokenizerBytes)
        self.uiReserveBytes = max(0, uiReserveBytes)
        self.safetyReserveBytes = max(0, safetyReserveBytes)
        self.rollingSummaryBytes = max(0, rollingSummaryBytes)
        self.totalRuntimeBytes =
            max(0, modelResidentBytes)
            + max(0, mlxCacheBytes)
            + max(0, compressedKVBytes)
            + max(0, rawShadowBytes)
            + max(0, fallbackReserveBytes)
            + max(0, scratchBytes)
            + max(0, promptAndTokenizerBytes)
            + max(0, uiReserveBytes)
            + max(0, safetyReserveBytes)
            + max(0, rollingSummaryBytes)
        self.headroomBytes = max(0, availableAppMemoryBytes) - self.totalRuntimeBytes
    }

    public var fitsAvailableMemory: Bool {
        totalRuntimeBytes <= availableAppMemoryBytes
    }
}

public enum TurboQuantAdmissionDowngradeReason: String, Codable, Sendable, CaseIterable {
    case releasedRawShadow
    case disabledPackedFallback
    case loweredValueBits
    case movedBalancedToMaxContext
    case reducedContext
    case rollingSummaryMemory
    case thermalOrBatterySaver
    case refusedInsufficientMemory
}

public struct TurboQuantAdmissionDowngrade: Codable, Equatable, Sendable {
    public var reason: TurboQuantAdmissionDowngradeReason
    public var message: String

    public init(reason: TurboQuantAdmissionDowngradeReason, message: String) {
        self.reason = reason
        self.message = message
    }
}

public struct TurboQuantMemoryPlan: Codable, Equatable, Sendable {
    public var requestedContextLength: Int
    public var admittedContextLength: Int
    public var requestedMode: TurboQuantUserMode
    public var effectiveMode: TurboQuantUserMode
    public var preset: TurboQuantPreset
    public var valueBits: Int
    public var groupSize: Int
    public var fallbackPolicy: TurboQuantFallbackPolicy
    public var rawBytesPerToken: Int
    public var packedFallbackBytesPerToken: Int
    public var compressedBytesPerToken: Int
    public var layerFootprint: TurboQuantLayerCacheFootprint
    public var usesRawShadow: Bool
    public var packedFallbackEnabled: Bool
    public var usesRollingSummaryMemory: Bool
    public var runtimeZones: TurboQuantRuntimeMemoryZones

    public var fitsRuntimeBudget: Bool {
        runtimeZones.fitsAvailableMemory
    }

    public init(
        requestedContextLength: Int,
        admittedContextLength: Int,
        requestedMode: TurboQuantUserMode,
        effectiveMode: TurboQuantUserMode,
        preset: TurboQuantPreset,
        valueBits: Int,
        groupSize: Int,
        fallbackPolicy: TurboQuantFallbackPolicy,
        rawBytesPerToken: Int,
        packedFallbackBytesPerToken: Int,
        compressedBytesPerToken: Int,
        layerFootprint: TurboQuantLayerCacheFootprint,
        usesRawShadow: Bool,
        packedFallbackEnabled: Bool,
        usesRollingSummaryMemory: Bool,
        runtimeZones: TurboQuantRuntimeMemoryZones
    ) {
        self.requestedContextLength = max(1, requestedContextLength)
        self.admittedContextLength = max(0, admittedContextLength)
        self.requestedMode = requestedMode
        self.effectiveMode = effectiveMode
        self.preset = preset
        self.valueBits = min(8, max(2, valueBits))
        self.groupSize = max(1, groupSize)
        self.fallbackPolicy = fallbackPolicy
        self.rawBytesPerToken = max(0, rawBytesPerToken)
        self.packedFallbackBytesPerToken = max(0, packedFallbackBytesPerToken)
        self.compressedBytesPerToken = max(0, compressedBytesPerToken)
        self.layerFootprint = layerFootprint
        self.usesRawShadow = usesRawShadow
        self.packedFallbackEnabled = packedFallbackEnabled
        self.usesRollingSummaryMemory = usesRollingSummaryMemory
        self.runtimeZones = runtimeZones
    }
}

public struct TurboQuantAdmission: Codable, Equatable, Sendable {
    public var admitted: Bool
    public var requestedContextLength: Int
    public var admittedContextLength: Int
    public var requestedMode: TurboQuantUserMode
    public var selectedMode: TurboQuantUserMode
    public var memoryPlan: TurboQuantMemoryPlan
    public var downgradeReasons: [TurboQuantAdmissionDowngrade]
    public var rejectedPaths: [RejectedPath]
    public var userMessage: String

    public var primaryDowngradeReason: TurboQuantAdmissionDowngradeReason? {
        downgradeReasons.first?.reason
    }

    public init(
        admitted: Bool,
        requestedContextLength: Int,
        admittedContextLength: Int,
        requestedMode: TurboQuantUserMode,
        selectedMode: TurboQuantUserMode,
        memoryPlan: TurboQuantMemoryPlan,
        downgradeReasons: [TurboQuantAdmissionDowngrade] = [],
        rejectedPaths: [RejectedPath] = [],
        userMessage: String
    ) {
        self.admitted = admitted
        self.requestedContextLength = max(1, requestedContextLength)
        self.admittedContextLength = max(0, admittedContextLength)
        self.requestedMode = requestedMode
        self.selectedMode = selectedMode
        self.memoryPlan = memoryPlan
        self.downgradeReasons = downgradeReasons
        self.rejectedPaths = rejectedPaths
        self.userMessage = userMessage
    }
}

public struct TurboQuantAdmissionPlanner: Sendable {
    public struct Options: Codable, Equatable, Sendable {
        public var defaultTokenizerBytes: Int
        public var defaultUIReserveBytes: Int
        public var defaultScratchBytes: Int
        public var exactFallbackDecodeLayerCount: Int
        public var rawShadowPrefillChunkLength: Int
        public var minimumContextLength: Int
        public var rollingSummaryContextLength: Int
        public var minimumValueBits: Int
        public var fastestContextCap: Int
        public var batterySaverContextCap: Int

        public init(
            defaultTokenizerBytes: Int = 64 * 1024 * 1024,
            defaultUIReserveBytes: Int = 256 * 1024 * 1024,
            defaultScratchBytes: Int = 96 * 1024 * 1024,
            exactFallbackDecodeLayerCount: Int = 1,
            rawShadowPrefillChunkLength: Int = 512,
            minimumContextLength: Int = 512,
            rollingSummaryContextLength: Int = 1024,
            minimumValueBits: Int = 2,
            fastestContextCap: Int = 8192,
            batterySaverContextCap: Int = 4096
        ) {
            self.defaultTokenizerBytes = max(0, defaultTokenizerBytes)
            self.defaultUIReserveBytes = max(0, defaultUIReserveBytes)
            self.defaultScratchBytes = max(0, defaultScratchBytes)
            self.exactFallbackDecodeLayerCount = max(1, exactFallbackDecodeLayerCount)
            self.rawShadowPrefillChunkLength = max(0, rawShadowPrefillChunkLength)
            self.minimumContextLength = max(1, minimumContextLength)
            self.rollingSummaryContextLength = max(1, rollingSummaryContextLength)
            self.minimumValueBits = min(8, max(2, minimumValueBits))
            self.fastestContextCap = max(1, fastestContextCap)
            self.batterySaverContextCap = max(1, batterySaverContextCap)
        }
    }

    private struct Candidate {
        var contextLength: Int
        var mode: TurboQuantUserMode
        var preset: TurboQuantPreset
        var valueBits: Int
        var usesRawShadow: Bool
        var packedFallbackEnabled: Bool
        var usesRollingSummaryMemory: Bool
    }

    public static let defaultOptions = Options()

    public var options: Options

    public init(options: Options = TurboQuantAdmissionPlanner.defaultOptions) {
        self.options = options
    }

    public func admit(
        profile: ModelMemoryProfile,
        requestedContextLength: Int,
        promptTokenCount: Int = 0,
        userMode: TurboQuantUserMode = .balanced,
        fallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed,
        preset: TurboQuantPreset = .turbo3_5,
        valueBits: Int? = nil,
        groupSize: Int = 64,
        memorySample: TurboQuantRuntimeMemorySample? = nil
    ) -> TurboQuantAdmission {
        let sample =
            memorySample
            ?? Self.currentRuntimeMemorySample(
                modelResidentBytes: profile.resolvedWeightBytes,
                tokenizerBytes: options.defaultTokenizerBytes,
                promptBytes: Self.promptBytes(promptTokenCount: promptTokenCount),
                uiReserveBytes: options.defaultUIReserveBytes
            )
        let requestedContext = max(1, requestedContextLength)
        let requestedValueBits = min(
            8, max(options.minimumValueBits, valueBits ?? preset.defaultValueBits))
        var downgrades: [TurboQuantAdmissionDowngrade] = []
        var candidate = initialCandidate(
            requestedContextLength: requestedContext,
            userMode: userMode,
            fallbackPolicy: fallbackPolicy,
            preset: preset,
            valueBits: requestedValueBits,
            thermalState: sample.thermalState,
            lowPowerModeEnabled: sample.lowPowerModeEnabled,
            downgrades: &downgrades
        )

        var plan = memoryPlan(
            profile: profile,
            requestedContextLength: requestedContext,
            candidate: candidate,
            requestedMode: userMode,
            fallbackPolicy: fallbackPolicy,
            groupSize: groupSize,
            sample: sample
        )
        if plan.fitsRuntimeBudget {
            return admittedResult(
                requestedContextLength: requestedContext,
                requestedMode: userMode,
                plan: plan,
                downgrades: downgrades
            )
        }

        if candidate.usesRawShadow {
            candidate.usesRawShadow = false
            downgrades.append(
                TurboQuantAdmissionDowngrade(
                    reason: .releasedRawShadow,
                    message: "Released the raw prefill shadow reserve."
                )
            )
            plan = memoryPlan(
                profile: profile,
                requestedContextLength: requestedContext,
                candidate: candidate,
                requestedMode: userMode,
                fallbackPolicy: fallbackPolicy,
                groupSize: groupSize,
                sample: sample
            )
            if plan.fitsRuntimeBudget {
                return admittedResult(
                    requestedContextLength: requestedContext,
                    requestedMode: userMode,
                    plan: plan,
                    downgrades: downgrades
                )
            }
        }

        if candidate.packedFallbackEnabled {
            candidate.packedFallbackEnabled = false
            downgrades.append(
                TurboQuantAdmissionDowngrade(
                    reason: .disabledPackedFallback,
                    message: "Disabled the packed fallback reserve."
                )
            )
            plan = memoryPlan(
                profile: profile,
                requestedContextLength: requestedContext,
                candidate: candidate,
                requestedMode: userMode,
                fallbackPolicy: fallbackPolicy,
                groupSize: groupSize,
                sample: sample
            )
            if plan.fitsRuntimeBudget {
                return admittedResult(
                    requestedContextLength: requestedContext,
                    requestedMode: userMode,
                    plan: plan,
                    downgrades: downgrades
                )
            }
        }

        if candidate.valueBits > options.minimumValueBits {
            let previousBits = candidate.valueBits
            candidate.valueBits = options.minimumValueBits
            downgrades.append(
                TurboQuantAdmissionDowngrade(
                    reason: .loweredValueBits,
                    message:
                        "Lowered TurboQuant value bits from \(previousBits) to \(candidate.valueBits)."
                )
            )
            plan = memoryPlan(
                profile: profile,
                requestedContextLength: requestedContext,
                candidate: candidate,
                requestedMode: userMode,
                fallbackPolicy: fallbackPolicy,
                groupSize: groupSize,
                sample: sample
            )
            if plan.fitsRuntimeBudget {
                return admittedResult(
                    requestedContextLength: requestedContext,
                    requestedMode: userMode,
                    plan: plan,
                    downgrades: downgrades
                )
            }
        }

        if candidate.mode == .balanced {
            candidate.mode = .maxContext
            candidate.preset = .turbo2_5
            candidate.valueBits = options.minimumValueBits
            candidate.usesRawShadow = false
            candidate.packedFallbackEnabled = false
            downgrades.append(
                TurboQuantAdmissionDowngrade(
                    reason: .movedBalancedToMaxContext,
                    message: "Moved Balanced mode to Max Context memory settings."
                )
            )
            plan = memoryPlan(
                profile: profile,
                requestedContextLength: requestedContext,
                candidate: candidate,
                requestedMode: userMode,
                fallbackPolicy: fallbackPolicy,
                groupSize: groupSize,
                sample: sample
            )
            if plan.fitsRuntimeBudget {
                return admittedResult(
                    requestedContextLength: requestedContext,
                    requestedMode: userMode,
                    plan: plan,
                    downgrades: downgrades
                )
            }
        }

        let reducedContext = maxAdmittedContext(
            profile: profile,
            requestedContextLength: candidate.contextLength,
            candidate: candidate,
            requestedMode: userMode,
            fallbackPolicy: fallbackPolicy,
            groupSize: groupSize,
            sample: sample
        )
        if reducedContext < candidate.contextLength, reducedContext >= options.minimumContextLength
        {
            candidate.contextLength = reducedContext
            downgrades.append(
                TurboQuantAdmissionDowngrade(
                    reason: .reducedContext,
                    message: "Reduced admitted context to \(reducedContext) tokens."
                )
            )
            plan = memoryPlan(
                profile: profile,
                requestedContextLength: requestedContext,
                candidate: candidate,
                requestedMode: userMode,
                fallbackPolicy: fallbackPolicy,
                groupSize: groupSize,
                sample: sample
            )
            if plan.fitsRuntimeBudget {
                return admittedResult(
                    requestedContextLength: requestedContext,
                    requestedMode: userMode,
                    plan: plan,
                    downgrades: downgrades
                )
            }
        }

        candidate.contextLength = min(candidate.contextLength, options.rollingSummaryContextLength)
        candidate.usesRollingSummaryMemory = true
        downgrades.append(
            TurboQuantAdmissionDowngrade(
                reason: .rollingSummaryMemory,
                message: "Using rolling summary memory for older turns."
            )
        )
        plan = memoryPlan(
            profile: profile,
            requestedContextLength: requestedContext,
            candidate: candidate,
            requestedMode: userMode,
            fallbackPolicy: fallbackPolicy,
            groupSize: groupSize,
            sample: sample
        )
        if plan.fitsRuntimeBudget {
            return admittedResult(
                requestedContextLength: requestedContext,
                requestedMode: userMode,
                plan: plan,
                downgrades: downgrades
            )
        }

        let refusal = TurboQuantAdmissionDowngrade(
            reason: .refusedInsufficientMemory,
            message:
                "Refused generation because the model, cache, and reserves exceed available memory."
        )
        downgrades.append(refusal)
        return TurboQuantAdmission(
            admitted: false,
            requestedContextLength: requestedContext,
            admittedContextLength: 0,
            requestedMode: userMode,
            selectedMode: plan.effectiveMode,
            memoryPlan: plan,
            downgradeReasons: downgrades,
            rejectedPaths: [
                RejectedPath(
                    path: "turboquant-context-admission",
                    reason: refusal.message
                )
            ],
            userMessage:
                "This model cannot safely run at the requested context on the current memory budget. Reduce context, switch models, or free memory before generation."
        )
    }

    public static func currentRuntimeMemorySample(
        modelResidentBytes: Int? = nil,
        tokenizerBytes: Int = TurboQuantAdmissionPlanner.defaultOptions.defaultTokenizerBytes,
        promptBytes: Int = 0,
        uiReserveBytes: Int = TurboQuantAdmissionPlanner.defaultOptions.defaultUIReserveBytes
    ) -> TurboQuantRuntimeMemorySample {
        let snapshot = Memory.snapshot()
        let sampledAvailable = sampleAvailableAppMemoryBytes(
            mlxActiveBytes: snapshot.activeMemory,
            mlxCacheBytes: snapshot.cacheMemory
        )
        return TurboQuantRuntimeMemorySample(
            availableAppMemoryBytes: sampledAvailable,
            mlxActiveBytes: snapshot.activeMemory,
            mlxCacheBytes: snapshot.cacheMemory,
            modelResidentBytes: modelResidentBytes,
            tokenizerBytes: tokenizerBytes,
            promptBytes: promptBytes,
            uiReserveBytes: uiReserveBytes,
            thermalState: currentThermalState(),
            lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
    }

    public static func runtimeBudgetBytes(availableAppMemoryBytes: Int) -> Int {
        let available = max(0, availableAppMemoryBytes)
        let safetyReserve = max(512 * 1024 * 1024, available / 5)
        return max(0, available - safetyReserve)
    }

    public static func safetyReserveBytes(availableAppMemoryBytes: Int) -> Int {
        let available = max(0, availableAppMemoryBytes)
        return min(available, max(512 * 1024 * 1024, available / 5))
    }

    private func initialCandidate(
        requestedContextLength: Int,
        userMode: TurboQuantUserMode,
        fallbackPolicy: TurboQuantFallbackPolicy,
        preset: TurboQuantPreset,
        valueBits: Int,
        thermalState: TurboQuantThermalState,
        lowPowerModeEnabled: Bool,
        downgrades: inout [TurboQuantAdmissionDowngrade]
    ) -> Candidate {
        var mode = userMode
        if userMode == .balanced,
            lowPowerModeEnabled || thermalState == .serious || thermalState == .critical
        {
            mode = .batterySaver
            downgrades.append(
                TurboQuantAdmissionDowngrade(
                    reason: .thermalOrBatterySaver,
                    message:
                        "Selected Battery Saver settings because power or thermal state is constrained."
                )
            )
        }

        let contextLength: Int
        let selectedPreset: TurboQuantPreset
        let selectedValueBits: Int
        let usesRawShadow: Bool
        let packedFallbackEnabled: Bool

        switch mode {
        case .fastest:
            contextLength = min(requestedContextLength, options.fastestContextCap)
            selectedPreset = preset
            selectedValueBits = max(valueBits, preset.defaultValueBits)
            usesRawShadow = false
            packedFallbackEnabled = false
        case .balanced:
            contextLength = requestedContextLength
            selectedPreset = preset
            selectedValueBits = valueBits
            usesRawShadow = false
            packedFallbackEnabled = fallbackPolicy == .packedAllowed
        case .maxContext:
            contextLength = requestedContextLength
            selectedPreset = .turbo2_5
            selectedValueBits = options.minimumValueBits
            usesRawShadow = false
            packedFallbackEnabled = false
        case .batterySaver:
            contextLength = min(requestedContextLength, options.batterySaverContextCap)
            selectedPreset = preset
            selectedValueBits = min(valueBits, max(options.minimumValueBits, 4))
            usesRawShadow = false
            packedFallbackEnabled = false
        }

        return Candidate(
            contextLength: max(1, contextLength),
            mode: mode,
            preset: selectedPreset,
            valueBits: selectedValueBits,
            usesRawShadow: usesRawShadow,
            packedFallbackEnabled: packedFallbackEnabled,
            usesRollingSummaryMemory: false
        )
    }

    private func memoryPlan(
        profile: ModelMemoryProfile,
        requestedContextLength: Int,
        candidate: Candidate,
        requestedMode: TurboQuantUserMode,
        fallbackPolicy: TurboQuantFallbackPolicy,
        groupSize: Int,
        sample: TurboQuantRuntimeMemorySample
    ) -> TurboQuantMemoryPlan {
        let footprint = profile.turboQuantLayerCacheFootprint(
            preset: candidate.preset,
            valueBits: candidate.valueBits,
            groupSize: groupSize
        )
        let compressedKVBytes = footprint.bytesPerTokenAllLayers * candidate.contextLength
        let rawBytesPerToken = profile.kvCacheBytes(contextLength: 1)
        let rawShadowTokens =
            candidate.usesRawShadow
            ? min(candidate.contextLength, options.rawShadowPrefillChunkLength)
            : 0
        let rawShadowBytes = rawBytesPerToken * rawShadowTokens
        let packedFallbackBytesPerToken = packedFallbackBytesPerToken(
            profile: profile,
            bits: candidate.valueBits,
            groupSize: groupSize
        )
        let fallbackReserveBytes = fallbackBytes(
            profile: profile,
            contextLength: candidate.contextLength,
            fallbackPolicy: fallbackPolicy,
            packedFallbackEnabled: candidate.packedFallbackEnabled,
            packedFallbackBytesPerToken: packedFallbackBytesPerToken,
            rawBytesPerToken: rawBytesPerToken
        )
        let available = sample.availableAppMemoryBytes
        let safetyReserve = Self.safetyReserveBytes(availableAppMemoryBytes: available)
        let runtimeBudget = Self.runtimeBudgetBytes(availableAppMemoryBytes: available)
        let modelResidentBytes = sample.modelResidentBytes ?? profile.resolvedWeightBytes
        let promptAndTokenizerBytes = sample.promptBytes + sample.tokenizerBytes
        let scratchBytes = scratchBytes(
            profile: profile,
            contextLength: candidate.contextLength,
            mode: candidate.mode
        )
        let rollingSummaryBytes =
            candidate.usesRollingSummaryMemory
            ? max(16 * 1024 * 1024, promptAndTokenizerBytes / 4)
            : 0
        let zones = TurboQuantRuntimeMemoryZones(
            availableAppMemoryBytes: available,
            runtimeBudgetBytes: runtimeBudget,
            mlxActiveBytes: sample.mlxActiveBytes,
            mlxCacheBytes: sample.mlxCacheBytes,
            modelResidentBytes: max(modelResidentBytes, sample.mlxActiveBytes),
            compressedKVBytes: compressedKVBytes,
            rawShadowBytes: rawShadowBytes,
            fallbackReserveBytes: fallbackReserveBytes,
            scratchBytes: scratchBytes,
            promptAndTokenizerBytes: promptAndTokenizerBytes,
            uiReserveBytes: sample.uiReserveBytes,
            safetyReserveBytes: safetyReserve,
            rollingSummaryBytes: rollingSummaryBytes
        )

        return TurboQuantMemoryPlan(
            requestedContextLength: requestedContextLength,
            admittedContextLength: candidate.contextLength,
            requestedMode: requestedMode,
            effectiveMode: candidate.mode,
            preset: candidate.preset,
            valueBits: candidate.valueBits,
            groupSize: max(1, groupSize),
            fallbackPolicy: fallbackPolicy,
            rawBytesPerToken: rawBytesPerToken,
            packedFallbackBytesPerToken: packedFallbackBytesPerToken,
            compressedBytesPerToken: footprint.bytesPerTokenAllLayers,
            layerFootprint: footprint,
            usesRawShadow: candidate.usesRawShadow,
            packedFallbackEnabled: candidate.packedFallbackEnabled,
            usesRollingSummaryMemory: candidate.usesRollingSummaryMemory,
            runtimeZones: zones
        )
    }

    private func fallbackBytes(
        profile: ModelMemoryProfile,
        contextLength: Int,
        fallbackPolicy: TurboQuantFallbackPolicy,
        packedFallbackEnabled: Bool,
        packedFallbackBytesPerToken: Int,
        rawBytesPerToken: Int
    ) -> Int {
        switch fallbackPolicy {
        case .exactRequired:
            return rawBytesPerToken * max(0, contextLength)
        case .packedAllowed:
            return packedFallbackEnabled ? packedFallbackBytesPerToken * max(0, contextLength) : 0
        case .compressedDecodeAllowed:
            guard profile.layerCount > 0 else { return rawBytesPerToken * max(0, contextLength) }
            let decodedLayers = min(profile.layerCount, options.exactFallbackDecodeLayerCount)
            return (rawBytesPerToken / max(1, profile.layerCount)) * decodedLayers
                * max(0, contextLength)
        case .fatalOnFailure:
            return 0
        }
    }

    private func packedFallbackBytesPerToken(
        profile: ModelMemoryProfile,
        bits: Int,
        groupSize: Int
    ) -> Int {
        let groupSize = max(1, groupSize)
        let bits = min(8, max(2, bits))
        let groups = (max(1, profile.headDimension) + groupSize - 1) / groupSize
        let packedWords = (groupSize * bits + 31) / 32
        let bytesPerGroup =
            packedWords * MemoryLayout<UInt32>.stride
            + MemoryLayout<Float>.stride
            + MemoryLayout<Float>.stride
        let bytesPerTensorPerLayer = max(1, profile.kvHeadCount) * groups * bytesPerGroup
        return bytesPerTensorPerLayer * 2 * max(0, profile.layerCount)
    }

    private func scratchBytes(
        profile: ModelMemoryProfile,
        contextLength: Int,
        mode: TurboQuantUserMode
    ) -> Int {
        let perLayerRaw =
            profile.layerCount > 0 ? profile.kvCacheBytes(contextLength: 1) / profile.layerCount : 0
        let tokenWindow: Int
        switch mode {
        case .fastest:
            tokenWindow = min(contextLength, 256)
        case .balanced, .maxContext:
            tokenWindow = min(contextLength, 512)
        case .batterySaver:
            tokenWindow = min(contextLength, 128)
        }
        return max(options.defaultScratchBytes, perLayerRaw * max(1, tokenWindow))
    }

    private func maxAdmittedContext(
        profile: ModelMemoryProfile,
        requestedContextLength: Int,
        candidate: Candidate,
        requestedMode: TurboQuantUserMode,
        fallbackPolicy: TurboQuantFallbackPolicy,
        groupSize: Int,
        sample: TurboQuantRuntimeMemorySample
    ) -> Int {
        var low = 0
        var high = max(0, requestedContextLength)
        while low < high {
            let mid = (low + high + 1) / 2
            var candidate = candidate
            candidate.contextLength = mid
            let plan = memoryPlan(
                profile: profile,
                requestedContextLength: requestedContextLength,
                candidate: candidate,
                requestedMode: requestedMode,
                fallbackPolicy: fallbackPolicy,
                groupSize: groupSize,
                sample: sample
            )
            if plan.fitsRuntimeBudget {
                low = mid
            } else {
                high = mid - 1
            }
        }
        return low
    }

    private func admittedResult(
        requestedContextLength: Int,
        requestedMode: TurboQuantUserMode,
        plan: TurboQuantMemoryPlan,
        downgrades: [TurboQuantAdmissionDowngrade]
    ) -> TurboQuantAdmission {
        let contextPart =
            plan.admittedContextLength == requestedContextLength
            ? "\(plan.admittedContextLength) tokens"
            : "\(plan.admittedContextLength) of \(requestedContextLength) requested tokens"
        let downgradePart =
            downgrades.isEmpty
            ? "No memory downgrade was needed."
            : downgrades.map(\.message).joined(separator: " ")
        return TurboQuantAdmission(
            admitted: true,
            requestedContextLength: requestedContextLength,
            admittedContextLength: plan.admittedContextLength,
            requestedMode: requestedMode,
            selectedMode: plan.effectiveMode,
            memoryPlan: plan,
            downgradeReasons: downgrades,
            rejectedPaths: [],
            userMessage:
                "TurboQuant can run \(contextPart) in \(plan.effectiveMode.rawValue) mode. \(downgradePart)"
        )
    }

    private static func promptBytes(promptTokenCount: Int) -> Int {
        max(0, promptTokenCount) * 16
    }

    private static func sampleAvailableAppMemoryBytes(
        mlxActiveBytes: Int,
        mlxCacheBytes: Int
    ) -> Int {
        #if os(iOS) || os(tvOS) || os(visionOS)
            if #available(iOS 13.0, tvOS 13.0, visionOS 1.0, *) {
                let value = UInt64(os_proc_available_memory())
                let available = value > UInt64(Int.max) ? Int.max : Int(value)
                return max(0, available + mlxActiveBytes + mlxCacheBytes)
            }
        #endif
        return ModelFitPlanner.currentSystemMemoryBytes()
    }

    private static func currentThermalState() -> TurboQuantThermalState {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            return .nominal
        case .fair:
            return .fair
        case .serious:
            return .serious
        case .critical:
            return .critical
        @unknown default:
            return .unknown
        }
    }
}
