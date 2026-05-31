// Copyright © 2026 RNT56.

import Foundation
import MLX

private let turboQuantHybridValueSeedSalt: UInt64 = 0xD1B5_4A32_D192_ED03

public enum TurboQuantColdAttentionMode: String, Codable, Sendable, CaseIterable {
    case off
    case selected
    case exhaustive
}

public enum TurboQuantLayerPolicy: Codable, Sendable, Equatable {
    case auto
    case all
    case finalQuarter
    case custom([Int: Int])
}

public enum TurboQuantColdSelectorEscalation: String, Codable, Sendable, Hashable {
    case none
    case maxBudget
    case exhaustive
}

public struct TurboQuantColdSelectorPolicy: Hashable, Codable, Sendable {
    public var nearestBlockCount: Int
    public var semanticWeight: Float
    public var lexicalWeight: Float
    public var keyNormWeight: Float
    public var recencyWeight: Float
    public var minimumConfidence: Float
    public var allowMaxBudgetEscalation: Bool
    public var allowExhaustiveEscalation: Bool

    public init(
        nearestBlockCount: Int = 2,
        semanticWeight: Float = 100,
        lexicalWeight: Float = 10,
        keyNormWeight: Float = 1,
        recencyWeight: Float = 0.25,
        minimumConfidence: Float = 0.35,
        allowMaxBudgetEscalation: Bool = true,
        allowExhaustiveEscalation: Bool = false
    ) {
        self.nearestBlockCount = max(0, nearestBlockCount)
        self.semanticWeight = max(0, semanticWeight)
        self.lexicalWeight = max(0, lexicalWeight)
        self.keyNormWeight = max(0, keyNormWeight)
        self.recencyWeight = max(0, recencyWeight)
        self.minimumConfidence = max(0, min(1, minimumConfidence))
        self.allowMaxBudgetEscalation = allowMaxBudgetEscalation
        self.allowExhaustiveEscalation = allowExhaustiveEscalation
    }

    public static let automatic = TurboQuantColdSelectorPolicy()
}

public struct TurboQuantColdSelectorHint: Hashable, Codable, Sendable {
    public var startToken: Int
    public var endToken: Int
    public var lexicalScore: Float
    public var semanticScore: Float
    public var anchorFlags: TurboQuantColdBlockAnchorFlags
    public var sourceID: String?

    public init(
        startToken: Int,
        endToken: Int,
        lexicalScore: Float = 0,
        semanticScore: Float = 0,
        anchorFlags: TurboQuantColdBlockAnchorFlags = [],
        sourceID: String? = nil
    ) {
        self.startToken = max(0, min(startToken, endToken))
        self.endToken = max(0, max(startToken, endToken))
        self.lexicalScore = max(0, lexicalScore)
        self.semanticScore = max(0, semanticScore)
        self.anchorFlags = anchorFlags
        self.sourceID = sourceID
    }

    public var isEmpty: Bool {
        startToken >= endToken
            || (lexicalScore == 0 && semanticScore == 0 && anchorFlags.isEmpty)
    }
}

public struct TurboQuantColdBlockAnchorFlags: OptionSet, Hashable, Codable, Sendable {
    public let rawValue: Int

    public static let system = Self(rawValue: 1 << 0)
    public static let tool = Self(rawValue: 1 << 1)
    public static let pinnedFile = Self(rawValue: 1 << 2)
    public static let userAnchor = Self(rawValue: 1 << 3)

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
}

public struct TurboQuantColdBlockDescriptor: Hashable, Codable, Sendable {
    public var blockID: Int
    public var startToken: Int
    public var endToken: Int
    public var compressedSlotStart: Int
    public var compressedSlotEnd: Int
    public var logicalTokenCount: Int
    public var recencyRank: Int
    public var anchorFlags: TurboQuantColdBlockAnchorFlags
    public var keyMeanSummary: [Float]
    public var maxKeyNormEstimate: Float
    public var scoreUpperBoundEstimate: Float
    public var lexicalScore: Float
    public var semanticScore: Float
    public var relevanceScore: Float

    private enum CodingKeys: String, CodingKey {
        case blockID
        case startToken
        case endToken
        case compressedSlotStart
        case compressedSlotEnd
        case logicalTokenCount
        case recencyRank
        case anchorFlags
        case keyMeanSummary
        case maxKeyNormEstimate
        case scoreUpperBoundEstimate
        case lexicalScore
        case semanticScore
        case relevanceScore
    }

    public init(
        blockID: Int,
        startToken: Int,
        endToken: Int,
        compressedSlotStart: Int,
        compressedSlotEnd: Int,
        logicalTokenCount: Int,
        recencyRank: Int,
        anchorFlags: TurboQuantColdBlockAnchorFlags,
        keyMeanSummary: [Float] = [],
        maxKeyNormEstimate: Float,
        scoreUpperBoundEstimate: Float? = nil,
        lexicalScore: Float = 0,
        semanticScore: Float = 0,
        relevanceScore: Float
    ) {
        self.blockID = blockID
        self.startToken = startToken
        self.endToken = endToken
        self.compressedSlotStart = compressedSlotStart
        self.compressedSlotEnd = compressedSlotEnd
        self.logicalTokenCount = max(0, logicalTokenCount)
        self.recencyRank = max(0, recencyRank)
        self.anchorFlags = anchorFlags
        self.keyMeanSummary = keyMeanSummary
        self.maxKeyNormEstimate = max(0, maxKeyNormEstimate)
        self.scoreUpperBoundEstimate = max(0, scoreUpperBoundEstimate ?? maxKeyNormEstimate)
        self.lexicalScore = max(0, lexicalScore)
        self.semanticScore = max(0, semanticScore)
        self.relevanceScore = relevanceScore
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let maxKeyNormEstimate =
            try container.decodeIfPresent(Float.self, forKey: .maxKeyNormEstimate) ?? 0
        self.init(
            blockID: try container.decode(Int.self, forKey: .blockID),
            startToken: try container.decode(Int.self, forKey: .startToken),
            endToken: try container.decode(Int.self, forKey: .endToken),
            compressedSlotStart: try container.decode(Int.self, forKey: .compressedSlotStart),
            compressedSlotEnd: try container.decode(Int.self, forKey: .compressedSlotEnd),
            logicalTokenCount: try container.decode(Int.self, forKey: .logicalTokenCount),
            recencyRank: try container.decodeIfPresent(Int.self, forKey: .recencyRank) ?? 0,
            anchorFlags: try container.decodeIfPresent(
                TurboQuantColdBlockAnchorFlags.self,
                forKey: .anchorFlags
            ) ?? [],
            keyMeanSummary: try container.decodeIfPresent(
                [Float].self,
                forKey: .keyMeanSummary
            ) ?? [],
            maxKeyNormEstimate: maxKeyNormEstimate,
            scoreUpperBoundEstimate: try container.decodeIfPresent(
                Float.self,
                forKey: .scoreUpperBoundEstimate
            ) ?? maxKeyNormEstimate,
            lexicalScore: try container.decodeIfPresent(Float.self, forKey: .lexicalScore) ?? 0,
            semanticScore: try container.decodeIfPresent(Float.self, forKey: .semanticScore) ?? 0,
            relevanceScore: try container.decodeIfPresent(Float.self, forKey: .relevanceScore)
                ?? 0
        )
    }

    public var isAnchor: Bool {
        !anchorFlags.isEmpty
    }
}

public struct TurboQuantColdSelection: Hashable, Codable, Sendable {
    public var selectedBlockIDs: [Int]
    public var selectedTokenBudget: Int
    public var selectedTokenCount: Int
    public var anchorTokenCount: Int
    public var budgetedTokenCount: Int
    public var anchorOverflowTokens: Int
    public var confidence: Float
    public var initialConfidence: Float
    public var finalConfidence: Float
    public var selectorEscalation: TurboQuantColdSelectorEscalation
    public var reasonFlags: [String]
    public var requiresExhaustiveFallback: Bool

    public init(
        selectedBlockIDs: [Int],
        selectedTokenBudget: Int,
        selectedTokenCount: Int,
        anchorTokenCount: Int = 0,
        budgetedTokenCount: Int? = nil,
        anchorOverflowTokens: Int = 0,
        confidence: Float,
        initialConfidence: Float? = nil,
        finalConfidence: Float? = nil,
        selectorEscalation: TurboQuantColdSelectorEscalation = .none,
        reasonFlags: [String],
        requiresExhaustiveFallback: Bool
    ) {
        self.selectedBlockIDs = selectedBlockIDs
        self.selectedTokenBudget = max(0, selectedTokenBudget)
        self.selectedTokenCount = max(0, selectedTokenCount)
        self.anchorTokenCount = max(0, anchorTokenCount)
        self.budgetedTokenCount = max(
            0,
            budgetedTokenCount ?? max(0, selectedTokenCount - anchorTokenCount)
        )
        self.anchorOverflowTokens = max(0, anchorOverflowTokens)
        self.confidence = max(0, min(1, confidence))
        self.initialConfidence = max(0, min(1, initialConfidence ?? confidence))
        self.finalConfidence = max(0, min(1, finalConfidence ?? confidence))
        self.selectorEscalation = selectorEscalation
        self.reasonFlags = reasonFlags
        self.requiresExhaustiveFallback = requiresExhaustiveFallback
    }

    public static let empty = TurboQuantColdSelection(
        selectedBlockIDs: [],
        selectedTokenBudget: 0,
        selectedTokenCount: 0,
        anchorTokenCount: 0,
        budgetedTokenCount: 0,
        anchorOverflowTokens: 0,
        confidence: 1,
        initialConfidence: 1,
        finalConfidence: 1,
        selectorEscalation: .none,
        reasonFlags: ["empty"],
        requiresExhaustiveFallback: false
    )

    public var isEmpty: Bool {
        selectedBlockIDs.isEmpty
    }
}

public struct TurboQuantSegmentStats {
    public var m: MLXArray
    public var l: MLXArray
    public var weightedValueSum: MLXArray
    public var outputDType: DType
    public var logicalTokenCount: Int
}

public struct TurboQuantHybridDiagnostics: Hashable, Codable, Sendable {
    public var route: String
    public var hotTokens: Int
    public var coldBlockCount: Int
    public var selectedColdTokens: Int
    public var selectedBudgetedColdTokens: Int
    public var anchorColdTokens: Int
    public var anchorOverflowTokens: Int
    public var selectedColdBlocks: [Int]
    public var selectorConfidence: Float
    public var selectorInitialConfidence: Float
    public var selectorFinalConfidence: Float
    public var selectorEscalation: TurboQuantColdSelectorEscalation
    public var selectorReasonFlags: [String]
    public var coldBudgetTokens: Int
    public var maxColdBudgetTokens: Int
    public var fallbackReason: String?
    public var fullScanFallbackCount: Int
    public var lastLayerColdBudget: Int?

    public init(
        route: String,
        hotTokens: Int,
        coldBlockCount: Int,
        selectedColdTokens: Int,
        selectedBudgetedColdTokens: Int = 0,
        anchorColdTokens: Int = 0,
        anchorOverflowTokens: Int = 0,
        selectedColdBlocks: [Int],
        selectorConfidence: Float,
        selectorInitialConfidence: Float? = nil,
        selectorFinalConfidence: Float? = nil,
        selectorEscalation: TurboQuantColdSelectorEscalation = .none,
        selectorReasonFlags: [String] = [],
        coldBudgetTokens: Int,
        maxColdBudgetTokens: Int,
        fallbackReason: String?,
        fullScanFallbackCount: Int,
        lastLayerColdBudget: Int?
    ) {
        self.route = route
        self.hotTokens = max(0, hotTokens)
        self.coldBlockCount = max(0, coldBlockCount)
        self.selectedColdTokens = max(0, selectedColdTokens)
        self.selectedBudgetedColdTokens = max(0, selectedBudgetedColdTokens)
        self.anchorColdTokens = max(0, anchorColdTokens)
        self.anchorOverflowTokens = max(0, anchorOverflowTokens)
        self.selectedColdBlocks = selectedColdBlocks
        self.selectorConfidence = max(0, min(1, selectorConfidence))
        self.selectorInitialConfidence = max(
            0,
            min(1, selectorInitialConfidence ?? selectorConfidence)
        )
        self.selectorFinalConfidence = max(
            0,
            min(1, selectorFinalConfidence ?? selectorConfidence)
        )
        self.selectorEscalation = selectorEscalation
        self.selectorReasonFlags = selectorReasonFlags
        self.coldBudgetTokens = max(0, coldBudgetTokens)
        self.maxColdBudgetTokens = max(0, maxColdBudgetTokens)
        self.fallbackReason = fallbackReason
        self.fullScanFallbackCount = max(0, fullScanFallbackCount)
        self.lastLayerColdBudget = lastLayerColdBudget.map { max(0, $0) }
    }

    private enum CodingKeys: String, CodingKey {
        case route
        case hotTokens
        case coldBlockCount
        case selectedColdTokens
        case selectedBudgetedColdTokens
        case anchorColdTokens
        case anchorOverflowTokens
        case selectedColdBlocks
        case selectorConfidence
        case selectorInitialConfidence
        case selectorFinalConfidence
        case selectorEscalation
        case selectorReasonFlags
        case coldBudgetTokens
        case maxColdBudgetTokens
        case fallbackReason
        case fullScanFallbackCount
        case lastLayerColdBudget
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            route: container.decode(String.self, forKey: .route),
            hotTokens: container.decode(Int.self, forKey: .hotTokens),
            coldBlockCount: container.decode(Int.self, forKey: .coldBlockCount),
            selectedColdTokens: container.decode(Int.self, forKey: .selectedColdTokens),
            selectedBudgetedColdTokens: container.decodeIfPresent(
                Int.self,
                forKey: .selectedBudgetedColdTokens
            ) ?? 0,
            anchorColdTokens: container.decodeIfPresent(
                Int.self,
                forKey: .anchorColdTokens
            ) ?? 0,
            anchorOverflowTokens: container.decodeIfPresent(
                Int.self,
                forKey: .anchorOverflowTokens
            ) ?? 0,
            selectedColdBlocks: container.decode([Int].self, forKey: .selectedColdBlocks),
            selectorConfidence: container.decode(Float.self, forKey: .selectorConfidence),
            selectorInitialConfidence: container.decodeIfPresent(
                Float.self,
                forKey: .selectorInitialConfidence
            ),
            selectorFinalConfidence: container.decodeIfPresent(
                Float.self,
                forKey: .selectorFinalConfidence
            ),
            selectorEscalation: container.decodeIfPresent(
                TurboQuantColdSelectorEscalation.self,
                forKey: .selectorEscalation
            ) ?? .none,
            selectorReasonFlags: container.decodeIfPresent(
                [String].self,
                forKey: .selectorReasonFlags
            ) ?? [],
            coldBudgetTokens: container.decode(Int.self, forKey: .coldBudgetTokens),
            maxColdBudgetTokens: container.decode(Int.self, forKey: .maxColdBudgetTokens),
            fallbackReason: container.decodeIfPresent(String.self, forKey: .fallbackReason),
            fullScanFallbackCount: container.decode(Int.self, forKey: .fullScanFallbackCount),
            lastLayerColdBudget: container.decodeIfPresent(Int.self, forKey: .lastLayerColdBudget)
        )
    }
}

private struct TurboQuantColdBlock {
    var descriptor: TurboQuantColdBlockDescriptor
    var keyCode: TurboQuantAttentionCode
    var valueCode: TurboQuantAttentionCode
}

private struct TurboQuantHybridCheckpoint {
    var offset: Int
    var hotStartToken: Int
    var hotKeys: MLXArray?
    var hotValues: MLXArray?
    var blocks: [TurboQuantColdBlock]
    var lastSelection: TurboQuantColdSelection
    var selectorPolicy: TurboQuantColdSelectorPolicy
    var selectorHints: [TurboQuantColdSelectorHint]
    var nextBlockID: Int
    var fullScanFallbackCount: Int
    var lastSealFailure: String?
    var diagnostics: TurboQuantHybridDiagnostics
}

public final class HybridTurboQuantKVCache: BaseKVCache {
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let preset: TurboQuantPreset
    public let groupSize: Int
    public let seed: UInt64
    public let valueBits: Int?
    public let optimizationPolicy: TurboQuantOptimizationPolicy
    public let fallbackPolicy: TurboQuantFallbackPolicy
    public let maxContextLength: Int?
    public let residentBudgetBytes: Int?
    public let layerIndex: Int?
    public let layerCount: Int?
    public let sparseValuePolicy: TurboQuantSparseValuePolicy
    public private(set) var hotWindowTokens: Int
    public private(set) var coldBlockTokens: Int
    public private(set) var coldBudgetTokens: Int
    public private(set) var maxColdBudgetTokens: Int
    public var coldAttentionMode: TurboQuantColdAttentionMode
    public var layerPolicy: TurboQuantLayerPolicy
    public var selectorPolicy: TurboQuantColdSelectorPolicy

    private var hotKeys: MLXArray?
    private var hotValues: MLXArray?
    private var hotStartToken = 0
    private var coldBlocks: [TurboQuantColdBlock] = []
    public private(set) var selectorHints: [TurboQuantColdSelectorHint] = []
    private var nextBlockID = 0
    private var fullScanFallbackCount = 0
    private var lastSealFailure: String?

    private var canSealColdBlocks: Bool {
        activeBackend == .metalPolarQJL
            && TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec
    }

    public private(set) var lastSelection: TurboQuantColdSelection = .empty
    public private(set) var diagnostics: TurboQuantHybridDiagnostics

    public init(
        maxSize: Int? = nil,
        hotWindowTokens: Int? = nil,
        coldBlockTokens: Int = 1024,
        coldBudgetTokens: Int = 4096,
        maxColdBudgetTokens: Int = 8192,
        coldAttentionMode: TurboQuantColdAttentionMode = .selected,
        layerPolicy: TurboQuantLayerPolicy = .auto,
        preset: TurboQuantPreset = .turbo4v2,
        groupSize: Int = 64,
        backend: TurboQuantBackend = .metalPolarQJL,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        fallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil,
        residentBudgetBytes: Int? = nil,
        layerIndex: Int? = nil,
        layerCount: Int? = nil,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        selectorPolicy: TurboQuantColdSelectorPolicy = .automatic,
        selectorHints: [TurboQuantColdSelectorHint] = []
    ) {
        self.maxContextLength = maxSize
        self.hotWindowTokens = Self.resolvedHotWindowTokens(
            requested: hotWindowTokens,
            maxSize: maxSize
        )
        self.coldBlockTokens = Self.resolvedColdBlockTokens(
            requested: coldBlockTokens,
            maxSize: maxSize
        )
        self.coldBudgetTokens = max(0, coldBudgetTokens)
        self.maxColdBudgetTokens = max(max(0, coldBudgetTokens), maxColdBudgetTokens)
        self.coldAttentionMode = Self.modeOverriddenByEnvironment(coldAttentionMode)
        self.layerPolicy = layerPolicy
        self.preset = Self.presetOverriddenByEnvironment(preset)
        self.groupSize = groupSize
        self.requestedBackend = backend
        self.activeBackend = TurboQuantKernelAvailability.current.runtimeBackend(for: backend)
        self.optimizationPolicy = optimizationPolicy
        self.fallbackPolicy = fallbackPolicy
        self.seed = seed
        self.valueBits = valueBits
        self.residentBudgetBytes = residentBudgetBytes
        self.layerIndex = layerIndex
        self.layerCount = layerCount
        self.sparseValuePolicy = sparseValuePolicy
        self.selectorPolicy = selectorPolicy
        self.selectorHints = selectorHints.filter { !$0.isEmpty }
        self.diagnostics = TurboQuantHybridDiagnostics(
            route: "hybrid",
            hotTokens: 0,
            coldBlockCount: 0,
            selectedColdTokens: 0,
            selectedBudgetedColdTokens: 0,
            anchorColdTokens: 0,
            anchorOverflowTokens: 0,
            selectedColdBlocks: [],
            selectorConfidence: 1,
            selectorInitialConfidence: 1,
            selectorFinalConfidence: 1,
            selectorEscalation: .none,
            selectorReasonFlags: ["empty"],
            coldBudgetTokens: self.coldBudgetTokens,
            maxColdBudgetTokens: self.maxColdBudgetTokens,
            fallbackReason: nil,
            fullScanFallbackCount: 0,
            lastLayerColdBudget: nil
        )
        super.init()
    }

    public override var maxSize: Int? {
        maxContextLength
    }

    public var rawHotLength: Int {
        hotKeys?.dim(2) ?? 0
    }

    public var coldBlockCount: Int {
        coldBlocks.count
    }

    public var coldTokenCount: Int {
        coldBlocks.reduce(0) { $0 + $1.descriptor.logicalTokenCount }
    }

    public var coldBlockDescriptors: [TurboQuantColdBlockDescriptor] {
        coldBlocks.map(\.descriptor)
    }

    public var cacheFootprint: TurboQuantRuntimeCacheFootprint {
        let compressedBytes = coldBlocks.reduce(0) {
            $0 + $1.keyCode.storageByteCount + $1.valueCode.storageByteCount
        }
        let rawShadowBytes = (hotKeys?.nbytes ?? 0) + (hotValues?.nbytes ?? 0)
        return TurboQuantRuntimeCacheFootprint(
            logicalLength: offset,
            capacity: maxContextLength ?? offset,
            compressedBytes: compressedBytes,
            rawShadowBytes: rawShadowBytes,
            lifecycle: coldBlocks.isEmpty ? .rawPrefillChunkOpen : .decodeCompressed
        )
    }

    public func runtimeSnapshot() -> TurboQuantCacheRuntimeSnapshot {
        let keyBytes =
            (hotKeys?.nbytes ?? 0)
            + coldBlocks.reduce(0) { $0 + $1.keyCode.storageByteCount }
        let valueBytes =
            (hotValues?.nbytes ?? 0)
            + coldBlocks.reduce(0) { $0 + $1.valueCode.storageByteCount }
        return TurboQuantCacheRuntimeSnapshot(
            lifecycleDescription:
                coldBlocks.isEmpty
                ? "hybridRawHot(logicalLength:\(offset),hotTokens:\(rawHotLength))"
                : "hybridSelectedCold(logicalLength:\(offset),hotTokens:\(rawHotLength),coldBlocks:\(coldBlocks.count))",
            logicalLength: offset,
            capacity: maxContextLength ?? offset,
            pinnedPrefixLength: 0,
            ringOffset: 0,
            keyBytes: keyBytes,
            valueBytes: valueBytes,
            rawShadowAllocated: hotKeys != nil || hotValues != nil,
            packedFallbackAllocated: false,
            lastAttentionPath: diagnostics.route,
            lastFailure: diagnostics.fallbackReason,
            kvCodec: .polarQJL,
            selectedPath: diagnostics.route,
            fallbackReason: diagnostics.fallbackReason,
            hybridDiagnostics: diagnostics,
            sparseValuePolicy: sparseValuePolicy,
            boundaryPolicy: nil,
            boundaryProtectedLayerCount: 0,
            boundaryProtectionReason: nil
        )
    }

    public override var state: [MLXArray] {
        get {
            var arrays: [MLXArray] = []
            if let hotKeys, let hotValues {
                arrays.append(hotKeys)
                arrays.append(hotValues)
            }
            for block in coldBlocks {
                arrays.append(contentsOf: Self.arrays(for: block.keyCode))
                arrays.append(contentsOf: Self.arrays(for: block.valueCode))
            }
            return arrays
        }
        set {
            guard newValue.isEmpty else {
                fatalError("HybridTurboQuantKVCache restore requires metaState-aware loading.")
            }
            hotKeys = nil
            hotValues = nil
            coldBlocks.removeAll()
            selectorHints.removeAll()
            offset = 0
            nextBlockID = 0
            refreshDiagnostics(selection: .empty, fallbackReason: nil)
        }
    }

    public override var metaState: [String] {
        get {
            let maxContextLengthString = maxContextLength.map(String.init) ?? "None"
            let valueBitsString = valueBits.map(String.init) ?? "None"
            let layerIndexString = layerIndex.map(String.init) ?? "None"
            let layerCountString = layerCount.map(String.init) ?? "None"
            let sparseValuePolicyString = String(describing: sparseValuePolicy)
            let selectorPolicyString = HybridTurboQuantKVCache.encodedMetadata(selectorPolicy)
            let hotStateCount = hotKeys == nil ? 0 : 2
            let coldBlockCount = coldBlocks.count
            var entries = [
                "HybridTurboQuantKVCache.v3",
                "offset=\(offset)",
                "hotStartToken=\(hotStartToken)",
                "maxSize=\(maxContextLengthString)",
                "hotWindowTokens=\(hotWindowTokens)",
                "coldBlockTokens=\(coldBlockTokens)",
                "coldBudgetTokens=\(coldBudgetTokens)",
                "maxColdBudgetTokens=\(maxColdBudgetTokens)",
                "coldAttentionMode=\(coldAttentionMode.rawValue)",
                "preset=\(preset.rawValue)",
                "backend=\(requestedBackend.rawValue)",
                "activeBackend=\(activeBackend.rawValue)",
                "groupSize=\(groupSize)",
                "seed=\(seed)",
                "valueBits=\(valueBitsString)",
                "layerIndex=\(layerIndexString)",
                "layerCount=\(layerCountString)",
                "sparseValuePolicy=\(sparseValuePolicyString)",
                "selectorPolicy=\(selectorPolicyString)",
                "hotStateCount=\(hotStateCount)",
                "coldBlockCount=\(coldBlockCount)",
            ]
            entries.append(
                contentsOf: selectorHints.map { hint in
                    "selectorHint=\(HybridTurboQuantKVCache.encodedMetadata(hint))"
                })
            entries.append(
                contentsOf: coldBlocks.map { block in
                    "block=\(HybridTurboQuantKVCache.encodedMetadata(block.descriptor))"
                })
            return entries
        }
        set {
            guard newValue.isEmpty || (newValue.count == 1 && newValue[0].isEmpty) else {
                fatalError("HybridTurboQuantKVCache metaState restore is not supported without state decoding.")
            }
        }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        appendHotInBoundedChunks(keys: keys, values: values)
        refreshDiagnostics(selection: lastSelection, fallbackReason: lastSealFailure)
        return rawHotState(defaultKeys: keys, defaultValues: values)
    }

    public func updateHybrid(keys: MLXArray, values: MLXArray) {
        _ = update(keys: keys, values: values)
    }

    public func rawHotState(
        defaultKeys: MLXArray,
        defaultValues: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        (hotKeys ?? defaultKeys, hotValues ?? defaultValues)
    }

    public func selectColdBlocks(
        query: MLXArray? = nil,
        budgetOverride: Int? = nil,
        layerIndex: Int? = nil,
        layerCount: Int? = nil
    ) -> TurboQuantColdSelection {
        let layerBudget = coldBudgetForLayer(
            layerIndex: layerIndex ?? self.layerIndex,
            layerCount: layerCount ?? self.layerCount,
            defaultBudget: budgetOverride ?? coldBudgetTokens
        )
        let budget = min(maxColdBudgetTokens, max(0, layerBudget))
        let descriptors = coldBlockDescriptorsScored(for: query)
        let selection = Self.selectColdBlocks(
            descriptors: descriptors,
            mode: coldAttentionMode,
            budgetTokens: budget,
            maxBudgetTokens: maxColdBudgetTokens,
            selectorPolicy: selectorPolicy
        )
        lastSelection = selection
        if selection.selectorEscalation == .exhaustive {
            fullScanFallbackCount += 1
        }
        refreshDiagnostics(selection: selection, fallbackReason: lastSealFailure)
        return selection
    }

    public static func selectColdBlocks(
        descriptors: [TurboQuantColdBlockDescriptor],
        mode: TurboQuantColdAttentionMode,
        budgetTokens: Int,
        maxBudgetTokens: Int,
        selectorPolicy: TurboQuantColdSelectorPolicy = .automatic
    ) -> TurboQuantColdSelection {
        guard !descriptors.isEmpty else { return .empty }
        guard mode != .off else {
            return TurboQuantColdSelection(
                selectedBlockIDs: [],
                selectedTokenBudget: budgetTokens,
                selectedTokenCount: 0,
                confidence: 1,
                reasonFlags: ["cold_off"],
                requiresExhaustiveFallback: false
            )
        }

        if mode == .exhaustive {
            return exhaustiveSelection(
                descriptors: descriptors,
                budgetTokens: max(0, maxBudgetTokens),
                initialConfidence: 1,
                reasonFlags: ["exhaustive"]
            )
        }

        let cappedBudget = min(maxBudgetTokens, max(0, budgetTokens))
        var selection = budgetedSelection(
            descriptors: descriptors,
            budgetTokens: cappedBudget,
            selectorPolicy: selectorPolicy,
            escalation: .none,
            initialConfidence: nil,
            extraReasons: []
        )
        let initialConfidence = selection.confidence

        if selection.confidence < selectorPolicy.minimumConfidence,
            selectorPolicy.allowMaxBudgetEscalation,
            maxBudgetTokens > cappedBudget
        {
            selection = budgetedSelection(
                descriptors: descriptors,
                budgetTokens: max(0, maxBudgetTokens),
                selectorPolicy: selectorPolicy,
                escalation: .maxBudget,
                initialConfidence: initialConfidence,
                extraReasons: ["max_budget_escalation"]
            )
        }

        if selection.confidence < selectorPolicy.minimumConfidence {
            if selectorPolicy.allowExhaustiveEscalation {
                return exhaustiveSelection(
                    descriptors: descriptors,
                    budgetTokens: max(0, maxBudgetTokens),
                    initialConfidence: initialConfidence,
                    reasonFlags: selection.reasonFlags + ["quality_guard_exhaustive"]
                )
            }
            if !selection.reasonFlags.contains("low_confidence_exhaustive_disabled") {
                selection.reasonFlags.append("low_confidence_exhaustive_disabled")
            }
        }

        return selection
    }

    private static func exhaustiveSelection(
        descriptors: [TurboQuantColdBlockDescriptor],
        budgetTokens: Int,
        initialConfidence: Float,
        reasonFlags: [String]
    ) -> TurboQuantColdSelection {
        let tokenCount = descriptors.reduce(0) { $0 + $1.logicalTokenCount }
        let anchorTokens = descriptors.filter(\.isAnchor).reduce(0) {
            $0 + $1.logicalTokenCount
        }
        return TurboQuantColdSelection(
            selectedBlockIDs: descriptors.sorted { $0.startToken < $1.startToken }.map(\.blockID),
            selectedTokenBudget: tokenCount,
            selectedTokenCount: tokenCount,
            anchorTokenCount: anchorTokens,
            budgetedTokenCount: max(0, tokenCount - anchorTokens),
            anchorOverflowTokens: max(0, anchorTokens - budgetTokens),
            confidence: 1,
            initialConfidence: initialConfidence,
            finalConfidence: 1,
            selectorEscalation: .exhaustive,
            reasonFlags: uniqueReasonFlags(reasonFlags.isEmpty ? ["exhaustive"] : reasonFlags),
            requiresExhaustiveFallback: true
        )
    }

    private static func budgetedSelection(
        descriptors: [TurboQuantColdBlockDescriptor],
        budgetTokens: Int,
        selectorPolicy: TurboQuantColdSelectorPolicy,
        escalation: TurboQuantColdSelectorEscalation,
        initialConfidence: Float?,
        extraReasons: [String]
    ) -> TurboQuantColdSelection {
        let cappedBudget = max(0, budgetTokens)
        var selected: [TurboQuantColdBlockDescriptor] = []
        var selectedIDs = Set<Int>()
        var budgetedTokens = 0
        var reasons = extraReasons

        func addReason(_ reason: String) {
            if !reasons.contains(reason) {
                reasons.append(reason)
            }
        }

        func addAnchor(_ descriptor: TurboQuantColdBlockDescriptor) {
            guard !selectedIDs.contains(descriptor.blockID) else { return }
            selectedIDs.insert(descriptor.blockID)
            selected.append(descriptor)
            addReason("anchor")
        }

        func addBudgeted(_ descriptor: TurboQuantColdBlockDescriptor, reason: String) {
            guard !selectedIDs.contains(descriptor.blockID) else { return }
            guard budgetedTokens + descriptor.logicalTokenCount <= cappedBudget else { return }
            selectedIDs.insert(descriptor.blockID)
            selected.append(descriptor)
            budgetedTokens += descriptor.logicalTokenCount
            addReason(reason)
        }

        for descriptor in descriptors where descriptor.isAnchor {
            addAnchor(descriptor)
        }

        let nonAnchors = descriptors.filter { !$0.isAnchor }
        let nearest = nonAnchors
            .sorted {
                if $0.endToken == $1.endToken {
                    return $0.blockID < $1.blockID
                }
                return $0.endToken > $1.endToken
            }
            .prefix(selectorPolicy.nearestBlockCount)
        for descriptor in nearest {
            addBudgeted(descriptor, reason: "nearest")
        }

        let maxKeyNorm = max(0, nonAnchors.map(\.maxKeyNormEstimate).max() ?? 0)
        let scored = nonAnchors
            .filter { !selectedIDs.contains($0.blockID) }
            .sorted {
                let left = selectorScore($0, policy: selectorPolicy, maxKeyNorm: maxKeyNorm)
                let right = selectorScore($1, policy: selectorPolicy, maxKeyNorm: maxKeyNorm)
                if left == right {
                    if $0.endToken == $1.endToken {
                        return $0.blockID < $1.blockID
                    }
                    return $0.endToken > $1.endToken
                }
                return left > right
            }
        for descriptor in scored {
            guard budgetedTokens < cappedBudget else { break }
            addBudgeted(
                descriptor,
                reason: selectorReason(descriptor, policy: selectorPolicy, maxKeyNorm: maxKeyNorm)
            )
        }

        selected.sort { $0.startToken < $1.startToken }
        let anchorTokens = selected.filter(\.isAnchor).reduce(0) {
            $0 + $1.logicalTokenCount
        }
        let selectedTokenCount = anchorTokens + budgetedTokens
        let confidence = selectorConfidence(
            descriptors: descriptors,
            selectedIDs: selectedIDs,
            selectedTokenCount: selectedTokenCount,
            policy: selectorPolicy,
            maxKeyNorm: maxKeyNorm
        )
        if selected.isEmpty {
            addReason(cappedBudget == 0 ? "budget_empty" : "no_selectable_blocks")
        }
        return TurboQuantColdSelection(
            selectedBlockIDs: selected.map(\.blockID),
            selectedTokenBudget: cappedBudget,
            selectedTokenCount: selectedTokenCount,
            anchorTokenCount: anchorTokens,
            budgetedTokenCount: budgetedTokens,
            anchorOverflowTokens: max(0, anchorTokens - cappedBudget),
            confidence: confidence,
            initialConfidence: initialConfidence ?? confidence,
            finalConfidence: confidence,
            selectorEscalation: escalation,
            reasonFlags: uniqueReasonFlags(reasons.isEmpty ? ["budget_empty"] : reasons),
            requiresExhaustiveFallback: false
        )
    }

    private static func selectorScore(
        _ descriptor: TurboQuantColdBlockDescriptor,
        policy: TurboQuantColdSelectorPolicy,
        maxKeyNorm: Float
    ) -> Float {
        let keyNormScore = maxKeyNorm > 0 ? descriptor.maxKeyNormEstimate / maxKeyNorm : 0
        let recencyScore = descriptor.recencyRank > 0 ? 1 / Float(descriptor.recencyRank) : 0
        return policy.semanticWeight * descriptor.semanticScore
            + policy.lexicalWeight * descriptor.lexicalScore
            + policy.keyNormWeight * keyNormScore
            + policy.recencyWeight * recencyScore
    }

    private static func selectorReason(
        _ descriptor: TurboQuantColdBlockDescriptor,
        policy: TurboQuantColdSelectorPolicy,
        maxKeyNorm: Float
    ) -> String {
        let semantic = policy.semanticWeight * descriptor.semanticScore
        let lexical = policy.lexicalWeight * descriptor.lexicalScore
        let keyNorm =
            maxKeyNorm > 0
            ? policy.keyNormWeight * descriptor.maxKeyNormEstimate / maxKeyNorm
            : 0
        if semantic >= lexical, semantic >= keyNorm, semantic > 0 {
            return "semantic"
        }
        if lexical >= keyNorm, lexical > 0 {
            return "lexical"
        }
        if keyNorm > 0 {
            return "key_norm"
        }
        return "recency"
    }

    private static func selectorConfidence(
        descriptors: [TurboQuantColdBlockDescriptor],
        selectedIDs: Set<Int>,
        selectedTokenCount: Int,
        policy: TurboQuantColdSelectorPolicy,
        maxKeyNorm: Float
    ) -> Float {
        let totalTokens = descriptors.reduce(0) { $0 + $1.logicalTokenCount }
        let tokenCoverage =
            totalTokens > 0 ? Float(selectedTokenCount) / Float(totalTokens) : 1
        let totalScore = descriptors.reduce(Float(0)) {
            $0 + max(0, selectorScore($1, policy: policy, maxKeyNorm: maxKeyNorm))
        }
        let selectedScore = descriptors.reduce(Float(0)) {
            guard selectedIDs.contains($1.blockID) else { return $0 }
            return $0 + max(0, selectorScore($1, policy: policy, maxKeyNorm: maxKeyNorm))
        }
        let scoreCoverage = totalScore > 0 ? selectedScore / totalScore : tokenCoverage
        return max(0, min(1, max(tokenCoverage, scoreCoverage)))
    }

    private static func uniqueReasonFlags(_ reasons: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for reason in reasons where !reason.isEmpty {
            if seen.insert(reason).inserted {
                result.append(reason)
            }
        }
        return result.isEmpty ? ["unknown"] : result
    }

    public func selectedColdState(
        selection: TurboQuantColdSelection,
        outputDType: DType
    ) throws -> (keys: MLXArray, values: MLXArray)? {
        guard !selection.selectedBlockIDs.isEmpty else { return nil }
        let selectedIDs = Set(selection.selectedBlockIDs)
        let selectedBlocks = coldBlocks
            .filter { selectedIDs.contains($0.descriptor.blockID) }
            .sorted { $0.descriptor.startToken < $1.descriptor.startToken }
        guard !selectedBlocks.isEmpty else { return nil }

        let decodedKeys = try selectedBlocks.map {
            try MLX.turboQuantMetalDecodeAttention($0.keyCode, outputDType: outputDType)
        }
        let decodedValues = try selectedBlocks.map {
            try MLX.turboQuantMetalDecodeAttention($0.valueCode, outputDType: outputDType)
        }
        return (
            keys: decodedKeys.count == 1 ? decodedKeys[0] : concatenated(decodedKeys, axis: 2),
            values: decodedValues.count == 1 ? decodedValues[0] : concatenated(decodedValues, axis: 2)
        )
    }

    public func selectedColdCompressedSegments(
        selection: TurboQuantColdSelection
    ) -> [(key: TurboQuantAttentionCode, value: TurboQuantAttentionCode)] {
        guard !selection.selectedBlockIDs.isEmpty else { return [] }
        let selectedIDs = Set(selection.selectedBlockIDs)
        return coldBlocks
            .filter { selectedIDs.contains($0.descriptor.blockID) }
            .sorted { $0.descriptor.startToken < $1.descriptor.startToken }
            .map { (key: $0.keyCode, value: $0.valueCode) }
    }

    public func recordFallbackReason(_ reason: String?) {
        lastSealFailure = reason
        refreshDiagnostics(selection: lastSelection, fallbackReason: reason)
    }

    public func markAnchor(
        tokenRange: Range<Int>,
        flags: TurboQuantColdBlockAnchorFlags
    ) {
        guard !flags.isEmpty else { return }
        mergeSelectorHints([
            TurboQuantColdSelectorHint(
                startToken: tokenRange.lowerBound,
                endToken: tokenRange.upperBound,
                anchorFlags: flags,
                sourceID: "anchor"
            )
        ])
    }

    public func applySelectorHints(_ hints: [TurboQuantColdSelectorHint]) {
        selectorHints = hints.filter { !$0.isEmpty }
        refreshHintScores()
        refreshDiagnostics(selection: lastSelection, fallbackReason: lastSealFailure)
    }

    private func mergeSelectorHints(_ hints: [TurboQuantColdSelectorHint]) {
        selectorHints.append(contentsOf: hints.filter { !$0.isEmpty })
        refreshHintScores()
        refreshDiagnostics(selection: lastSelection, fallbackReason: lastSealFailure)
    }

    private func refreshHintScores() {
        for index in coldBlocks.indices {
            applySelectorHintsToDescriptor(&coldBlocks[index].descriptor, resetScores: true)
        }
        refreshRecency()
    }

    public override var isTrimmable: Bool {
        true
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0, offset > 0 else { return 0 }
        let targetTrim = min(n, offset)
        var remaining = targetTrim
        var actualTrimmed = 0

        if let hotKeys, let hotValues, remaining > 0 {
            let hotLength = hotKeys.dim(2)
            let keepHot = max(0, hotLength - remaining)
            if keepHot == 0 {
                self.hotKeys = nil
                self.hotValues = nil
                hotStartToken += hotLength
                remaining -= hotLength
                actualTrimmed += hotLength
            } else {
                self.hotKeys = hotKeys[.ellipsis, ..<keepHot, 0...]
                self.hotValues = hotValues[.ellipsis, ..<keepHot, 0...]
                actualTrimmed += remaining
                remaining = 0
            }
        }

        while remaining > 0, let last = coldBlocks.last {
            let count = last.descriptor.logicalTokenCount
            if remaining >= count {
                coldBlocks.removeLast()
                remaining -= count
                actualTrimmed += count
            } else {
                // Partial compressed block rollback is intentionally conservative:
                // drop the whole block and report the extra trim to keep state coherent.
                coldBlocks.removeLast()
                actualTrimmed += count
                remaining = 0
            }
        }

        let trimmed = min(offset, actualTrimmed)
        offset = max(0, offset - trimmed)
        if hotKeys == nil {
            hotStartToken = offset
        }
        refreshRecency()
        lastSelection = .empty
        refreshDiagnostics(selection: lastSelection, fallbackReason: lastSealFailure)
        return trimmed
    }

    public override func copy() -> any KVCache {
        let copied = HybridTurboQuantKVCache(
            maxSize: maxContextLength,
            hotWindowTokens: hotWindowTokens,
            coldBlockTokens: coldBlockTokens,
            coldBudgetTokens: coldBudgetTokens,
            maxColdBudgetTokens: maxColdBudgetTokens,
            coldAttentionMode: coldAttentionMode,
            layerPolicy: layerPolicy,
            preset: preset,
            groupSize: groupSize,
            backend: requestedBackend,
            optimizationPolicy: optimizationPolicy,
            fallbackPolicy: fallbackPolicy,
            seed: seed,
            valueBits: valueBits,
            residentBudgetBytes: residentBudgetBytes,
            layerIndex: layerIndex,
            layerCount: layerCount,
            sparseValuePolicy: sparseValuePolicy,
            selectorPolicy: selectorPolicy,
            selectorHints: selectorHints
        )
        copied.offset = offset
        copied.hotStartToken = hotStartToken
        copied.hotKeys = hotKeys
        copied.hotValues = hotValues
        copied.coldBlocks = coldBlocks
        copied.nextBlockID = nextBlockID
        copied.lastSelection = lastSelection
        copied.fullScanFallbackCount = fullScanFallbackCount
        copied.lastSealFailure = lastSealFailure
        copied.diagnostics = diagnostics
        copied.selectorHints = selectorHints
        return copied
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 1 else { return .none }
        if returnArray || windowSize != nil {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    public func makeCompressedUpdateCheckpoint(appendingTokenCount tokenCount: Int)
        -> Any
    {
        TurboQuantHybridCheckpoint(
            offset: offset,
            hotStartToken: hotStartToken,
            hotKeys: hotKeys,
            hotValues: hotValues,
            blocks: coldBlocks,
            lastSelection: lastSelection,
            selectorPolicy: selectorPolicy,
            selectorHints: selectorHints,
            nextBlockID: nextBlockID,
            fullScanFallbackCount: fullScanFallbackCount,
            lastSealFailure: lastSealFailure,
            diagnostics: diagnostics
        )
    }

    public func restoreCompressedUpdateCheckpoint(_ checkpoint: Any) {
        guard let checkpoint = checkpoint as? TurboQuantHybridCheckpoint else { return }
        offset = checkpoint.offset
        hotStartToken = checkpoint.hotStartToken
        hotKeys = checkpoint.hotKeys
        hotValues = checkpoint.hotValues
        coldBlocks = checkpoint.blocks
        lastSelection = checkpoint.lastSelection
        selectorPolicy = checkpoint.selectorPolicy
        selectorHints = checkpoint.selectorHints
        nextBlockID = checkpoint.nextBlockID
        fullScanFallbackCount = checkpoint.fullScanFallbackCount
        lastSealFailure = checkpoint.lastSealFailure
        diagnostics = checkpoint.diagnostics
    }

    static func restoreFromMetaState(
        state: [MLXArray],
        metaState: [String]
    ) throws -> HybridTurboQuantKVCache {
        let version = metaState.first
        guard version == "HybridTurboQuantKVCache.v1"
            || version == "HybridTurboQuantKVCache.v2"
            || version == "HybridTurboQuantKVCache.v3"
        else {
            throw KVCacheError(message: "Invalid HybridTurboQuantKVCache metaState version")
        }
        var metadata: [String: String] = [:]
        var encodedDescriptorEntries: [String] = []
        var encodedHintEntries: [String] = []
        var legacyBlockDescriptorEntries: [String] = []
        for entry in metaState.dropFirst() {
            guard let equalIndex = entry.firstIndex(of: "=") else {
                legacyBlockDescriptorEntries.append(entry)
                continue
            }
            let key = String(entry[..<equalIndex])
            let value = String(entry[entry.index(after: equalIndex)...])
            switch key {
            case "block":
                encodedDescriptorEntries.append(value)
            case "selectorHint":
                encodedHintEntries.append(value)
            default:
                metadata[key] = value
            }
        }
        let maxSize =
            metadata["maxSize"].flatMap { $0 == "None" ? nil : Int($0) }
        let preset = metadata["preset"].flatMap(TurboQuantPreset.init(rawValue:)) ?? .turbo4v2
        let backend =
            metadata["backend"].flatMap(TurboQuantBackend.init(rawValue:)) ?? .metalPolarQJL
        let mode =
            metadata["coldAttentionMode"].flatMap(TurboQuantColdAttentionMode.init(rawValue:))
            ?? .selected
        let valueBits =
            metadata["valueBits"].flatMap { $0 == "None" ? nil : Int($0) }
        let seed = metadata["seed"].flatMap(UInt64.init) ?? defaultTurboQuantSeed
        let selectorPolicy =
            metadata["selectorPolicy"].flatMap {
                try? Self.decodedMetadata(TurboQuantColdSelectorPolicy.self, from: $0)
            } ?? .automatic
        let selectorHints = encodedHintEntries.compactMap {
            try? Self.decodedMetadata(TurboQuantColdSelectorHint.self, from: $0)
        }
        let cache = HybridTurboQuantKVCache(
            maxSize: maxSize,
            hotWindowTokens: metadata["hotWindowTokens"].flatMap(Int.init),
            coldBlockTokens: metadata["coldBlockTokens"].flatMap(Int.init) ?? 1024,
            coldBudgetTokens: metadata["coldBudgetTokens"].flatMap(Int.init) ?? 4096,
            maxColdBudgetTokens: metadata["maxColdBudgetTokens"].flatMap(Int.init) ?? 8192,
            coldAttentionMode: mode,
            preset: preset,
            groupSize: metadata["groupSize"].flatMap(Int.init) ?? 64,
            backend: backend,
            seed: seed,
            valueBits: valueBits,
            layerIndex: metadata["layerIndex"].flatMap { $0 == "None" ? nil : Int($0) },
            layerCount: metadata["layerCount"].flatMap { $0 == "None" ? nil : Int($0) },
            selectorPolicy: selectorPolicy,
            selectorHints: selectorHints
        )
        cache.offset = metadata["offset"].flatMap(Int.init) ?? 0
        cache.hotStartToken = metadata["hotStartToken"].flatMap(Int.init) ?? cache.offset

        var stateIndex = 0
        let hotStateCount = metadata["hotStateCount"].flatMap(Int.init) ?? 0
        if hotStateCount == 2 {
            guard state.count >= 2 else {
                throw KVCacheError(message: "HybridTurboQuantKVCache hot state is truncated")
            }
            cache.hotKeys = state[0]
            cache.hotValues = state[1]
            if metadata["hotStartToken"] == nil {
                cache.hotStartToken = max(0, cache.offset - state[0].dim(2))
            }
            stateIndex = 2
        }

        let descriptors: [TurboQuantColdBlockDescriptor]
        if version == "HybridTurboQuantKVCache.v2" || version == "HybridTurboQuantKVCache.v3" {
            descriptors = try encodedDescriptorEntries.map {
                try Self.decodedMetadata(TurboQuantColdBlockDescriptor.self, from: $0)
            }
        } else {
            descriptors = try legacyBlockDescriptorEntries.map(Self.descriptor(from:))
        }
        for descriptor in descriptors {
            guard stateIndex + 10 <= state.count else {
                throw KVCacheError(message: "HybridTurboQuantKVCache compressed block state is truncated")
            }
            let keyArrays = Array(state[stateIndex ..< stateIndex + 5])
            stateIndex += 5
            let valueArrays = Array(state[stateIndex ..< stateIndex + 5])
            stateIndex += 5
            let keyCode = try Self.attentionCode(
                arrays: keyArrays,
                descriptor: descriptor,
                preset: preset,
                role: .key,
                groupSize: cache.groupSize,
                seed: seed,
                valueBits: valueBits
            )
            let valueCode = try Self.attentionCode(
                arrays: valueArrays,
                descriptor: descriptor,
                preset: preset,
                role: .value,
                groupSize: cache.groupSize,
                seed: seed ^ turboQuantHybridValueSeedSalt,
                valueBits: valueBits
            )
            cache.coldBlocks.append(
                TurboQuantColdBlock(
                    descriptor: descriptor,
                    keyCode: keyCode,
                    valueCode: valueCode
                )
            )
            cache.nextBlockID = max(cache.nextBlockID, descriptor.blockID + 1)
        }
        cache.refreshRecency()
        cache.refreshDiagnostics(selection: .empty, fallbackReason: nil)
        return cache
    }

    private func appendHotInBoundedChunks(keys: MLXArray, values: MLXArray) {
        let tokenCount = keys.dim(2)
        guard tokenCount > 0 else { return }
        guard canSealColdBlocks else {
            appendHot(keys: keys, values: values)
            return
        }
        let maxBufferedHotTokens = max(1, hotWindowTokens + coldBlockTokens)
        var consumed = 0
        while consumed < tokenCount {
            do {
                try sealReadyColdBlocks(sealAtCapacity: true)
                lastSealFailure = nil
            } catch {
                lastSealFailure = String(describing: error)
                let tailRange = consumed ..< tokenCount
                appendHot(
                    keys: keys[.ellipsis, tailRange, 0...],
                    values: values[.ellipsis, tailRange, 0...]
                )
                return
            }
            let remaining = tokenCount - consumed
            let currentHot = rawHotLength
            let room = max(1, maxBufferedHotTokens - currentHot)
            let chunkCount = min(remaining, room)
            let chunkRange = consumed ..< (consumed + chunkCount)
            appendHot(
                keys: keys[.ellipsis, chunkRange, 0...],
                values: values[.ellipsis, chunkRange, 0...]
            )
            do {
                try sealReadyColdBlocks(sealAtCapacity: false)
                lastSealFailure = nil
            } catch {
                lastSealFailure = String(describing: error)
                if consumed + chunkCount < tokenCount {
                    let tailRange = (consumed + chunkCount) ..< tokenCount
                    appendHot(
                        keys: keys[.ellipsis, tailRange, 0...],
                        values: values[.ellipsis, tailRange, 0...]
                    )
                }
                return
            }
            consumed += chunkCount
        }
    }

    private func appendHot(keys: MLXArray, values: MLXArray) {
        if hotKeys == nil {
            hotStartToken = offset
        }
        if let currentKeys = hotKeys, let currentValues = hotValues {
            hotKeys = concatenated([currentKeys, keys], axis: 2)
            hotValues = concatenated([currentValues, values], axis: 2)
        } else {
            hotKeys = keys
            hotValues = values
        }
        offset += keys.dim(2)
    }

    private func sealReadyColdBlocks(sealAtCapacity _: Bool) throws {
        guard canSealColdBlocks else {
            return
        }
        let maxResidentTokens = max(1, hotWindowTokens)
        while let hotKeys, let hotValues,
            hotKeys.dim(2) >= coldBlockTokens
        {
            let hotLength = hotKeys.dim(2)
            let shouldSeal = hotLength > maxResidentTokens
            guard shouldSeal else { break }
            let blockKeys = hotKeys[.ellipsis, ..<coldBlockTokens, 0...]
            let blockValues = hotValues[.ellipsis, ..<coldBlockTokens, 0...]
            try sealBlock(keys: blockKeys, values: blockValues)
            let remainingStart = coldBlockTokens
            self.hotKeys = hotKeys[.ellipsis, remainingStart..., 0...]
            self.hotValues = hotValues[.ellipsis, remainingStart..., 0...]
            hotStartToken += coldBlockTokens
        }
    }

    private func sealBlock(keys: MLXArray, values: MLXArray) throws {
        let tokenCount = keys.dim(2)
        let startToken = hotStartToken
        let endToken = startToken + tokenCount
        let keyConfig = TurboQuantConfiguration(
            preset: preset,
            role: .key,
            groupSize: groupSize,
            mode: .affine,
            backend: .metalPolarQJL,
            seed: seed,
            valueBits: valueBits
        )
        let valueConfig = TurboQuantConfiguration(
            preset: preset,
            role: .value,
            groupSize: groupSize,
            mode: .affine,
            backend: .metalPolarQJL,
            seed: seed ^ turboQuantHybridValueSeedSalt,
            valueBits: valueBits
        )
        let keyCode = try MLX.turboQuantMetalEncodeAttention(
            keys,
            configuration: keyConfig,
            capacity: tokenCount,
            logicalLength: tokenCount
        )
        let valueCode = try MLX.turboQuantMetalEncodeAttention(
            values,
            configuration: valueConfig,
            capacity: tokenCount,
            logicalLength: tokenCount
        )
        let maxKeyNormEstimate = Self.maxKeyNormEstimate(keys)
        var descriptor = TurboQuantColdBlockDescriptor(
            blockID: nextBlockID,
            startToken: startToken,
            endToken: endToken,
            compressedSlotStart: 0,
            compressedSlotEnd: tokenCount,
            logicalTokenCount: tokenCount,
            recencyRank: 0,
            anchorFlags: [],
            keyMeanSummary: Self.keyMeanSummary(keys),
            maxKeyNormEstimate: maxKeyNormEstimate,
            scoreUpperBoundEstimate: maxKeyNormEstimate,
            relevanceScore: Float(endToken)
        )
        applySelectorHintsToDescriptor(&descriptor, resetScores: true)
        coldBlocks.append(
            TurboQuantColdBlock(
                descriptor: descriptor,
                keyCode: keyCode,
                valueCode: valueCode
            )
        )
        nextBlockID += 1
        refreshRecency()
    }

    private func refreshRecency() {
        let maxKeyNorm = max(0, coldBlocks.map(\.descriptor.maxKeyNormEstimate).max() ?? 0)
        for index in coldBlocks.indices {
            coldBlocks[index].descriptor.recencyRank = coldBlocks.count - index
            coldBlocks[index].descriptor.relevanceScore =
                Self.staticRelevanceScore(
                    coldBlocks[index].descriptor,
                    policy: selectorPolicy,
                    maxKeyNorm: maxKeyNorm
                )
        }
    }

    private func coldBlockDescriptorsScored(for query: MLXArray?) -> [TurboQuantColdBlockDescriptor] {
        var descriptors = coldBlockDescriptors
        guard let query,
            let querySummary = Self.meanSummary(query),
            !querySummary.isEmpty
        else {
            return descriptors
        }
        let queryNorm = Self.vectorNorm(querySummary)
        guard queryNorm > 0 else { return descriptors }
        for index in descriptors.indices {
            let queryScore = Self.summarySimilarity(
                lhs: querySummary,
                rhs: descriptors[index].keyMeanSummary
            )
            descriptors[index].semanticScore = max(descriptors[index].semanticScore, queryScore)
            descriptors[index].scoreUpperBoundEstimate =
                descriptors[index].maxKeyNormEstimate * queryNorm
            descriptors[index].relevanceScore += queryScore
        }
        return descriptors
    }

    private static func staticRelevanceScore(
        _ descriptor: TurboQuantColdBlockDescriptor,
        policy: TurboQuantColdSelectorPolicy,
        maxKeyNorm: Float
    ) -> Float {
        let keyNormScore = maxKeyNorm > 0 ? descriptor.maxKeyNormEstimate / maxKeyNorm : 0
        let recencyScore = descriptor.recencyRank > 0 ? 1 / Float(descriptor.recencyRank) : 0
        return policy.semanticWeight * descriptor.semanticScore
            + policy.lexicalWeight * descriptor.lexicalScore
            + policy.keyNormWeight * keyNormScore
            + policy.recencyWeight * recencyScore
            + (descriptor.isAnchor ? 10_000 : 0)
    }

    private func applySelectorHintsToDescriptor(
        _ descriptor: inout TurboQuantColdBlockDescriptor,
        resetScores: Bool
    ) {
        if resetScores {
            descriptor.lexicalScore = 0
            descriptor.semanticScore = 0
        }
        for hint in selectorHints {
            let overlapStart = max(descriptor.startToken, hint.startToken)
            let overlapEnd = min(descriptor.endToken, hint.endToken)
            guard overlapStart < overlapEnd, descriptor.logicalTokenCount > 0 else { continue }
            let overlap = Float(overlapEnd - overlapStart) / Float(descriptor.logicalTokenCount)
            descriptor.lexicalScore += hint.lexicalScore * overlap
            descriptor.semanticScore += hint.semanticScore * overlap
            descriptor.anchorFlags.formUnion(hint.anchorFlags)
        }
    }

    private func refreshDiagnostics(
        selection: TurboQuantColdSelection,
        fallbackReason: String?,
        route: String? = nil
    ) {
        let route = route ?? defaultAttentionRoute(selection: selection)
        diagnostics = TurboQuantHybridDiagnostics(
            route: route,
            hotTokens: rawHotLength,
            coldBlockCount: coldBlocks.count,
            selectedColdTokens: selection.selectedTokenCount,
            selectedBudgetedColdTokens: selection.budgetedTokenCount,
            anchorColdTokens: selection.anchorTokenCount,
            anchorOverflowTokens: selection.anchorOverflowTokens,
            selectedColdBlocks: selection.selectedBlockIDs,
            selectorConfidence: selection.confidence,
            selectorInitialConfidence: selection.initialConfidence,
            selectorFinalConfidence: selection.finalConfidence,
            selectorEscalation: selection.selectorEscalation,
            selectorReasonFlags: selection.reasonFlags,
            coldBudgetTokens: selection.selectedTokenBudget,
            maxColdBudgetTokens: maxColdBudgetTokens,
            fallbackReason: diagnosticFallbackReason(
                selection: selection,
                runtimeFallbackReason: fallbackReason
            ),
            fullScanFallbackCount: fullScanFallbackCount,
            lastLayerColdBudget: selection.selectedTokenBudget
        )
    }

    private func defaultAttentionRoute(selection: TurboQuantColdSelection) -> String {
        if coldBlocks.isEmpty || selection.selectedBlockIDs.isEmpty {
            return "hybrid_raw_hot"
        }
        if selection.selectorEscalation == .exhaustive || coldAttentionMode == .exhaustive {
            return "hybrid_exhaustive_cold"
        }
        return "hybrid_selected_cold"
    }

    func segmentedAttentionRoute(selection: TurboQuantColdSelection) -> String {
        if selection.selectorEscalation == .exhaustive || coldAttentionMode == .exhaustive {
            return "hybrid_exhaustive_segmented_attention"
        }
        return "hybrid_selected_segmented_attention"
    }

    func decodedAttentionRoute(selection: TurboQuantColdSelection) -> String {
        if selection.selectorEscalation == .exhaustive || coldAttentionMode == .exhaustive {
            return "hybrid_exhaustive_decoded_sdpa"
        }
        if selection.selectedBlockIDs.isEmpty {
            return "hybrid_raw_hot"
        }
        return "hybrid_selected_decoded_sdpa"
    }

    func recordAttentionRoute(
        _ route: String,
        selection: TurboQuantColdSelection? = nil,
        fallbackReason: String? = nil
    ) {
        refreshDiagnostics(
            selection: selection ?? lastSelection,
            fallbackReason: fallbackReason ?? lastSealFailure,
            route: route
        )
    }

    private func diagnosticFallbackReason(
        selection: TurboQuantColdSelection,
        runtimeFallbackReason: String?
    ) -> String? {
        var reasons = [String]()
        if let runtimeFallbackReason, !runtimeFallbackReason.isEmpty {
            reasons.append(runtimeFallbackReason)
        }
        if coldAttentionMode == .selected, selection.selectorEscalation == .exhaustive {
            let flags = selection.reasonFlags.isEmpty
                ? "unknown"
                : selection.reasonFlags.joined(separator: ",")
            reasons.append("selected_mode_exhaustive_fallback:\(flags)")
        }
        guard !reasons.isEmpty else { return nil }
        return Self.uniqueReasonFlags(reasons).joined(separator: "; ")
    }

    func coldBudgetForLayer(
        layerIndex: Int?,
        layerCount: Int?,
        defaultBudget: Int
    ) -> Int {
        switch layerPolicy {
        case .all:
            return defaultBudget
        case .finalQuarter:
            guard let layerIndex, let layerCount, layerCount > 0 else {
                return defaultBudget
            }
            return layerIndex >= (layerCount * 3 / 4) ? defaultBudget : 0
        case .custom(let budgets):
            guard let layerIndex else { return defaultBudget }
            return budgets[layerIndex] ?? 0
        case .auto:
            guard let layerIndex, let layerCount, layerCount > 0 else {
                return defaultBudget
            }
            if layerIndex < layerCount / 4 {
                return 0
            }
            if layerIndex >= (layerCount * 3 / 4) {
                return defaultBudget
            }
            return layerIndex.isMultiple(of: 4) ? min(2048, defaultBudget) : 0
        }
    }

    private static func resolvedHotWindowTokens(requested: Int?, maxSize: Int?) -> Int {
        if let requested {
            return max(1, requested)
        }
        #if os(macOS)
            let defaultWindow = 16_384
        #else
            let defaultWindow = ProcessInfo.processInfo.isLowPowerModeEnabled ? 4096 : 8192
        #endif
        if let maxSize {
            return max(1, min(defaultWindow, maxSize))
        }
        return defaultWindow
    }

    private static func resolvedColdBlockTokens(requested: Int, maxSize: Int?) -> Int {
        let requested = max(1, requested)
        if let maxSize, maxSize < 32_768 {
            return min(requested, 512)
        }
        return requested
    }

    private static func modeOverriddenByEnvironment(
        _ mode: TurboQuantColdAttentionMode
    ) -> TurboQuantColdAttentionMode {
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FORCE_HYBRID_OFF") {
            return .off
        }
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FORCE_HYBRID_EXHAUSTIVE") {
            return .exhaustive
        }
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FORCE_HYBRID_SELECTED") {
            return .selected
        }
        return mode
    }

    private static func presetOverriddenByEnvironment(_ preset: TurboQuantPreset) -> TurboQuantPreset {
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FORCE_TURBO3_5") {
            return .turbo3_5
        }
        return preset
    }

    private static func arrays(for code: TurboQuantAttentionCode) -> [MLXArray] {
        [
            code.packedMagnitudes,
            code.signs,
            code.highPrecisionMask,
            code.residualSigns,
            code.scales,
        ]
    }

    private static func encodedMetadata<T: Encodable>(_ value: T) -> String {
        guard let data = try? JSONEncoder().encode(value) else { return "" }
        return data.base64EncodedString()
    }

    private static func decodedMetadata<T: Decodable>(
        _ type: T.Type,
        from value: String
    ) throws -> T {
        guard let data = Data(base64Encoded: value) else {
            throw KVCacheError(message: "Invalid HybridTurboQuantKVCache metadata encoding")
        }
        return try JSONDecoder().decode(type, from: data)
    }

    private static func maxKeyNormEstimate(_ keys: MLXArray) -> Float {
        let fp32 = keys.asType(.float32)
        return sqrt((fp32 * fp32).sum(axis: -1)).max().item(Float.self)
    }

    private static func keyMeanSummary(_ keys: MLXArray) -> [Float] {
        meanSummary(keys) ?? []
    }

    private static func meanSummary(_ array: MLXArray, maxDimensions: Int = 16) -> [Float]? {
        guard array.ndim >= 4 else { return nil }
        let batchCount = max(1, array.dim(0))
        let headCount = max(1, array.dim(1))
        let tokenCount = max(1, array.dim(2))
        let headDimension = max(1, array.dim(3))
        let summaryCount = min(maxDimensions, headDimension)
        guard summaryCount > 0 else { return nil }
        let values = array.asType(.float32).asArray(Float.self)
        guard values.count >= batchCount * headCount * tokenCount * headDimension else {
            return nil
        }
        let dimensions = (0 ..< summaryCount).map {
            min(headDimension - 1, ($0 * headDimension) / summaryCount)
        }
        let denominator = Float(batchCount * headCount * tokenCount)
        return dimensions.map { dimension in
            var sum = Float(0)
            for batch in 0 ..< batchCount {
                for head in 0 ..< headCount {
                    for token in 0 ..< tokenCount {
                        let index =
                            (((batch * headCount + head) * tokenCount + token) * headDimension)
                            + dimension
                        sum += values[index]
                    }
                }
            }
            return sum / max(denominator, 1)
        }
    }

    private static func vectorNorm(_ values: [Float]) -> Float {
        sqrt(values.reduce(Float(0)) { $0 + $1 * $1 })
    }

    private static func summarySimilarity(lhs: [Float], rhs: [Float]) -> Float {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }
        var dot = Float(0)
        var leftNorm = Float(0)
        var rightNorm = Float(0)
        for index in 0 ..< count {
            dot += lhs[index] * rhs[index]
            leftNorm += lhs[index] * lhs[index]
            rightNorm += rhs[index] * rhs[index]
        }
        let denominator = sqrt(leftNorm) * sqrt(rightNorm)
        guard denominator > Float.leastNonzeroMagnitude else { return 0 }
        return max(0, min(1, (dot / denominator + 1) * 0.5))
    }

    private static func descriptor(from entry: String) throws -> TurboQuantColdBlockDescriptor {
        let values = entry.split(separator: ":").compactMap { Int($0) }
        guard values.count == 7 else {
            throw KVCacheError(message: "Invalid HybridTurboQuantKVCache block descriptor")
        }
        return TurboQuantColdBlockDescriptor(
            blockID: values[0],
            startToken: values[1],
            endToken: values[2],
            compressedSlotStart: values[3],
            compressedSlotEnd: values[4],
            logicalTokenCount: values[5],
            recencyRank: 0,
            anchorFlags: TurboQuantColdBlockAnchorFlags(rawValue: values[6]),
            maxKeyNormEstimate: 0,
            relevanceScore: Float(values[2])
        )
    }

    private static func attentionCode(
        arrays: [MLXArray],
        descriptor: TurboQuantColdBlockDescriptor,
        preset: TurboQuantPreset,
        role: TurboQuantTensorRole,
        groupSize: Int,
        seed: UInt64,
        valueBits: Int?
    ) throws -> TurboQuantAttentionCode {
        guard arrays.count == 5, arrays[0].ndim == 5 else {
            throw KVCacheError(message: "Invalid HybridTurboQuantKVCache compressed code arrays")
        }
        let bitsetWords =
            arrays[1].ndim == 5
            ? arrays[1].dim(4)
            : max(1, (groupSize + 31) / 32)
        let scalesPerGroup =
            arrays[4].ndim == 5
            ? arrays[4].dim(4)
            : (role == .value ? 2 : 3)
        let layout = TurboQuantAttentionLayout(
            batchSize: arrays[0].dim(0),
            kvHeadCount: arrays[0].dim(1),
            capacity: arrays[0].dim(2),
            logicalLength: descriptor.logicalTokenCount,
            ringOffset: 0,
            pinnedPrefixLength: 0,
            headDimension: arrays[0].dim(3) * groupSize,
            groupsPerVector: arrays[0].dim(3),
            magnitudeWordsPerGroup: arrays[0].dim(4),
            bitsetWordsPerGroup: bitsetWords
        )
        return TurboQuantAttentionCode(
            layout: layout,
            preset: preset,
            role: role,
            groupSize: groupSize,
            seed: seed,
            valueBits: valueBits,
            scalesPerGroup: scalesPerGroup,
            packedMagnitudes: arrays[0],
            signs: arrays[1],
            highPrecisionMask: arrays[2],
            residualSigns: arrays[3],
            scales: arrays[4]
        )
    }
}

func turboQuantHybridAttentionThrowing(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: HybridTurboQuantKVCache,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) throws -> (output: MLXArray, state: AttentionKVState) {
    cache.updateHybrid(keys: keys, values: values)
    let hot = cache.rawHotState(defaultKeys: keys, defaultValues: values)
    let selection = cache.selectColdBlocks(query: queries)
    var segmentedFallbackReason: String?

    if queries.dim(2) == 1, sinks == nil {
        let coldSegments = cache.selectedColdCompressedSegments(selection: selection)
        if !coldSegments.isEmpty {
            do {
                let output = try MLX.turboQuantMetalSegmentedScaledDotProductAttention(
                    queries: queries,
                    rawKeys: hot.keys,
                    rawValues: hot.values,
                    coldSegments: coldSegments,
                    scale: scale,
                    outputDType: queries.dtype,
                    sparseVThreshold: cache.sparseValuePolicy.resolvedThreshold(
                        runtimeMode: .capacityTurboQuant,
                        contextLength: cache.offset
                    )
                )
                cache.recordFallbackReason(nil)
                cache.recordAttentionRoute(
                    cache.segmentedAttentionRoute(selection: selection),
                    selection: selection
                )
                let state = AttentionKVState.hybridTurboQuant(
                    keys: hot.keys,
                    values: hot.values,
                    selection: selection,
                    cache: cache
                )
                return (output, state)
            } catch {
                segmentedFallbackReason = "selected_segmented_attention_fallback:\(error)"
                cache.recordFallbackReason(segmentedFallbackReason)
            }
        }
    }

    let selectedCold = try cache.selectedColdState(
        selection: selection,
        outputDType: hot.keys.dtype
    )

    let attentionKeys: MLXArray
    let attentionValues: MLXArray
    if let selectedCold {
        attentionKeys = concatenated([selectedCold.keys, hot.keys], axis: 2)
        attentionValues = concatenated([selectedCold.values, hot.values], axis: 2)
    } else {
        attentionKeys = hot.keys
        attentionValues = hot.values
    }

    let adjustedMask = turboQuantHybridMask(
        original: mask,
        queryLength: queries.dim(2),
        keyLength: attentionKeys.dim(2)
    )
    let output = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: attentionKeys,
        values: attentionValues,
        scale: scale,
        mask: adjustedMask,
        sinks: sinks
    )
    if segmentedFallbackReason == nil {
        cache.recordFallbackReason(nil)
    }
    cache.recordAttentionRoute(
        cache.decodedAttentionRoute(selection: selection),
        selection: selection
    )
    let state = AttentionKVState.hybridTurboQuant(
        keys: attentionKeys,
        values: attentionValues,
        selection: selection,
        cache: cache
    )
    return (output, state)
}

func turboQuantHybridMask(
    original: MLXFast.ScaledDotProductAttentionMaskMode,
    queryLength: Int,
    keyLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    guard queryLength > 1 else { return .none }
    if queryLength == keyLength {
        return original
    }
    return .array(createCausalMask(n: queryLength, offset: max(0, keyLength - queryLength)))
}
