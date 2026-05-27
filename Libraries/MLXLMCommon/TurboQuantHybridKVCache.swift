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
    public var maxKeyNormEstimate: Float
    public var relevanceScore: Float

    public var isAnchor: Bool {
        !anchorFlags.isEmpty
    }
}

public struct TurboQuantColdSelection: Hashable, Codable, Sendable {
    public var selectedBlockIDs: [Int]
    public var selectedTokenBudget: Int
    public var selectedTokenCount: Int
    public var confidence: Float
    public var reasonFlags: [String]
    public var requiresExhaustiveFallback: Bool

    public static let empty = TurboQuantColdSelection(
        selectedBlockIDs: [],
        selectedTokenBudget: 0,
        selectedTokenCount: 0,
        confidence: 1,
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

public struct TurboQuantHybridDiagnostics: Equatable, Codable, Sendable {
    public var route: String
    public var hotTokens: Int
    public var coldBlockCount: Int
    public var selectedColdTokens: Int
    public var selectedColdBlocks: [Int]
    public var selectorConfidence: Float
    public var coldBudgetTokens: Int
    public var maxColdBudgetTokens: Int
    public var fallbackReason: String?
    public var fullScanFallbackCount: Int
    public var lastLayerColdBudget: Int?
}

private struct TurboQuantColdBlock {
    var descriptor: TurboQuantColdBlockDescriptor
    var keyCode: TurboQuantAttentionCode
    var valueCode: TurboQuantAttentionCode
}

private struct TurboQuantHybridCheckpoint {
    var offset: Int
    var hotKeys: MLXArray?
    var hotValues: MLXArray?
    var blocks: [TurboQuantColdBlock]
    var nextBlockID: Int
    var fullScanFallbackCount: Int
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
    public private(set) var hotWindowTokens: Int
    public private(set) var coldBlockTokens: Int
    public private(set) var coldBudgetTokens: Int
    public private(set) var maxColdBudgetTokens: Int
    public var coldAttentionMode: TurboQuantColdAttentionMode
    public var layerPolicy: TurboQuantLayerPolicy

    private var hotKeys: MLXArray?
    private var hotValues: MLXArray?
    private var coldBlocks: [TurboQuantColdBlock] = []
    private var nextBlockID = 0
    private var fullScanFallbackCount = 0
    private var lastSealFailure: String?

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
        layerCount: Int? = nil
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
        self.diagnostics = TurboQuantHybridDiagnostics(
            route: "hybrid",
            hotTokens: 0,
            coldBlockCount: 0,
            selectedColdTokens: 0,
            selectedColdBlocks: [],
            selectorConfidence: 1,
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
            offset = 0
            nextBlockID = 0
            refreshDiagnostics(selection: .empty, fallbackReason: nil)
        }
    }

    public override var metaState: [String] {
        get {
            [
                "HybridTurboQuantKVCache.v1",
                "offset=\(offset)",
                "maxSize=\(maxContextLength.map(String.init) ?? "None")",
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
                "valueBits=\(valueBits.map(String.init) ?? "None")",
                "layerIndex=\(layerIndex.map(String.init) ?? "None")",
                "layerCount=\(layerCount.map(String.init) ?? "None")",
                "hotStateCount=\(hotKeys == nil ? 0 : 2)",
                "coldBlockCount=\(coldBlocks.count)",
            ] + coldBlocks.map { block in
                let descriptor = block.descriptor
                return [
                    descriptor.blockID,
                    descriptor.startToken,
                    descriptor.endToken,
                    descriptor.compressedSlotStart,
                    descriptor.compressedSlotEnd,
                    descriptor.logicalTokenCount,
                    descriptor.anchorFlags.rawValue,
                ].map(String.init).joined(separator: ":")
            }
        }
        set {
            guard newValue.isEmpty || (newValue.count == 1 && newValue[0].isEmpty) else {
                fatalError("HybridTurboQuantKVCache metaState restore is not supported without state decoding.")
            }
        }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        appendHot(keys: keys, values: values)
        do {
            try sealReadyColdBlocks()
        } catch {
            lastSealFailure = String(describing: error)
        }
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
        let selection = Self.selectColdBlocks(
            descriptors: coldBlockDescriptors,
            mode: coldAttentionMode,
            budgetTokens: budget,
            maxBudgetTokens: maxColdBudgetTokens
        )
        lastSelection = selection
        if selection.requiresExhaustiveFallback {
            fullScanFallbackCount += 1
        }
        refreshDiagnostics(selection: selection, fallbackReason: lastSealFailure)
        return selection
    }

    public static func selectColdBlocks(
        descriptors: [TurboQuantColdBlockDescriptor],
        mode: TurboQuantColdAttentionMode,
        budgetTokens: Int,
        maxBudgetTokens: Int
    ) -> TurboQuantColdSelection {
        guard !descriptors.isEmpty else { return .empty }
        guard mode != .off, budgetTokens > 0 || mode == .exhaustive else {
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
            let tokenCount = descriptors.reduce(0) { $0 + $1.logicalTokenCount }
            return TurboQuantColdSelection(
                selectedBlockIDs: descriptors.map(\.blockID),
                selectedTokenBudget: tokenCount,
                selectedTokenCount: tokenCount,
                confidence: 1,
                reasonFlags: ["exhaustive"],
                requiresExhaustiveFallback: false
            )
        }

        let cappedBudget = min(maxBudgetTokens, max(0, budgetTokens))
        var selected: [TurboQuantColdBlockDescriptor] = []
        var selectedIDs = Set<Int>()
        var tokens = 0
        var reasons: [String] = []

        func add(_ descriptor: TurboQuantColdBlockDescriptor, reason: String) {
            guard !selectedIDs.contains(descriptor.blockID) else { return }
            guard tokens + descriptor.logicalTokenCount <= cappedBudget || descriptor.isAnchor else {
                return
            }
            selectedIDs.insert(descriptor.blockID)
            selected.append(descriptor)
            tokens += descriptor.logicalTokenCount
            if !reasons.contains(reason) {
                reasons.append(reason)
            }
        }

        for descriptor in descriptors where descriptor.isAnchor {
            add(descriptor, reason: "anchor")
        }

        if let nearest = descriptors.max(by: { $0.endToken < $1.endToken }) {
            add(nearest, reason: "nearest")
        }

        let scored = descriptors.sorted {
            if $0.relevanceScore == $1.relevanceScore {
                return $0.endToken > $1.endToken
            }
            return $0.relevanceScore > $1.relevanceScore
        }
        for descriptor in scored {
            guard tokens < cappedBudget else { break }
            add(descriptor, reason: "score")
        }

        selected.sort { $0.startToken < $1.startToken }
        let coverage = descriptors.isEmpty ? 1 : Float(selected.count) / Float(descriptors.count)
        let requiresExhaustive =
            selected.isEmpty && !descriptors.isEmpty && cappedBudget < descriptors[0].logicalTokenCount
        return TurboQuantColdSelection(
            selectedBlockIDs: selected.map(\.blockID),
            selectedTokenBudget: cappedBudget,
            selectedTokenCount: tokens,
            confidence: max(0.05, min(1, coverage)),
            reasonFlags: reasons.isEmpty ? ["budget_empty"] : reasons,
            requiresExhaustiveFallback: requiresExhaustive
        )
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
        for index in coldBlocks.indices {
            let descriptor = coldBlocks[index].descriptor
            guard descriptor.startToken < tokenRange.upperBound,
                tokenRange.lowerBound < descriptor.endToken
            else {
                continue
            }
            coldBlocks[index].descriptor.anchorFlags.formUnion(flags)
            coldBlocks[index].descriptor.relevanceScore += 10
        }
        refreshDiagnostics(selection: lastSelection, fallbackReason: lastSealFailure)
    }

    public override var isTrimmable: Bool {
        true
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0, offset > 0 else { return 0 }
        let targetTrim = min(n, offset)
        var remaining = targetTrim

        if let hotKeys, let hotValues, remaining > 0 {
            let hotLength = hotKeys.dim(2)
            let keepHot = max(0, hotLength - remaining)
            if keepHot == 0 {
                self.hotKeys = nil
                self.hotValues = nil
                remaining -= hotLength
            } else {
                self.hotKeys = hotKeys[.ellipsis, ..<keepHot, 0...]
                self.hotValues = hotValues[.ellipsis, ..<keepHot, 0...]
                remaining = 0
            }
        }

        while remaining > 0, let last = coldBlocks.last {
            let count = last.descriptor.logicalTokenCount
            if remaining >= count {
                coldBlocks.removeLast()
                remaining -= count
            } else {
                // Partial compressed block rollback is intentionally conservative:
                // drop the whole block and report the extra trim to keep state coherent.
                coldBlocks.removeLast()
                remaining = 0
            }
        }

        let trimmed = targetTrim - remaining
        offset = max(0, offset - trimmed)
        refreshRecency()
        refreshDiagnostics(selection: .empty, fallbackReason: lastSealFailure)
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
            layerCount: layerCount
        )
        copied.offset = offset
        copied.hotKeys = hotKeys
        copied.hotValues = hotValues
        copied.coldBlocks = coldBlocks
        copied.nextBlockID = nextBlockID
        copied.lastSelection = lastSelection
        copied.fullScanFallbackCount = fullScanFallbackCount
        copied.lastSealFailure = lastSealFailure
        copied.diagnostics = diagnostics
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
            hotKeys: hotKeys,
            hotValues: hotValues,
            blocks: coldBlocks,
            nextBlockID: nextBlockID,
            fullScanFallbackCount: fullScanFallbackCount,
            diagnostics: diagnostics
        )
    }

    public func restoreCompressedUpdateCheckpoint(_ checkpoint: Any) {
        guard let checkpoint = checkpoint as? TurboQuantHybridCheckpoint else { return }
        offset = checkpoint.offset
        hotKeys = checkpoint.hotKeys
        hotValues = checkpoint.hotValues
        coldBlocks = checkpoint.blocks
        nextBlockID = checkpoint.nextBlockID
        fullScanFallbackCount = checkpoint.fullScanFallbackCount
        diagnostics = checkpoint.diagnostics
    }

    static func restoreFromMetaState(
        state: [MLXArray],
        metaState: [String]
    ) throws -> HybridTurboQuantKVCache {
        guard metaState.first == "HybridTurboQuantKVCache.v1" else {
            throw KVCacheError(message: "Invalid HybridTurboQuantKVCache metaState version")
        }
        let metadata = Dictionary(
            uniqueKeysWithValues: metaState.compactMap { entry -> (String, String)? in
                guard let equalIndex = entry.firstIndex(of: "=") else { return nil }
                return (
                    String(entry[..<equalIndex]),
                    String(entry[entry.index(after: equalIndex)...])
                )
            }
        )
        let blockDescriptorEntries = metaState.filter {
            !$0.contains("=") && $0 != "HybridTurboQuantKVCache.v1"
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
            layerCount: metadata["layerCount"].flatMap { $0 == "None" ? nil : Int($0) }
        )
        cache.offset = metadata["offset"].flatMap(Int.init) ?? 0

        var stateIndex = 0
        let hotStateCount = metadata["hotStateCount"].flatMap(Int.init) ?? 0
        if hotStateCount == 2 {
            guard state.count >= 2 else {
                throw KVCacheError(message: "HybridTurboQuantKVCache hot state is truncated")
            }
            cache.hotKeys = state[0]
            cache.hotValues = state[1]
            stateIndex = 2
        }

        let descriptors = try blockDescriptorEntries.map(Self.descriptor(from:))
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

    private func appendHot(keys: MLXArray, values: MLXArray) {
        if let currentKeys = hotKeys, let currentValues = hotValues {
            hotKeys = concatenated([currentKeys, keys], axis: 2)
            hotValues = concatenated([currentValues, values], axis: 2)
        } else {
            hotKeys = keys
            hotValues = values
        }
        offset += keys.dim(2)
    }

    private func sealReadyColdBlocks() throws {
        guard activeBackend == .metalPolarQJL,
            TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec
        else {
            return
        }
        while let hotKeys, let hotValues,
            hotKeys.dim(2) > hotWindowTokens + coldBlockTokens
        {
            let blockKeys = hotKeys[.ellipsis, ..<coldBlockTokens, 0...]
            let blockValues = hotValues[.ellipsis, ..<coldBlockTokens, 0...]
            try sealBlock(keys: blockKeys, values: blockValues)
            let remainingStart = coldBlockTokens
            self.hotKeys = hotKeys[.ellipsis, remainingStart..., 0...]
            self.hotValues = hotValues[.ellipsis, remainingStart..., 0...]
        }
    }

    private func sealBlock(keys: MLXArray, values: MLXArray) throws {
        let tokenCount = keys.dim(2)
        let startToken = coldTokenCount
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
        let descriptor = TurboQuantColdBlockDescriptor(
            blockID: nextBlockID,
            startToken: startToken,
            endToken: endToken,
            compressedSlotStart: 0,
            compressedSlotEnd: tokenCount,
            logicalTokenCount: tokenCount,
            recencyRank: 0,
            anchorFlags: [],
            maxKeyNormEstimate: 0,
            relevanceScore: Float(endToken)
        )
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
        for index in coldBlocks.indices {
            coldBlocks[index].descriptor.recencyRank = coldBlocks.count - index
            coldBlocks[index].descriptor.relevanceScore =
                Float(coldBlocks[index].descriptor.endToken)
                + (coldBlocks[index].descriptor.isAnchor ? 10_000 : 0)
        }
    }

    private func refreshDiagnostics(
        selection: TurboQuantColdSelection,
        fallbackReason: String?
    ) {
        diagnostics = TurboQuantHybridDiagnostics(
            route: coldBlocks.isEmpty ? "hybrid_raw_hot" : "hybrid_selected_cold",
            hotTokens: rawHotLength,
            coldBlockCount: coldBlocks.count,
            selectedColdTokens: selection.selectedTokenCount,
            selectedColdBlocks: selection.selectedBlockIDs,
            selectorConfidence: selection.confidence,
            coldBudgetTokens: selection.selectedTokenBudget,
            maxColdBudgetTokens: maxColdBudgetTokens,
            fallbackReason: fallbackReason,
            fullScanFallbackCount: fullScanFallbackCount,
            lastLayerColdBudget: selection.selectedTokenBudget
        )
    }

    private func coldBudgetForLayer(
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
                    outputDType: queries.dtype
                )
                cache.recordFallbackReason(nil)
                let state = AttentionKVState.hybridTurboQuant(
                    keys: hot.keys,
                    values: hot.values,
                    selection: selection,
                    cache: cache
                )
                return (output, state)
            } catch {
                cache.recordFallbackReason("segmented_attention_fallback:\(error)")
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
    let state = AttentionKVState.hybridTurboQuant(
        keys: attentionKeys,
        values: attentionValues,
        selection: selection,
        cache: cache
    )
    return (output, state)
}

private func turboQuantHybridMask(
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
