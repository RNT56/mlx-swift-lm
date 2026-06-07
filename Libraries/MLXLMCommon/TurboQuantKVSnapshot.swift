// Copyright (c) 2026 RNT56.

import Foundation
import MLX

public struct TurboQuantKVSnapshotIdentity: Hashable, Codable, Sendable {
    public var modelID: String
    public var modelRevision: String?
    public var tokenizerHash: String
    public var profileHash: String
    public var ropeConfigHash: String
    public var tokenPrefixHash: String
    public var fallbackContractHash: String?

    public init(
        modelID: String,
        modelRevision: String? = nil,
        tokenizerHash: String,
        profileHash: String,
        ropeConfigHash: String,
        tokenPrefixHash: String,
        fallbackContractHash: String? = nil
    ) {
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.tokenizerHash = tokenizerHash
        self.profileHash = profileHash
        self.ropeConfigHash = ropeConfigHash
        self.tokenPrefixHash = tokenPrefixHash
        self.fallbackContractHash = fallbackContractHash
    }
}

public struct TurboQuantKVSnapshotArrayDescriptor: Hashable, Codable, Sendable {
    public var name: String
    public var dtype: String
    public var shape: [Int]
    public var byteCount: Int64

    public init(name: String, dtype: String, shape: [Int], byteCount: Int64 = 0) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.byteCount = byteCount
    }

    public init(name: String, array: MLXArray) {
        self.init(
            name: name,
            dtype: String(describing: array.dtype),
            shape: array.shape,
            byteCount: Int64(array.nbytes)
        )
    }
}

public struct TurboQuantKVSnapshotManifest: Hashable, Codable, Sendable {
    public static let currentSchemaVersion = 4

    public var schemaVersion: Int
    public var snapshotID: UUID
    public var conversationID: UUID
    public var modelID: String
    public var modelRevision: String?
    public var tokenizerHash: String
    public var profileHash: String
    public var turboQuantLayoutVersion: Int
    public var ropeConfigHash: String
    public var tokenPrefixHash: String
    public var fallbackContractHash: String?
    public var logicalLength: Int
    public var pinnedPrefixLength: Int
    public var compressedKeyBytes: Int64
    public var compressedValueBytes: Int64
    public var blobByteCount: Int64
    public var encryptionKeyID: String
    public var createdAt: Date

    public var cacheKind: String
    public var kvCodec: TurboQuantKVCodec
    public var preset: String
    public var requestedBackend: String
    public var activeBackend: String
    public var quantizationMode: String
    public var keyBits: Int
    public var groupSize: Int
    public var valueBits: Int
    public var seed: UInt64
    public var mode: String
    public var capacity: Int
    public var ringOffset: Int
    public var batchSize: Int
    public var kvHeadCount: Int
    public var keyHeadDimension: Int
    public var valueHeadDimension: Int
    public var rotatingKeep: Int?
    public var rotatingStep: Int?
    public var requestedRuntimeMode: TurboQuantRuntimeMode?
    public var resolvedRuntimeMode: TurboQuantRuntimeMode?
    public var precisionPolicy: TurboQuantKVPrecisionPolicy?
    public var sparseValuePolicy: TurboQuantSparseValuePolicy?
    public var boundaryPolicy: TurboQuantBoundaryPolicy?
    public var boundaryProtectedLayerCount: Int
    public var boundaryProtectionReason: String?
    public var runtimeFallbackReason: String?
    public var selectedPath: String?
    public var fallbackReason: String?
    public var polarWHTKeyBytes: Int64
    public var polarWHTKeyPayloadAllocated: Bool
    public var polarWHTKeyBits: Int?
    public var polarWHTKeySeed: UInt64?
    public var polarWHTKeyPackedWordsPerVector: Int?
    public var polarWHTValueBytes: Int64
    public var polarWHTValuePayloadAllocated: Bool
    public var polarWHTValueBits: Int?
    public var polarWHTValueSeed: UInt64?
    public var polarWHTValuePackedWordsPerVector: Int?
    public var arrays: [TurboQuantKVSnapshotArrayDescriptor]

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        snapshotID: UUID = UUID(),
        conversationID: UUID,
        identity: TurboQuantKVSnapshotIdentity,
        turboQuantLayoutVersion: Int,
        logicalLength: Int,
        pinnedPrefixLength: Int,
        compressedKeyBytes: Int64,
        compressedValueBytes: Int64,
        blobByteCount: Int64,
        encryptionKeyID: String,
        createdAt: Date = Date(),
        cacheKind: String,
        kvCodec: TurboQuantKVCodec = .polarQJL,
        preset: String,
        requestedBackend: String,
        activeBackend: String,
        quantizationMode: String = QuantizationMode.affine.rawValue,
        keyBits: Int? = nil,
        groupSize: Int,
        valueBits: Int,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        mode: String = QuantizationMode.affine.rawValue,
        capacity: Int,
        ringOffset: Int,
        batchSize: Int,
        kvHeadCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int,
        rotatingKeep: Int? = nil,
        rotatingStep: Int? = nil,
        requestedRuntimeMode: TurboQuantRuntimeMode? = nil,
        resolvedRuntimeMode: TurboQuantRuntimeMode? = nil,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        sparseValuePolicy: TurboQuantSparseValuePolicy? = nil,
        boundaryPolicy: TurboQuantBoundaryPolicy? = nil,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil,
        runtimeFallbackReason: String? = nil,
        selectedPath: String? = nil,
        fallbackReason: String? = nil,
        polarWHTKeyBytes: Int64 = 0,
        polarWHTKeyPayloadAllocated: Bool = false,
        polarWHTKeyBits: Int? = nil,
        polarWHTKeySeed: UInt64? = nil,
        polarWHTKeyPackedWordsPerVector: Int? = nil,
        polarWHTValueBytes: Int64 = 0,
        polarWHTValuePayloadAllocated: Bool = false,
        polarWHTValueBits: Int? = nil,
        polarWHTValueSeed: UInt64? = nil,
        polarWHTValuePackedWordsPerVector: Int? = nil,
        arrays: [TurboQuantKVSnapshotArrayDescriptor]
    ) {
        self.schemaVersion = schemaVersion
        self.snapshotID = snapshotID
        self.conversationID = conversationID
        self.modelID = identity.modelID
        self.modelRevision = identity.modelRevision
        self.tokenizerHash = identity.tokenizerHash
        self.profileHash = identity.profileHash
        self.turboQuantLayoutVersion = turboQuantLayoutVersion
        self.ropeConfigHash = identity.ropeConfigHash
        self.tokenPrefixHash = identity.tokenPrefixHash
        self.fallbackContractHash = identity.fallbackContractHash
        self.logicalLength = logicalLength
        self.pinnedPrefixLength = pinnedPrefixLength
        self.compressedKeyBytes = compressedKeyBytes
        self.compressedValueBytes = compressedValueBytes
        self.blobByteCount = blobByteCount
        self.encryptionKeyID = encryptionKeyID
        self.createdAt = createdAt
        self.cacheKind = cacheKind
        self.kvCodec = kvCodec
        self.preset = preset
        self.requestedBackend = requestedBackend
        self.activeBackend = activeBackend
        self.quantizationMode = quantizationMode
        self.keyBits = keyBits ?? (TurboQuantPreset(rawValue: preset)?.effectiveBits ?? valueBits)
        self.groupSize = groupSize
        self.valueBits = valueBits
        self.seed = seed
        self.mode = mode
        self.capacity = capacity
        self.ringOffset = ringOffset
        self.batchSize = batchSize
        self.kvHeadCount = kvHeadCount
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
        self.rotatingKeep = rotatingKeep
        self.rotatingStep = rotatingStep
        self.requestedRuntimeMode = requestedRuntimeMode
        self.resolvedRuntimeMode = resolvedRuntimeMode
        self.precisionPolicy =
            precisionPolicy
            ?? TurboQuantKVPrecisionPolicy.legacy(
                preset: TurboQuantPreset(rawValue: preset) ?? .turbo3_5,
                valueBits: valueBits
            )
        self.sparseValuePolicy = sparseValuePolicy
        self.boundaryPolicy = boundaryPolicy ?? self.precisionPolicy?.boundary
        self.boundaryProtectedLayerCount = max(0, boundaryProtectedLayerCount)
        self.boundaryProtectionReason = boundaryProtectionReason
        self.runtimeFallbackReason = runtimeFallbackReason
        self.selectedPath = selectedPath
        self.fallbackReason = fallbackReason ?? runtimeFallbackReason
        self.polarWHTKeyBytes = max(0, polarWHTKeyBytes)
        self.polarWHTKeyPayloadAllocated = polarWHTKeyPayloadAllocated
        self.polarWHTKeyBits = polarWHTKeyBits
        self.polarWHTKeySeed = polarWHTKeySeed
        self.polarWHTKeyPackedWordsPerVector = polarWHTKeyPackedWordsPerVector
        self.polarWHTValueBytes = max(0, polarWHTValueBytes)
        self.polarWHTValuePayloadAllocated = polarWHTValuePayloadAllocated
        self.polarWHTValueBits = polarWHTValueBits
        self.polarWHTValueSeed = polarWHTValueSeed
        self.polarWHTValuePackedWordsPerVector = polarWHTValuePackedWordsPerVector
        self.arrays = arrays
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case snapshotID
        case conversationID
        case modelID
        case modelRevision
        case tokenizerHash
        case profileHash
        case turboQuantLayoutVersion
        case ropeConfigHash
        case tokenPrefixHash
        case fallbackContractHash
        case logicalLength
        case pinnedPrefixLength
        case compressedKeyBytes
        case compressedValueBytes
        case blobByteCount
        case encryptionKeyID
        case createdAt
        case cacheKind
        case kvCodec
        case preset
        case requestedBackend
        case activeBackend
        case quantizationMode
        case keyBits
        case groupSize
        case valueBits
        case seed
        case mode
        case capacity
        case ringOffset
        case batchSize
        case kvHeadCount
        case keyHeadDimension
        case valueHeadDimension
        case rotatingKeep
        case rotatingStep
        case requestedRuntimeMode
        case resolvedRuntimeMode
        case precisionPolicy
        case sparseValuePolicy
        case boundaryPolicy
        case boundaryProtectedLayerCount
        case boundaryProtectionReason
        case runtimeFallbackReason
        case selectedPath
        case fallbackReason
        case polarWHTKeyBytes
        case polarWHTKeyPayloadAllocated
        case polarWHTKeyBits
        case polarWHTKeySeed
        case polarWHTKeyPackedWordsPerVector
        case polarWHTValueBytes
        case polarWHTValuePayloadAllocated
        case polarWHTValueBits
        case polarWHTValueSeed
        case polarWHTValuePackedWordsPerVector
        case arrays
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let identity = TurboQuantKVSnapshotIdentity(
            modelID: try container.decode(String.self, forKey: .modelID),
            modelRevision: try container.decodeIfPresent(String.self, forKey: .modelRevision),
            tokenizerHash: try container.decode(String.self, forKey: .tokenizerHash),
            profileHash: try container.decode(String.self, forKey: .profileHash),
            ropeConfigHash: try container.decode(String.self, forKey: .ropeConfigHash),
            tokenPrefixHash: try container.decode(String.self, forKey: .tokenPrefixHash),
            fallbackContractHash: try container.decodeIfPresent(
                String.self,
                forKey: .fallbackContractHash
            )
        )
        let preset = try container.decode(String.self, forKey: .preset)
        let valueBits = try container.decode(Int.self, forKey: .valueBits)
        self.init(
            schemaVersion: try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1,
            snapshotID: try container.decode(UUID.self, forKey: .snapshotID),
            conversationID: try container.decode(UUID.self, forKey: .conversationID),
            identity: identity,
            turboQuantLayoutVersion: try container.decode(
                Int.self,
                forKey: .turboQuantLayoutVersion
            ),
            logicalLength: try container.decode(Int.self, forKey: .logicalLength),
            pinnedPrefixLength: try container.decode(Int.self, forKey: .pinnedPrefixLength),
            compressedKeyBytes: try container.decode(Int64.self, forKey: .compressedKeyBytes),
            compressedValueBytes: try container.decode(
                Int64.self,
                forKey: .compressedValueBytes
            ),
            blobByteCount: try container.decode(Int64.self, forKey: .blobByteCount),
            encryptionKeyID: try container.decode(String.self, forKey: .encryptionKeyID),
            createdAt: try container.decode(Date.self, forKey: .createdAt),
            cacheKind: try container.decode(String.self, forKey: .cacheKind),
            kvCodec: try container.decodeIfPresent(TurboQuantKVCodec.self, forKey: .kvCodec)
                ?? .polarQJL,
            preset: preset,
            requestedBackend: try container.decode(String.self, forKey: .requestedBackend),
            activeBackend: try container.decode(String.self, forKey: .activeBackend),
            quantizationMode: try container.decodeIfPresent(String.self, forKey: .quantizationMode)
                ?? (try container.decodeIfPresent(String.self, forKey: .mode)
                    ?? QuantizationMode.affine.rawValue),
            keyBits: try container.decodeIfPresent(Int.self, forKey: .keyBits),
            groupSize: try container.decode(Int.self, forKey: .groupSize),
            valueBits: valueBits,
            seed: try container.decodeIfPresent(UInt64.self, forKey: .seed)
                ?? defaultTurboQuantSeed,
            mode: try container.decodeIfPresent(String.self, forKey: .mode)
                ?? QuantizationMode.affine.rawValue,
            capacity: try container.decode(Int.self, forKey: .capacity),
            ringOffset: try container.decode(Int.self, forKey: .ringOffset),
            batchSize: try container.decode(Int.self, forKey: .batchSize),
            kvHeadCount: try container.decode(Int.self, forKey: .kvHeadCount),
            keyHeadDimension: try container.decode(Int.self, forKey: .keyHeadDimension),
            valueHeadDimension: try container.decode(Int.self, forKey: .valueHeadDimension),
            rotatingKeep: try container.decodeIfPresent(Int.self, forKey: .rotatingKeep),
            rotatingStep: try container.decodeIfPresent(Int.self, forKey: .rotatingStep),
            requestedRuntimeMode: try container.decodeIfPresent(
                TurboQuantRuntimeMode.self,
                forKey: .requestedRuntimeMode
            ),
            resolvedRuntimeMode: try container.decodeIfPresent(
                TurboQuantRuntimeMode.self,
                forKey: .resolvedRuntimeMode
            ),
            precisionPolicy: try container.decodeIfPresent(
                TurboQuantKVPrecisionPolicy.self,
                forKey: .precisionPolicy
            ),
            sparseValuePolicy: try container.decodeIfPresent(
                TurboQuantSparseValuePolicy.self,
                forKey: .sparseValuePolicy
            ),
            boundaryPolicy: try container.decodeIfPresent(
                TurboQuantBoundaryPolicy.self,
                forKey: .boundaryPolicy
            ),
            boundaryProtectedLayerCount: try container.decodeIfPresent(
                Int.self,
                forKey: .boundaryProtectedLayerCount
            ) ?? 0,
            boundaryProtectionReason: try container.decodeIfPresent(
                String.self,
                forKey: .boundaryProtectionReason
            ),
            runtimeFallbackReason: try container.decodeIfPresent(
                String.self,
                forKey: .runtimeFallbackReason
            ),
            selectedPath: try container.decodeIfPresent(String.self, forKey: .selectedPath),
            fallbackReason: try container.decodeIfPresent(String.self, forKey: .fallbackReason),
            polarWHTKeyBytes: try container.decodeIfPresent(
                Int64.self,
                forKey: .polarWHTKeyBytes
            ) ?? 0,
            polarWHTKeyPayloadAllocated: try container.decodeIfPresent(
                Bool.self,
                forKey: .polarWHTKeyPayloadAllocated
            ) ?? false,
            polarWHTKeyBits: try container.decodeIfPresent(
                Int.self,
                forKey: .polarWHTKeyBits
            ),
            polarWHTKeySeed: try container.decodeIfPresent(
                UInt64.self,
                forKey: .polarWHTKeySeed
            ),
            polarWHTKeyPackedWordsPerVector: try container.decodeIfPresent(
                Int.self,
                forKey: .polarWHTKeyPackedWordsPerVector
            ),
            polarWHTValueBytes: try container.decodeIfPresent(
                Int64.self,
                forKey: .polarWHTValueBytes
            ) ?? 0,
            polarWHTValuePayloadAllocated: try container.decodeIfPresent(
                Bool.self,
                forKey: .polarWHTValuePayloadAllocated
            ) ?? false,
            polarWHTValueBits: try container.decodeIfPresent(
                Int.self,
                forKey: .polarWHTValueBits
            ),
            polarWHTValueSeed: try container.decodeIfPresent(
                UInt64.self,
                forKey: .polarWHTValueSeed
            ),
            polarWHTValuePackedWordsPerVector: try container.decodeIfPresent(
                Int.self,
                forKey: .polarWHTValuePackedWordsPerVector
            ),
            arrays: try container.decode(
                [TurboQuantKVSnapshotArrayDescriptor].self,
                forKey: .arrays
            )
        )
    }

    public var identity: TurboQuantKVSnapshotIdentity {
        TurboQuantKVSnapshotIdentity(
            modelID: modelID,
            modelRevision: modelRevision,
            tokenizerHash: tokenizerHash,
            profileHash: profileHash,
            ropeConfigHash: ropeConfigHash,
            tokenPrefixHash: tokenPrefixHash,
            fallbackContractHash: fallbackContractHash
        )
    }
}

public struct TurboQuantKVSnapshotPayload: @unchecked Sendable {
    public var manifest: TurboQuantKVSnapshotManifest
    public var compressedArrays: [String: MLXArray]

    public init(
        manifest: TurboQuantKVSnapshotManifest,
        compressedArrays: [String: MLXArray]
    ) {
        self.manifest = manifest
        self.compressedArrays = compressedArrays
    }
}

public enum TurboQuantKVSnapshotError: Error, Equatable, CustomStringConvertible {
    case invalidSchemaVersion(Int)
    case identityMismatch(field: String, expected: String, actual: String)
    case invalidManifest(String)
    case invalidLayout(String)
    case invalidCacheKind(expected: String, actual: String)
    case missingArray(String)
    case invalidArrayMetadata(String)
    case unsupportedBackend(String)

    public var description: String {
        switch self {
        case .invalidSchemaVersion(let version):
            "unsupported TurboQuant KV snapshot schema version \(version)"
        case .identityMismatch(let field, let expected, let actual):
            "TurboQuant KV snapshot identity mismatch for \(field): expected \(expected), got \(actual)"
        case .invalidManifest(let message):
            "invalid TurboQuant KV snapshot manifest: \(message)"
        case .invalidLayout(let message):
            "invalid TurboQuant KV snapshot layout: \(message)"
        case .invalidCacheKind(let expected, let actual):
            "invalid TurboQuant KV snapshot cache kind: expected \(expected), got \(actual)"
        case .missingArray(let name):
            "TurboQuant KV snapshot missing compressed array \(name)"
        case .invalidArrayMetadata(let message):
            "invalid TurboQuant KV snapshot array metadata: \(message)"
        case .unsupportedBackend(let message):
            "unsupported TurboQuant KV snapshot backend: \(message)"
        }
    }
}

extension TurboQuantKVSnapshotManifest {
    public func validate(
        expectedIdentity: TurboQuantKVSnapshotIdentity,
        expectedCacheKind: String? = nil,
        expectedLayoutVersion: Int = TurboQuantAttentionLayout.currentVersion
    ) throws {
        guard schemaVersion >= 1, schemaVersion <= Self.currentSchemaVersion else {
            throw TurboQuantKVSnapshotError.invalidSchemaVersion(schemaVersion)
        }
        try validateIdentity(expectedIdentity)
        if let expectedCacheKind, cacheKind != expectedCacheKind {
            throw TurboQuantKVSnapshotError.invalidCacheKind(
                expected: expectedCacheKind,
                actual: cacheKind
            )
        }
        guard turboQuantLayoutVersion == expectedLayoutVersion else {
            throw TurboQuantKVSnapshotError.invalidLayout(
                "unsupported layout version \(turboQuantLayoutVersion)"
            )
        }
        guard !modelID.isEmpty, !tokenizerHash.isEmpty, !profileHash.isEmpty,
            !ropeConfigHash.isEmpty, !tokenPrefixHash.isEmpty, !encryptionKeyID.isEmpty
        else {
            throw TurboQuantKVSnapshotError.invalidManifest("identity fields must be non-empty")
        }
        guard logicalLength >= 0, capacity > 0, logicalLength <= capacity else {
            throw TurboQuantKVSnapshotError.invalidLayout(
                "logical length \(logicalLength) is outside capacity \(capacity)"
            )
        }
        guard pinnedPrefixLength >= 0, pinnedPrefixLength <= logicalLength else {
            throw TurboQuantKVSnapshotError.invalidLayout(
                "pinned prefix \(pinnedPrefixLength) is outside logical length \(logicalLength)"
            )
        }
        let ringCapacity = capacity - pinnedPrefixLength
        if ringCapacity == 0 {
            guard ringOffset == 0 else {
                throw TurboQuantKVSnapshotError.invalidLayout(
                    "ring offset must be zero when pinned prefix consumes capacity"
                )
            }
        } else {
            guard ringOffset >= 0, ringOffset < ringCapacity else {
                throw TurboQuantKVSnapshotError.invalidLayout(
                    "ring offset \(ringOffset) is outside rotating region \(ringCapacity)"
                )
            }
        }
        guard batchSize > 0, kvHeadCount > 0, keyHeadDimension > 0,
            valueHeadDimension > 0, groupSize > 0, valueBits > 0
        else {
            throw TurboQuantKVSnapshotError.invalidLayout(
                "batch, head, group, and value bit fields must be positive"
            )
        }
        guard compressedKeyBytes >= 0, compressedValueBytes >= 0, polarWHTKeyBytes >= 0,
            polarWHTValueBytes >= 0,
            blobByteCount
                >= compressedKeyBytes + compressedValueBytes + polarWHTKeyBytes
                    + polarWHTValueBytes
        else {
            throw TurboQuantKVSnapshotError.invalidManifest(
                "compressed byte counts are inconsistent"
            )
        }
        if polarWHTKeyPayloadAllocated {
            guard kvCodec == .polarWHT else {
                throw TurboQuantKVSnapshotError.invalidManifest(
                    "PolarWHT key payload requires polarWHT codec"
                )
            }
            guard polarWHTKeyBytes > 0,
                (polarWHTKeyBits ?? 0) > 0,
                polarWHTKeySeed != nil,
                (polarWHTKeyPackedWordsPerVector ?? 0) > 0
            else {
                throw TurboQuantKVSnapshotError.invalidManifest(
                    "PolarWHT key payload metadata is incomplete"
                )
            }
        }
        if polarWHTValuePayloadAllocated {
            guard kvCodec == .polarWHT else {
                throw TurboQuantKVSnapshotError.invalidManifest(
                    "PolarWHT value payload requires polarWHT codec"
                )
            }
            guard polarWHTValueBytes > 0,
                (polarWHTValueBits ?? 0) > 0,
                polarWHTValueSeed != nil,
                (polarWHTValuePackedWordsPerVector ?? 0) > 0
            else {
                throw TurboQuantKVSnapshotError.invalidManifest(
                    "PolarWHT value payload metadata is incomplete"
                )
            }
        }
    }

    public func validateArrayDescriptors(against arraysByName: [String: MLXArray]) throws {
        let required = Set(TurboQuantKVSnapshotArrayName.ordered)
        let optionalKey = Set(TurboQuantKVSnapshotArrayName.polarWHTKeyOrdered)
        let optionalValue = Set(TurboQuantKVSnapshotArrayName.polarWHTValueOrdered)
        let optional = optionalKey.union(optionalValue)
        let known = required.union(optional)
        let descriptorNames = Set(arrays.map(\.name))
        guard required.isSubset(of: descriptorNames),
            descriptorNames.isSubset(of: known)
        else {
            let missing = required.subtracting(descriptorNames).sorted().joined(separator: ",")
            let extra = descriptorNames.subtracting(known).sorted().joined(separator: ",")
            throw TurboQuantKVSnapshotError.invalidArrayMetadata(
                "expected compressed KV arrays plus optional PolarWHT payloads; missing [\(missing)] extra [\(extra)]"
            )
        }
        let descriptorKeySidecarNames = descriptorNames.intersection(optionalKey)
        guard descriptorKeySidecarNames.isEmpty || descriptorKeySidecarNames == optionalKey else {
            throw TurboQuantKVSnapshotError.invalidArrayMetadata(
                "PolarWHT key payload descriptors must be all-or-none"
            )
        }
        let descriptorValueSidecarNames = descriptorNames.intersection(optionalValue)
        guard descriptorValueSidecarNames.isEmpty || descriptorValueSidecarNames == optionalValue else {
            throw TurboQuantKVSnapshotError.invalidArrayMetadata(
                "PolarWHT value payload descriptors must be all-or-none"
            )
        }
        let payloadNames = Set(arraysByName.keys)
        guard payloadNames == descriptorNames else {
            let names = Set(arraysByName.keys)
            let missing = descriptorNames.subtracting(names).sorted().joined(separator: ",")
            let extra = names.subtracting(descriptorNames).sorted().joined(separator: ",")
            throw TurboQuantKVSnapshotError.invalidArrayMetadata(
                "payload arrays do not match manifest; missing [\(missing)] extra [\(extra)]"
            )
        }

        for descriptor in arrays {
            guard let array = arraysByName[descriptor.name] else {
                throw TurboQuantKVSnapshotError.missingArray(descriptor.name)
            }
            guard descriptor.shape == array.shape else {
                throw TurboQuantKVSnapshotError.invalidArrayMetadata(
                    "\(descriptor.name) shape \(array.shape) does not match manifest \(descriptor.shape)"
                )
            }
            let dtype = String(describing: array.dtype)
            guard descriptor.dtype == dtype else {
                throw TurboQuantKVSnapshotError.invalidArrayMetadata(
                    "\(descriptor.name) dtype \(dtype) does not match manifest \(descriptor.dtype)"
                )
            }
            guard descriptor.byteCount == Int64(array.nbytes) else {
                throw TurboQuantKVSnapshotError.invalidArrayMetadata(
                    "\(descriptor.name) byte count \(array.nbytes) does not match manifest \(descriptor.byteCount)"
                )
            }
        }
    }

    private func validateIdentity(_ expected: TurboQuantKVSnapshotIdentity) throws {
        try requireEqual("modelID", expected.modelID, modelID)
        try requireEqual("modelRevision", expected.modelRevision ?? "", modelRevision ?? "")
        try requireEqual("tokenizerHash", expected.tokenizerHash, tokenizerHash)
        try requireEqual("profileHash", expected.profileHash, profileHash)
        try requireEqual("ropeConfigHash", expected.ropeConfigHash, ropeConfigHash)
        try requireEqual("tokenPrefixHash", expected.tokenPrefixHash, tokenPrefixHash)
        try requireEqual(
            "fallbackContractHash",
            expected.fallbackContractHash ?? "",
            fallbackContractHash ?? ""
        )
    }

    private func requireEqual(_ field: String, _ expected: String, _ actual: String) throws {
        guard expected == actual else {
            throw TurboQuantKVSnapshotError.identityMismatch(
                field: field,
                expected: expected,
                actual: actual
            )
        }
    }
}

public enum TurboQuantKVSnapshotArrayName {
    public static let ordered = [
        "key.packedMagnitudes",
        "key.signs",
        "key.highPrecisionMask",
        "key.residualSigns",
        "key.scales",
        "value.packedMagnitudes",
        "value.signs",
        "value.highPrecisionMask",
        "value.residualSigns",
        "value.scales",
    ]

    public static let polarWHTValuePackedIndices = "polarWHTValue.packedIndices"
    public static let polarWHTValueNorms = "polarWHTValue.norms"
    public static let polarWHTValueOrdered = [
        polarWHTValuePackedIndices,
        polarWHTValueNorms,
    ]

    public static let polarWHTKeyPackedIndices = "polarWHTKey.packedIndices"
    public static let polarWHTKeyNorms = "polarWHTKey.norms"
    public static let polarWHTKeyOrdered = [
        polarWHTKeyPackedIndices,
        polarWHTKeyNorms,
    ]

    public static let allKnown = ordered + polarWHTKeyOrdered + polarWHTValueOrdered
}
