import Foundation
import MLX

public enum TurboQuantRuntimeFailure: Error, Codable, Sendable, Equatable,
    CustomStringConvertible
{
    case compressedAttentionUnavailable(String)
    case unsupportedBackend(String)
    case packedFallbackUnavailable(String)
    case decodedFallbackUnavailable(String)
    case unsupportedAttentionShape(String)
    case unsupportedAttentionMask(String)
    case unsupportedTensorDType(String)
    case cacheLayoutInvalid(String)
    case cacheLifecycleInvalid(String)
    case noBudgetedFallback(String)
    case fallbackBudgetExceeded(String)
    case modelProfileMismatch(String)

    public init(attentionStateError error: TurboQuantAttentionStateError) {
        switch error {
        case .compressedAttentionUnavailable(let message):
            self = .compressedAttentionUnavailable(message)
        case .noSemanticallyCorrectFallback(let message):
            self = .noBudgetedFallback(message)
        }
    }

    public init(_ error: Error) {
        if let runtimeFailure = error as? TurboQuantRuntimeFailure {
            self = runtimeFailure
        } else if let attentionError = error as? TurboQuantAttentionStateError {
            self.init(attentionStateError: attentionError)
        } else if let turboQuantError = error as? TurboQuantError {
            self.init(turboQuantError)
        } else {
            self = TurboQuantRuntimeFailure.classify(message: String(describing: error))
        }
    }

    public init(_ error: TurboQuantError) {
        switch error {
        case .unsupportedBackend(let backend, let message):
            self = .unsupportedBackend("\(backend.rawValue): \(message)")
        case .invalidGroupSize, .invalidMetalConfiguration, .invalidQualityInput,
            .invalidReferenceCode:
            self = TurboQuantRuntimeFailure.classify(message: error.description)
        }
    }

    public var pinesFailureKind: TurboQuantPinesFailureKind {
        switch self {
        case .compressedAttentionUnavailable, .unsupportedBackend:
            .turboQuantPathUnavailable
        case .packedFallbackUnavailable, .decodedFallbackUnavailable, .noBudgetedFallback:
            .turboQuantFallbackUnavailable
        case .fallbackBudgetExceeded:
            .fallbackBudgetExceeded
        case .unsupportedAttentionShape:
            .unsupportedAttentionShape
        case .unsupportedAttentionMask:
            .unsupportedAttentionMask
        case .unsupportedTensorDType:
            .unsupportedTensorDType
        case .cacheLayoutInvalid:
            .cacheLayoutInvalid
        case .cacheLifecycleInvalid:
            .cacheLifecycleInvalid
        case .modelProfileMismatch:
            .modelProfileMismatch
        }
    }

    public var description: String {
        switch self {
        case .compressedAttentionUnavailable(let message):
            "TurboQuant compressed attention unavailable: \(message)"
        case .unsupportedBackend(let message):
            "TurboQuant unsupported backend: \(message)"
        case .packedFallbackUnavailable(let message):
            "TurboQuant packed fallback unavailable: \(message)"
        case .decodedFallbackUnavailable(let message):
            "TurboQuant decoded fallback unavailable: \(message)"
        case .unsupportedAttentionShape(let message):
            "TurboQuant unsupported attention shape: \(message)"
        case .unsupportedAttentionMask(let message):
            "TurboQuant unsupported attention mask: \(message)"
        case .unsupportedTensorDType(let message):
            "TurboQuant unsupported tensor dtype: \(message)"
        case .cacheLayoutInvalid(let message):
            "TurboQuant cache layout invalid: \(message)"
        case .cacheLifecycleInvalid(let message):
            "TurboQuant cache lifecycle invalid: \(message)"
        case .noBudgetedFallback(let message):
            "TurboQuant has no budgeted fallback: \(message)"
        case .fallbackBudgetExceeded(let message):
            "TurboQuant fallback budget exceeded: \(message)"
        case .modelProfileMismatch(let message):
            "TurboQuant model profile mismatch: \(message)"
        }
    }

    private static func classify(message: String) -> TurboQuantRuntimeFailure {
        let lowercased = message.lowercased()
        if lowercased.contains("unsupported turboquant backend") {
            return .unsupportedBackend(message)
        }
        if lowercased.contains("resident bytes")
            && lowercased.contains("admitted budget")
        {
            return .fallbackBudgetExceeded(message)
        }
        if lowercased.contains("cache lifecycle") {
            return .cacheLifecycleInvalid(message)
        }
        if lowercased.contains("cache storage invalid")
            || lowercased.contains("compressed storage invalid")
            || lowercased.contains("layout")
            || lowercased.contains("ring offset")
        {
            return .cacheLayoutInvalid(message)
        }
        if lowercased.contains("mask") {
            return .unsupportedAttentionMask(message)
        }
        if lowercased.contains("dtype") || lowercased.contains("data type") {
            return .unsupportedTensorDType(message)
        }
        if lowercased.contains("shape")
            || lowercased.contains("head dimension")
            || lowercased.contains("unsupported")
        {
            return .unsupportedAttentionShape(message)
        }
        return .compressedAttentionUnavailable(message)
    }
}

public enum TurboQuantPinesFailureKind: String, Codable, Sendable, Equatable {
    case turboQuantPathUnavailable
    case turboQuantFallbackUnavailable
    case fallbackBudgetExceeded
    case unsupportedAttentionShape
    case unsupportedAttentionMask
    case unsupportedTensorDType
    case cacheLayoutInvalid
    case cacheLifecycleInvalid
    case modelProfileMismatch
    case mlxRuntimeFailure
}
