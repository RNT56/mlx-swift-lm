import Dispatch
import Foundation

public struct TurboQuantTimingSnapshot: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var attentionWithCacheUpdateCalls: Int
    public var compressedCacheUpdateCalls: Int
    public var compressedAttentionCalls: Int
    public var exactPrefillAttentionCalls: Int
    public var dynamicCacheQuantizationCalls: Int
    public var attentionWithCacheUpdateSeconds: Double
    public var compressedCacheUpdateSeconds: Double
    public var compressedAttentionSeconds: Double
    public var exactPrefillAttentionSeconds: Double
    public var dynamicCacheQuantizationSeconds: Double
    public var accountedCompressedSeconds: Double
    public var unaccountedAttentionWithCacheUpdateSeconds: Double

    fileprivate init(
        enabled: Bool,
        counters: TurboQuantTimingCounters
    ) {
        self.enabled = enabled
        self.attentionWithCacheUpdateCalls = counters.attentionWithCacheUpdateCalls
        self.compressedCacheUpdateCalls = counters.compressedCacheUpdateCalls
        self.compressedAttentionCalls = counters.compressedAttentionCalls
        self.exactPrefillAttentionCalls = counters.exactPrefillAttentionCalls
        self.dynamicCacheQuantizationCalls = counters.dynamicCacheQuantizationCalls
        self.attentionWithCacheUpdateSeconds = counters.attentionWithCacheUpdateSeconds
        self.compressedCacheUpdateSeconds = counters.compressedCacheUpdateSeconds
        self.compressedAttentionSeconds = counters.compressedAttentionSeconds
        self.exactPrefillAttentionSeconds = counters.exactPrefillAttentionSeconds
        self.dynamicCacheQuantizationSeconds = counters.dynamicCacheQuantizationSeconds
        self.accountedCompressedSeconds =
            counters.compressedCacheUpdateSeconds
            + counters.compressedAttentionSeconds
            + counters.exactPrefillAttentionSeconds
        self.unaccountedAttentionWithCacheUpdateSeconds = max(
            0,
            counters.attentionWithCacheUpdateSeconds - self.accountedCompressedSeconds
        )
    }
}

private struct TurboQuantTimingCounters {
    var attentionWithCacheUpdateCalls: Int = 0
    var compressedCacheUpdateCalls: Int = 0
    var compressedAttentionCalls: Int = 0
    var exactPrefillAttentionCalls: Int = 0
    var dynamicCacheQuantizationCalls: Int = 0
    var attentionWithCacheUpdateSeconds: Double = 0
    var compressedCacheUpdateSeconds: Double = 0
    var compressedAttentionSeconds: Double = 0
    var exactPrefillAttentionSeconds: Double = 0
    var dynamicCacheQuantizationSeconds: Double = 0
}

public enum TurboQuantTimingScope: Sendable {
    case attentionWithCacheUpdate
    case compressedCacheUpdate
    case compressedAttention
    case exactPrefillAttention
    case dynamicCacheQuantization
}

private final class TurboQuantTimingCollector: @unchecked Sendable {
    static let shared = TurboQuantTimingCollector()

    private let lock = NSLock()
    private let environmentEnabled: Bool
    private var enabledOverride: Bool?
    private var counters = TurboQuantTimingCounters()

    private init() {
        let environment = ProcessInfo.processInfo.environment
        self.environmentEnabled = TurboQuantTimingCollector.truthy(
            environment["TQ_TURBOQUANT_TIMING"])
            || TurboQuantTimingCollector.truthy(environment["TURBOQUANT_TIMING"])
    }

    var isEnabled: Bool {
        lock.lock()
        let enabled = enabledOverride ?? environmentEnabled
        lock.unlock()
        return enabled
    }

    @discardableResult
    func setEnabledOverride(_ enabled: Bool?) -> Bool? {
        lock.lock()
        defer { lock.unlock() }
        let previous = enabledOverride
        enabledOverride = enabled
        return previous
    }

    func reset() {
        lock.lock()
        counters = TurboQuantTimingCounters()
        lock.unlock()
    }

    func snapshot() -> TurboQuantTimingSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return TurboQuantTimingSnapshot(
            enabled: enabledOverride ?? environmentEnabled,
            counters: counters
        )
    }

    func record(_ scope: TurboQuantTimingScope, seconds: Double) {
        lock.lock()
        defer { lock.unlock() }
        switch scope {
        case .attentionWithCacheUpdate:
            counters.attentionWithCacheUpdateCalls += 1
            counters.attentionWithCacheUpdateSeconds += seconds
        case .compressedCacheUpdate:
            counters.compressedCacheUpdateCalls += 1
            counters.compressedCacheUpdateSeconds += seconds
        case .compressedAttention:
            counters.compressedAttentionCalls += 1
            counters.compressedAttentionSeconds += seconds
        case .exactPrefillAttention:
            counters.exactPrefillAttentionCalls += 1
            counters.exactPrefillAttentionSeconds += seconds
        case .dynamicCacheQuantization:
            counters.dynamicCacheQuantizationCalls += 1
            counters.dynamicCacheQuantizationSeconds += seconds
        }
    }

    private static func truthy(_ raw: String?) -> Bool {
        guard let raw = raw?.lowercased() else { return false }
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }
}

public enum TurboQuantTiming {
    public static var isEnabled: Bool {
        TurboQuantTimingCollector.shared.isEnabled
    }

    @discardableResult
    public static func setEnabledOverride(_ enabled: Bool?) -> Bool? {
        TurboQuantTimingCollector.shared.setEnabledOverride(enabled)
    }

    public static func reset() {
        TurboQuantTimingCollector.shared.reset()
    }

    public static func snapshot() -> TurboQuantTimingSnapshot {
        TurboQuantTimingCollector.shared.snapshot()
    }

    @inline(__always)
    public static func start() -> UInt64? {
        guard isEnabled else { return nil }
        return DispatchTime.now().uptimeNanoseconds
    }

    @inline(__always)
    public static func record(
        _ scope: TurboQuantTimingScope,
        startedAt start: UInt64?
    ) {
        guard let start else { return }
        let end = DispatchTime.now().uptimeNanoseconds
        TurboQuantTimingCollector.shared.record(
            scope,
            seconds: Double(end - start) / 1_000_000_000
        )
    }

    @inline(__always)
    public static func measure<T>(
        _ scope: TurboQuantTimingScope,
        _ body: () throws -> T
    ) rethrows -> T {
        let start = start()
        defer { record(scope, startedAt: start) }
        return try body()
    }
}
