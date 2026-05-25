import Foundation

/// Process-wide MTP load policy.
///
/// The default keeps MTP-specific weights disabled so existing model loads keep
/// their current memory behavior. Apps that want MTP should set
/// ``retainMTPWeights`` before loading an MTP-capable model.
public enum MTPConfig {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var retainedMTPWeights = false

    public static var retainMTPWeights: Bool {
        get {
            lock.withLock { retainedMTPWeights }
        }
        set {
            lock.withLock { retainedMTPWeights = newValue }
        }
    }
}
