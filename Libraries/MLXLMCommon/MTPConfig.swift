import Foundation

/// Process-wide MTP load policy.
///
/// The default keeps MTP-specific weights disabled so existing model loads keep
/// their current memory behavior. Apps that want MTP should set
/// ``retainMTPWeights`` before loading an MTP-capable model.
public enum MTPConfig {
    public nonisolated(unsafe) static var retainMTPWeights = false
}
