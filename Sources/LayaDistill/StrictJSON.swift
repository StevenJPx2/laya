import CryptoKit
import Foundation

public enum DistillError: Error, LocalizedError, Sendable, Equatable {
    case invalidSpec(String)
    case invalidData(String)
    case budget(String)
    case teacher(String)
    case training(String)
    case artifact(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSpec(let message): return "invalid task spec: \(message)"
        case .invalidData(let message): return "invalid data: \(message)"
        case .budget(let message): return "budget: \(message)"
        case .teacher(let message): return "teacher: \(message)"
        case .training(let message): return "training: \(message)"
        case .artifact(let message): return "artifact: \(message)"
        }
    }
}

struct AnyKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

extension Decoder {
    /// A keyed container that rejects fields not declared in `Keys`, so a typo in
    /// a spec or dataset fails loudly instead of silently falling back to a default.
    func strictContainer<Keys: CodingKey & CaseIterable>(keyedBy type: Keys.Type, context: String) throws -> KeyedDecodingContainer<Keys> {
        let allowed = Set(Keys.allCases.map(\.stringValue))
        let present = try container(keyedBy: AnyKey.self).allKeys.map(\.stringValue)
        let unknown = present.filter { !allowed.contains($0) }.sorted()

        guard unknown.isEmpty else {
            throw DistillError.invalidSpec("\(context): unknown field(s) \(unknown.joined(separator: ", "))")
        }

        return try container(keyedBy: type)
    }
}

enum Canonical {
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

        return encoder
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func sha256(_ text: String) -> String { sha256(Data(text.utf8)) }

    /// Uniform value in [0, 1) derived from a string; used for deterministic splits.
    static func unitInterval(_ text: String) -> Double {
        let digest = Array(SHA256.hash(data: Data(text.utf8)))
        let value = digest.prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }

        return Double(value >> 11) / Double(UInt64(1) << 53)
    }
}

func matches(_ value: String, _ pattern: String) -> Bool {
    value.range(of: pattern, options: .regularExpression) != nil
}
