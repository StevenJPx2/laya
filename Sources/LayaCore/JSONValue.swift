import Foundation

public enum JSONValue: Codable, Sendable, Equatable {
    case string(String), number(Double), bool(Bool), object([(String, JSONValue)]), array([JSONValue]), null

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let v = try? c.decode(Bool.self) { self = .bool(v); return }
        if let v = try? c.decode(Double.self) { self = .number(v); return }
        if let v = try? c.decode(String.self) { self = .string(v); return }
        if let keyed = try? decoder.container(keyedBy: DynamicKey.self) {
            self = .object(keyed.allKeys.compactMap { key in try? (key.stringValue, keyed.decode(JSONValue.self, forKey: key)) })
            return
        }
        self = .array(try c.decode([JSONValue].self))
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .object(let values):
            var keyed = encoder.container(keyedBy: DynamicKey.self)
            for (key, value) in values { try keyed.encode(value, forKey: DynamicKey(stringValue: key)!) }
        case .array(let v): try c.encode(v)
        case .null: try c.encodeNil()
        }
    }

    public var stringValue: String? { if case .string(let v) = self { return v }; return nil }

    public static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case (.string(let a), .string(let b)): return a == b
        case (.number(let a), .number(let b)): return a == b
        case (.bool(let a), .bool(let b)): return a == b
        case (.null, .null): return true
        case (.array(let a), .array(let b)): return a == b
        case (.object(let a), .object(let b)): return a.count == b.count && zip(a, b).allSatisfy { pair in pair.0.0 == pair.1.0 && pair.0.1 == pair.1.1 }
        default: return false
        }
    }
}

private struct DynamicKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}
