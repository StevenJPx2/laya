import Foundation

public enum Prompt {
    private static let maxLen = 512
    private static let headMaxLen = 192
    public static let temperatures = [1.0, 1.0, 1.0]

    public static func serialize(_ value: JSONValue) -> String {
        if case .string(let v) = value { return v }
        return jsonText(value)
    }

    public static func renderedOptions(_ q: Question) throws -> [String] {
        switch q.type {
        case "choice":
            if case .array(let labels)? = q.criteria {
                return try labels.map {
                    guard case .string(let label) = $0 else { throw LayaError.invalid("choice labels must be strings") }

                    return label
                }
            }

            guard case .object(let rawValues)? = q.criteria, !rawValues.isEmpty else { throw LayaError.invalid("choice criteria must be a nonempty object or array") }
            return ordered(rawValues).map { key, value in
                switch value { case .null: return key; case .string(let s) where s.isEmpty: return key; default: return "\(key): \(render(value))" }
            }
        case "score":
            guard case .array(let values)? = q.criteria, !values.isEmpty else { throw LayaError.invalid("score criteria must be a nonempty array") }
            return values.enumerated().map { "level \($0.offset): \(render($0.element))" }
        case "noul":
            let values = q.criteria?.objectValue ?? []
            return ["false: \(render(values.first(where: { $0.0 == "false" })?.1, fallback: "no, the statement does not hold"))", "true: \(render(values.first(where: { $0.0 == "true" })?.1, fallback: "yes, the statement holds"))"]
        default: throw LayaError.invalid("unknown question type \(q.type)")
        }
    }

    private static func render(_ value: JSONValue?, fallback: String? = nil) -> String {
        guard let value, value != .null, value.stringValue != "" else { return fallback ?? "" }
        if case .string(let s) = value { return s }
        return jsonText(value)
    }

    private static func jsonText(_ value: JSONValue) -> String {
        switch value {
        case .string(let value): return try! String(decoding: JSONEncoder().encode(value), as: UTF8.self)
        case .number(let value): return value == value.rounded() && abs(value) < 1e15 ? String(Int64(value)) : String(value)
        case .bool(let value): return value ? "true" : "false"
        case .null: return "null"
        case .array(let values): return "[" + values.map(jsonText).joined(separator: ", ") + "]"
        case .object(let values): return "{" + ordered(values).map { key, value in "\(jsonText(.string(key))): \(jsonText(value))" }.joined(separator: ", ") + "}"
        }
    }

    static func ordered(_ values: [(String, JSONValue)]) -> [(String, JSONValue)] {
        values.sorted {
            let lhs = keyRank($0.0)
            let rhs = keyRank($1.0)

            if lhs.0 != rhs.0 { return lhs.0 < rhs.0 }
            if lhs.1 != rhs.1 { return lhs.1 < rhs.1 }

            return lhs.2 < rhs.2
        }
    }

    private static func keyRank(_ key: String) -> (Int, Int, String) {
        let preferred = ["from", "subject", "body", "message", "role", "content", "billing", "technical", "sales", "other", "false", "true"]

        if let index = preferred.firstIndex(of: key) { return (0, index, key) }
        if key.hasPrefix("department_"), let index = Int(key.dropFirst("department_".count)) { return (1, index, key) }

        return (2, 0, key)
    }

    public static func confidence(_ probabilities: [Double]) -> Double {
        guard probabilities.count > 1 else { return 1 }
        let entropy = probabilities.reduce(0) { total, p in p > 0 ? total - p * log(p) : total }
        return max(0, min(1, 1 - entropy / log(Double(probabilities.count))))
    }
}

private extension JSONValue {
    var objectValue: [(String, JSONValue)]? { if case .object(let v) = self { return v }; return nil }
}
