import Foundation

public struct Question: Codable, Sendable {
    public let type: String
    public let instructions: JSONValue
    public let criteria: JSONValue?

    public init(type: String, instructions: JSONValue, criteria: JSONValue?) {
        self.type = type
        self.instructions = instructions
        self.criteria = criteria
    }
}

public struct PredictRequest: Codable, Sendable {
    public let state: JSONValue
    public let questions: [String: Question]

    public init(state: JSONValue, questions: [String: Question]) {
        self.state = state
        self.questions = questions
    }
}

/// Wire request: `{"op":"health"}` for a health check, otherwise a predict
/// request carrying `state` and `questions`.
public struct DaemonRequest: Decodable, Sendable {
    public let op: String?
    public let state: JSONValue?
    public let questions: [String: Question]?
}

public struct Answer: Codable, Sendable {
    public let type: String
    public var confidence: Double
    public let action: Action
    public var choice: String? = nil
    public var probabilities: [String: Double]? = nil
    public var score: Double? = nil
    public var legend: [String: JSONValue]? = nil
    public var noul: Double? = nil
}

public struct Action: Codable, Sendable { public let act_probability: Double }
public struct Usage: Codable, Sendable { public let input_tokens: Int; public let output_tokens: Int }
public struct PredictResponse: Codable, Sendable {
    public let model: String
    public let answers: [String: Answer]
    public let usage: Usage
}

public struct HealthResponse: Codable, Sendable {
    public let status: String
    public let model: String
    public let warm: Bool
}

public enum LayaError: Error, LocalizedError, Sendable {
    case invalid(String), modelUnavailable(String), protocolError(String)
    public var errorDescription: String? {
        switch self { case .invalid(let s), .modelUnavailable(let s), .protocolError(let s): return s }
    }
}
