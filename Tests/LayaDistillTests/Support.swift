import Foundation
import LayaCore
@testable import LayaDistill

/// In-process teacher transport: records requests and answers from a closure.
/// Tests never reach the network.
final class StubTransport: HTTPTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [URLRequest] = []
    private let respond: @Sendable (URLRequest) -> (Int, Data)

    init(respond: @escaping @Sendable (URLRequest) -> (Int, Data)) {
        self.respond = respond
    }

    var requests: [URLRequest] { lock.withLock { recorded } }

    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        lock.withLock { recorded.append(request) }

        let (status, body) = respond(request)
        return (body, HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!)
    }
}

enum Fixture {
    static func spec(provider: String = "anthropic", features: String = "hashed", maxUsd: Double = 1, maxRequests: Int = 500, extra: String = "") -> String {
        let teacher: String
        switch provider {
        case "dataset":
            teacher = #"{"provider": "dataset", "model": "gold"}"#
        case "openai-compatible":
            teacher = #"{"provider": "openai-compatible", "model": "local-model", "base_url": "http://127.0.0.1:8080/v1", "api_key_env": "LOCAL_KEY", "max_output_tokens": 16, "pricing": {"input_usd_per_mtok": 1, "output_usd_per_mtok": 2}}"#
        default:
            teacher = #"{"provider": "anthropic", "model": "test-teacher", "api_key_env": "TEST_TEACHER_KEY", "max_output_tokens": 16, "max_retries": 0, "pricing": {"input_usd_per_mtok": 10, "output_usd_per_mtok": 50}}"#
        }

        return """
        {
          "spec_version": 1, "name": "tiny", "version": "1.0.0",
          "instructions": "Gate the requested action.",
          "input": {"fields": [{"name": "request", "type": "string", "max_chars": 200}, {"name": "note", "type": "string", "required": false, "max_chars": 200}]},
          "labels": [{"name": "allow", "description": "safe"}, {"name": "deny", "description": "dangerous"}, {"name": "ask", "description": "needs confirmation"}],
          "abstain": {"label": "ask", "min_confidence": 0.4},
          "examples": [{"input": {"request": "read the changelog"}, "label": "allow"}],
          "teacher": \(teacher),
          "budget": {"max_requests": \(maxRequests), "max_usd": \(maxUsd)},
          "dataset": {"holdout_fraction": 0.3, "split_seed": "tiny-seed"},
          "student": {"features": "\(features)", "hash_dimensions": 1024, "epochs": 200, "learning_rate": 0.1, "l2": 0.0001}
          \(extra)
        }
        """
    }

    static func loadSpec(_ text: String) throws -> TaskSpec { try TaskSpec.decode(Data(text.utf8)) }

    /// Separable synthetic rows: the verb decides the label. Every verb cycles
    /// through the same objects, so objects carry no label signal.
    static func dataset(_ spec: TaskSpec, count: Int = 60) throws -> [DatasetExample] {
        let verbs = [("read", "allow"), ("list", "allow"), ("view", "allow"), ("delete", "deny"), ("wipe", "deny"), ("exfiltrate", "deny"), ("install", "ask"), ("deploy", "ask"), ("publish", "ask")]
        let objects = ["the logs", "project files", "a config", "the database", "user secrets", "a package", "release notes", "the cache", "test fixtures", "the build", "billing records", "a branch"]

        return try (0..<count).map { index in
            let (verb, gold) = verbs[index % verbs.count]
            let object = objects[(index / verbs.count) % objects.count]
            let input = try spec.input.parse(.object([("request", .string("\(verb) \(object) item \(index)"))]))

            return DatasetExample(id: "row-\(index)", input: input, group: nil, gold: gold)
        }
    }

    /// Anthropic-shaped reply choosing a label from the request text. Uses
    /// capitalized labels on purpose to exercise casing normalization.
    static func anthropicTeacher() -> StubTransport {
        StubTransport { request in
            let body = try! JSONSerialization.jsonObject(with: request.httpBody ?? Data()) as! [String: Any]
            let messages = body["messages"] as! [[String: Any]]
            let user = messages.last!["content"] as! String
            let label = ["delete", "wipe", "exfiltrate"].contains { user.contains($0) } ? "Deny" : ["install", "deploy", "publish"].contains { user.contains($0) } ? "ask" : "allow"
            let reply: [String: Any] = [
                "content": [["type": "text", "text": "{\"label\": \"\(label)\"}"]],
                "stop_reason": "end_turn", "usage": ["input_tokens": 120, "output_tokens": 6],
            ]

            return (200, try! JSONSerialization.data(withJSONObject: reply))
        }
    }

    static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("laya-distill-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)

        return url
    }
}
