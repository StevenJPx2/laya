import Foundation
import LayaCore

public protocol HTTPTransport: Sendable {
    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse)
}

public struct URLSessionTransport: HTTPTransport {
    public init() {}

    public func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse else { throw DistillError.teacher("non-HTTP response") }

        return (data, http)
    }
}

/// Outcome of labeling one example. Token counts are the provider's reported usage.
public struct TeacherResult: Sendable {
    public let label: String?
    public let status: LabelStatus
    public let reason: String?
    public let inputTokens: Int
    public let outputTokens: Int
}

enum TeacherPrompt {
    static func system(_ spec: TaskSpec) -> String {
        var lines = [
            "You label data for training a classifier. Choose exactly one label for the input.",
            "Task: \(spec.instructions)",
            "Labels:",
        ]
        lines += spec.labels.map { "- \($0.name): \($0.description)" }

        if let abstain = spec.abstain {
            lines.append("If the input is ambiguous or lacks the information needed to decide, answer \"\(abstain.label)\".")
        }
        if !spec.examples.isEmpty {
            lines.append("Examples:")
            for example in spec.examples {
                let rendered = (try? spec.input.parse(example.input).rendered) ?? ""
                lines.append("Input:\n\(rendered)\nLabel: \(example.label)")
            }
        }

        lines.append("Respond only with JSON of the form {\"label\": \"<label>\"}.")

        return lines.joined(separator: "\n")
    }

    static func schema(_ spec: TaskSpec) -> [String: Any] {
        [
            "type": "object",
            "properties": ["label": ["type": "string", "enum": spec.labelNames]],
            "required": ["label"],
            "additionalProperties": false,
        ]
    }

    /// Conservative pre-flight token estimate (about 3 UTF-8 bytes per token plus
    /// framing). Only used to refuse or stop before spending; real usage is billed.
    static func estimatedInputTokens(_ spec: TaskSpec, _ input: TaskInput) -> Int {
        (system(spec).utf8.count + input.rendered.utf8.count) / 3 + 32
    }
}

public struct TeacherClient: Sendable {
    let spec: TaskSpec
    let apiKey: String
    let transport: HTTPTransport
    let sleep: @Sendable (Double) async -> Void

    public init(spec: TaskSpec, apiKey: String, transport: HTTPTransport = URLSessionTransport(),
                sleep: @escaping @Sendable (Double) async -> Void = { try? await Task.sleep(nanoseconds: UInt64($0 * 1e9)) }) {
        self.spec = spec
        self.apiKey = apiKey
        self.transport = transport
        self.sleep = sleep
    }

    public func label(_ input: TaskInput) async throws -> TeacherResult {
        let request = try makeRequest(input)
        var attempt = 0

        while true {
            let data: Data
            let response: HTTPURLResponse

            do {
                (data, response) = try await transport.send(request)
            } catch {
                guard attempt < spec.teacher.maxRetries else {
                    return TeacherResult(label: nil, status: .error, reason: "transport: \((error as? URLError)?.code.rawValue ?? -1)", inputTokens: 0, outputTokens: 0)
                }

                attempt += 1
                await sleep(Double(1 << attempt))
                continue
            }

            if response.statusCode == 200 { return try parse(data) }

            let retryable = response.statusCode == 429 || response.statusCode >= 500
            guard retryable, attempt < spec.teacher.maxRetries else {
                return TeacherResult(label: nil, status: .error, reason: "http \(response.statusCode): \(Redactor.errorSummary(data))", inputTokens: 0, outputTokens: 0)
            }

            attempt += 1
            let hinted = Double(response.value(forHTTPHeaderField: "retry-after") ?? "") ?? 0
            await sleep(min(30, max(hinted, Double(1 << attempt))))
        }
    }

    func makeRequest(_ input: TaskInput) throws -> URLRequest {
        let teacher = spec.teacher
        let system = TeacherPrompt.system(spec)
        let user = "Input:\n\(input.rendered)"
        var request: URLRequest
        var body: [String: Any]

        switch teacher.provider {
        case .anthropic:
            request = URLRequest(url: endpoint(default: "https://api.anthropic.com", path: "/v1/messages"))
            request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
            request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
            body = [
                "model": teacher.model, "max_tokens": teacher.maxOutputTokens, "system": system,
                "messages": [["role": "user", "content": user]],
                "output_config": ["format": ["type": "json_schema", "schema": TeacherPrompt.schema(spec)]],
            ]
        case .openaiCompatible:
            request = URLRequest(url: endpoint(default: "", path: "/chat/completions"))
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "authorization")
            body = [
                "model": teacher.model, teacher.maxTokensField: teacher.maxOutputTokens,
                "messages": [["role": "system", "content": system], ["role": "user", "content": user]],
                "response_format": ["type": "json_schema", "json_schema": ["name": "label", "strict": true, "schema": TeacherPrompt.schema(spec)]],
            ]
        case .dataset:
            throw DistillError.teacher("the dataset teacher does not make requests")
        }

        request.httpMethod = "POST"
        request.timeoutInterval = TimeInterval(teacher.timeoutSeconds)
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])

        return request
    }

    private func endpoint(default fallback: String, path: String) -> URL {
        let base = (spec.teacher.baseURL ?? fallback).trimmingCharacters(in: CharacterSet(charactersIn: "/"))

        return URL(string: base + path)!
    }

    func parse(_ data: Data) throws -> TeacherResult {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return TeacherResult(label: nil, status: .error, reason: "response is not JSON", inputTokens: 0, outputTokens: 0)
        }

        let reply = spec.teacher.provider == .anthropic ? Self.anthropicReply(object) : Self.chatReply(object)

        if let refusal = reply.refusal {
            return TeacherResult(label: nil, status: .refused, reason: refusal, inputTokens: reply.input, outputTokens: reply.output)
        }

        let resolved = resolve(reply.text)

        return TeacherResult(label: resolved.label, status: resolved.label == nil ? .invalid : .ok,
                             reason: resolved.reason, inputTokens: reply.input, outputTokens: reply.output)
    }

    /// Match the reply to a declared label. Casing is normalized because
    /// provider docs note enum casing is not guaranteed; anything else is invalid.
    private func resolve(_ text: String?) -> (label: String?, reason: String?) {
        guard let text, let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let raw = object["label"] as? String else { return (nil, "reply is not {\"label\": ...}") }

        guard let label = spec.labelNames.first(where: { $0.caseInsensitiveCompare(raw.trimmingCharacters(in: .whitespaces)) == .orderedSame }) else {
            return (nil, "reply label is not declared")
        }

        return (label, nil)
    }

    private struct Reply { let text: String?; let refusal: String?; let input: Int; let output: Int }

    private static func anthropicReply(_ object: [String: Any]) -> Reply {
        let usage = object["usage"] as? [String: Any]
        let blocks = object["content"] as? [[String: Any]] ?? []
        let text = blocks.first { $0["type"] as? String == "text" }?["text"] as? String
        let stop = object["stop_reason"] as? String
        let refusal = stop == "refusal" ? "refusal" : stop == "max_tokens" ? "max_tokens reached" : nil

        return Reply(text: text, refusal: refusal, input: usage?["input_tokens"] as? Int ?? 0, output: usage?["output_tokens"] as? Int ?? 0)
    }

    private static func chatReply(_ object: [String: Any]) -> Reply {
        let usage = object["usage"] as? [String: Any]
        let choice = (object["choices"] as? [[String: Any]])?.first
        let message = choice?["message"] as? [String: Any]
        let finish = choice?["finish_reason"] as? String
        let refusal = (message?["refusal"] as? String).map { _ in "refusal" } ?? (finish == "length" ? "max tokens reached" : finish == "content_filter" ? "content_filter" : nil)

        return Reply(text: message?["content"] as? String, refusal: refusal, input: usage?["prompt_tokens"] as? Int ?? 0, output: usage?["completion_tokens"] as? Int ?? 0)
    }
}

/// Keeps inputs, prompts, and secrets out of logs and label files.
public enum Redactor {
    private static let patterns = [
        #"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#,
        #"\b(sk|pk|rk|ghp|gho|xox[abp])[-_][A-Za-z0-9_-]{8,}"#,
        #"\b[A-Za-z0-9_-]{32,}\b"#,
        #"\+?\d[\d\s().-]{7,}\d"#,
    ]

    public static func redact(_ text: String) -> String {
        patterns.reduce(text) { $0.replacingOccurrences(of: $1, with: "[redacted]", options: .regularExpression) }
    }

    /// Provider error type plus a short, redacted message.
    static func errorSummary(_ data: Data) -> String {
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let error = object?["error"] as? [String: Any]
        let type = error?["type"] as? String ?? "unknown"
        let message = String((error?["message"] as? String ?? "").prefix(160))

        return redact("\(type) \(message)").trimmingCharacters(in: .whitespaces)
    }
}
