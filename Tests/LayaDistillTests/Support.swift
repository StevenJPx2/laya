import Foundation
import LayaCore
@testable import LayaDistill

/// Deterministic stand-in for Laya. Returns real `Answer` values (decoded
/// through LayaCore's own response type) from a closure over the request.
final class FakeTeacher: LayaTeacher, @unchecked Sendable {
    let identity = "fake-laya"
    private let lock = NSLock()
    private var recorded: [(JSONValue, Question)] = []
    private let respond: @Sendable (String, Question) throws -> Answer

    init(respond: @escaping @Sendable (String, Question) throws -> Answer) {
        self.respond = respond
    }

    var calls: Int { lock.withLock { recorded.count } }

    func answer(state: JSONValue, question: Question) async throws -> Answer {
        lock.withLock { recorded.append((state, question)) }

        return try respond(Prompt.serialize(state), question)
    }
}

enum Fixture {
    static func spec(question: String = "choice", labels: [String] = ["allow", "deny", "ask"], abstain: String? = "ask",
                     minConfidence: Double = 0.6, minMargin: Double = 0.2, uncertain: String = "abstain", extra: String = "") -> String {
        let labelJSON = labels.map { #"{"name": "\#($0)", "description": "\#($0) description"}"# }.joined(separator: ", ")
        let abstainJSON = abstain.map { #""abstain": {"label": "\#($0)", "min_confidence": 0.4},"# } ?? ""

        return """
        {
          "spec_version": 2, "name": "tiny", "version": "1.0.0",
          "instructions": "Gate the requested action.",
          "input": {"fields": [{"name": "request", "type": "string", "max_chars": 200}, {"name": "note", "type": "string", "required": false, "max_chars": 200}]},
          "labels": [\(labelJSON)],
          \(abstainJSON)
          "teacher": {"question_type": "\(question)", "min_confidence": \(minConfidence), "min_margin": \(minMargin), "uncertain": "\(uncertain)"},
          "dataset": {"holdout_fraction": 0.3, "split_seed": "tiny-seed"},
          "student": {"hash_dimensions": 1024, "epochs": 200, "learning_rate": 0.1, "l2": 0.0001}
          \(extra)
        }
        """
    }

    static func loadSpec(_ text: String) throws -> TaskSpec { try TaskSpec.decode(Data(text.utf8)) }

    static let verbs = [("read", "allow"), ("list", "allow"), ("view", "allow"), ("delete", "deny"), ("wipe", "deny"),
                        ("exfiltrate", "deny"), ("install", "ask"), ("deploy", "ask"), ("publish", "ask")]
    static let objects = ["the logs", "project files", "a config", "the database", "user secrets", "a package",
                          "release notes", "the cache", "test fixtures", "the build", "billing records", "a branch"]

    /// Unlabeled pool rows (the verb decides the label; objects carry no signal)
    /// plus human gold rows phrased differently so no content overlaps.
    static func dataset(_ spec: TaskSpec, pool: Int = 60, gold: Int = 18) throws -> [DatasetExample] {
        let poolRows = try (0..<pool).map { index in
            let (verb, _) = verbs[index % verbs.count]
            let object = objects[(index / verbs.count) % objects.count]
            let input = try spec.input.parse(.object([("request", .string("\(verb) \(object) item \(index)"))]))
            return DatasetExample(id: "pool-\(index)", input: input, group: nil, gold: nil)
        }
        let goldRows = try (0..<gold).map { index in
            let (verb, label) = verbs[index % verbs.count]
            let input = try spec.input.parse(.object([("request", .string("please \(verb) \(objects[index % objects.count]) now, case \(index)"))]))
            return DatasetExample(id: "gold-\(index)", input: input, group: nil, gold: label)
        }

        return poolRows + goldRows
    }

    static func answer(_ probabilities: [String: Double], type: String = "choice", noul: Double? = nil, confidence: Double = 0.5) -> Answer {
        var object: [String: Any] = ["type": type, "confidence": confidence, "action": ["act_probability": 0.9], "probabilities": probabilities]
        if let noul { object["noul"] = noul }

        return try! JSONDecoder().decode(Answer.self, from: JSONSerialization.data(withJSONObject: object))
    }

    /// Confident on the verb, except rows containing "maybe", which are ambiguous.
    static func verbTeacher() -> FakeTeacher {
        FakeTeacher { state, _ in
            if state.contains("maybe") { return answer(["allow": 0.4, "deny": 0.35, "ask": 0.25]) }

            let label = verbs.first { state.contains($0.0) }?.1 ?? "allow"
            var probabilities = ["allow": 0.05, "deny": 0.05, "ask": 0.05]
            probabilities[label] = 0.9

            return answer(probabilities)
        }
    }

    static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("laya-distill-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)

        return url
    }

    static func label(_ spec: TaskSpec, _ examples: [DatasetExample], teacher: LayaTeacher = verbTeacher()) async throws -> [String: LabelRecord] {
        var records: [String: LabelRecord] = [:]
        _ = try await Labeler(spec: spec, examples: examples, existing: [:]).run(teacher: teacher, sink: { records[$0.id] = $0 }, log: { _ in })

        return records
    }
}
