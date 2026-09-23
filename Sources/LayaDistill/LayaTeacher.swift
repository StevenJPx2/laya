import CryptoKit
import Foundation
import LayaCore

/// The teacher boundary: one typed Laya question in, one Laya answer out.
/// Implemented by the local runtime, the installed daemon, and test fakes.
public protocol LayaTeacher: Sendable {
    /// Recorded with every label, e.g. `laya-daemon:laya-typed-decisions`.
    var identity: String { get }
    func answer(state: JSONValue, question: Question) async throws -> Answer
}

/// Builds the task's Laya question and maps answers back to label order.
public enum LayaQuestion {
    public static func make(_ spec: TaskSpec) -> Question {
        let instructions = JSONValue.string(spec.instructions)

        switch spec.teacher.questionType {
        case .choice:
            return Question(type: "choice", instructions: instructions, criteria: .object(spec.labels.map { ($0.name, .string($0.description)) }))
        case .score:
            return Question(type: "score", instructions: instructions, criteria: .array(spec.labels.map { .string("\($0.name): \($0.description)") }))
        case .noul:
            return Question(type: "noul", instructions: instructions,
                            criteria: .object([("false", .string(spec.labels[0].description)), ("true", .string(spec.labels[1].description))]))
        }
    }

    /// Labels are tied to the exact question asked; changing instructions or
    /// labels makes old labels stale.
    public static func sha256(_ spec: TaskSpec) -> String {
        Canonical.sha256((try? Canonical.encoder().encode(make(spec))) ?? Data())
    }

    /// Per-label probabilities in spec label order.
    public static func probabilities(_ answer: Answer, spec: TaskSpec) throws -> [Double] {
        switch spec.teacher.questionType {
        case .choice:
            return try spec.labelNames.map { name in
                guard let p = answer.probabilities?[name] else { throw DistillError.teacher("Laya answer is missing label \(name)") }
                return p
            }
        case .score:
            return try spec.labels.indices.map { level in
                guard let p = answer.probabilities?["\(level)"] else { throw DistillError.teacher("Laya answer is missing score level \(level)") }
                return p
            }
        case .noul:
            guard let p = answer.noul else { throw DistillError.teacher("Laya answer is missing noul") }
            return [1 - p, p]
        }
    }
}

public enum LabelStatus: String, Codable, Sendable {
    /// Confident Laya answer; its label is used for training.
    case accepted
    /// Failed the confidence gate; trained as the abstain label or dropped.
    case uncertain
    /// Teacher call failed or returned an unusable answer.
    case error
}

public struct TeacherDecision: Sendable, Equatable {
    public let status: LabelStatus
    /// The teacher's own top label, before the gate.
    public let layaLabel: String
    /// The training label, or nil when the row must not be trained on.
    public let label: String?
    public let top: Double
    public let margin: Double
}

/// The explicit gate that keeps ambiguous teacher output (Laya or imported)
/// from becoming a hard label.
public enum TeacherPolicy {
    /// Probabilities from the Laya API are rounded to 4 decimals.
    static let tolerance = 1e-9

    public static func decide(_ probabilities: [Double], spec: TaskSpec) throws -> TeacherDecision {
        guard probabilities.count == spec.labels.count, probabilities.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw DistillError.teacher("Laya probabilities do not match the task labels")
        }
        guard abs(probabilities.reduce(0, +) - 1) <= 0.02 else { throw DistillError.teacher("Laya probabilities do not sum to 1") }

        let order = probabilities.indices.sorted { probabilities[$0] > probabilities[$1] }
        let top = probabilities[order[0]]
        let margin = top - probabilities[order[1]]
        let layaLabel = spec.labelNames[order[0]]
        let confident = top + tolerance >= spec.teacher.minConfidence && margin + tolerance >= spec.teacher.minMargin

        if confident {
            return TeacherDecision(status: .accepted, layaLabel: layaLabel, label: layaLabel, top: top, margin: margin)
        }

        let fallback = spec.teacher.uncertain == .abstain ? spec.abstain?.label : nil

        return TeacherDecision(status: .uncertain, layaLabel: layaLabel, label: fallback, top: top, margin: margin)
    }
}

/// Laya running in this process on the Neural Engine.
public struct RuntimeTeacher: LayaTeacher {
    public let identity: String
    let runtime: LayaRuntime

    public init(runtime: LayaRuntime, assets: URL) async throws {
        self.runtime = runtime
        identity = "laya-runtime:\(await runtime.health().model)@\(try Self.fingerprint(assets: assets).prefix(12))"
    }

    public func answer(state: JSONValue, question: Question) async throws -> Answer {
        let response = try await runtime.predict(PredictRequest(state: state, questions: ["q": question]))
        guard let answer = response.answers["q"] else { throw DistillError.teacher("Laya returned no answer") }

        return answer
    }

    /// Identifies the asset set cheaply without hashing ~800 MB of weights:
    /// runtime manifest, action-head weights, and the first 4 MiB of embeddings.
    static func fingerprint(assets: URL) throws -> String {
        var hasher = SHA256()

        for (name, limit) in [("runtime_manifest.json", Int.max), ("act_head.f32.bin", Int.max), ("embeddings.f16.bin", 4 << 20)] {
            let handle = try FileHandle(forReadingFrom: assets.appendingPathComponent(name))
            defer { try? handle.close() }

            hasher.update(data: Data(name.utf8))
            hasher.update(data: try handle.read(upToCount: limit) ?? Data())
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// The installed `laya-daemon`, reached over its Unix socket.
public struct DaemonTeacher: LayaTeacher {
    public let identity: String
    let path: String

    public init(path: String = SocketClient.defaultPath) throws {
        let reply = try SocketClient.request(Data(#"{"op":"health"}"#.utf8), path: path)
        guard let health = try? JSONDecoder().decode(HealthResponse.self, from: reply), health.status == "ok" else {
            throw DistillError.teacher("laya-daemon at \(path) is not healthy")
        }

        self.path = path
        identity = "laya-daemon:\(health.model)"
    }

    public func answer(state: JSONValue, question: Question) async throws -> Answer {
        let payload = try JSONEncoder().encode(PredictRequest(state: state, questions: ["q": question]))
        let reply = try SocketClient.request(payload, path: path)

        if let failure = try? JSONDecoder().decode([String: String].self, from: reply), let message = failure["error"] {
            throw DistillError.teacher("laya-daemon: \(message)")
        }
        guard let answer = try JSONDecoder().decode(PredictResponse.self, from: reply).answers["q"] else {
            throw DistillError.teacher("laya-daemon returned no answer")
        }

        return answer
    }
}
