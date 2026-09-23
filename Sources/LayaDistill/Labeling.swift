import Foundation
import LayaCore

public struct LabelingSummary: Codable, Sendable {
    public var teacher: String
    public var questionType: String
    public var pending = 0
    public var selected = 0
    public var accepted = 0
    public var uncertainToAbstain = 0
    public var uncertainDropped = 0
    public var errors = 0
    public var stopReason: String?

    enum CodingKeys: String, CodingKey {
        case teacher, pending, selected, accepted, errors
        case questionType = "question_type", uncertainToAbstain = "uncertain_to_abstain", uncertainDropped = "uncertain_dropped", stopReason = "stop_reason"
    }
}

/// Asks Laya the task question for each pending row and records the gated result.
public struct Labeler: Sendable {
    public static let maxConsecutiveErrors = 5

    let spec: TaskSpec
    let examples: [DatasetExample]
    let existing: [String: LabelRecord]

    public init(spec: TaskSpec, examples: [DatasetExample], existing: [String: LabelRecord]) {
        self.spec = spec
        self.examples = examples
        self.existing = existing
    }

    /// Rows without a usable record for the current content and question.
    public func pending() -> [DatasetExample] {
        let question = LayaQuestion.sha256(spec)

        return examples.filter { example in
            guard let record = existing[example.id] else { return true }

            return record.status == .error || record.contentHash != example.contentHash || record.questionSha256 != question
        }
    }

    public func run(teacher: LayaTeacher, limit: Int? = nil, sink: (LabelRecord) throws -> Void, log: (String) -> Void) async throws -> LabelingSummary {
        let pending = pending()
        let batch = Array(pending.prefix(limit ?? pending.count))
        let question = LayaQuestion.make(spec)
        var summary = LabelingSummary(teacher: teacher.identity, questionType: spec.teacher.questionType.rawValue, pending: pending.count, selected: batch.count)
        var failures = 0

        for (index, example) in batch.enumerated() {
            let record = await label(example, teacher: teacher, question: question)
            try sink(record)
            count(record, into: &summary)
            log("[label] \(index + 1)/\(batch.count) \(String(example.contentHash.prefix(10))) \(record.status.rawValue) \(record.label ?? "-")")

            failures = record.status == .error ? failures + 1 : 0
            if failures >= Self.maxConsecutiveErrors {
                summary.stopReason = "stopped after \(failures) consecutive teacher errors (last: \(record.reason ?? "unknown"))"
                break
            }
        }

        return summary
    }

    private func label(_ example: DatasetExample, teacher: LayaTeacher, question: Question) async -> LabelRecord {
        let questionHash = LayaQuestion.sha256(spec)

        do {
            let answer = try await teacher.answer(state: example.input.jsonValue, question: question)
            let probabilities = try LayaQuestion.probabilities(answer, spec: spec)
            let decision = try TeacherPolicy.decide(probabilities, spec: spec)

            return LabelRecord(
                id: example.id, contentHash: example.contentHash, questionSha256: questionHash, teacher: teacher.identity, status: decision.status,
                layaLabel: decision.layaLabel, label: decision.label, probabilities: Dictionary(uniqueKeysWithValues: zip(spec.labelNames, probabilities)),
                top: decision.top, margin: decision.margin, layaConfidence: answer.confidence, actProbability: answer.action.act_probability,
                reason: decision.status == .uncertain ? "below min_confidence \(spec.teacher.minConfidence) or min_margin \(spec.teacher.minMargin)" : nil
            )
        } catch {
            return LabelRecord(
                id: example.id, contentHash: example.contentHash, questionSha256: questionHash, teacher: teacher.identity, status: .error,
                layaLabel: nil, label: nil, probabilities: nil, top: nil, margin: nil, layaConfidence: nil, actProbability: nil,
                reason: String(error.localizedDescription.prefix(200))
            )
        }
    }

    private func count(_ record: LabelRecord, into summary: inout LabelingSummary) {
        switch record.status {
        case .accepted: summary.accepted += 1
        case .uncertain where record.label != nil: summary.uncertainToAbstain += 1
        case .uncertain: summary.uncertainDropped += 1
        case .error: summary.errors += 1
        }
    }
}
