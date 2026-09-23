import Foundation
import LayaCore

public struct ImportSummary: Codable, Sendable, Equatable {
    public var teacher = ""
    /// Ledger lines read; later lines for an id replace earlier ones.
    public var ledgerRows = 0
    /// Label records written, one per dataset row found in the ledger.
    public var imported = 0
    public var accepted = 0
    public var uncertainToAbstain = 0
    public var uncertainDropped = 0
    public var forbidden = 0
    public var failed = 0
    /// Labeled ledger rows the confidence gate could not use (for example probabilities that do not sum to 1).
    public var invalid = 0
    /// Ledger ids with no dataset row; skipped.
    public var unknownIds = 0
    /// Dataset rows with no ledger line.
    public var missing = 0

    enum CodingKeys: String, CodingKey {
        case teacher, imported, accepted, forbidden, failed, invalid, missing
        case ledgerRows = "ledger_rows", uncertainToAbstain = "uncertain_to_abstain", uncertainDropped = "uncertain_dropped", unknownIds = "unknown_ids"
    }
}

/// Imports labels from a jev-distill ledger (`jev-distill.labels` v1). Labeled
/// rows go through the same `TeacherPolicy` gate as Laya answers; forbidden and
/// failed rows become `error` records and are never trained on.
public enum JevImport {
    public static let ledger = "jev-distill.labels"
    public static let ledgerVersion = 1

    public static func records(_ url: URL, spec: TaskSpec, examples: [DatasetExample]) throws -> (records: [LabelRecord], summary: ImportSummary) {
        let lines = try JSONLines.read(url, maxLines: spec.dataset.maxExamples * 8, maxLineBytes: 65_536)
        var latest: [String: (line: Int, row: JevRow)] = [:]

        for line in lines {
            let row: JevRow
            do {
                row = try JSONDecoder().decode(JevRow.self, from: line.data)
            } catch {
                throw DistillError.invalidData("\(url.lastPathComponent) line \(line.line): \(describe(error))")
            }
            guard row.ledger == ledger, row.ledgerVersion == ledgerVersion else {
                throw DistillError.invalidData("\(url.lastPathComponent) line \(line.line): expected ledger \(ledger) v\(ledgerVersion)")
            }

            latest[row.id] = (line.line, row)
        }

        let known = Set(examples.map(\.id))
        let question = LayaQuestion.sha256(spec)
        var summary = ImportSummary(ledgerRows: lines.count, unknownIds: latest.keys.filter { !known.contains($0) }.count)
        var records: [LabelRecord] = []

        for example in examples {
            guard let (line, row) = latest[example.id] else {
                summary.missing += 1
                continue
            }

            let record = try record(row, line: line, example: example, spec: spec, question: question)
            records.append(record)
            count(record, row: row, into: &summary)
        }

        summary.imported = records.count
        summary.teacher = Set(records.map(\.teacher)).sorted().joined(separator: ",")

        return (records, summary)
    }

    private static func record(_ row: JevRow, line: Int, example: DatasetExample, spec: TaskSpec, question: String) throws -> LabelRecord {
        let teacher = "jev:\(row.answeredModel ?? row.requestedModel ?? "unknown")"

        func make(_ status: LabelStatus, _ decision: TeacherDecision?, probabilities: [Double]?, reason: String?) -> LabelRecord {
            LabelRecord(
                id: example.id, contentHash: example.contentHash, questionSha256: question, teacher: teacher, status: status,
                layaLabel: decision?.layaLabel, label: decision?.label, probabilities: probabilities.map { Dictionary(uniqueKeysWithValues: zip(spec.labelNames, $0)) },
                top: decision?.top, margin: decision?.margin, layaConfidence: nil, actProbability: nil, reason: reason
            )
        }

        switch row.status {
        case "forbidden", "failed":
            let detail = row.error.map { ": " + Prompt.serialize($0) } ?? ""
            return make(.error, nil, probabilities: nil, reason: String("jev \(row.status)\(detail)".prefix(200)))

        case "labeled":
            guard let raw = row.probabilities, Set(raw.keys) == Set(spec.labelNames), raw.count == spec.labels.count,
                  row.choice.map(spec.labelNames.contains) ?? true else {
                let keys = row.probabilities.map { $0.keys.sorted().joined(separator: ", ") } ?? "none"
                throw DistillError.invalidData("ledger line \(line) (id \(row.id)): probabilities cover [\(keys)], task labels are [\(spec.labelNames.joined(separator: ", "))]")
            }

            let probabilities = spec.labelNames.map { raw[$0]! }
            do {
                let decision = try TeacherPolicy.decide(probabilities, spec: spec)
                let reason = decision.status == .uncertain ? "below min_confidence \(spec.teacher.minConfidence) or min_margin \(spec.teacher.minMargin)" : nil
                return make(decision.status, decision, probabilities: probabilities, reason: reason)
            } catch {
                return make(.error, nil, probabilities: nil, reason: "jev invalid: \(error.localizedDescription)")
            }

        default:
            throw DistillError.invalidData("ledger line \(line) (id \(row.id)): unknown status \(row.status)")
        }
    }

    private static func count(_ record: LabelRecord, row: JevRow, into summary: inout ImportSummary) {
        switch (record.status, row.status) {
        case (.accepted, _): summary.accepted += 1
        case (.uncertain, _) where record.label != nil: summary.uncertainToAbstain += 1
        case (.uncertain, _): summary.uncertainDropped += 1
        case (.error, "forbidden"): summary.forbidden += 1
        case (.error, "failed"): summary.failed += 1
        case (.error, _): summary.invalid += 1
        }
    }
}

/// The ledger fields the import reads. Other fields (usage, hashes, request
/// ids) are the ledger's own bookkeeping and are ignored.
private struct JevRow: Decodable {
    let ledger: String
    let ledgerVersion: Int
    let id: String
    let status: String
    let choice: String?
    let probabilities: [String: Double]?
    let requestedModel: String?
    let answeredModel: String?
    let error: JSONValue?

    enum CodingKeys: String, CodingKey {
        case ledger, id, status, choice, probabilities, error
        case ledgerVersion = "ledger_version", requestedModel = "requested_model", answeredModel = "answered_model"
    }
}
