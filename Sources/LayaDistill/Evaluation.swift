import Foundation

public struct ClassMetrics: Codable, Sendable {
    public let precision: Double
    public let recall: Double
    public let f1: Double
    public let support: Int
}

public struct Metrics: Codable, Sendable {
    public let examples: Int
    public let accuracy: Double
    public let macroF1: Double
    public let perLabel: [String: ClassMetrics]
    /// Rows are the reference label, columns the prediction, in `labels` order.
    public let confusion: [[Int]]
    public let labels: [String]

    enum CodingKeys: String, CodingKey { case examples, accuracy, labels, confusion, macroF1 = "macro_f1", perLabel = "per_label" }

    public init(reference: [Int], predicted: [Int], labels: [String]) {
        var confusion = [[Int]](repeating: [Int](repeating: 0, count: labels.count), count: labels.count)
        for (r, p) in zip(reference, predicted) { confusion[r][p] += 1 }

        var perLabel: [String: ClassMetrics] = [:]
        for (c, name) in labels.enumerated() {
            let tp = Double(confusion[c][c])
            let predictedCount = Double(confusion.reduce(0) { $0 + $1[c] })
            let support = confusion[c].reduce(0, +)
            let precision = predictedCount == 0 ? 0 : tp / predictedCount
            let recall = support == 0 ? 0 : tp / Double(support)
            let f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall)
            perLabel[name] = ClassMetrics(precision: precision, recall: recall, f1: f1, support: support)
        }

        let present = labels.filter { perLabel[$0]!.support > 0 }
        examples = reference.count
        accuracy = reference.isEmpty ? 0 : Double(zip(reference, predicted).filter { $0 == $1 }.count) / Double(reference.count)
        macroF1 = present.isEmpty ? 0 : present.reduce(0) { $0 + perLabel[$1]!.f1 } / Double(present.count)
        self.perLabel = perLabel
        self.confusion = confusion
        self.labels = labels
    }
}

/// Student versus Laya on Laya-labeled rows the student did not train on.
public struct AgreementReport: Codable, Sendable {
    /// Student argmax against the gated Laya training labels.
    public let student: Metrics
    /// Student with its own abstain policy applied (what `predict` serves).
    public let served: Metrics
    public let abstainRate: Double
    public let majority: Metrics

    enum CodingKeys: String, CodingKey { case student, served, majority, abstainRate = "abstain_rate" }
}

/// Accuracy against human `gold` labels on rows nobody trained on.
public struct GoldReport: Codable, Sendable {
    public let studentServed: Metrics
    public let studentArgmax: Metrics
    /// Laya's own top label on the gold rows that Laya labeled.
    public let layaRaw: Metrics?
    /// Laya after the confidence gate; rows the gate dropped are excluded.
    public let layaGated: Metrics?
    public let majority: Metrics

    enum CodingKeys: String, CodingKey {
        case majority, studentServed = "student_served", studentArgmax = "student_argmax", layaRaw = "laya_raw", layaGated = "laya_gated"
    }
}

public struct EvaluationReport: Codable, Sendable {
    public let teacher: String
    public let split: SplitReport
    public let agreement: AgreementReport?
    public let gold: GoldReport?

    public var markdown: String {
        var lines = [
            "Teacher: \(teacher). Split: \(split.train) train / \(split.holdout) Laya holdout / \(split.gold) gold (evaluation-only).",
            "",
            "| Reference | Model | n | Accuracy | Macro-F1 |",
            "|---|---|---:|---:|---:|",
        ]

        if let agreement {
            lines.append(row("Laya labels", "Student (argmax)", agreement.student))
            lines.append(row("Laya labels", "Student (served, abstain rate \(percent(agreement.abstainRate)))", agreement.served))
            lines.append(row("Laya labels", "Majority class", agreement.majority))
        }
        if let gold {
            lines.append(row("Human gold", "Student (served)", gold.studentServed))
            lines.append(row("Human gold", "Student (argmax)", gold.studentArgmax))
            if let raw = gold.layaRaw { lines.append(row("Human gold", "Laya teacher (raw argmax)", raw)) }
            if let gated = gold.layaGated { lines.append(row("Human gold", "Laya teacher (confidence-gated)", gated)) }
            lines.append(row("Human gold", "Majority class", gold.majority))
        }

        lines.append("")
        lines.append("Teacher gate on training pool: uncertain→abstain \(split.uncertainToAbstain), uncertain dropped \(split.uncertainDropped), errors \(split.teacherErrors), unlabeled \(split.unlabeled), stale \(split.staleLabels).")
        lines.append("Leakage controls: gold-overlap excluded \(split.goldOverlapExcluded), duplicates removed \(split.duplicatesRemoved), conflicts dropped \(split.conflictsDropped), train-overlap excluded \(split.trainOverlapExcluded).")

        if let gold {
            lines.append("")
            lines.append("Confusion vs human gold (rows = gold, columns = student served): \(gold.studentServed.labels.joined(separator: ", "))")
            lines += gold.studentServed.confusion.map { "    \($0.map { String(format: "%4d", $0) }.joined())" }
        }

        return lines.joined(separator: "\n")
    }

    private func row(_ reference: String, _ name: String, _ metrics: Metrics) -> String {
        "| \(reference) | \(name) | \(metrics.examples) | \(percent(metrics.accuracy)) | \(String(format: "%.3f", metrics.macroF1)) |"
    }
}

func percent(_ value: Double) -> String { String(format: "%.1f%%", value * 100) }
