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

public struct Baseline: Codable, Sendable {
    public let name: String
    public let status: String
    public let metrics: Metrics?
}

public struct EvaluationReport: Codable, Sendable {
    public let reference: String
    public let split: SplitReport
    /// Student argmax against the teacher's labels on held-out rows.
    public let student: Metrics
    /// Student with the abstain policy applied (what `predict` serves).
    public let served: Metrics
    public let abstainRate: Double
    public let baselines: [Baseline]
    /// Only present when held-out rows carry human `gold` labels.
    public let studentVsGold: Metrics?
    public let teacherVsGold: Metrics?

    enum CodingKeys: String, CodingKey {
        case reference, split, student, served, baselines
        case abstainRate = "abstain_rate", studentVsGold = "student_vs_gold", teacherVsGold = "teacher_vs_gold"
    }

    public var markdown: String {
        var lines = [
            "| Model (holdout, n=\(student.examples), reference: \(reference)) | Accuracy | Macro-F1 |",
            "|---|---:|---:|",
            row("Student (argmax)", student),
            row("Student (served, abstain rate \(percent(abstainRate)))", served),
        ]

        for baseline in baselines {
            if let metrics = baseline.metrics { lines.append(row(baseline.name, metrics)) } else { lines.append("| \(baseline.name) | \(baseline.status) | — |") }
        }
        if let studentVsGold, let teacherVsGold {
            lines.append(row("Student vs gold", studentVsGold))
            lines.append(row("Teacher vs gold", teacherVsGold))
        }

        lines.append("")
        lines.append("Split: \(split.train) train / \(split.holdout) holdout; duplicates removed \(split.duplicatesRemoved), conflicts dropped \(split.conflictsDropped), unlabeled \(split.unlabeled), stale labels \(split.staleLabels), few-shot kept out of holdout \(split.fewShotForcedToTrain), train-overlap excluded \(split.trainOverlapExcluded).")
        lines.append("")
        lines.append("Confusion (rows = \(reference), columns = student served): labels \(served.labels.joined(separator: ", "))")
        lines += served.confusion.map { "    \($0.map { String(format: "%4d", $0) }.joined())" }

        return lines.joined(separator: "\n")
    }

    private func row(_ name: String, _ metrics: Metrics) -> String {
        "| \(name) | \(percent(metrics.accuracy)) | \(String(format: "%.3f", metrics.macroF1)) |"
    }
}

func percent(_ value: Double) -> String { String(format: "%.1f%%", value * 100) }
