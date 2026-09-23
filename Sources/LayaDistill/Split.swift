import Foundation

public struct LabeledExample: Sendable {
    public let example: DatasetExample
    public let label: Int
}

/// Counts from joining Laya labels to the dataset and splitting it.
public struct SplitReport: Codable, Sendable, Equatable {
    public var total = 0
    public var gold = 0
    public var goldOverlapExcluded = 0
    public var unlabeled = 0
    public var teacherErrors = 0
    public var staleLabels = 0
    public var uncertainToAbstain = 0
    public var uncertainDropped = 0
    public var duplicatesRemoved = 0
    public var conflictsDropped = 0
    public var trainOverlapExcluded = 0
    public var train = 0
    public var holdout = 0

    enum CodingKeys: String, CodingKey {
        case total, gold, unlabeled, train, holdout
        case goldOverlapExcluded = "gold_overlap_excluded", teacherErrors = "teacher_errors", staleLabels = "stale_labels"
        case uncertainToAbstain = "uncertain_to_abstain", uncertainDropped = "uncertain_dropped", duplicatesRemoved = "duplicates_removed"
        case conflictsDropped = "conflicts_dropped", trainOverlapExcluded = "train_overlap_excluded"
    }
}

public struct Split: Sendable {
    /// Laya-labeled rows the student trains on.
    public let train: [LabeledExample]
    /// Laya-labeled rows held out to measure student-vs-Laya agreement.
    public let holdout: [LabeledExample]
    /// Human-labeled rows; never trained on. `label` is the human label.
    public let gold: [LabeledExample]
    public let report: SplitReport
}

public enum Splitter {
    /// Leakage controls:
    /// - gold rows are evaluation-only, and pool rows sharing a gold row's content
    ///   or `group` are excluded from training;
    /// - Laya labels must match the row's content and the current question;
    /// - uncertain Laya answers train only as the abstain label (or are dropped);
    /// - identical inputs collapse, and conflicting labels are dropped;
    /// - train/holdout assignment is a deterministic hash of the seed and the
    ///   row's `group` (or content), so a group never straddles the split;
    /// - `excluding` removes evaluation rows whose content was trained on.
    public static func split(_ examples: [DatasetExample], labels: [String: LabelRecord], spec: TaskSpec, excluding trained: Set<String> = []) -> Split {
        var report = SplitReport(total: examples.count)
        let goldRows = examples.filter { $0.gold != nil }
        let goldHashes = Set(goldRows.map(\.contentHash))
        let goldGroups = Set(goldRows.compactMap(\.group))

        let pool = examples.filter { example in
            guard example.gold == nil else { return false }
            guard !goldHashes.contains(example.contentHash), !(example.group.map(goldGroups.contains) ?? false) else {
                report.goldOverlapExcluded += 1
                return false
            }
            return true
        }

        let labeled = deduplicate(join(pool, labels: labels, spec: spec, report: &report), report: &report)
        var train: [LabeledExample] = []
        var holdout: [LabeledExample] = []

        for item in labeled {
            let key = item.example.group.map { "group:\($0)" } ?? "content:\(item.example.contentHash)"
            let isHoldout = Canonical.unitInterval("\(spec.dataset.splitSeed)\u{0}\(key)") < spec.dataset.holdoutFraction

            if !isHoldout {
                train.append(item)
            } else if trained.contains(item.example.contentHash) {
                report.trainOverlapExcluded += 1
            } else {
                holdout.append(item)
            }
        }

        let gold = deduplicate(goldRows.map { LabeledExample(example: $0, label: spec.labelIndex($0.gold!)!) }, report: &report).filter { item in
            guard trained.contains(item.example.contentHash) else { return true }
            report.trainOverlapExcluded += 1
            return false
        }

        report.gold = gold.count
        report.train = train.count
        report.holdout = holdout.count

        return Split(train: train, holdout: holdout, gold: gold, report: report)
    }

    private static func join(_ examples: [DatasetExample], labels: [String: LabelRecord], spec: TaskSpec, report: inout SplitReport) -> [LabeledExample] {
        let question = LayaQuestion.sha256(spec)

        return examples.compactMap { example in
            guard let record = labels[example.id] else {
                report.unlabeled += 1
                return nil
            }
            guard record.contentHash == example.contentHash, record.questionSha256 == question else {
                report.staleLabels += 1
                return nil
            }

            switch (record.status, record.label.flatMap(spec.labelIndex)) {
            case (.error, _):
                report.teacherErrors += 1
                return nil
            case (.uncertain, nil):
                report.uncertainDropped += 1
                return nil
            case (.uncertain, let index?):
                report.uncertainToAbstain += 1
                return LabeledExample(example: example, label: index)
            case (.accepted, let index?):
                return LabeledExample(example: example, label: index)
            case (.accepted, nil):
                report.teacherErrors += 1
                return nil
            }
        }
    }

    private static func deduplicate(_ items: [LabeledExample], report: inout SplitReport) -> [LabeledExample] {
        var byHash: [String: [LabeledExample]] = [:]
        var order: [String] = []

        for item in items {
            if byHash[item.example.contentHash] == nil { order.append(item.example.contentHash) }
            byHash[item.example.contentHash, default: []].append(item)
        }

        return order.compactMap { hash in
            let copies = byHash[hash]!

            guard Set(copies.map(\.label)).count == 1 else {
                report.conflictsDropped += copies.count
                return nil
            }

            report.duplicatesRemoved += copies.count - 1
            return copies[0]
        }
    }
}
