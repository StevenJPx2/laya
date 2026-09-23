import Foundation

public struct LabeledExample: Sendable {
    public let example: DatasetExample
    public let label: Int
}

/// Counts from joining teacher labels to the dataset and splitting it.
public struct SplitReport: Codable, Sendable, Equatable {
    public var total = 0
    public var unlabeled = 0
    public var staleLabels = 0
    public var duplicatesRemoved = 0
    public var conflictsDropped = 0
    public var fewShotForcedToTrain = 0
    public var trainOverlapExcluded = 0
    public var train = 0
    public var holdout = 0

    enum CodingKeys: String, CodingKey {
        case total, unlabeled, train, holdout
        case staleLabels = "stale_labels", duplicatesRemoved = "duplicates_removed", conflictsDropped = "conflicts_dropped"
        case fewShotForcedToTrain = "few_shot_forced_to_train", trainOverlapExcluded = "train_overlap_excluded"
    }
}

public struct Split: Sendable {
    public let train: [LabeledExample]
    public let holdout: [LabeledExample]
    public let report: SplitReport
}

public enum Splitter {
    /// Join labels, de-duplicate by content, then assign each group to train or
    /// holdout deterministically. Leakage controls:
    /// - identical (normalized) inputs are collapsed; conflicting labels are dropped;
    /// - rows sharing a `group` always land on the same side;
    /// - inputs equal to a few-shot example shown to the teacher never enter holdout;
    /// - `excluding` removes holdout rows whose content was trained on (used by eval).
    public static func split(_ examples: [DatasetExample], labels: [String: LabelRecord], spec: TaskSpec, excluding trained: Set<String> = []) -> Split {
        var report = SplitReport(total: examples.count)
        let unique = deduplicate(join(examples, labels: labels, spec: spec, report: &report), report: &report)
        let fewShot = Set(spec.examples.compactMap { try? spec.input.parse($0.input).contentHash })

        var train: [LabeledExample] = []
        var holdout: [LabeledExample] = []

        for item in unique {
            let key = item.example.group.map { "group:\($0)" } ?? "content:\(item.example.contentHash)"
            let isHoldout = Canonical.unitInterval("\(spec.dataset.splitSeed)\u{0}\(key)") < spec.dataset.holdoutFraction

            if isHoldout, fewShot.contains(item.example.contentHash) {
                report.fewShotForcedToTrain += 1
                train.append(item)
            } else if isHoldout, trained.contains(item.example.contentHash) {
                report.trainOverlapExcluded += 1
            } else if isHoldout {
                holdout.append(item)
            } else {
                train.append(item)
            }
        }

        report.train = train.count
        report.holdout = holdout.count

        return Split(train: train, holdout: holdout, report: report)
    }

    private static func join(_ examples: [DatasetExample], labels: [String: LabelRecord], spec: TaskSpec, report: inout SplitReport) -> [LabeledExample] {
        examples.compactMap { example in
            guard let record = labels[example.id], record.status == .ok, let name = record.label, let index = spec.labelIndex(name) else {
                report.unlabeled += 1
                return nil
            }
            guard record.contentHash == example.contentHash else {
                report.staleLabels += 1
                return nil
            }

            return LabeledExample(example: example, label: index)
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
            report.duplicatesRemoved += copies.count - 1

            guard Set(copies.map(\.label)).count == 1 else {
                report.conflictsDropped += copies.count
                report.duplicatesRemoved -= copies.count - 1
                return nil
            }

            return copies[0]
        }
    }
}
