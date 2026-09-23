import Foundation

public enum Workflow {
    /// Train a student on gated Laya labels and evaluate it on the Laya holdout
    /// and on human gold rows. Requires no Laya model: labels are already on disk.
    public static func train(spec: TaskSpec, examples: [DatasetExample], labels: [String: LabelRecord],
                             now: Date = Date(), log: (String) -> Void = { _ in }) throws -> ClassifierArtifact {
        let split = Splitter.split(examples, labels: labels, spec: spec)
        try checkCoverage(split, spec: spec)

        let features = HashedFeatures(dimensions: spec.student.hashDimensions)
        let rows = split.train.map { features.extract($0.example.input) }
        let selection = try ModelSelection.selectL2(split.train, rows: rows, spec: spec, dimensions: features.descriptor.dimensions)
        let l2 = selection?.l2 ?? spec.student.l2

        log("[train] \(split.train.count) train / \(split.holdout.count) Laya holdout / \(split.gold.count) gold")
        if let selection {
            log("[train] l2 \(l2) selected by \(selection.folds)-fold CV on train: " + selection.candidates.map { "\($0.l2)=\(percent($0.accuracy))" }.joined(separator: " "))
        }

        let result = try Trainer.train(rows, labels: split.train.map(\.label), classes: spec.labels.count,
                                       dimensions: features.descriptor.dimensions, config: spec.student, l2: l2)
        log("[train] final loss \(String(format: "%.4f", result.finalLoss)), train accuracy \(percent(result.trainAccuracy))")

        let present = Set(split.train.map(\.label))
        let info = TrainingInfo(
            teacher: teachers(split.train, labels), questionSha256: LayaQuestion.sha256(spec), examples: split.train.count,
            epochs: spec.student.epochs, l2: l2, selection: selection, finalLoss: result.finalLoss, trainAccuracy: result.trainAccuracy,
            labelsWithoutTrainingRows: spec.labels.indices.filter { !present.contains($0) }.map { spec.labelNames[$0] },
            datasetSha256: Dataset.fingerprint(examples), labelsSha256: LabelStore.fingerprint(labels),
            trainHashes: split.train.map(\.example.contentHash).sorted(), createdAt: ISO8601DateFormatter().string(from: now)
        )
        let artifact = ClassifierArtifact(spec: spec, features: features.descriptor, model: result.model, training: info, evaluation: nil)

        return artifact.withEvaluation(report(artifact, split: split, labels: labels))
    }

    /// Re-evaluate an artifact. Evaluation rows whose content was trained on are
    /// excluded and counted, so a changed dataset cannot leak into the metrics.
    public static func evaluate(_ classifier: Classifier, examples: [DatasetExample], labels: [String: LabelRecord]) throws -> EvaluationReport {
        let artifact = classifier.artifact
        let split = Splitter.split(examples, labels: labels, spec: artifact.spec, excluding: Set(artifact.training.trainHashes))

        guard !split.holdout.isEmpty || !split.gold.isEmpty else { throw DistillError.training("no evaluation rows (Laya holdout or gold)") }

        return report(artifact, split: split, labels: labels)
    }

    private static func report(_ artifact: ClassifierArtifact, split: Split, labels: [String: LabelRecord]) -> EvaluationReport {
        let names = artifact.spec.labelNames
        let features = HashedFeatures(dimensions: artifact.features.dimensions)
        let majority = majorityLabel(split.train, classes: names.count)

        func predictions(_ items: [LabeledExample]) -> (argmax: [Int], served: [Int]) {
            let probabilities = items.map { artifact.model.probabilities(features.extract($0.example.input)) }
            return (probabilities.map(argmax), probabilities.map { artifact.spec.labelIndex(artifact.decide($0).label)! })
        }

        var agreement: AgreementReport?
        if !split.holdout.isEmpty {
            let reference = split.holdout.map(\.label)
            let predicted = predictions(split.holdout)
            agreement = AgreementReport(
                student: Metrics(reference: reference, predicted: predicted.argmax, labels: names),
                served: Metrics(reference: reference, predicted: predicted.served, labels: names),
                abstainRate: Double(zip(predicted.argmax, predicted.served).filter { $0 != $1 }.count) / Double(reference.count),
                majority: Metrics(reference: reference, predicted: reference.map { _ in majority }, labels: names)
            )
        }

        return EvaluationReport(teacher: teachers(split.train + split.holdout, labels), split: split.report,
                                agreement: agreement, gold: goldReport(split.gold, labels: labels, artifact: artifact, predictions: predictions, majority: majority))
    }

    private static func goldReport(_ gold: [LabeledExample], labels: [String: LabelRecord], artifact: ClassifierArtifact,
                                   predictions: ([LabeledExample]) -> (argmax: [Int], served: [Int]), majority: Int) -> GoldReport? {
        guard !gold.isEmpty else { return nil }

        let spec = artifact.spec
        let names = spec.labelNames
        let reference = gold.map(\.label)
        let predicted = predictions(gold)
        let question = LayaQuestion.sha256(spec)

        func laya(_ pick: (LabelRecord) -> String?) -> Metrics? {
            let pairs = gold.compactMap { item -> (Int, Int)? in
                guard let record = labels[item.example.id], record.contentHash == item.example.contentHash, record.questionSha256 == question,
                      let index = pick(record).flatMap(spec.labelIndex) else { return nil }
                return (item.label, index)
            }
            return pairs.isEmpty ? nil : Metrics(reference: pairs.map(\.0), predicted: pairs.map(\.1), labels: names)
        }

        return GoldReport(
            studentServed: Metrics(reference: reference, predicted: predicted.served, labels: names),
            studentArgmax: Metrics(reference: reference, predicted: predicted.argmax, labels: names),
            layaRaw: laya(\.layaLabel), layaGated: laya(\.label),
            majority: Metrics(reference: reference, predicted: reference.map { _ in majority }, labels: names)
        )
    }

    private static func majorityLabel(_ train: [LabeledExample], classes: Int) -> Int {
        let counts = (0..<classes).map { c in train.filter { $0.label == c }.count }
        return counts.indices.max { counts[$0] < counts[$1] || (counts[$0] == counts[$1] && $0 > $1) } ?? 0
    }

    private static func teachers(_ items: [LabeledExample], _ labels: [String: LabelRecord]) -> String {
        Set(items.compactMap { labels[$0.example.id]?.teacher }).sorted().joined(separator: ",")
    }

    private static func checkCoverage(_ split: Split, spec: TaskSpec) throws {
        guard Set(split.train.map(\.label)).count >= 2 else {
            throw DistillError.training("need Laya-labeled training rows for at least 2 labels (have \(split.train.count) rows); run label, or relax teacher.min_confidence")
        }
        guard !split.holdout.isEmpty else { throw DistillError.training("Laya holdout is empty; add unlabeled rows or raise dataset.holdout_fraction") }
    }
}
