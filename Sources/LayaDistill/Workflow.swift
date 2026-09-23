import Foundation

public enum Workflow {
    public static func train(spec: TaskSpec, examples: [DatasetExample], labels: [String: LabelRecord], extractor: FeatureExtractor,
                             baseline: FeatureExtractor? = nil, now: Date = Date(), log: (String) -> Void = { _ in }) async throws -> ClassifierArtifact {
        let split = Splitter.split(examples, labels: labels, spec: spec)
        try checkCoverage(split, spec: spec)

        log("[train] \(split.train.count) train / \(split.holdout.count) holdout; extracting \(extractor.descriptor.kind.rawValue) features")
        let trainRows = try await extract(split.train, extractor, log: log)
        let vectors = trainRows.map(\.vector)
        let dense = extractor.descriptor.kind == .laya
        let selection = try ModelSelection.selectL2(split.train, rows: vectors, spec: spec, dense: dense, dimensions: extractor.descriptor.dimensions)
        let l2 = selection?.l2 ?? spec.student.l2

        if let selection {
            log("[train] l2 \(l2) selected by \(selection.folds)-fold CV on train: " + selection.candidates.map { "\($0.l2)=\(percent($0.accuracy))" }.joined(separator: " "))
        }

        let result = try Trainer.train(vectors, labels: split.train.map(\.label), classes: spec.labels.count,
                                       dimensions: extractor.descriptor.dimensions, dense: dense, config: spec.student, l2: l2)
        log("[train] final loss \(String(format: "%.4f", result.finalLoss)), train accuracy \(percent(result.trainAccuracy))")

        let teachers = Set(split.train.compactMap { labels[$0.example.id]?.teacher }).sorted().joined(separator: ",")
        let info = TrainingInfo(
            teacher: teachers, examples: split.train.count, epochs: spec.student.epochs, l2: l2, selection: selection,
            finalLoss: result.finalLoss, trainAccuracy: result.trainAccuracy,
            datasetSha256: Dataset.fingerprint(examples), labelsSha256: LabelStore.fingerprint(labels),
            trainHashes: split.train.map(\.example.contentHash).sorted(), createdAt: ISO8601DateFormatter().string(from: now)
        )
        let artifact = ClassifierArtifact(spec: spec, features: extractor.descriptor, model: result.model, training: info, evaluation: nil)
        let holdoutRows = try await extract(split.holdout, extractor, log: log)
        let report = try await evaluate(artifact, split: split, rows: holdoutRows, labels: labels, baseline: baseline)

        return artifact.withEvaluation(report)
    }

    /// Re-evaluate an artifact on a dataset. Holdout rows whose content was
    /// trained on are excluded and counted, so a changed dataset cannot leak.
    public static func evaluate(_ classifier: Classifier, examples: [DatasetExample], labels: [String: LabelRecord],
                                baseline: FeatureExtractor? = nil) async throws -> EvaluationReport {
        let artifact = classifier.artifact
        let split = Splitter.split(examples, labels: labels, spec: artifact.spec, excluding: Set(artifact.training.trainHashes))

        guard !split.holdout.isEmpty else { throw DistillError.training("no held-out rows to evaluate") }

        let rows = try await extract(split.holdout, classifier.extractor, log: { _ in })

        return try await evaluate(artifact, split: split, rows: rows, labels: labels, baseline: baseline)
    }

    private static func evaluate(_ artifact: ClassifierArtifact, split: Split, rows: [FeatureRow], labels: [String: LabelRecord],
                                 baseline: FeatureExtractor?) async throws -> EvaluationReport {
        let spec = artifact.spec
        let names = spec.labelNames
        let reference = split.holdout.map(\.label)
        let probabilities = rows.map { artifact.model.probabilities($0.vector) }
        let served = probabilities.map { spec.labelIndex(artifact.decide($0).label)! }
        let abstained = zip(probabilities, served).filter { argmax($0) != $1 }.count

        var baselines = [majority(split, reference: reference, names: names)]
        baselines.append(try await zeroShot(split.holdout, rows: rows, baseline: baseline, reference: reference, names: names))

        let gold = split.holdout.indices.compactMap { i in split.holdout[i].example.gold.flatMap(spec.labelIndex).map { (i, $0) } }
        let studentVsGold = gold.isEmpty ? nil : Metrics(reference: gold.map(\.1), predicted: gold.map { served[$0.0] }, labels: names)
        let teacherVsGold = gold.isEmpty ? nil : Metrics(reference: gold.map(\.1), predicted: gold.map { reference[$0.0] }, labels: names)

        return EvaluationReport(
            reference: "teacher (\(Set(split.holdout.compactMap { labels[$0.example.id]?.teacher }).sorted().joined(separator: ",")))",
            split: split.report, student: Metrics(reference: reference, predicted: probabilities.map(argmax), labels: names),
            served: Metrics(reference: reference, predicted: served, labels: names), abstainRate: Double(abstained) / Double(max(1, served.count)),
            baselines: baselines, studentVsGold: studentVsGold, teacherVsGold: teacherVsGold
        )
    }

    private static func majority(_ split: Split, reference: [Int], names: [String]) -> Baseline {
        let counts = Dictionary(grouping: split.train.map(\.label), by: { $0 }).mapValues(\.count)
        let top = counts.max { $0.value < $1.value || ($0.value == $1.value && $0.key > $1.key) }?.key ?? 0

        return Baseline(name: "Majority class (\(names[top]))", status: "ok",
                        metrics: Metrics(reference: reference, predicted: reference.map { _ in top }, labels: names))
    }

    /// The untrained baseline: the stock Laya checkpoint answering the task as a
    /// generic `choice` question. It is not a distilled model.
    private static func zeroShot(_ holdout: [LabeledExample], rows: [FeatureRow], baseline: FeatureExtractor?,
                                 reference: [Int], names: [String]) async throws -> Baseline {
        let name = "Laya zero-shot choice (untrained)"
        var picks = rows.compactMap(\.zeroShot)

        if picks.count != rows.count, let baseline {
            picks = try await extract(holdout, baseline, log: { _ in }).compactMap(\.zeroShot)
        }
        guard picks.count == reference.count else { return Baseline(name: name, status: "not run (needs --model/--assets)", metrics: nil) }

        return Baseline(name: name, status: "ok", metrics: Metrics(reference: reference, predicted: picks, labels: names))
    }

    private static func extract(_ items: [LabeledExample], _ extractor: FeatureExtractor, log: (String) -> Void) async throws -> [FeatureRow] {
        var rows: [FeatureRow] = []
        rows.reserveCapacity(items.count)

        for (index, item) in items.enumerated() {
            rows.append(try await extractor.extract(item.example.input))
            if (index + 1) % 100 == 0 { log("[features] \(index + 1)/\(items.count)") }
        }

        return rows
    }

    private static func checkCoverage(_ split: Split, spec: TaskSpec) throws {
        guard !split.holdout.isEmpty else { throw DistillError.training("holdout is empty; add rows or raise dataset.holdout_fraction") }

        let present = Set(split.train.map(\.label))
        let missing = spec.labels.indices.filter { !present.contains($0) }.map { spec.labels[$0].name }

        guard missing.isEmpty else { throw DistillError.training("no training rows for label(s) \(missing.joined(separator: ", "))") }
    }
}
