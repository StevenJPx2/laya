import Foundation

public enum Workflow {
    /// Train a student on gated teacher labels and evaluate it on the teacher
    /// holdout and on human gold rows. Labels are already on disk; Laya runs
    /// only when `representations` is given (required for `laya-logits-v1`,
    /// optional for hashed students, where it adds Laya zero-shot on gold).
    public static func train(spec: TaskSpec, examples: [DatasetExample], labels: [String: LabelRecord],
                             representations: RepresentationProvider? = nil, now: Date = Date(),
                             log: (String) -> Void = { _ in }) async throws -> ClassifierArtifact {
        let split = Splitter.split(examples, labels: labels, spec: spec)
        try checkCoverage(split, spec: spec)

        log("[train] \(split.train.count) train / \(split.holdout.count) teacher holdout / \(split.gold.count) gold")

        let logits = try await collectLogits(split, spec: spec, includeTrain: true, representations: representations, log: log)
        let encoder = FeatureEncoder(descriptor(spec, train: split.train, logits: logits, representations: representations))
        let dimensions = encoder.descriptor.dimensions
        let rows = try split.train.map { try encoder.vector($0.example.input, laya: logits[$0.example.contentHash]) }
        let selection = try ModelSelection.selectL2(split.train, rows: rows, spec: spec, dimensions: dimensions)
        let l2 = selection?.l2 ?? spec.student.l2

        if let selection {
            log("[train] l2 \(l2) selected by \(selection.folds)-fold CV on train: " + selection.candidates.map { "\($0.l2)=\(percent($0.accuracy))" }.joined(separator: " "))
        }

        let result = try Trainer.train(rows, labels: split.train.map(\.label), classes: spec.labels.count,
                                       dimensions: dimensions, config: spec.student, l2: l2)
        log("[train] final loss \(String(format: "%.4f", result.finalLoss)), train accuracy \(percent(result.trainAccuracy))")

        let present = Set(split.train.map(\.label))
        let info = TrainingInfo(
            teacher: teachers(split.train, labels), questionSha256: LayaQuestion.sha256(spec), examples: split.train.count,
            epochs: spec.student.epochs, l2: l2, selection: selection, finalLoss: result.finalLoss, trainAccuracy: result.trainAccuracy,
            labelsWithoutTrainingRows: spec.labels.indices.filter { !present.contains($0) }.map { spec.labelNames[$0] },
            datasetSha256: Dataset.fingerprint(examples), labelsSha256: LabelStore.fingerprint(labels),
            trainHashes: split.train.map(\.example.contentHash).sorted(), createdAt: ISO8601DateFormatter().string(from: now)
        )
        let artifact = ClassifierArtifact(spec: spec, features: encoder.descriptor, model: result.model, training: info, evaluation: nil)

        return artifact.withEvaluation(try report(artifact, encoder: encoder, split: split, labels: labels, logits: logits))
    }

    /// Re-evaluate an artifact. Evaluation rows whose content was trained on are
    /// excluded and counted, so a changed dataset cannot leak into the metrics.
    public static func evaluate(_ classifier: Classifier, examples: [DatasetExample], labels: [String: LabelRecord],
                                log: (String) -> Void = { _ in }) async throws -> EvaluationReport {
        let artifact = classifier.artifact
        let split = Splitter.split(examples, labels: labels, spec: artifact.spec, excluding: Set(artifact.training.trainHashes))

        guard !split.holdout.isEmpty || !split.gold.isEmpty else { throw DistillError.training("no evaluation rows (teacher holdout or gold)") }

        let logits = try await collectLogits(split, spec: artifact.spec, includeTrain: false, representations: classifier.representations, log: log)

        return try report(artifact, encoder: classifier.encoder, split: split, labels: labels, logits: logits)
    }

    /// Logit students need logits for every row they are fit or scored on;
    /// hashed students with a runtime need them on gold only, for zero-shot.
    private static func collectLogits(_ split: Split, spec: TaskSpec, includeTrain: Bool, representations: RepresentationProvider?,
                                      log: (String) -> Void) async throws -> [String: [Double]] {
        guard let representations else {
            guard spec.student.features == .hashedNgram else { throw DistillError.training("student.features \(spec.student.features.rawValue) needs a Laya runtime") }
            return [:]
        }

        let rows = spec.student.features.usesLaya ? (includeTrain ? split.train : []) + split.holdout + split.gold : split.gold

        return try await LayaLogits.collect(rows.map(\.example), spec: spec, provider: representations, log: log)
    }

    private static func descriptor(_ spec: TaskSpec, train: [LabeledExample], logits: [String: [Double]],
                                   representations: RepresentationProvider?) -> FeatureDescriptor {
        guard spec.student.features.usesLaya, let representations else { return .hashed(dimensions: spec.student.hashDimensions) }

        let scheme = spec.student.features
        let rows = train.compactMap { logits[$0.example.contentHash] }
        let dense = FeatureEncoder.denseDimensions(scheme, labels: spec.labels.count, laya: rows.first?.count ?? spec.labels.count)
        let statistics = LayaLogits.statistics(rows.map { Array($0.prefix(dense)) })
        let dimensions = dense + (scheme == .layaHybrid ? spec.student.hashDimensions : 0)

        return FeatureDescriptor(scheme: scheme, dimensions: dimensions, mean: statistics.mean, sd: statistics.sd,
                                 questionSha256: LayaQuestion.sha256(spec), layaFingerprint: representations.fingerprint)
    }

    private static func report(_ artifact: ClassifierArtifact, encoder: FeatureEncoder, split: Split, labels: [String: LabelRecord],
                               logits: [String: [Double]]) throws -> EvaluationReport {
        let names = artifact.spec.labelNames
        let majority = majorityLabel(split.train, classes: names.count)

        func predictions(_ items: [LabeledExample]) throws -> (argmax: [Int], served: [Int]) {
            let probabilities = try items.map { artifact.model.probabilities(try encoder.vector($0.example.input, laya: logits[$0.example.contentHash])) }
            return (probabilities.map(argmax), probabilities.map { artifact.spec.labelIndex(artifact.decide($0).label)! })
        }

        var agreement: AgreementReport?
        if !split.holdout.isEmpty {
            let reference = split.holdout.map(\.label)
            let predicted = try predictions(split.holdout)
            agreement = AgreementReport(
                student: Metrics(reference: reference, predicted: predicted.argmax, labels: names),
                served: Metrics(reference: reference, predicted: predicted.served, labels: names),
                abstainRate: Double(zip(predicted.argmax, predicted.served).filter { $0 != $1 }.count) / Double(reference.count),
                majority: Metrics(reference: reference, predicted: reference.map { _ in majority }, labels: names)
            )
        }

        let gold = split.gold.isEmpty ? nil : try goldReport(split.gold, labels: labels, artifact: artifact, predicted: predictions(split.gold),
                                                              logits: logits, majority: majority)

        return EvaluationReport(teacher: teachers(split.train + split.holdout, labels), split: split.report, agreement: agreement, gold: gold)
    }

    private static func goldReport(_ gold: [LabeledExample], labels: [String: LabelRecord], artifact: ClassifierArtifact,
                                   predicted: (argmax: [Int], served: [Int]), logits: [String: [Double]], majority: Int) -> GoldReport {
        let spec = artifact.spec
        let names = spec.labelNames
        let reference = gold.map(\.label)
        let question = LayaQuestion.sha256(spec)
        let current = gold.filter { item in
            guard let record = labels[item.example.id] else { return false }
            return record.contentHash == item.example.contentHash && record.questionSha256 == question
        }

        func metrics(_ pairs: [(Int, Int)]) -> Metrics? {
            pairs.isEmpty ? nil : Metrics(reference: pairs.map(\.0), predicted: pairs.map(\.1), labels: names)
        }
        func teacher(_ pick: (LabelRecord) -> String?) -> Metrics? {
            metrics(current.compactMap { item in pick(labels[item.example.id]!).flatMap(spec.labelIndex).map { (item.label, $0) } })
        }

        let served = Metrics(reference: reference, predicted: predicted.served, labels: names)
        let zeroShot = metrics(gold.compactMap { item in logits[item.example.contentHash].map { (item.label, argmax(Array($0.prefix(names.count)))) } })

        return GoldReport(
            studentServed: served,
            studentArgmax: Metrics(reference: reference, predicted: predicted.argmax, labels: names),
            teacher: current.isEmpty ? nil : teachers(current, labels),
            teacherRaw: teacher(\.layaLabel), teacherGated: teacher(\.label), layaZeroShot: zeroShot,
            majority: Metrics(reference: reference, predicted: reference.map { _ in majority }, labels: names),
            beatsLayaZeroShot: zeroShot.map { served.accuracy > $0.accuracy }
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
            throw DistillError.training("need teacher-labeled training rows for at least 2 labels (have \(split.train.count) rows); run label or import, or relax teacher.min_confidence")
        }
        guard !split.holdout.isEmpty else { throw DistillError.training("teacher holdout is empty; add unlabeled rows or raise dataset.holdout_fraction") }
    }
}
