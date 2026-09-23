import Foundation
import LayaCore

public struct TrainingInfo: Codable, Sendable {
    /// Teacher identities whose labels trained this student.
    public let teacher: String
    public let questionSha256: String
    public let examples: Int
    public let epochs: Int
    public let l2: Double
    /// Present when L2 was chosen by cross-validation on the training split.
    public let selection: SelectionResult?
    public let finalLoss: Double
    public let trainAccuracy: Double
    public let labelsWithoutTrainingRows: [String]
    public let datasetSha256: String
    public let labelsSha256: String
    /// Content hashes of training rows, so evaluation can prove disjointness.
    public let trainHashes: [String]
    public let createdAt: String

    enum CodingKeys: String, CodingKey {
        case teacher, examples, epochs, l2, selection
        case questionSha256 = "question_sha256", finalLoss = "final_loss", trainAccuracy = "train_accuracy"
        case labelsWithoutTrainingRows = "labels_without_training_rows", datasetSha256 = "dataset_sha256"
        case labelsSha256 = "labels_sha256", trainHashes = "train_hashes", createdAt = "created_at"
    }
}

/// A versioned student. Hashed-feature students are self-contained; logit
/// students also need the Laya runtime whose fingerprint they record.
public struct ClassifierArtifact: Codable, Sendable {
    public static let format = "laya.classifier"
    public static let formatVersion = 3

    public var format = Self.format
    public var formatVersion = Self.formatVersion
    public let spec: TaskSpec
    public let specSha256: String
    public let features: FeatureDescriptor
    public let model: LinearModel
    public let training: TrainingInfo
    public let evaluation: EvaluationReport?
    public var integrity = ""

    enum CodingKeys: String, CodingKey {
        case format, spec, features, model, training, evaluation, integrity
        case formatVersion = "format_version", specSha256 = "spec_sha256"
    }

    init(spec: TaskSpec, features: FeatureDescriptor, model: LinearModel, training: TrainingInfo, evaluation: EvaluationReport?) {
        self.spec = spec
        specSha256 = spec.sha256
        self.features = features
        self.model = model
        self.training = training
        self.evaluation = evaluation
    }

    func withEvaluation(_ report: EvaluationReport) -> ClassifierArtifact {
        ClassifierArtifact(spec: spec, features: features, model: model, training: training, evaluation: report)
    }

    private func payloadHash() throws -> String {
        var copy = self
        copy.integrity = ""

        return Canonical.sha256(try Canonical.encoder().encode(copy))
    }

    public func save(to url: URL) throws {
        var sealed = self
        sealed.integrity = try payloadHash()

        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Canonical.encoder().encode(sealed).write(to: url, options: .atomic)
    }

    public static func load(_ url: URL) throws -> ClassifierArtifact {
        let data = try Data(contentsOf: url)
        guard data.count <= 256 << 20 else { throw DistillError.artifact("artifact exceeds 256 MiB") }

        let artifact: ClassifierArtifact
        do {
            artifact = try JSONDecoder().decode(ClassifierArtifact.self, from: data)
        } catch {
            throw DistillError.artifact("\(url.lastPathComponent): \(describe(error))")
        }

        try artifact.verify()

        return artifact
    }

    func verify() throws {
        guard format == Self.format, formatVersion == Self.formatVersion else {
            throw DistillError.artifact("unsupported artifact format \(format) v\(formatVersion); retrain with this version")
        }
        guard try payloadHash() == integrity else { throw DistillError.artifact("integrity check failed; the artifact was modified or truncated") }
        guard spec.sha256 == specSha256 else { throw DistillError.artifact("embedded spec does not match spec_sha256") }

        try spec.validate()
        try model.validate()
        try features.validate(spec: spec)

        guard features.scheme == spec.student.features, model.classes == spec.labels.count, model.dimensions == features.dimensions else {
            throw DistillError.artifact("model shape does not match the task labels or feature scheme")
        }
    }

    /// Apply the abstain policy to student probabilities. Shared by serving and evaluation.
    func decide(_ probabilities: [Double]) -> Prediction {
        let top = argmax(probabilities)
        let abstain = spec.abstain.flatMap { policy in probabilities[top] < policy.minConfidence ? policy.label : nil }
        let rounded = Dictionary(uniqueKeysWithValues: zip(spec.labelNames, probabilities.map { ($0 * 10_000).rounded() / 10_000 }))

        return Prediction(classifier: spec.name, version: spec.version, label: abstain ?? spec.labelNames[top], argmax: spec.labelNames[top],
                          confidence: (probabilities[top] * 10_000).rounded() / 10_000, probabilities: rounded, abstained: abstain != nil)
    }
}

public struct Prediction: Codable, Sendable {
    public let classifier: String
    public let version: String
    public let label: String
    /// Student's top label before the abstain policy.
    public let argmax: String
    public let confidence: Double
    public let probabilities: [String: Double]
    public let abstained: Bool
}

/// A loaded student. Hashed students run on the input alone; logit students
/// run one Laya forward pass per prediction through `representations`.
public struct Classifier: Sendable {
    public let artifact: ClassifierArtifact
    let encoder: FeatureEncoder
    let representations: RepresentationProvider?
    private let question: Question

    /// Rejects a logit student without a runtime, or with a runtime whose
    /// asset fingerprint differs from the one it was trained on.
    public init(artifact: ClassifierArtifact, representations: RepresentationProvider? = nil) throws {
        if artifact.features.scheme.usesLaya {
            guard let representations else {
                throw DistillError.artifact("\(artifact.spec.name) uses \(artifact.features.scheme.rawValue) features and needs a Laya runtime")
            }
            guard representations.fingerprint == artifact.features.layaFingerprint else {
                throw DistillError.artifact("\(artifact.spec.name) was trained on Laya assets \(artifact.features.layaFingerprint?.prefix(12) ?? "-") "
                                            + "but the runtime has \(representations.fingerprint.prefix(12)); retrain against this runtime")
            }
        }

        self.artifact = artifact
        self.representations = representations
        encoder = FeatureEncoder(artifact.features)
        question = LayaQuestion.make(artifact.spec)
    }

    public func predict(_ raw: JSONValue) async throws -> Prediction {
        let input = try artifact.spec.input.parse(raw)

        return artifact.decide(try probabilities(input, logits: try await logits(input)))
    }

    /// Laya features (label-ordered logits, then pooled) when this student consumes them.
    func logits(_ input: TaskInput) async throws -> [Double]? {
        guard artifact.features.scheme.usesLaya, let representations else { return nil }

        return try await LayaLogits.features(input, spec: artifact.spec, question: question, provider: representations)
    }

    func probabilities(_ input: TaskInput, logits: [Double]?) throws -> [Double] {
        artifact.model.probabilities(try encoder.vector(input, laya: logits))
    }
}
