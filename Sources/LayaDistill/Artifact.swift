import Foundation
import LayaCore

public struct TrainingInfo: Codable, Sendable {
    public let teacher: String
    public let examples: Int
    public let epochs: Int
    public let l2: Double
    /// Present when L2 was chosen by cross-validation on the training split.
    public let selection: SelectionResult?
    public let finalLoss: Double
    public let trainAccuracy: Double
    public let datasetSha256: String
    public let labelsSha256: String
    /// Content hashes of training rows, so evaluation can prove holdout disjointness.
    public let trainHashes: [String]
    public let createdAt: String

    enum CodingKeys: String, CodingKey {
        case teacher, examples, epochs, l2, selection
        case finalLoss = "final_loss", trainAccuracy = "train_accuracy", datasetSha256 = "dataset_sha256"
        case labelsSha256 = "labels_sha256", trainHashes = "train_hashes", createdAt = "created_at"
    }
}

/// A self-contained, versioned classifier. Everything needed to validate
/// inputs, extract features, and apply the abstain policy travels with it.
public struct ClassifierArtifact: Codable, Sendable {
    public static let format = "laya.classifier"
    public static let formatVersion = 1

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
            throw DistillError.artifact("unsupported artifact format \(format) v\(formatVersion)")
        }
        guard try payloadHash() == integrity else { throw DistillError.artifact("integrity check failed; the artifact was modified or truncated") }
        guard spec.sha256 == specSha256 else { throw DistillError.artifact("embedded spec does not match spec_sha256") }

        try spec.validate()
        try model.validate()

        guard model.classes == spec.labels.count, model.dimensions == features.dimensions else {
            throw DistillError.artifact("model shape does not match the task labels or feature dimensions")
        }
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

/// Loaded artifact plus the feature extractor it was trained with.
public struct Classifier: Sendable {
    public let artifact: ClassifierArtifact
    let extractor: FeatureExtractor

    /// `runtime`/`assets` are required only for `laya` feature artifacts; the
    /// asset fingerprint must match the one recorded at training time.
    public init(artifact: ClassifierArtifact, runtime: LayaRuntime?, assets: URL?) throws {
        switch artifact.features.kind {
        case .hashed:
            extractor = HashedFeatures(dimensions: artifact.features.dimensions)
        case .laya:
            guard let runtime, let assets else { throw DistillError.artifact("\(artifact.spec.name) needs the Laya model (--model/--assets)") }

            let fingerprint = try LayaFeatures.fingerprint(assets: assets)
            guard fingerprint == artifact.features.modelFingerprint else {
                throw DistillError.artifact("\(artifact.spec.name) was trained on different Laya assets (fingerprint mismatch)")
            }

            extractor = try LayaFeatures(runtime: runtime, spec: artifact.spec, fingerprint: fingerprint, hidden: artifact.features.dimensions - artifact.spec.labels.count)
        }

        self.artifact = artifact
    }

    public func predict(_ raw: JSONValue) async throws -> Prediction {
        let input = try artifact.spec.input.parse(raw)
        let probabilities = artifact.model.probabilities(try await extractor.extract(input).vector)

        return artifact.decide(probabilities)
    }
}

extension ClassifierArtifact {
    /// Apply the abstain policy to student probabilities. Shared by serving and evaluation.
    func decide(_ probabilities: [Double]) -> Prediction {
        let top = argmax(probabilities)
        let abstain = spec.abstain.flatMap { policy in probabilities[top] < policy.minConfidence ? policy.label : nil }
        let rounded = Dictionary(uniqueKeysWithValues: zip(spec.labelNames, probabilities.map { ($0 * 10_000).rounded() / 10_000 }))

        return Prediction(classifier: spec.name, version: spec.version, label: abstain ?? spec.labelNames[top], argmax: spec.labelNames[top],
                          confidence: (probabilities[top] * 10_000).rounded() / 10_000, probabilities: rounded, abstained: abstain != nil)
    }
}
