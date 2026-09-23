import CryptoKit
import Foundation
import LayaCore

public struct SparseVector: Sendable {
    public let indices: [Int]
    public let values: [Double]

    init(dense: [Double]) {
        indices = Array(dense.indices)
        values = dense
    }

    init(indices: [Int], values: [Double]) {
        self.indices = indices
        self.values = values
    }
}

public struct FeatureRow: Sendable {
    public let vector: SparseVector
    /// Laya's own zero-shot pick (spec label index); the untrained baseline.
    public let zeroShot: Int?
}

public struct FeatureDescriptor: Codable, Sendable, Equatable {
    public let kind: FeatureKind
    public let dimensions: Int
    /// Laya asset fingerprint; artifacts refuse to load against other assets.
    public let modelFingerprint: String?

    enum CodingKeys: String, CodingKey { case kind, dimensions, modelFingerprint = "model_fingerprint" }
}

public protocol FeatureExtractor: Sendable {
    var descriptor: FeatureDescriptor { get }
    func extract(_ input: TaskInput) async throws -> FeatureRow
}

/// Hashed word unigrams and bigrams, globally and per field, with log term
/// frequency and L2 normalization. Deterministic (FNV-1a), no model required.
public struct HashedFeatures: FeatureExtractor {
    public let descriptor: FeatureDescriptor

    public init(dimensions: Int) {
        descriptor = FeatureDescriptor(kind: .hashed, dimensions: dimensions, modelFingerprint: nil)
    }

    public func extract(_ input: TaskInput) async throws -> FeatureRow {
        var counts: [Int: Double] = [:]

        for (name, value) in input.fields {
            let words = Prompt.serialize(value).lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)

            for (index, word) in words.enumerated() {
                counts[bucket(word), default: 0] += 1
                counts[bucket("\(name)=\(word)"), default: 0] += 1
                if index > 0 { counts[bucket("\(words[index - 1]) \(word)"), default: 0] += 1 }
            }
        }

        let indices = counts.keys.sorted()
        let raw = indices.map { log1p(counts[$0]!) }
        let norm = max(1e-12, sqrt(raw.reduce(0) { $0 + $1 * $1 }))

        return FeatureRow(vector: SparseVector(indices: indices, values: raw.map { $0 / norm }), zeroShot: nil)
    }

    private func bucket(_ token: String) -> Int {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325

        for byte in token.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100_0000_01b3
        }

        return Int(hash % UInt64(descriptor.dimensions))
    }
}

/// Frozen Laya encoder features: pooled decision vector (hidden size) followed
/// by the checkpoint's raw logit for each task label. Only the head trained on
/// top is task-specific; the checkpoint weights are unchanged.
public struct LayaFeatures: FeatureExtractor {
    public let descriptor: FeatureDescriptor
    let runtime: LayaRuntime
    let question: Question
    let order: [Int]

    public init(runtime: LayaRuntime, spec: TaskSpec, fingerprint: String, hidden: Int = 1024) throws {
        let criteria = JSONValue.object(spec.labels.map { ($0.name, .string($0.description)) })
        let question = Question(type: "choice", instructions: .string(spec.instructions), criteria: criteria)
        let rendered = try Prompt.renderedOptions(question)

        order = try spec.labels.map { label in
            guard let index = rendered.firstIndex(of: "\(label.name): \(label.description)") else {
                throw DistillError.training("label '\(label.name)' is missing from the rendered Laya question")
            }

            return index
        }

        self.runtime = runtime
        self.question = question
        descriptor = FeatureDescriptor(kind: .laya, dimensions: hidden + spec.labels.count, modelFingerprint: fingerprint)
    }

    public func extract(_ input: TaskInput) async throws -> FeatureRow {
        let representation = try await runtime.representation(state: input.jsonValue, question: question)
        let logits = order.map { representation.logits[$0] }
        let zeroShot = logits.indices.max { logits[$0] < logits[$1] }

        guard representation.pooled.count + logits.count == descriptor.dimensions else {
            throw DistillError.training("Laya representation has \(representation.pooled.count) dims; expected \(descriptor.dimensions - logits.count)")
        }

        return FeatureRow(vector: SparseVector(dense: representation.pooled + logits), zeroShot: zeroShot)
    }

    /// Identifies the Laya asset set cheaply without hashing ~800 MB of weights:
    /// runtime manifest, action-head weights, and the first 4 MiB of embeddings.
    public static func fingerprint(assets: URL) throws -> String {
        var hasher = SHA256()

        for (name, limit) in [("runtime_manifest.json", Int.max), ("act_head.f32.bin", Int.max), ("embeddings.f16.bin", 4 << 20)] {
            let handle = try FileHandle(forReadingFrom: assets.appendingPathComponent(name))
            defer { try? handle.close() }

            hasher.update(data: Data(name.utf8))
            hasher.update(data: try handle.read(upToCount: limit) ?? Data())
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
