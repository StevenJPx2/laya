import Foundation
import LayaCore

public struct SparseVector: Sendable {
    public let indices: [Int]
    public let values: [Double]
}

/// How an artifact turns an input into its feature vector. The Laya fields are
/// set only for Laya feature schemes.
public struct FeatureDescriptor: Codable, Sendable, Equatable {
    public let scheme: FeatureScheme
    public let dimensions: Int
    /// Training-split mean and standard deviation per logit, in label order.
    public let mean: [Double]?
    public let sd: [Double]?
    public let questionSha256: String?
    /// Asset fingerprint of the Laya runtime the logits came from.
    public let layaFingerprint: String?

    enum CodingKeys: String, CodingKey {
        case scheme, dimensions, mean, sd, questionSha256 = "question_sha256", layaFingerprint = "laya_fingerprint"
    }

    static func hashed(dimensions: Int) -> FeatureDescriptor {
        FeatureDescriptor(scheme: .hashedNgram, dimensions: dimensions, mean: nil, sd: nil, questionSha256: nil, layaFingerprint: nil)
    }

    func validate(spec: TaskSpec) throws {
        switch scheme {
        case .hashedNgram:
            guard mean == nil, sd == nil, questionSha256 == nil, layaFingerprint == nil else {
                throw DistillError.artifact("hashed-ngram-v1 features carry Laya statistics")
            }

        case .layaLogits, .layaEmbedding, .layaHybrid:
            let dense = mean?.count ?? 0
            let shape = switch scheme {
            case .layaLogits: dense == spec.labels.count && dimensions == dense
            case .layaEmbedding: dense > spec.labels.count && dimensions == dense
            default: dense > spec.labels.count && dimensions > dense
            }
            guard let mean, let sd, let questionSha256, let layaFingerprint, !layaFingerprint.isEmpty,
                  shape, sd.count == mean.count,
                  mean.allSatisfy(\.isFinite), sd.allSatisfy({ $0.isFinite && $0 >= LayaLogits.sdFloor }) else {
                throw DistillError.artifact("\(scheme.rawValue) features need finite mean/sd per dimension and a Laya fingerprint")
            }
            guard questionSha256 == LayaQuestion.sha256(spec) else { throw DistillError.artifact("\(scheme.rawValue) question hash does not match the spec") }
        }
    }
}

/// Hashed word unigrams and bigrams, globally and per field, with log term
/// frequency and L2 normalization. Deterministic (FNV-1a) and model-free, so a
/// trained student runs without Laya.
public struct HashedFeatures: Sendable {
    public let descriptor: FeatureDescriptor

    public init(dimensions: Int) {
        descriptor = .hashed(dimensions: dimensions)
    }

    public func extract(_ input: TaskInput) -> SparseVector {
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

        return SparseVector(indices: indices, values: raw.map { $0 / norm })
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
