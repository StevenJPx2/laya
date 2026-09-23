import Foundation
import LayaCore

public struct SparseVector: Sendable {
    public let indices: [Int]
    public let values: [Double]
}

public struct FeatureDescriptor: Codable, Sendable, Equatable {
    public static let scheme = "hashed-ngram-v1"

    public let scheme: String
    public let dimensions: Int
}

/// Hashed word unigrams and bigrams, globally and per field, with log term
/// frequency and L2 normalization. Deterministic (FNV-1a) and model-free, so a
/// trained student runs without Laya.
public struct HashedFeatures: Sendable {
    public let descriptor: FeatureDescriptor

    public init(dimensions: Int) {
        descriptor = FeatureDescriptor(scheme: FeatureDescriptor.scheme, dimensions: dimensions)
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
