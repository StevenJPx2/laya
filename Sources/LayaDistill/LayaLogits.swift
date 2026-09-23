import Foundation
import LayaCore

/// The frozen-Laya feature boundary: one typed question in, its representation
/// out. Implemented by the in-process runtime and by test fakes.
public protocol RepresentationProvider: Sendable {
    /// Identifies the Laya assets; a logit student only runs on the assets it was trained on.
    var fingerprint: String { get }
    func representation(state: JSONValue, question: Question) async throws -> Representation
}

/// Laya running in this process on the Neural Engine.
public struct RuntimeRepresentations: RepresentationProvider {
    public let fingerprint: String
    let runtime: LayaRuntime

    public init(runtime: LayaRuntime, assets: URL) throws {
        self.runtime = runtime
        fingerprint = try RuntimeTeacher.fingerprint(assets: assets)
    }

    public func representation(state: JSONValue, question: Question) async throws -> Representation {
        try await runtime.representation(state: state, question: question)
    }
}

/// `laya-logits-v1`: Laya's raw (uncalibrated) option logits for the task
/// question, in spec label order.
enum LayaLogits {
    static let sdFloor = 1e-6

    static func extract(_ input: TaskInput, spec: TaskSpec, question: Question, provider: RepresentationProvider) async throws -> [Double] {
        Array(try await features(input, spec: spec, question: question, provider: provider).prefix(spec.labels.count))
    }

    /// Label-ordered logits followed by the pooled vector: everything a Laya
    /// student can consume, from one forward pass.
    static func features(_ input: TaskInput, spec: TaskSpec, question: Question, provider: RepresentationProvider) async throws -> [Double] {
        let representation = try await provider.representation(state: input.jsonValue, question: question)
        guard representation.pooled.allSatisfy(\.isFinite) else { throw DistillError.training("Laya returned a non-finite pooled vector") }

        return try labelOrder(representation, spec: spec).map { representation.logits[$0] } + representation.pooled
    }

    /// Option index of each label. Laya orders choice options itself, so choice
    /// labels are matched by their rendered option; score and noul keep label order.
    static func labelOrder(_ representation: Representation, spec: TaskSpec) throws -> [Int] {
        let count = spec.labels.count

        guard representation.logits.count == count, representation.options.count == count, representation.logits.allSatisfy(\.isFinite) else {
            throw DistillError.training("Laya returned \(representation.logits.count) logits for \(count) labels")
        }
        guard spec.teacher.questionType == .choice else { return Array(0..<count) }

        return try spec.labels.map { label in
            guard let index = representation.options.firstIndex(of: "\(label.name): \(label.description)") else {
                throw DistillError.training("Laya options do not include label \(label.name)")
            }
            return index
        }
    }

    /// Logits for every distinct input, keyed by content hash. This is the
    /// per-run cache: CV, the final fit, and evaluation never re-run Laya.
    static func collect(_ examples: [DatasetExample], spec: TaskSpec, provider: RepresentationProvider,
                        log: (String) -> Void) async throws -> [String: [Double]] {
        let question = LayaQuestion.make(spec)
        var seen = Set<String>()
        let pending = examples.filter { seen.insert($0.contentHash).inserted }
        var cache: [String: [Double]] = [:]

        for (index, example) in pending.enumerated() {
            cache[example.contentHash] = try await features(example.input, spec: spec, question: question, provider: provider)
            if (index + 1) % 100 == 0 || index + 1 == pending.count { log("[laya] features \(index + 1)/\(pending.count)") }
        }

        return cache
    }

    /// Per-dimension mean and population standard deviation, floored at `sdFloor`.
    static func statistics(_ rows: [[Double]]) -> (mean: [Double], sd: [Double]) {
        let dimensions = rows.first?.count ?? 0
        let n = Double(max(1, rows.count))
        let mean = (0..<dimensions).map { d in rows.reduce(0) { $0 + $1[d] } / n }
        let sd = (0..<dimensions).map { d in max(sdFloor, sqrt(rows.reduce(0) { $0 + pow($1[d] - mean[d], 2) } / n)) }

        return (mean, sd)
    }
}

/// Maps a validated input (plus its cached Laya features, for Laya schemes) to
/// the student's feature vector. Shared by training, evaluation, and serving.
/// Standardized Laya values come first; hybrid students append hashed n-grams.
struct FeatureEncoder: Sendable {
    let descriptor: FeatureDescriptor
    private let hashed: HashedFeatures?
    private let dense: Int

    init(_ descriptor: FeatureDescriptor) {
        self.descriptor = descriptor
        dense = descriptor.mean?.count ?? 0

        let hashDimensions = descriptor.dimensions - dense
        hashed = descriptor.scheme == .hashedNgram || descriptor.scheme == .layaHybrid ? HashedFeatures(dimensions: hashDimensions) : nil
    }

    /// `laya` is the cached Laya feature vector (logits, then pooled); a Laya
    /// scheme standardizes its leading `mean.count` values.
    func vector(_ input: TaskInput, laya: [Double]?) throws -> SparseVector {
        if descriptor.scheme == .hashedNgram, let hashed { return hashed.extract(input) }

        guard let laya, let mean = descriptor.mean, let sd = descriptor.sd, dense > 0, laya.count >= dense, sd.count == dense else {
            throw DistillError.training("\(descriptor.scheme.rawValue) features need Laya features for every row")
        }

        let values = zip(laya.prefix(dense), zip(mean, sd)).map { ($0 - $1.0) / $1.1 }
        guard let hashed else { return SparseVector(indices: Array(0..<dense), values: values) }

        let words = hashed.extract(input)

        return SparseVector(indices: Array(0..<dense) + words.indices.map { $0 + dense }, values: values + words.values)
    }

    /// How many leading Laya values a scheme standardizes.
    static func denseDimensions(_ scheme: FeatureScheme, labels: Int, laya: Int) -> Int {
        switch scheme {
        case .hashedNgram: 0
        case .layaLogits: labels
        case .layaEmbedding, .layaHybrid: laya
        }
    }
}
