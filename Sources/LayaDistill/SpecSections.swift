import Foundation
import LayaCore

public enum FieldType: String, Codable, Sendable { case string, number, boolean, json }

public struct FieldSpec: Codable, Sendable {
    public let name: String
    public let type: FieldType
    public let required: Bool
    public let maxChars: Int
    public let description: String?

    enum CodingKeys: String, CodingKey, CaseIterable { case name, type, required, maxChars = "max_chars", description }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "input.fields[]")
        name = try c.decode(String.self, forKey: .name)
        type = try c.decode(FieldType.self, forKey: .type)
        required = try c.decodeIfPresent(Bool.self, forKey: .required) ?? true
        maxChars = try c.decodeIfPresent(Int.self, forKey: .maxChars) ?? 4000
        description = try c.decodeIfPresent(String.self, forKey: .description)
    }
}

public struct InputSchema: Codable, Sendable {
    public let fields: [FieldSpec]

    enum CodingKeys: String, CodingKey, CaseIterable { case fields }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "input")
        fields = try c.decode([FieldSpec].self, forKey: .fields)
    }

    func validate() throws {
        guard (1...TaskSpec.maxFields).contains(fields.count) else { throw DistillError.invalidSpec("input.fields must contain 1-\(TaskSpec.maxFields) fields") }
        guard Set(fields.map(\.name)).count == fields.count else { throw DistillError.invalidSpec("input field names must be unique") }

        for field in fields {
            guard matches(field.name, "^[a-z][a-z0-9_]{0,63}$") else { throw DistillError.invalidSpec("field '\(field.name)' must match ^[a-z][a-z0-9_]{0,63}$") }
            guard (1...20_000).contains(field.maxChars) else { throw DistillError.invalidSpec("field '\(field.name)' max_chars must be 1-20000") }
        }
    }

    /// Validate an input object against the schema. Unknown fields, missing
    /// required fields, wrong types, and oversized values are rejected.
    public func parse(_ value: JSONValue, context: String = "input") throws -> TaskInput {
        guard case .object(let pairs) = value else { throw DistillError.invalidData("\(context) must be a JSON object") }

        let declared = Set(fields.map(\.name))
        let unknown = pairs.map(\.0).filter { !declared.contains($0) }.sorted()
        guard unknown.isEmpty else { throw DistillError.invalidData("\(context) has undeclared field(s) \(unknown.joined(separator: ", "))") }

        var values: [(String, JSONValue)] = []

        for field in fields {
            guard let fieldValue = pairs.first(where: { $0.0 == field.name })?.1, fieldValue != .null else {
                if field.required { throw DistillError.invalidData("\(context).\(field.name) is required") }
                continue
            }

            try check(fieldValue, against: field, context: "\(context).\(field.name)")
            values.append((field.name, fieldValue))
        }

        return TaskInput(fields: values)
    }

    private func check(_ value: JSONValue, against field: FieldSpec, context: String) throws {
        switch (field.type, value) {
        case (.string, .string), (.number, .number), (.boolean, .bool), (.json, _): break
        default: throw DistillError.invalidData("\(context) must be of type \(field.type.rawValue)")
        }

        guard Prompt.serialize(value).count <= field.maxChars else {
            throw DistillError.invalidData("\(context) exceeds max_chars \(field.maxChars)")
        }
    }
}

/// A schema-validated input. Field order follows the spec, so rendering and
/// hashing are deterministic regardless of the caller's key order.
public struct TaskInput: Sendable {
    public let fields: [(String, JSONValue)]

    public var jsonValue: JSONValue { .object(fields) }

    public var rendered: String {
        fields.map { "\($0.0): \(Prompt.serialize($0.1))" }.joined(separator: "\n")
    }

    /// Identity used for de-duplication and leakage checks: case- and
    /// whitespace-insensitive, so trivially reformatted copies collide.
    public var contentHash: String {
        let normalized = rendered.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")

        return Canonical.sha256(normalized)
    }
}

/// Which typed Laya question the task is asked as.
/// - `choice`: one option per label (`name: description`).
/// - `noul`: exactly two labels; the first is "false", the second "true".
/// - `score`: labels are ordered rubric levels, lowest first.
public enum QuestionType: String, Codable, Sendable { case choice, noul, score }

/// What happens to a Laya answer that fails the confidence gate.
/// - `abstain`: train on the task's abstain label (for example `ask`) instead.
/// - `drop`: exclude the row from training.
public enum UncertainPolicy: String, Codable, Sendable { case abstain, drop }

/// Where training labels come from.
/// - `laya`: `laya-distill label` asks Laya the task question.
/// - `import`: `laya-distill import` reads labels from an external teacher's ledger.
public enum TeacherSource: String, Codable, Sendable { case laya, `import` }

/// The labeling teacher (Laya by default, or imported labels). An answer
/// becomes a hard training label only when its top probability is at least
/// `min_confidence` and it leads the runner-up by at least `min_margin`.
public struct TeacherSpec: Codable, Sendable {
    public let questionType: QuestionType
    public let minConfidence: Double
    public let minMargin: Double
    public let uncertain: UncertainPolicy
    /// Stored only when declared, so specs without it keep their hash.
    let declaredSource: TeacherSource?

    public var source: TeacherSource { declaredSource ?? .laya }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case questionType = "question_type", minConfidence = "min_confidence", minMargin = "min_margin", uncertain, declaredSource = "source"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "teacher")
        questionType = try c.decode(QuestionType.self, forKey: .questionType)
        minConfidence = try c.decode(Double.self, forKey: .minConfidence)
        minMargin = try c.decodeIfPresent(Double.self, forKey: .minMargin) ?? 0
        uncertain = try c.decode(UncertainPolicy.self, forKey: .uncertain)
        declaredSource = try c.decodeIfPresent(TeacherSource.self, forKey: .declaredSource)
    }

    func validate(labels: Int, hasAbstain: Bool) throws {
        guard minConfidence >= 0, minConfidence < 1, minMargin >= 0, minMargin < 1 else {
            throw DistillError.invalidSpec("teacher.min_confidence and teacher.min_margin must be in [0, 1)")
        }
        if questionType == .noul, labels != 2 {
            throw DistillError.invalidSpec("teacher.question_type noul needs exactly 2 labels (false first, true second)")
        }
        if uncertain == .abstain, !hasAbstain {
            throw DistillError.invalidSpec("teacher.uncertain abstain requires an abstain label")
        }
    }
}

public struct DatasetSpec: Codable, Sendable {
    public let maxExamples: Int
    public let maxLineBytes: Int
    public let holdoutFraction: Double
    public let splitSeed: String

    enum CodingKeys: String, CodingKey, CaseIterable {
        case maxExamples = "max_examples", maxLineBytes = "max_line_bytes", holdoutFraction = "holdout_fraction", splitSeed = "split_seed"
    }

    init() {
        maxExamples = 5000
        maxLineBytes = 65_536
        holdoutFraction = 0.25
        splitSeed = "laya"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "dataset")
        let defaults = DatasetSpec()
        maxExamples = try c.decodeIfPresent(Int.self, forKey: .maxExamples) ?? defaults.maxExamples
        maxLineBytes = try c.decodeIfPresent(Int.self, forKey: .maxLineBytes) ?? defaults.maxLineBytes
        holdoutFraction = try c.decodeIfPresent(Double.self, forKey: .holdoutFraction) ?? defaults.holdoutFraction
        splitSeed = try c.decodeIfPresent(String.self, forKey: .splitSeed) ?? defaults.splitSeed
    }

    func validate() throws {
        guard (1...100_000).contains(maxExamples), (256...1_048_576).contains(maxLineBytes) else {
            throw DistillError.invalidSpec("dataset: max_examples 1-100000, max_line_bytes 256-1048576")
        }
        guard holdoutFraction >= 0.05, holdoutFraction <= 0.5, !splitSeed.isEmpty else {
            throw DistillError.invalidSpec("dataset: holdout_fraction 0.05-0.5 and a nonempty split_seed")
        }
    }
}

public enum ClassWeighting: String, Codable, Sendable { case balanced, none }

/// Student input features.
/// - `hashed-ngram-v1`: hashed word n-grams; serving needs no Laya model.
/// - `laya-logits-v1`: Laya's raw option logits for the task question,
///   standardized on the training split; serving runs one Laya forward pass.
/// - `laya-embedding-v1`: those logits plus Laya's pooled decision vector,
///   standardized the same way; needs more labeled rows than logits alone.
/// - `laya-hybrid-v1`: the embedding features plus `hash_dimensions` hashed n-grams.
public enum FeatureScheme: String, Codable, Sendable {
    case hashedNgram = "hashed-ngram-v1", layaLogits = "laya-logits-v1", layaEmbedding = "laya-embedding-v1", layaHybrid = "laya-hybrid-v1"

    public var usesLaya: Bool { self != .hashedNgram }
}

/// The student: a softmax head over `features`.
public struct StudentSpec: Codable, Sendable {
    public let hashDimensions: Int
    public let epochs: Int
    public let learningRate: Double
    public let l2: Double
    /// When set, L2 is chosen from these values by group-aware k-fold
    /// cross-validation on the training split only; `l2` is then ignored.
    public let l2Grid: [Double]?
    public let classWeighting: ClassWeighting
    /// Stored only when declared, so specs without it keep their hash.
    let declaredFeatures: FeatureScheme?

    public var features: FeatureScheme { declaredFeatures ?? .hashedNgram }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case hashDimensions = "hash_dimensions", epochs, learningRate = "learning_rate", l2, l2Grid = "l2_grid", classWeighting = "class_weighting"
        case declaredFeatures = "features"
    }

    init() {
        hashDimensions = 4096
        epochs = 300
        learningRate = 0.05
        l2 = 1e-4
        l2Grid = nil
        classWeighting = .balanced
        declaredFeatures = nil
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "student")
        let defaults = StudentSpec()
        hashDimensions = try c.decodeIfPresent(Int.self, forKey: .hashDimensions) ?? defaults.hashDimensions
        epochs = try c.decodeIfPresent(Int.self, forKey: .epochs) ?? defaults.epochs
        learningRate = try c.decodeIfPresent(Double.self, forKey: .learningRate) ?? defaults.learningRate
        l2 = try c.decodeIfPresent(Double.self, forKey: .l2) ?? defaults.l2
        l2Grid = try c.decodeIfPresent([Double].self, forKey: .l2Grid)
        classWeighting = try c.decodeIfPresent(ClassWeighting.self, forKey: .classWeighting) ?? defaults.classWeighting
        declaredFeatures = try c.decodeIfPresent(FeatureScheme.self, forKey: .declaredFeatures)
    }

    func validate() throws {
        guard (64...262_144).contains(hashDimensions), (1...5000).contains(epochs) else {
            throw DistillError.invalidSpec("student: hash_dimensions 64-262144, epochs 1-5000")
        }
        guard learningRate > 0, learningRate <= 1, l2 >= 0, l2 <= 10 else {
            throw DistillError.invalidSpec("student: learning_rate in (0, 1], l2 in [0, 10]")
        }
        if let l2Grid {
            guard (2...12).contains(l2Grid.count), l2Grid.allSatisfy({ $0 >= 0 && $0 <= 10 }) else {
                throw DistillError.invalidSpec("student.l2_grid needs 2-12 values in [0, 10]")
            }
        }
    }
}
