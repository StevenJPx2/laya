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

public enum TeacherProvider: String, Codable, Sendable {
    case anthropic
    case openaiCompatible = "openai-compatible"
    /// Uses the dataset's `gold` labels as the teacher. No network, no cost.
    case dataset

    var isNetworked: Bool { self != .dataset }
}

public struct Pricing: Codable, Sendable {
    public let inputUsdPerMtok: Double
    public let outputUsdPerMtok: Double

    enum CodingKeys: String, CodingKey, CaseIterable { case inputUsdPerMtok = "input_usd_per_mtok", outputUsdPerMtok = "output_usd_per_mtok" }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "teacher.pricing")
        inputUsdPerMtok = try c.decode(Double.self, forKey: .inputUsdPerMtok)
        outputUsdPerMtok = try c.decode(Double.self, forKey: .outputUsdPerMtok)
    }

    func cost(input: Int, output: Int) -> Double {
        (Double(input) * inputUsdPerMtok + Double(output) * outputUsdPerMtok) / 1_000_000
    }
}

public struct TeacherSpec: Codable, Sendable {
    public let provider: TeacherProvider
    public let model: String
    public let baseURL: String?
    public let apiKeyEnv: String?
    public let maxOutputTokens: Int
    public let timeoutSeconds: Int
    public let maxRetries: Int
    public let maxTokensField: String
    public let pricing: Pricing?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case provider, model, baseURL = "base_url", apiKeyEnv = "api_key_env", maxOutputTokens = "max_output_tokens"
        case timeoutSeconds = "timeout_seconds", maxRetries = "max_retries", maxTokensField = "max_tokens_field", pricing
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "teacher")
        provider = try c.decode(TeacherProvider.self, forKey: .provider)
        model = try c.decode(String.self, forKey: .model)
        baseURL = try c.decodeIfPresent(String.self, forKey: .baseURL)
        apiKeyEnv = try c.decodeIfPresent(String.self, forKey: .apiKeyEnv)
        maxOutputTokens = try c.decodeIfPresent(Int.self, forKey: .maxOutputTokens) ?? 64
        timeoutSeconds = try c.decodeIfPresent(Int.self, forKey: .timeoutSeconds) ?? 60
        maxRetries = try c.decodeIfPresent(Int.self, forKey: .maxRetries) ?? 2
        maxTokensField = try c.decodeIfPresent(String.self, forKey: .maxTokensField) ?? "max_completion_tokens"
        pricing = try c.decodeIfPresent(Pricing.self, forKey: .pricing)
    }

    public var identity: String { "\(provider.rawValue)/\(model)" }

    func validate() throws {
        guard !model.isEmpty, model.count <= 128 else { throw DistillError.invalidSpec("teacher.model must be set explicitly (1-128 characters)") }
        guard provider.isNetworked else { return }

        guard let apiKeyEnv, matches(apiKeyEnv, "^[A-Z_][A-Z0-9_]{0,63}$") else {
            throw DistillError.invalidSpec("teacher.api_key_env must name an environment variable, e.g. ANTHROPIC_API_KEY; keys are never stored in the spec")
        }
        guard let pricing, pricing.inputUsdPerMtok >= 0, pricing.outputUsdPerMtok >= 0 else {
            throw DistillError.invalidSpec("teacher.pricing is required for networked teachers so spend can be bounded")
        }
        guard (1...1024).contains(maxOutputTokens), (1...300).contains(timeoutSeconds), (0...5).contains(maxRetries) else {
            throw DistillError.invalidSpec("teacher limits: max_output_tokens 1-1024, timeout_seconds 1-300, max_retries 0-5")
        }
        guard ["max_completion_tokens", "max_tokens"].contains(maxTokensField) else {
            throw DistillError.invalidSpec("teacher.max_tokens_field must be max_completion_tokens or max_tokens")
        }
        if let baseURL {
            guard let url = URL(string: baseURL), let scheme = url.scheme, ["https", "http"].contains(scheme) else {
                throw DistillError.invalidSpec("teacher.base_url must be an http(s) URL")
            }
            guard scheme == "https" || ["localhost", "127.0.0.1", "::1"].contains(url.host ?? "") else {
                throw DistillError.invalidSpec("teacher.base_url must use https unless it is a loopback address")
            }
        }
        if provider == .openaiCompatible, baseURL == nil { throw DistillError.invalidSpec("teacher.base_url is required for openai-compatible") }
    }
}

public struct BudgetSpec: Codable, Sendable {
    public let maxRequests: Int
    public let maxUsd: Double

    enum CodingKeys: String, CodingKey, CaseIterable { case maxRequests = "max_requests", maxUsd = "max_usd" }

    init() {
        maxRequests = 0
        maxUsd = 0
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "budget")
        maxRequests = try c.decode(Int.self, forKey: .maxRequests)
        maxUsd = try c.decode(Double.self, forKey: .maxUsd)
    }

    func validate(networked: Bool) throws {
        guard (0...100_000).contains(maxRequests), maxUsd >= 0, maxUsd <= 10_000 else {
            throw DistillError.invalidSpec("budget: max_requests 0-100000, max_usd 0-10000")
        }
        if networked, maxRequests == 0 || maxUsd == 0 {
            throw DistillError.invalidSpec("budget.max_requests and budget.max_usd must be positive for networked teachers")
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

public enum FeatureKind: String, Codable, Sendable {
    /// Frozen Laya encoder: pooled decision vector + per-label zero-shot logits.
    case laya
    /// Hashed word uni/bigrams. Model-free; useful for tiny tasks and tests.
    case hashed
}

public enum ClassWeighting: String, Codable, Sendable { case balanced, none }

public struct StudentSpec: Codable, Sendable {
    public let features: FeatureKind
    public let hashDimensions: Int
    public let epochs: Int
    public let learningRate: Double
    public let l2: Double
    /// When set, L2 is chosen from these values by group-aware k-fold
    /// cross-validation on the training split only; `l2` is then ignored.
    public let l2Grid: [Double]?
    public let classWeighting: ClassWeighting

    enum CodingKeys: String, CodingKey, CaseIterable {
        case features, hashDimensions = "hash_dimensions", epochs, learningRate = "learning_rate", l2, l2Grid = "l2_grid", classWeighting = "class_weighting"
    }

    init() {
        features = .laya
        hashDimensions = 4096
        epochs = 300
        learningRate = 0.05
        l2 = 1e-4
        l2Grid = nil
        classWeighting = .balanced
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "student")
        let defaults = StudentSpec()
        features = try c.decodeIfPresent(FeatureKind.self, forKey: .features) ?? defaults.features
        hashDimensions = try c.decodeIfPresent(Int.self, forKey: .hashDimensions) ?? defaults.hashDimensions
        epochs = try c.decodeIfPresent(Int.self, forKey: .epochs) ?? defaults.epochs
        learningRate = try c.decodeIfPresent(Double.self, forKey: .learningRate) ?? defaults.learningRate
        l2 = try c.decodeIfPresent(Double.self, forKey: .l2) ?? defaults.l2
        l2Grid = try c.decodeIfPresent([Double].self, forKey: .l2Grid)
        classWeighting = try c.decodeIfPresent(ClassWeighting.self, forKey: .classWeighting) ?? defaults.classWeighting
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
