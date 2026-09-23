import Foundation
import LayaCore

/// A versioned classification task: input schema, labels, abstain policy,
/// teacher, budget, dataset bounds, and student configuration.
public struct TaskSpec: Codable, Sendable {
    public static let currentVersion = 1
    public static let maxLabels = 32
    public static let maxFields = 32
    public static let maxExamples = 20

    public let specVersion: Int
    public let name: String
    public let version: String
    public let description: String?
    public let instructions: String
    public let input: InputSchema
    public let labels: [LabelSpec]
    public let abstain: AbstainPolicy?
    public let examples: [FewShotExample]
    public let teacher: TeacherSpec
    public let budget: BudgetSpec
    public let dataset: DatasetSpec
    public let student: StudentSpec

    enum CodingKeys: String, CodingKey, CaseIterable {
        case specVersion = "spec_version", name, version, description, instructions, input, labels, abstain, examples, teacher, budget, dataset, student
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "task")
        specVersion = try c.decode(Int.self, forKey: .specVersion)
        name = try c.decode(String.self, forKey: .name)
        version = try c.decode(String.self, forKey: .version)
        description = try c.decodeIfPresent(String.self, forKey: .description)
        instructions = try c.decode(String.self, forKey: .instructions)
        input = try c.decode(InputSchema.self, forKey: .input)
        labels = try c.decode([LabelSpec].self, forKey: .labels)
        abstain = try c.decodeIfPresent(AbstainPolicy.self, forKey: .abstain)
        examples = try c.decodeIfPresent([FewShotExample].self, forKey: .examples) ?? []
        teacher = try c.decode(TeacherSpec.self, forKey: .teacher)
        budget = try c.decodeIfPresent(BudgetSpec.self, forKey: .budget) ?? BudgetSpec()
        dataset = try c.decodeIfPresent(DatasetSpec.self, forKey: .dataset) ?? DatasetSpec()
        student = try c.decodeIfPresent(StudentSpec.self, forKey: .student) ?? StudentSpec()
    }

    public static func load(_ url: URL) throws -> TaskSpec {
        try decode(Data(contentsOf: url))
    }

    public static func decode(_ data: Data) throws -> TaskSpec {
        guard data.count <= 1_048_576 else { throw DistillError.invalidSpec("task file exceeds 1 MiB") }

        let spec: TaskSpec
        do {
            spec = try JSONDecoder().decode(TaskSpec.self, from: data)
        } catch let error as DistillError {
            throw error
        } catch {
            throw DistillError.invalidSpec(describe(error))
        }

        try spec.validate()

        return spec
    }

    public var labelNames: [String] { labels.map(\.name) }

    /// Hash of the normalized spec; artifacts record it so a model is never
    /// silently served against a different task definition.
    public var sha256: String { Canonical.sha256((try? Canonical.encoder().encode(self)) ?? Data()) }

    public func labelIndex(_ name: String) -> Int? { labels.firstIndex { $0.name == name } }

    func validate() throws {
        guard specVersion == Self.currentVersion else { throw DistillError.invalidSpec("spec_version must be \(Self.currentVersion)") }
        guard matches(name, "^[a-z][a-z0-9_-]{0,63}$") else { throw DistillError.invalidSpec("name must match ^[a-z][a-z0-9_-]{0,63}$") }
        guard !version.isEmpty, version.count <= 32 else { throw DistillError.invalidSpec("version must be 1-32 characters") }
        guard !instructions.isEmpty, instructions.count <= 4000 else { throw DistillError.invalidSpec("instructions must be 1-4000 characters") }

        try input.validate()
        try validateLabels()
        try teacher.validate()
        try budget.validate(networked: teacher.provider.isNetworked)
        try dataset.validate()
        try student.validate()

        for (index, example) in examples.enumerated() {
            _ = try input.parse(example.input, context: "examples[\(index)].input")
            guard labelIndex(example.label) != nil else { throw DistillError.invalidSpec("examples[\(index)].label '\(example.label)' is not a declared label") }
        }
    }

    private func validateLabels() throws {
        guard (2...Self.maxLabels).contains(labels.count) else { throw DistillError.invalidSpec("labels must contain 2-\(Self.maxLabels) entries") }
        guard Set(labels.map(\.name)).count == labels.count else { throw DistillError.invalidSpec("label names must be unique") }
        guard examples.count <= Self.maxExamples else { throw DistillError.invalidSpec("at most \(Self.maxExamples) few-shot examples") }

        for label in labels {
            guard matches(label.name, "^[a-z][a-z0-9_-]{0,31}$") else { throw DistillError.invalidSpec("label '\(label.name)' must match ^[a-z][a-z0-9_-]{0,31}$") }
            guard !label.description.isEmpty, label.description.count <= 500 else { throw DistillError.invalidSpec("label '\(label.name)' needs a 1-500 character description") }
        }

        if let abstain {
            guard labelIndex(abstain.label) != nil else { throw DistillError.invalidSpec("abstain.label '\(abstain.label)' must be a declared label") }
            guard abstain.minConfidence > 0, abstain.minConfidence < 1 else { throw DistillError.invalidSpec("abstain.min_confidence must be in (0, 1)") }
        }
    }
}

public struct LabelSpec: Codable, Sendable {
    public let name: String
    public let description: String

    enum CodingKeys: String, CodingKey, CaseIterable { case name, description }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "labels[]")
        name = try c.decode(String.self, forKey: .name)
        description = try c.decode(String.self, forKey: .description)
    }
}

/// When the student's top probability is below `min_confidence`, it answers
/// `label` instead (for example `ask`). The teacher is told about the same label.
public struct AbstainPolicy: Codable, Sendable {
    public let label: String
    public let minConfidence: Double

    enum CodingKeys: String, CodingKey, CaseIterable { case label, minConfidence = "min_confidence" }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "abstain")
        label = try c.decode(String.self, forKey: .label)
        minConfidence = try c.decode(Double.self, forKey: .minConfidence)
    }
}

public struct FewShotExample: Codable, Sendable {
    public let input: JSONValue
    public let label: String

    enum CodingKeys: String, CodingKey, CaseIterable { case input, label }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "examples[]")
        input = try c.decode(JSONValue.self, forKey: .input)
        label = try c.decode(String.self, forKey: .label)
    }
}

func describe(_ error: Error) -> String {
    guard let decoding = error as? DecodingError else { return error.localizedDescription }

    switch decoding {
    case .keyNotFound(let key, let context): return "missing field '\(key.stringValue)' at \(path(context))"
    case .typeMismatch(_, let context), .valueNotFound(_, let context), .dataCorrupted(let context): return "\(context.debugDescription) at \(path(context))"
    @unknown default: return "malformed JSON"
    }
}

private func path(_ context: DecodingError.Context) -> String {
    let components = context.codingPath.map(\.stringValue)

    return components.isEmpty ? "<root>" : components.joined(separator: ".")
}
