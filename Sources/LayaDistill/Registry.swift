import Foundation
import LayaCore

/// Classifiers served by the daemon under `{"op":"classify"}`.
public struct ClassifierRegistry: Sendable {
    public static let suffix = ".classifier.json"
    public static let maxClassifiers = 64

    let classifiers: [String: Classifier]

    public var names: [String] { classifiers.keys.sorted() }

    /// Load every `*.classifier.json` in `directory`. Any invalid artifact fails
    /// startup rather than being skipped silently.
    public static func load(directory: URL) throws -> ClassifierRegistry {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasSuffix(suffix) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard files.count <= maxClassifiers else { throw DistillError.artifact("more than \(maxClassifiers) classifiers in \(directory.path)") }

        var classifiers: [String: Classifier] = [:]

        for file in files {
            let artifact = try ClassifierArtifact.load(file)
            guard classifiers[artifact.spec.name] == nil else { throw DistillError.artifact("duplicate classifier name \(artifact.spec.name)") }

            classifiers[artifact.spec.name] = Classifier(artifact: artifact)
        }

        return ClassifierRegistry(classifiers: classifiers)
    }

    public init(classifiers: [String: Classifier]) {
        self.classifiers = classifiers
    }

    /// Socket handler for `{"op":"classify","classifier":<name>,"input":{...}}`
    /// and `{"op":"classifiers"}`.
    public func handle(op: String, line: Data) async throws -> Data {
        switch op {
        case "classifiers":
            let entries = names.map { ["name": $0, "version": classifiers[$0]!.artifact.spec.version, "teacher": classifiers[$0]!.artifact.training.teacher] }
            return try Canonical.encoder().encode(["classifiers": entries])

        case "classify":
            let request = try JSONDecoder().decode(ClassifyRequest.self, from: line)
            guard let classifier = classifiers[request.classifier] else { throw DistillError.invalidData("unknown classifier \(request.classifier)") }

            return try Canonical.encoder().encode(try classifier.predict(request.input))

        default:
            throw DistillError.invalidData("unknown op \(op)")
        }
    }
}

private struct ClassifyRequest: Decodable {
    let classifier: String
    let input: JSONValue

    enum CodingKeys: String, CodingKey, CaseIterable { case op, classifier, input }

    init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "classify request")
        classifier = try c.decode(String.self, forKey: .classifier)
        input = try c.decode(JSONValue.self, forKey: .input)
    }
}
