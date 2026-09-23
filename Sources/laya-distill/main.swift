import Foundation
import LayaCore
import LayaDistill

let usage = """
usage:
  laya-distill init <dir> [--template permission]
  laya-distill validate <task.json>
  laya-distill label <task.json> --data <data.jsonl> --labels <labels.jsonl> [--limit N] [--approve]
  laya-distill train <task.json> --data <data.jsonl> --labels <labels.jsonl> --out <name.classifier.json> [--report <file.md>]
  laya-distill eval <artifact> --data <data.jsonl> --labels <labels.jsonl> [--report <file.md>]
  laya-distill predict <artifact> [--input '<json object>']   (otherwise reads one JSON object per stdin line)

Laya model options (needed for `laya` features and the zero-shot baseline):
  --model <laya.mlpackage>  --assets <assets dir>   (defaults: $LAYA_MODEL/$LAYA_ASSETS, then build/…)
Teacher API keys are read only from the environment variable named by teacher.api_key_env,
and only when --approve is given.
"""

struct Arguments {
    let command: String
    let positional: [String]
    private let options: [String: String]
    private let flags: Set<String>

    init(_ raw: [String]) throws {
        guard let command = raw.first else { throw DistillError.invalidData("missing command") }

        var positional: [String] = []
        var options: [String: String] = [:]
        var flags = Set<String>()
        var index = 1

        while index < raw.count {
            let token = raw[index]

            if ["--approve"].contains(token) {
                flags.insert(token)
            } else if token.hasPrefix("--") {
                guard index + 1 < raw.count else { throw DistillError.invalidData("\(token) needs a value") }
                options[token] = raw[index + 1]
                index += 1
            } else {
                positional.append(token)
            }
            index += 1
        }

        self.command = command
        self.positional = positional
        self.options = options
        self.flags = flags
    }

    func value(_ name: String) -> String? { options[name] }
    func has(_ flag: String) -> Bool { flags.contains(flag) }

    func require(_ name: String) throws -> URL {
        guard let value = options[name] else { throw DistillError.invalidData("\(command) requires \(name)") }

        return URL(fileURLWithPath: value)
    }

    func first() throws -> URL {
        guard let value = positional.first else { throw DistillError.invalidData("\(command) requires a path") }

        return URL(fileURLWithPath: value)
    }
}

func emit<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    print(String(decoding: try encoder.encode(value), as: UTF8.self))
}

func log(_ message: String) { fputs(message + "\n", stderr) }

/// Loads the Laya runtime only when asked: explicitly via flags, or because
/// the task/artifact uses `laya` features.
func layaRuntime(_ args: Arguments, required: Bool) async throws -> (LayaRuntime, URL)? {
    let environment = ProcessInfo.processInfo.environment
    let explicit = args.value("--model") != nil || args.value("--assets") != nil
    guard required || explicit else { return nil }

    let model = URL(fileURLWithPath: args.value("--model") ?? environment["LAYA_MODEL"] ?? "build/laya.mlpackage")
    let assets = URL(fileURLWithPath: args.value("--assets") ?? environment["LAYA_ASSETS"] ?? "build/assets")
    log("[laya] loading \(model.lastPathComponent)")

    return (try await LayaRuntime(modelURL: model, assetsURL: assets), assets)
}

func initTask(_ args: Arguments) throws {
    let directory = try args.first()
    guard (args.value("--template") ?? "permission") == "permission" else { throw DistillError.invalidData("unknown template; available: permission") }

    let files = [("task.json", PermissionTemplate.spec + "\n"), ("data.jsonl", PermissionTemplate.dataset)]
    for (name, _) in files where FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path) {
        throw DistillError.invalidData("\(directory.appendingPathComponent(name).path) already exists; refusing to overwrite")
    }

    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    for (name, contents) in files { try contents.write(to: directory.appendingPathComponent(name), atomically: true, encoding: .utf8) }

    _ = try TaskSpec.load(directory.appendingPathComponent("task.json"))
    log("[init] wrote task.json and data.jsonl to \(directory.path)")
}

func label(_ args: Arguments) async throws {
    let spec = try TaskSpec.load(try args.first())
    let labelsURL = try args.require("--labels")
    let examples = try Dataset.load(try args.require("--data"), spec: spec)
    let labeler = Labeler(spec: spec, examples: examples, existing: try LabelStore.load(labelsURL, maxRows: spec.dataset.maxExamples))
    let limit = try args.value("--limit").map { value -> Int in
        guard let limit = Int(value), limit > 0 else { throw DistillError.invalidData("--limit must be a positive integer") }
        return limit
    }

    var client: TeacherClient?
    if spec.teacher.provider != .dataset, args.has("--approve") {
        let name = spec.teacher.apiKeyEnv!
        guard let key = ProcessInfo.processInfo.environment[name], !key.isEmpty else { throw DistillError.teacher("environment variable \(name) is not set") }
        client = TeacherClient(spec: spec, apiKey: key)
    }

    let summary = try await labeler.run(limit: limit, approve: args.has("--approve"), client: client,
                                        sink: { try LabelStore.append($0, to: labelsURL) }, log: log)
    try emit(summary)
}

func train(_ args: Arguments) async throws {
    let spec = try TaskSpec.load(try args.first())
    let examples = try Dataset.load(try args.require("--data"), spec: spec)
    let labels = try LabelStore.load(try args.require("--labels"), maxRows: spec.dataset.maxExamples)
    let out = try args.require("--out")
    guard out.lastPathComponent.hasSuffix(ClassifierRegistry.suffix) else { throw DistillError.invalidData("--out must end in \(ClassifierRegistry.suffix)") }

    let laya = try await layaRuntime(args, required: spec.student.features == .laya)
    let fingerprint = try laya.map { try LayaFeatures.fingerprint(assets: $0.1) }
    let zeroShot = try laya.map { try LayaFeatures(runtime: $0.0, spec: spec, fingerprint: fingerprint!) }
    let extractor: FeatureExtractor = spec.student.features == .laya ? zeroShot! : HashedFeatures(dimensions: spec.student.hashDimensions)

    let artifact = try await Workflow.train(spec: spec, examples: examples, labels: labels, extractor: extractor, baseline: zeroShot, log: log)
    try artifact.save(to: out)
    try writeReport(artifact.evaluation!, args)
    log("[train] wrote \(out.path)")
    try emit(artifact.evaluation!)
}

func evaluate(_ args: Arguments) async throws {
    let artifact = try ClassifierArtifact.load(try args.first())
    let examples = try Dataset.load(try args.require("--data"), spec: artifact.spec)
    let labels = try LabelStore.load(try args.require("--labels"), maxRows: artifact.spec.dataset.maxExamples)
    let laya = try await layaRuntime(args, required: artifact.features.kind == .laya)
    let classifier = try Classifier(artifact: artifact, runtime: laya?.0, assets: laya?.1)
    let baseline = try laya.map { try LayaFeatures(runtime: $0.0, spec: artifact.spec, fingerprint: try LayaFeatures.fingerprint(assets: $0.1)) }

    let report = try await Workflow.evaluate(classifier, examples: examples, labels: labels, baseline: baseline)
    try writeReport(report, args)
    try emit(report)
}

func predict(_ args: Arguments) async throws {
    let artifact = try ClassifierArtifact.load(try args.first())
    let laya = try await layaRuntime(args, required: artifact.features.kind == .laya)
    let classifier = try Classifier(artifact: artifact, runtime: laya?.0, assets: laya?.1)

    if let inline = args.value("--input") {
        try emit(try await classifier.predict(try JSONDecoder().decode(JSONValue.self, from: Data(inline.utf8))))
        return
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

    while let line = readLine(), !line.isEmpty {
        guard line.utf8.count <= artifact.spec.dataset.maxLineBytes else { throw DistillError.invalidData("stdin line exceeds max_line_bytes") }
        let prediction = try await classifier.predict(try JSONDecoder().decode(JSONValue.self, from: Data(line.utf8)))
        print(String(decoding: try encoder.encode(prediction), as: UTF8.self))
    }
}

func writeReport(_ report: EvaluationReport, _ args: Arguments) throws {
    guard let path = args.value("--report") else { return }

    try (report.markdown + "\n").write(toFile: path, atomically: true, encoding: .utf8)
}

do {
    let args = try Arguments(Array(CommandLine.arguments.dropFirst()))

    switch args.command {
    case "init": try initTask(args)
    case "validate": try emit(["name": try TaskSpec.load(try args.first()).name, "status": "valid"])
    case "label": try await label(args)
    case "train": try await train(args)
    case "eval": try await evaluate(args)
    case "predict": try await predict(args)
    case "help", "--help", "-h": print(usage)
    default: throw DistillError.invalidData("unknown command \(args.command)")
    }
} catch {
    fputs("error: \(error.localizedDescription)\n\n\(usage)\n", stderr)
    exit(1)
}
