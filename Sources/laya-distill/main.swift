import Foundation
import LayaCore
import LayaDistill

let usage = """
usage:
  laya-distill init <dir> [--template routing|permission]
  laya-distill validate <task.json>
  laya-distill label <task.json> --data <data.jsonl> --labels <labels.jsonl> [--limit N]
                     [--teacher daemon [--socket <path>] | --teacher runtime [--model <laya.mlpackage>] [--assets <dir>]]
  laya-distill train <task.json> --data <data.jsonl> --labels <labels.jsonl> --out <name.classifier.json> [--report <file.md>]
  laya-distill eval <artifact> --data <data.jsonl> --labels <labels.jsonl> [--report <file.md>]
  laya-distill predict <artifact> [--input '<json object>']   (otherwise reads one JSON object per stdin line)

Only `label` uses Laya (the teacher). The default teacher is the installed laya-daemon
($LAYA_SOCKET or ~/Library/Application Support/laya/laya.sock). `--teacher runtime` loads
Laya in-process from --model/--assets ($LAYA_MODEL/$LAYA_ASSETS, then build/…).
train, eval, and predict run the student without Laya.
"""

struct Arguments {
    let command: String
    let positional: [String]
    private let options: [String: String]

    init(_ raw: [String]) throws {
        guard let command = raw.first else { throw DistillError.invalidData("missing command") }

        var positional: [String] = []
        var options: [String: String] = [:]
        var index = 1

        while index < raw.count {
            let token = raw[index]

            if token.hasPrefix("--") {
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
    }

    func value(_ name: String) -> String? { options[name] }

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

func teacher(_ args: Arguments) async throws -> LayaTeacher {
    let environment = ProcessInfo.processInfo.environment

    switch args.value("--teacher") ?? "daemon" {
    case "daemon":
        return try DaemonTeacher(path: args.value("--socket") ?? SocketClient.defaultPath)
    case "runtime":
        let model = URL(fileURLWithPath: args.value("--model") ?? environment["LAYA_MODEL"] ?? "build/laya.mlpackage")
        let assets = URL(fileURLWithPath: args.value("--assets") ?? environment["LAYA_ASSETS"] ?? "build/assets")
        log("[laya] loading \(model.lastPathComponent)")
        return try await RuntimeTeacher(runtime: try await LayaRuntime(modelURL: model, assetsURL: assets), assets: assets)
    default:
        throw DistillError.invalidData("--teacher must be daemon or runtime")
    }
}

func initTask(_ args: Arguments) throws {
    let directory = try args.first()
    let name = args.value("--template") ?? "routing"
    guard let template = Templates.named(name) else {
        throw DistillError.invalidData("unknown template \(name); available: \(Templates.names.joined(separator: ", "))")
    }

    let files = [("task.json", template.spec + "\n"), ("data.jsonl", template.dataset)]
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

    let summary = try await labeler.run(teacher: try await teacher(args), limit: limit, sink: { try LabelStore.append($0, to: labelsURL) }, log: log)
    try emit(summary)
}

func train(_ args: Arguments) throws {
    let spec = try TaskSpec.load(try args.first())
    let examples = try Dataset.load(try args.require("--data"), spec: spec)
    let labels = try LabelStore.load(try args.require("--labels"), maxRows: spec.dataset.maxExamples)
    let out = try args.require("--out")
    guard out.lastPathComponent.hasSuffix(ClassifierRegistry.suffix) else { throw DistillError.invalidData("--out must end in \(ClassifierRegistry.suffix)") }

    let artifact = try Workflow.train(spec: spec, examples: examples, labels: labels, log: log)
    try artifact.save(to: out)
    try writeReport(artifact.evaluation!, args)
    log("[train] wrote \(out.path)")
    try emit(artifact.evaluation!)
}

func evaluate(_ args: Arguments) throws {
    let artifact = try ClassifierArtifact.load(try args.first())
    let examples = try Dataset.load(try args.require("--data"), spec: artifact.spec)
    let labels = try LabelStore.load(try args.require("--labels"), maxRows: artifact.spec.dataset.maxExamples)

    let report = try Workflow.evaluate(Classifier(artifact: artifact), examples: examples, labels: labels)
    try writeReport(report, args)
    try emit(report)
}

func predict(_ args: Arguments) throws {
    let artifact = try ClassifierArtifact.load(try args.first())
    let classifier = Classifier(artifact: artifact)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

    if let inline = args.value("--input") {
        try emit(try classifier.predict(try JSONDecoder().decode(JSONValue.self, from: Data(inline.utf8))))
        return
    }

    while let line = readLine(), !line.isEmpty {
        guard line.utf8.count <= artifact.spec.dataset.maxLineBytes else { throw DistillError.invalidData("stdin line exceeds max_line_bytes") }
        let prediction = try classifier.predict(try JSONDecoder().decode(JSONValue.self, from: Data(line.utf8)))
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
    case "train": try train(args)
    case "eval": try evaluate(args)
    case "predict": try predict(args)
    case "help", "--help", "-h": print(usage)
    default: throw DistillError.invalidData("unknown command \(args.command)")
    }
} catch {
    fputs("error: \(error.localizedDescription)\n\n\(usage)\n", stderr)
    exit(1)
}
