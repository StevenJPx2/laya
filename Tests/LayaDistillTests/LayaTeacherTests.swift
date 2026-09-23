import XCTest
import LayaCore
@testable import LayaDistill

/// Real Laya as the teacher. Requires the exported model:
///   LAYA_MODEL=build/laya.mlpackage LAYA_ASSETS=build/assets swift test --filter LayaTeacherTests
/// The daemon case additionally needs a running daemon at LAYA_TEST_SOCKET.
final class LayaTeacherTests: XCTestCase {
    func testRealRuntimeLabelsAndStudentRunsWithoutLaya() async throws {
        guard let model = ProcessInfo.processInfo.environment["LAYA_MODEL"],
              let assets = ProcessInfo.processInfo.environment["LAYA_ASSETS"] else { throw XCTSkip("set LAYA_MODEL and LAYA_ASSETS") }

        let spec = try TaskSpec.decode(Data(RoutingTemplate.spec.utf8))
        let directory = try Fixture.temporaryDirectory()
        let dataURL = directory.appendingPathComponent("data.jsonl")
        let artifactURL = directory.appendingPathComponent("routing.classifier.json")
        try RoutingTemplate.dataset.write(to: dataURL, atomically: true, encoding: .utf8)
        let examples = try Dataset.load(dataURL, spec: spec)

        let labels = try await labelWithRuntime(spec, examples, model: model, assets: assets)
        let records = Array(labels.values)

        XCTAssertEqual(records.count, examples.count)
        XCTAssertTrue(records.allSatisfy { $0.status != .error }, "\(records.compactMap(\.reason).prefix(3))")
        XCTAssertTrue(records.allSatisfy { $0.teacher.hasPrefix("laya-runtime:laya-typed-decisions@") })
        XCTAssertTrue(records.allSatisfy { abs(($0.probabilities?.values.reduce(0, +) ?? 0) - 1) < 0.01 })
        XCTAssertTrue(records.filter { $0.status == .uncertain }.allSatisfy { $0.label == "other" && ($0.top! < 0.6 || $0.margin! < 0.2 - 1e-9) },
                      "every uncertain answer failed the gate and trains as the fallback label, not as Laya's argmax")
        XCTAssertTrue(records.filter { $0.status == .accepted }.allSatisfy { $0.label == $0.layaLabel && $0.top! >= 0.6 - 1e-9 })

        try Workflow.train(spec: spec, examples: examples, labels: labels).save(to: artifactURL)

        // Nothing below references Laya: the student artifact is self-contained.
        let artifact = try ClassifierArtifact.load(artifactURL)
        let classifier = Classifier(artifact: artifact)
        let prediction = try classifier.predict(.object([("subject", .string("Refund")), ("body", .string("I was charged twice, please refund me."))]))
        let report = try XCTUnwrap(artifact.evaluation)
        let goldHashes = Set(examples.filter { $0.gold != nil }.map(\.contentHash))

        XCTAssertEqual(prediction.probabilities.count, 4)
        XCTAssertTrue(Set(artifact.training.trainHashes).isDisjoint(with: goldHashes), "gold rows are never trained on")
        XCTAssertEqual(report.split.gold, 48)
        XCTAssertNotNil(report.agreement)
        XCTAssertEqual(report.gold?.layaRaw?.examples, 48, "Laya itself is scored against every gold row")
        print("real-laya report:\n\(report.markdown)")
    }

    func testDaemonTeacherWhenAvailable() async throws {
        guard let socket = ProcessInfo.processInfo.environment["LAYA_TEST_SOCKET"] else { throw XCTSkip("set LAYA_TEST_SOCKET to a running laya-daemon") }

        let spec = try TaskSpec.decode(Data(RoutingTemplate.spec.utf8))
        let input = try spec.input.parse(.object([("body", .string("I was charged twice for my invoice."))]))
        let teacher = try DaemonTeacher(path: socket)

        let answer = try await teacher.answer(state: input.jsonValue, question: LayaQuestion.make(spec))
        let decision = try TeacherPolicy.decide(try LayaQuestion.probabilities(answer, spec: spec), spec: spec)

        XCTAssertEqual(teacher.identity, "laya-daemon:laya-typed-decisions")
        XCTAssertTrue(spec.labelNames.contains(decision.layaLabel))
    }

    /// The runtime lives only inside this function, so the student test above
    /// cannot accidentally depend on it.
    private func labelWithRuntime(_ spec: TaskSpec, _ examples: [DatasetExample], model: String, assets: String) async throws -> [String: LabelRecord] {
        let runtime = try await LayaRuntime(modelURL: URL(fileURLWithPath: model), assetsURL: URL(fileURLWithPath: assets))
        let teacher = try await RuntimeTeacher(runtime: runtime, assets: URL(fileURLWithPath: assets))

        return try await Fixture.label(spec, examples, teacher: teacher)
    }
}
