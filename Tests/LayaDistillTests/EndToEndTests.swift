import XCTest
import LayaCore
@testable import LayaDistill

final class EndToEndTests: XCTestCase {
    /// Fake Laya teacher -> label file -> gated split -> student training ->
    /// artifact on disk -> reload -> predictions and re-evaluation, with no
    /// Laya model anywhere after labeling.
    func testTinyStudentEndToEnd() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let directory = try Fixture.temporaryDirectory()
        let labelsURL = directory.appendingPathComponent("labels.jsonl")
        let artifactURL = directory.appendingPathComponent("tiny.classifier.json")
        let vague = try (0..<6).map { index in
            DatasetExample(id: "vague-\(index)", input: try spec.input.parse(.object([("request", .string("maybe handle thing \(index)"))])), group: nil, gold: nil)
        }
        let examples = try Fixture.dataset(spec) + vague

        let summary = try await Labeler(spec: spec, examples: examples, existing: [:])
            .run(teacher: Fixture.verbTeacher(), sink: { try LabelStore.append($0, to: labelsURL) }, log: { _ in })
        XCTAssertEqual(summary.accepted, 78, "60 pool + 18 gold rows are confident")
        XCTAssertEqual(summary.uncertainToAbstain, 6)

        let labels = try LabelStore.load(labelsURL, maxRows: 200)
        try await Workflow.train(spec: spec, examples: examples, labels: labels).save(to: artifactURL)

        let artifact = try ClassifierArtifact.load(artifactURL)
        let report = try XCTUnwrap(artifact.evaluation)
        let agreement = try XCTUnwrap(report.agreement)
        let gold = try XCTUnwrap(report.gold)
        let goldHashes = Set(examples.filter { $0.gold != nil }.map(\.contentHash))

        XCTAssertEqual(artifact.training.teacher, "fake-laya")
        XCTAssertEqual(artifact.training.questionSha256, LayaQuestion.sha256(spec))
        XCTAssertEqual(artifact.features.scheme, .hashedNgram)
        XCTAssertTrue(Set(artifact.training.trainHashes).isDisjoint(with: goldHashes), "gold rows are never trained on")
        XCTAssertEqual(report.split.gold, 18)
        XCTAssertEqual(report.split.uncertainToAbstain, 6)
        XCTAssertEqual(report.split.train + report.split.holdout, 66)
        XCTAssertGreaterThanOrEqual(agreement.student.accuracy, 0.9, "student reproduces the teacher on held-out rows")
        XCTAssertGreaterThan(agreement.student.accuracy, agreement.majority.accuracy + 0.3)
        XCTAssertGreaterThanOrEqual(gold.studentServed.accuracy, 0.8, "student generalizes to differently phrased gold rows")
        XCTAssertEqual(gold.teacherRaw?.accuracy, 1, "the fake teacher is perfect on gold by construction")
        XCTAssertEqual(gold.teacherRaw?.examples, 18)
        XCTAssertEqual(gold.teacher, "fake-laya")
        XCTAssertNil(gold.layaZeroShot, "no Laya runtime, no zero-shot row")
        XCTAssertNil(gold.beatsLayaZeroShot)

        let classifier = try Classifier(artifact: artifact)
        let wipe = try await classifier.predict(.object([("request", .string("wipe the backups"))]))
        let list = try await classifier.predict(.object([("request", .string("list files")), ("note", .string("routine"))]))
        XCTAssertEqual(wipe.label, "deny")
        XCTAssertEqual(list.label, "allow")
        await XCTAssertThrowsErrorAsync(try await classifier.predict(.object([("command", .string("x"))])))

        let again = try await Workflow.evaluate(classifier, examples: examples, labels: labels)
        XCTAssertEqual(again.agreement?.served.accuracy, agreement.served.accuracy, "re-evaluation from disk reproduces the stored report")
        XCTAssertEqual(again.gold?.studentServed.accuracy, gold.studentServed.accuracy)
        XCTAssertEqual(again.split.trainOverlapExcluded, 0)
    }

    func testRegistryServesClassifyWithoutLaya() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let examples = try Fixture.dataset(spec)
        let directory = try Fixture.temporaryDirectory()
        try await Workflow.train(spec: spec, examples: examples, labels: try await Fixture.label(spec, examples))
            .save(to: directory.appendingPathComponent("tiny.classifier.json"))

        let registry = try ClassifierRegistry.load(directory: directory)
        let line = Data(#"{"op":"classify","classifier":"tiny","input":{"request":"delete the database"}}"#.utf8)
        let reply = try JSONDecoder().decode(Prediction.self, from: try await registry.handle(op: "classify", line: line))
        let listing = String(decoding: try await registry.handle(op: "classifiers", line: Data("{}".utf8)), as: UTF8.self)

        XCTAssertEqual(registry.names, ["tiny"])
        XCTAssertEqual(reply.label, "deny")
        XCTAssertTrue(listing.contains(#""teacher":"fake-laya""#), listing)
        await XCTAssertThrowsErrorAsync(try await registry.handle(op: "classify", line: Data(#"{"op":"classify","classifier":"missing","input":{}}"#.utf8)))
    }

    func testTamperedArtifactIsRejected() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let examples = try Fixture.dataset(spec)
        let url = try Fixture.temporaryDirectory().appendingPathComponent("tiny.classifier.json")
        try await Workflow.train(spec: spec, examples: examples, labels: try await Fixture.label(spec, examples)).save(to: url)

        let text = try String(contentsOf: url, encoding: .utf8)
        try text.replacingOccurrences(of: #""min_confidence":0.4"#, with: #""min_confidence":0.1"#).write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(try ClassifierArtifact.load(url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("integrity"), error.localizedDescription)
        }
    }

    func testTrainingNeedsConfidentLabels() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec(uncertain: "drop"))
        let examples = try Fixture.dataset(spec)
        let unsure = FakeTeacher { _, _ in Fixture.answer(["allow": 0.4, "deny": 0.3, "ask": 0.3]) }

        let labels = try await Fixture.label(spec, examples, teacher: unsure)

        XCTAssertEqual(labels.values.filter { $0.status == .uncertain && $0.label == nil }.count, examples.count, "drop policy keeps no label")
        await XCTAssertThrowsErrorAsync(try await Workflow.train(spec: spec, examples: examples, labels: labels)) { error in
            XCTAssertTrue(error.localizedDescription.contains("at least 2 labels"), error.localizedDescription)
        }
    }

    func testL2IsSelectedByCrossValidationOnTrainOnly() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec().replacingOccurrences(of: #""l2": 0.0001"#, with: #""l2_grid": [0.0001, 0.1, 5]"#))
        let examples = try Fixture.dataset(spec)

        let artifact = try await Workflow.train(spec: spec, examples: examples, labels: try await Fixture.label(spec, examples))
        let selection = try XCTUnwrap(artifact.training.selection)

        XCTAssertEqual(selection.candidates.map(\.l2), [0.0001, 0.1, 5])
        XCTAssertEqual(selection.folds, 5)
        XCTAssertEqual(artifact.training.l2, selection.l2)
        XCTAssertEqual(selection.l2, selection.candidates.max { ($0.accuracy, $0.l2) < ($1.accuracy, $1.l2) }!.l2)
    }
}

func XCTAssertThrowsErrorAsync<T>(_ expression: @autoclosure () async throws -> T, file: StaticString = #filePath, line: UInt = #line,
                                  _ handler: (Error) -> Void = { _ in }) async {
    do {
        _ = try await expression()
        XCTFail("expected an error", file: file, line: line)
    } catch {
        handler(error)
    }
}
