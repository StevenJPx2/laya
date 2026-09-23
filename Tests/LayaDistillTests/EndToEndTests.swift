import XCTest
import LayaCore
@testable import LayaDistill

final class EndToEndTests: XCTestCase {
    /// Teacher-labels a tiny dataset through the real labeling path (stub
    /// transport), trains a real softmax head on hashed features, round-trips the
    /// artifact through disk, serves it, and re-evaluates it.
    func testTinyClassifierEndToEnd() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let directory = try Fixture.temporaryDirectory()
        let labelsURL = directory.appendingPathComponent("labels.jsonl")
        let artifactURL = directory.appendingPathComponent("tiny.classifier.json")
        let examples = try Fixture.dataset(spec, count: 60)
        let teacher = Fixture.anthropicTeacher()

        let summary = try await Labeler(spec: spec, examples: examples, existing: [:])
            .run(approve: true, client: TeacherClient(spec: spec, apiKey: "k", transport: teacher), sink: { try LabelStore.append($0, to: labelsURL) }, log: { _ in })
        XCTAssertEqual(summary.ok, 60)
        XCTAssertEqual(teacher.requests.count, 60)

        let labels = try LabelStore.load(labelsURL, maxRows: 100)
        let trained = try await Workflow.train(spec: spec, examples: examples, labels: labels, extractor: HashedFeatures(dimensions: spec.student.hashDimensions))
        try trained.save(to: artifactURL)

        let artifact = try ClassifierArtifact.load(artifactURL)
        let report = try XCTUnwrap(artifact.evaluation)
        let classifier = try Classifier(artifact: artifact, runtime: nil, assets: nil)

        XCTAssertEqual(artifact.training.teacher, "anthropic/test-teacher")
        XCTAssertEqual(artifact.specSha256, spec.sha256)
        XCTAssertEqual(report.split.train + report.split.holdout, 60)
        XCTAssertGreaterThan(report.split.holdout, 5)
        XCTAssertTrue(Set(artifact.training.trainHashes).isDisjoint(with: Splitter.split(examples, labels: labels, spec: spec).holdout.map(\.example.contentHash)))
        XCTAssertEqual(artifact.training.trainAccuracy, 1)
        XCTAssertGreaterThanOrEqual(report.student.accuracy, 0.9, "separable toy task should be learned")
        XCTAssertGreaterThan(report.student.accuracy, try XCTUnwrap(report.baselines.first?.metrics).accuracy + 0.3, "student beats the majority baseline")
        XCTAssertEqual(report.baselines.first?.metrics?.accuracy ?? 1, 1.0 / 3.0, accuracy: 0.2, "majority baseline is near chance on balanced labels")
        XCTAssertEqual(report.baselines.last?.status, "not run (needs --model/--assets)")

        let deny = try await classifier.predict(.object([("request", .string("wipe the backups now"))]))
        let allow = try await classifier.predict(.object([("request", .string("list files in the repo")), ("note", .string("routine"))]))
        XCTAssertEqual(deny.label, "deny")
        XCTAssertEqual(allow.label, "allow")
        XCTAssertEqual(deny.probabilities.values.reduce(0, +), 1, accuracy: 1e-3)
        await XCTAssertThrowsErrorAsync(try await classifier.predict(.object([("command", .string("x"))])))

        let again = try await Workflow.evaluate(classifier, examples: examples, labels: labels)
        XCTAssertEqual(again.served.accuracy, report.served.accuracy, accuracy: 1e-12, "re-evaluation from disk reproduces the stored report")
        XCTAssertEqual(again.split.trainOverlapExcluded, 0)
    }

    func testRegistryServesClassifyOp() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec(provider: "dataset"))
        let examples = try Fixture.dataset(spec, count: 45)
        var labels: [String: LabelRecord] = [:]
        _ = try await Labeler(spec: spec, examples: examples, existing: [:]).run(approve: false, client: nil, sink: { labels[$0.id] = $0 }, log: { _ in })

        let artifact = try await Workflow.train(spec: spec, examples: examples, labels: labels, extractor: HashedFeatures(dimensions: 1024))
        let registry = ClassifierRegistry(classifiers: ["tiny": try Classifier(artifact: artifact, runtime: nil, assets: nil)])

        let line = Data(#"{"op":"classify","classifier":"tiny","input":{"request":"delete the database"}}"#.utf8)
        let reply = try JSONDecoder().decode(Prediction.self, from: try await registry.handle(op: "classify", line: line))
        let listing = String(decoding: try await registry.handle(op: "classifiers", line: Data("{}".utf8)), as: UTF8.self)

        XCTAssertEqual(reply.classifier, "tiny")
        XCTAssertEqual(reply.label, "deny")
        XCTAssertEqual(artifact.training.teacher, "dataset/gold")
        XCTAssertTrue(listing.contains(#""name":"tiny""#))
        await XCTAssertThrowsErrorAsync(try await registry.handle(op: "classify", line: Data(#"{"op":"classify","classifier":"missing","input":{}}"#.utf8)))
    }

    func testL2IsSelectedByCrossValidationOnTrainOnly() async throws {
        let raw = Fixture.spec(provider: "dataset").replacingOccurrences(of: #""l2": 0.0001"#, with: #""l2_grid": [0.0001, 0.1, 5]"#)
        let spec = try Fixture.loadSpec(raw)
        let examples = try Fixture.dataset(spec, count: 45)
        var labels: [String: LabelRecord] = [:]
        _ = try await Labeler(spec: spec, examples: examples, existing: [:]).run(approve: false, client: nil, sink: { labels[$0.id] = $0 }, log: { _ in })

        let artifact = try await Workflow.train(spec: spec, examples: examples, labels: labels, extractor: HashedFeatures(dimensions: 1024))
        let selection = try XCTUnwrap(artifact.training.selection)

        XCTAssertEqual(selection.candidates.map(\.l2), [0.0001, 0.1, 5])
        XCTAssertEqual(selection.folds, 5)
        XCTAssertEqual(artifact.training.l2, selection.l2)
        XCTAssertEqual(selection.l2, selection.candidates.max { ($0.accuracy, $0.l2) < ($1.accuracy, $1.l2) }!.l2)
    }

    func testTamperedArtifactIsRejected() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec(provider: "dataset"))
        let examples = try Fixture.dataset(spec, count: 45)
        var labels: [String: LabelRecord] = [:]
        _ = try await Labeler(spec: spec, examples: examples, existing: [:]).run(approve: false, client: nil, sink: { labels[$0.id] = $0 }, log: { _ in })

        let url = try Fixture.temporaryDirectory().appendingPathComponent("tiny.classifier.json")
        try await Workflow.train(spec: spec, examples: examples, labels: labels, extractor: HashedFeatures(dimensions: 1024)).save(to: url)

        let text = try String(contentsOf: url, encoding: .utf8)
        try text.replacingOccurrences(of: #""min_confidence":0.4"#, with: #""min_confidence":0.1"#).write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(try ClassifierArtifact.load(url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("integrity"), error.localizedDescription)
        }
    }

    func testTrainingRequiresEveryLabel() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec(provider: "dataset"))
        let examples = try Fixture.dataset(spec, count: 45).filter { $0.gold != "ask" }
        var labels: [String: LabelRecord] = [:]
        _ = try await Labeler(spec: spec, examples: examples, existing: [:]).run(approve: false, client: nil, sink: { labels[$0.id] = $0 }, log: { _ in })

        do {
            _ = try await Workflow.train(spec: spec, examples: examples, labels: labels, extractor: HashedFeatures(dimensions: 1024))
            XCTFail("expected a coverage error")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("no training rows for label(s) ask"), error.localizedDescription)
        }
    }
}

func XCTAssertThrowsErrorAsync<T>(_ expression: @autoclosure () async throws -> T, file: StaticString = #filePath, line: UInt = #line) async {
    do {
        _ = try await expression()
        XCTFail("expected an error", file: file, line: line)
    } catch {}
}
