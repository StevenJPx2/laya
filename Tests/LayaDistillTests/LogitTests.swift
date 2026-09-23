import XCTest
import LayaCore
@testable import LayaDistill

final class LogitTests: XCTestCase {
    private func logitSpec(grid: Bool = true) throws -> TaskSpec {
        let student = grid ? #""l2_grid": [0.0001, 0.01, 1], "features": "laya-logits-v1""# : #""l2": 0.0001, "features": "laya-logits-v1""#
        return try Fixture.loadSpec(Fixture.spec().replacingOccurrences(of: #""l2": 0.0001"#, with: student))
    }

    /// Jev ledger -> import -> logit student over fake Laya logits -> artifact v3
    /// -> reload -> predictions and re-evaluation.
    func testLogitStudentOnImportedLabelsBeatsLayaZeroShot() async throws {
        let spec = try logitSpec()
        let examples = try Fixture.dataset(spec)
        let directory = try Fixture.temporaryDirectory()
        let ledger = directory.appendingPathComponent("jev.jsonl")
        let labelsURL = directory.appendingPathComponent("labels.jsonl")
        let artifactURL = directory.appendingPathComponent("tiny.classifier.json")
        try Fixture.jevLedger(examples).write(to: ledger, atomically: true, encoding: .utf8)
        try LabelStore.merge(try JevImport.records(ledger, spec: spec, examples: examples).records, into: labelsURL, maxRows: 200)
        let labels = try LabelStore.load(labelsURL, maxRows: 200)
        let laya = FakeRepresentations()

        await XCTAssertThrowsErrorAsync(try await Workflow.train(spec: spec, examples: examples, labels: labels)) { error in
            XCTAssertTrue(error.localizedDescription.contains("needs a Laya runtime"), error.localizedDescription)
        }

        try await Workflow.train(spec: spec, examples: examples, labels: labels, representations: laya).save(to: artifactURL)

        XCTAssertEqual(laya.calls, examples.count, "one forward pass per row despite 5-fold CV over 3 candidates, the final fit, and evaluation")

        let artifact = try ClassifierArtifact.load(artifactURL)
        let report = try XCTUnwrap(artifact.evaluation)
        let gold = try XCTUnwrap(report.gold)
        let zeroShot = try XCTUnwrap(gold.layaZeroShot)

        XCTAssertEqual(artifact.formatVersion, 3)
        XCTAssertEqual(artifact.features.scheme, .layaLogits)
        XCTAssertEqual(artifact.features.dimensions, 3)
        XCTAssertEqual(artifact.features.mean?.count, 3)
        XCTAssertEqual(artifact.features.questionSha256, LayaQuestion.sha256(spec))
        XCTAssertEqual(artifact.features.layaFingerprint, laya.fingerprint)
        XCTAssertEqual(artifact.training.teacher, "jev:jev-1.13.0")
        XCTAssertEqual(gold.teacher, "jev:jev-1.13.0")
        XCTAssertEqual(gold.teacherRaw?.accuracy, 1)
        XCTAssertEqual(zeroShot.examples, 18)
        XCTAssertEqual(zeroShot.accuracy, 1.0 / 3, accuracy: 1e-9, "the fake logits' allow bias makes zero-shot answer allow everywhere")
        XCTAssertGreaterThanOrEqual(gold.studentServed.accuracy, 0.9, "a standardized head removes the bias")
        XCTAssertEqual(gold.beatsLayaZeroShot, true)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(report.agreement).student.accuracy, 0.9)
        XCTAssertTrue(report.markdown.contains("Teacher jev:jev-1.13.0 (raw argmax)"), report.markdown)
        XCTAssertTrue(report.markdown.contains("Laya zero-shot (raw logit argmax)"), report.markdown)
        XCTAssertTrue(report.markdown.contains("beats Laya zero-shot on gold: yes"), report.markdown)

        let classifier = try Classifier(artifact: artifact, representations: laya)
        let before = laya.calls
        let wipe = try await classifier.predict(.object([("request", .string("wipe the backups"))]))
        let deploy = try await classifier.predict(.object([("request", .string("deploy the site"))]))
        XCTAssertEqual(wipe.label, "deny")
        XCTAssertEqual(deploy.label, "ask")
        XCTAssertEqual(laya.calls, before + 2)

        let again = try await Workflow.evaluate(classifier, examples: examples, labels: labels)
        XCTAssertEqual(again.gold?.studentServed.accuracy, gold.studentServed.accuracy, "re-evaluation from disk reproduces the stored report")
        XCTAssertEqual(again.gold?.layaZeroShot?.accuracy, zeroShot.accuracy)
        XCTAssertEqual(laya.calls, before + 2 + report.split.holdout + report.split.gold, "evaluation needs logits for holdout and gold only")
    }

    func testLogitArtifactRequiresTheSameLayaAssets() async throws {
        let spec = try logitSpec(grid: false)
        let examples = try Fixture.dataset(spec)
        let url = try Fixture.temporaryDirectory().appendingPathComponent("tiny.classifier.json")
        try await Workflow.train(spec: spec, examples: examples, labels: try await Fixture.label(spec, examples), representations: FakeRepresentations()).save(to: url)
        let artifact = try ClassifierArtifact.load(url)

        XCTAssertNoThrow(try Classifier(artifact: artifact, representations: FakeRepresentations()))
        XCTAssertThrowsError(try Classifier(artifact: artifact, representations: FakeRepresentations(fingerprint: "other-assets"))) { error in
            XCTAssertTrue(error.localizedDescription.contains("retrain against this runtime"), error.localizedDescription)
        }
        XCTAssertThrowsError(try Classifier(artifact: artifact)) { error in
            XCTAssertTrue(error.localizedDescription.contains("needs a Laya runtime"), error.localizedDescription)
        }

        let text = try String(contentsOf: url, encoding: .utf8)
        try text.replacingOccurrences(of: #""format_version":3"#, with: #""format_version":2"#).write(to: url, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try ClassifierArtifact.load(url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("v2; retrain"), error.localizedDescription)
        }
    }

    func testRegistryServesLogitStudentWithOneForwardPass() async throws {
        let spec = try logitSpec(grid: false)
        let examples = try Fixture.dataset(spec)
        let directory = try Fixture.temporaryDirectory()
        let laya = FakeRepresentations()
        try await Workflow.train(spec: spec, examples: examples, labels: try await Fixture.label(spec, examples), representations: laya)
            .save(to: directory.appendingPathComponent("tiny.classifier.json"))

        let registry = try ClassifierRegistry.load(directory: directory, representations: laya)
        let before = laya.calls
        let line = Data(#"{"op":"classify","classifier":"tiny","input":{"request":"delete the database"}}"#.utf8)
        let reply = try JSONDecoder().decode(Prediction.self, from: try await registry.handle(op: "classify", line: line))
        let listing = String(decoding: try await registry.handle(op: "classifiers", line: Data("{}".utf8)), as: UTF8.self)

        XCTAssertEqual(reply.label, "deny")
        XCTAssertEqual(laya.calls, before + 1)
        XCTAssertTrue(listing.contains(#""features":"laya-logits-v1""#), listing)
        XCTAssertThrowsError(try ClassifierRegistry.load(directory: directory), "a logit student needs the daemon's runtime")
        XCTAssertThrowsError(try ClassifierRegistry.load(directory: directory, representations: FakeRepresentations(fingerprint: "x"))) { error in
            XCTAssertTrue(error.localizedDescription.contains("tiny.classifier.json"), error.localizedDescription)
        }
    }

    /// Embedding and hybrid students: logits + pooled (8 fake dims), and for
    /// hybrid, hashed n-grams appended after the standardized Laya values.
    func testEmbeddingAndHybridStudentsRoundTripAndServe() async throws {
        for (scheme, dimensions) in [(FeatureScheme.layaEmbedding, 11), (.layaHybrid, 11 + 1024)] {
            let spec = try Fixture.loadSpec(Fixture.spec().replacingOccurrences(of: #""l2": 0.0001"#, with: #""l2": 0.0001, "features": "\#(scheme.rawValue)""#))
            let examples = try Fixture.dataset(spec)
            let url = try Fixture.temporaryDirectory().appendingPathComponent("tiny.classifier.json")
            let laya = FakeRepresentations()
            try await Workflow.train(spec: spec, examples: examples, labels: try await Fixture.label(spec, examples), representations: laya).save(to: url)

            let artifact = try ClassifierArtifact.load(url)
            let gold = try XCTUnwrap(artifact.evaluation?.gold)
            let classifier = try Classifier(artifact: artifact, representations: laya)

            XCTAssertEqual(artifact.features.scheme, scheme)
            XCTAssertEqual(artifact.features.dimensions, dimensions)
            XCTAssertEqual(artifact.features.mean?.count, 11, "logits and pooled values are standardized")
            XCTAssertEqual(laya.calls, examples.count, "\(scheme.rawValue): one forward pass per row")
            XCTAssertGreaterThanOrEqual(gold.studentServed.accuracy, 0.9, scheme.rawValue)
            XCTAssertEqual(gold.layaZeroShot?.accuracy ?? 0, 1.0 / 3, accuracy: 1e-9, "zero-shot reads only the logits")
            let wipe = try await classifier.predict(.object([("request", .string("wipe the backups"))]))
            XCTAssertEqual(wipe.label, "deny")
            XCTAssertThrowsError(try Classifier(artifact: artifact)) { error in
                XCTAssertTrue(error.localizedDescription.contains("\(scheme.rawValue) features and needs a Laya runtime"), error.localizedDescription)
            }
        }
    }

    func testHashedStudentAddsZeroShotWhenLayaIsAvailable() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let examples = try Fixture.dataset(spec)
        let laya = FakeRepresentations()

        let artifact = try await Workflow.train(spec: spec, examples: examples, labels: try await Fixture.label(spec, examples), representations: laya)

        XCTAssertEqual(artifact.features, .hashed(dimensions: 1024))
        XCTAssertEqual(laya.calls, 18, "hashed students ask Laya only about gold rows")
        XCTAssertEqual(artifact.evaluation?.gold?.layaZeroShot?.examples, 18)
        XCTAssertNotNil(artifact.evaluation?.gold?.beatsLayaZeroShot)
        XCTAssertNoThrow(try Classifier(artifact: artifact, representations: FakeRepresentations(fingerprint: "any")), "hashed students ignore Laya assets")
    }

    func testLogitsAreMappedToLabelOrder() throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let options = try Prompt.renderedOptions(LayaQuestion.make(spec))
        let representation = Representation(pooled: [], logits: [1, 2, 3], options: options)

        XCTAssertEqual(options, ["allow: allow description", "ask: ask description", "deny: deny description"], "Laya orders choice options itself")
        XCTAssertEqual(try LayaLogits.labelOrder(representation, spec: spec), [0, 2, 1])
        XCTAssertThrowsError(try LayaLogits.labelOrder(Representation(pooled: [], logits: [1, 2], options: Array(options.prefix(2))), spec: spec))

        let statistics = LayaLogits.statistics([[1, 5], [3, 5]])
        XCTAssertEqual(statistics.mean, [2, 5])
        XCTAssertEqual(statistics.sd, [1, LayaLogits.sdFloor], "constant logits are floored, not divided by zero")
    }
}
