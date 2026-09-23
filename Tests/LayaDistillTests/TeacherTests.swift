import XCTest
import LayaCore
@testable import LayaDistill

final class TeacherTests: XCTestCase {
    func testConfidenceGateBoundaries() throws {
        let spec = try Fixture.loadSpec(Fixture.spec(minConfidence: 0.6, minMargin: 0.2))

        let atThreshold = try TeacherPolicy.decide([0.6, 0.4, 0.0], spec: spec)
        XCTAssertEqual(atThreshold.status, .accepted, "top == min_confidence and margin == min_margin are accepted (0.6 - 0.4 is not exactly 0.2 in binary)")
        XCTAssertEqual(atThreshold.label, "allow")

        let lowTop = try TeacherPolicy.decide([0.5999, 0.2, 0.2001], spec: spec)
        XCTAssertEqual(lowTop.status, .uncertain)
        XCTAssertEqual(lowTop.layaLabel, "allow", "Laya's raw pick is kept for audit")
        XCTAssertEqual(lowTop.label, "ask", "uncertain answers train as the abstain label, never as Laya's argmax")

        let narrow = try TeacherPolicy.decide([0.1, 0.61, 0.29], spec: spec)
        XCTAssertEqual(narrow.status, .accepted)
        let tooNarrow = try TeacherPolicy.decide([0.0, 0.6, 0.4001], spec: spec)
        XCTAssertEqual(tooNarrow.status, .uncertain, "a high top probability still needs the margin")
        XCTAssertEqual(tooNarrow.margin, 0.1999, accuracy: 1e-9)

        let confidentAsk = try TeacherPolicy.decide([0.1, 0.1, 0.8], spec: spec)
        XCTAssertEqual(confidentAsk.status, .accepted, "a confident 'ask' is a real ask label")

        let drop = try Fixture.loadSpec(Fixture.spec(uncertain: "drop"))
        XCTAssertNil(try TeacherPolicy.decide([0.5, 0.3, 0.2], spec: drop).label, "drop policy excludes the row")

        XCTAssertThrowsError(try TeacherPolicy.decide([0.5, 0.5], spec: spec), "wrong label count")
        XCTAssertThrowsError(try TeacherPolicy.decide([0.5, 0.3, 0.3], spec: spec), "does not sum to 1")
        XCTAssertThrowsError(try TeacherPolicy.decide([.nan, 0.5, 0.5], spec: spec), "not finite")
    }

    func testQuestionConstructionAndAnswerMapping() throws {
        let choice = try Fixture.loadSpec(Fixture.spec())
        let question = LayaQuestion.make(choice)
        XCTAssertEqual(question.type, "choice")
        // Laya orders choice options itself; answers are mapped back by label name, so spec order is preserved.
        XCTAssertEqual(Set(try Prompt.renderedOptions(question)), ["allow: allow description", "deny: deny description", "ask: ask description"])
        XCTAssertEqual(try LayaQuestion.probabilities(Fixture.answer(["ask": 0.2, "allow": 0.5, "deny": 0.3]), spec: choice), [0.5, 0.3, 0.2])
        XCTAssertThrowsError(try LayaQuestion.probabilities(Fixture.answer(["allow": 1]), spec: choice))

        let score = try Fixture.loadSpec(Fixture.spec(question: "score", labels: ["low", "medium", "high"], abstain: nil, uncertain: "drop"))
        XCTAssertEqual(try Prompt.renderedOptions(LayaQuestion.make(score)).first, "level 0: low: low description")
        XCTAssertEqual(try LayaQuestion.probabilities(Fixture.answer(["0": 0.1, "1": 0.2, "2": 0.7], type: "score"), spec: score), [0.1, 0.2, 0.7])

        let noul = try Fixture.loadSpec(Fixture.spec(question: "noul", labels: ["safe", "unsafe"], abstain: nil, uncertain: "drop"))
        XCTAssertEqual(try Prompt.renderedOptions(LayaQuestion.make(noul)), ["false: safe description", "true: unsafe description"])
        let mapped = try LayaQuestion.probabilities(Fixture.answer([:], type: "noul", noul: 0.8), spec: noul)
        XCTAssertEqual(mapped[0], 0.2, accuracy: 1e-12)
        XCTAssertEqual(mapped[1], 0.8, accuracy: 1e-12)

        let reworded = try Fixture.loadSpec(Fixture.spec().replacingOccurrences(of: "Gate the requested action.", with: "Gate it."))
        XCTAssertNotEqual(LayaQuestion.sha256(choice), LayaQuestion.sha256(reworded), "labels are tied to the exact question")
    }

    func testLabelerPersistsProvenanceAndResumes() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let examples = try Fixture.dataset(spec, pool: 9, gold: 0) + [
            DatasetExample(id: "vague", input: try spec.input.parse(.object([("request", .string("maybe do something"))])), group: nil, gold: nil),
        ]
        let file = try Fixture.temporaryDirectory().appendingPathComponent("labels.jsonl")
        let teacher = Fixture.verbTeacher()

        let summary = try await Labeler(spec: spec, examples: examples, existing: [:])
            .run(teacher: teacher, sink: { try LabelStore.append($0, to: file) }, log: { _ in })
        let records = try LabelStore.load(file, maxRows: 100)
        let vague = try XCTUnwrap(records["vague"])
        let text = try String(contentsOf: file, encoding: .utf8)

        XCTAssertEqual(summary.accepted, 9)
        XCTAssertEqual(summary.uncertainToAbstain, 1)
        XCTAssertEqual(vague.status, .uncertain)
        XCTAssertEqual(vague.layaLabel, "allow")
        XCTAssertEqual(vague.label, "ask")
        XCTAssertEqual(vague.probabilities, ["allow": 0.4, "deny": 0.35, "ask": 0.25])
        XCTAssertEqual(vague.top, 0.4)
        XCTAssertEqual(vague.teacher, "fake-laya")
        XCTAssertEqual(vague.questionSha256, LayaQuestion.sha256(spec))
        XCTAssertEqual(vague.actProbability, 0.9)
        XCTAssertFalse(text.contains("item 0") || text.contains("maybe"), "inputs are not persisted")

        let resumed = Labeler(spec: spec, examples: examples, existing: records)
        XCTAssertTrue(resumed.pending().isEmpty, "labeled rows are not re-asked")

        let reworded = try Fixture.loadSpec(Fixture.spec().replacingOccurrences(of: "Gate the requested action.", with: "Gate it."))
        XCTAssertEqual(Labeler(spec: reworded, examples: examples, existing: records).pending().count, examples.count, "a new question relabels everything")

        let limited = try await Labeler(spec: reworded, examples: examples, existing: records).run(teacher: teacher, limit: 3, sink: { _ in }, log: { _ in })
        XCTAssertEqual(limited.selected, 3)
        XCTAssertEqual(teacher.calls, examples.count + 3)
    }

    func testLabelerStopsAfterConsecutiveTeacherErrors() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let failing = FakeTeacher { _, _ in throw LayaError.protocolError("daemon unavailable") }
        var records: [LabelRecord] = []

        let summary = try await Labeler(spec: spec, examples: try Fixture.dataset(spec, pool: 20, gold: 0), existing: [:])
            .run(teacher: failing, sink: { records.append($0) }, log: { _ in })

        XCTAssertEqual(failing.calls, Labeler.maxConsecutiveErrors)
        XCTAssertEqual(summary.errors, Labeler.maxConsecutiveErrors)
        XCTAssertTrue(summary.stopReason?.contains("daemon unavailable") == true)
        XCTAssertTrue(records.allSatisfy { $0.status == .error && $0.label == nil })
    }
}
