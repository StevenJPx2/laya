import XCTest
import LayaCore
@testable import LayaDistill

final class SpecAndDataTests: XCTestCase {
    func testTemplatesAreValidAndKeepPoolAndGoldDisjoint() throws {
        let expected = ["routing": (["billing", "technical", "sales", "other"], 60, 48), "permission": (["allow", "deny", "ask"], 60, 60)]

        for name in Templates.names {
            let template = try XCTUnwrap(Templates.named(name))
            let spec = try TaskSpec.decode(Data(template.spec.utf8))
            let file = try Fixture.temporaryDirectory().appendingPathComponent("data.jsonl")
            try template.dataset.write(to: file, atomically: true, encoding: .utf8)
            let examples = try Dataset.load(file, spec: spec)

            let gold = examples.filter { $0.gold != nil }
            let pool = examples.filter { $0.gold == nil }
            let (labels, poolCount, goldCount) = try XCTUnwrap(expected[name])

            XCTAssertEqual(spec.labelNames, labels, name)
            XCTAssertEqual(pool.count, poolCount, name)
            XCTAssertEqual(gold.count, goldCount, name)
            XCTAssertEqual(spec.abstain?.label, labels.last, "\(name): uncertain Laya answers fall back to the abstain label")
            XCTAssertTrue(Set(gold.map(\.contentHash)).isDisjoint(with: pool.map(\.contentHash)), "\(name): pool rows never duplicate gold rows")
        }
    }

    func testSpecIsStrict() {
        XCTAssertNoThrow(try Fixture.loadSpec(Fixture.spec()))

        let rejected: [(String, String)] = [
            (Fixture.spec(extra: #", "temperature": 0.2"#), "unknown field(s) temperature"),
            (Fixture.spec().replacingOccurrences(of: #""spec_version": 2"#, with: #""spec_version": 1"#), "spec_version must be 2"),
            (Fixture.spec().replacingOccurrences(of: #""uncertain": "abstain""#, with: #""uncertain": "abstain", "provider": "anthropic""#), "unknown field(s) provider"),
            (Fixture.spec(question: "noul"), "exactly 2 labels"),
            (Fixture.spec(abstain: nil), "requires an abstain label"),
            (Fixture.spec(minConfidence: 1.2), "min_confidence"),
            (Fixture.spec().replacingOccurrences(of: #""l2": 0.0001"#, with: #""l2": 0.0001, "features": "laya""#), "student.features"),
            (Fixture.spec().replacingOccurrences(of: #""uncertain": "abstain""#, with: #""uncertain": "abstain", "source": "jev""#), "teacher.source"),
        ]

        for (text, message) in rejected {
            XCTAssertThrowsError(try Fixture.loadSpec(text), message) { error in
                XCTAssertTrue(error.localizedDescription.contains(message), "\(message) not in: \(error.localizedDescription)")
            }
        }

        XCTAssertNoThrow(try Fixture.loadSpec(Fixture.spec(question: "noul", labels: ["safe", "unsafe"], abstain: nil, uncertain: "drop")))

        let plain = try? Fixture.loadSpec(Fixture.spec())
        let declared = try? Fixture.loadSpec(Fixture.spec().replacingOccurrences(of: #""uncertain": "abstain""#, with: #""uncertain": "abstain", "source": "import""#)
            .replacingOccurrences(of: #""l2": 0.0001"#, with: #""l2": 0.0001, "features": "laya-logits-v1""#))
        XCTAssertEqual(plain?.teacher.source, .laya)
        XCTAssertEqual(plain?.student.features, .hashedNgram)
        XCTAssertEqual(declared?.teacher.source, .import)
        XCTAssertEqual(declared?.student.features, .layaLogits)
        XCTAssertFalse(plain?.sha256.isEmpty ?? true)
        XCTAssertFalse(Fixture.specJSON(plain).contains("source") || Fixture.specJSON(plain).contains("features"), "undeclared optional fields keep existing spec hashes")
    }

    func testInputSchemaValidation() throws {
        let spec = try Fixture.loadSpec(Fixture.spec())

        XCTAssertThrowsError(try spec.input.parse(.object([("request", .string("x")), ("extra", .string("y"))])))
        XCTAssertThrowsError(try spec.input.parse(.object([("note", .string("x"))])))
        XCTAssertThrowsError(try spec.input.parse(.object([("request", .number(3))])))
        XCTAssertThrowsError(try spec.input.parse(.object([("request", .string(String(repeating: "a", count: 201)))])))

        let reordered = try spec.input.parse(.object([("note", .string("n")), ("request", .string("Read   Logs"))]))
        let canonical = try spec.input.parse(.object([("request", .string("read logs")), ("note", .string("N"))]))
        XCTAssertEqual(reordered.rendered, "request: Read   Logs\nnote: n")
        XCTAssertEqual(reordered.contentHash, canonical.contentHash, "hash ignores key order, case, and whitespace")
    }

    func testDatasetLoaderIsBounded() throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let file = try Fixture.temporaryDirectory().appendingPathComponent("data.jsonl")

        try #"{"id": "a", "input": {"request": "read"}, "gold": "allow"}"#.appending("\n").write(to: file, atomically: true, encoding: .utf8)
        XCTAssertEqual(try Dataset.load(file, spec: spec).count, 1)

        try #"{"id": "a", "input": {"request": "read"}, "label": "allow"}"#.write(to: file, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try Dataset.load(file, spec: spec), "unknown row field")

        try "{\"id\": \"a\", \"input\": {\"request\": \"x\"}}\n{\"id\": \"a\", \"input\": {\"request\": \"y\"}}\n".write(to: file, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try Dataset.load(file, spec: spec), "duplicate id")

        let tight = try Fixture.loadSpec(Fixture.spec().replacingOccurrences(of: #""holdout_fraction""#, with: #""max_examples": 1, "holdout_fraction""#))
        XCTAssertThrowsError(try Dataset.load(file, spec: tight), "row count bound")
    }

    func testSplitKeepsGoldEvaluationOnlyAndBlocksLeakage() throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let question = LayaQuestion.sha256(spec)

        func example(_ id: String, _ text: String, group: String? = nil, gold: String? = nil) throws -> DatasetExample {
            DatasetExample(id: id, input: try spec.input.parse(.object([("request", .string(text))])), group: group, gold: gold)
        }
        func record(_ example: DatasetExample, _ label: String?, status: LabelStatus = .accepted, question: String? = nil) -> LabelRecord {
            LabelRecord(id: example.id, contentHash: example.contentHash, questionSha256: question ?? LayaQuestion.sha256(spec), teacher: "fake",
                        status: status, layaLabel: label ?? "allow", label: label, probabilities: nil, top: nil, margin: nil,
                        layaConfidence: nil, actProbability: nil, reason: nil)
        }

        let gold = [try example("g1", "gold row", group: "shared", gold: "deny"), try example("g2", "copied text", gold: "allow")]
        let pool = [try example("leak-group", "different text", group: "shared"), try example("leak-copy", "COPIED   text"),
                    try example("dup1", "same text"), try example("dup2", "Same text"), try example("c1", "conflict"), try example("c2", "CONFLICT"),
                    try example("stale", "old question"), try example("unsure", "ambiguous"), try example("dropped", "very ambiguous"),
                    try example("error", "teacher failed")]
            + (try (0..<40).map { try example("p\($0)", "pool \($0)", group: "family-\($0 % 5)") })

        var labels = Dictionary(uniqueKeysWithValues: (gold + pool).map { ($0.id, record($0, "allow")) })
        labels["c2"] = record(pool[5], "deny")
        labels["stale"] = record(pool[6], "allow", question: "other-question")
        labels["unsure"] = record(pool[7], "ask", status: .uncertain)
        labels["dropped"] = record(pool[8], nil, status: .uncertain)
        labels["error"] = record(pool[9], nil, status: .error)

        let split = Splitter.split(gold + pool, labels: labels, spec: spec)
        let trained = Set(split.train.map(\.example.id))

        XCTAssertEqual(question, LayaQuestion.sha256(spec))
        XCTAssertEqual(Set(split.gold.map(\.example.id)), ["g1", "g2"])
        XCTAssertEqual(split.gold.first { $0.example.id == "g1" }?.label, spec.labelIndex("deny"), "gold rows keep the human label")
        XCTAssertTrue(trained.isDisjoint(with: ["g1", "g2", "leak-group", "leak-copy"]), "gold rows and pool rows sharing gold content or group are never trained on")
        XCTAssertEqual(split.report.goldOverlapExcluded, 2)
        XCTAssertEqual(split.report.duplicatesRemoved, 1)
        XCTAssertEqual(split.report.conflictsDropped, 2)
        XCTAssertEqual(split.report.staleLabels, 1)
        XCTAssertEqual(split.report.uncertainToAbstain, 1)
        XCTAssertEqual(split.report.uncertainDropped, 1)
        XCTAssertEqual(split.report.teacherErrors, 1)

        let trainGroups = Set(split.train.compactMap(\.example.group))
        XCTAssertTrue(trainGroups.isDisjoint(with: split.holdout.compactMap(\.example.group)), "a group never straddles the split")
        XCTAssertEqual(Splitter.split(gold + pool, labels: labels, spec: spec).holdout.map(\.example.id), split.holdout.map(\.example.id), "deterministic")

        let excluded = Splitter.split(gold + pool, labels: labels, spec: spec, excluding: Set((split.holdout + split.gold).map(\.example.contentHash)))
        XCTAssertTrue(excluded.holdout.isEmpty)
        XCTAssertTrue(excluded.gold.isEmpty)
        XCTAssertEqual(excluded.report.trainOverlapExcluded, split.holdout.count + split.gold.count)
    }
}
