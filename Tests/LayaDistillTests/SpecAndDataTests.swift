import XCTest
import LayaCore
@testable import LayaDistill

final class SpecAndDataTests: XCTestCase {
    func testTemplatesAndFixturesAreValid() throws {
        let permission = try TaskSpec.decode(Data(PermissionTemplate.spec.utf8))
        XCTAssertEqual(permission.labelNames, ["allow", "deny", "ask"])
        XCTAssertEqual(permission.teacher.provider, .dataset)

        for provider in ["anthropic", "openai-compatible", "dataset"] {
            XCTAssertNoThrow(try Fixture.loadSpec(Fixture.spec(provider: provider)), provider)
        }
    }

    func testSpecIsStrict() {
        let unknownTop = Fixture.spec(extra: #", "temperature": 0.2"#)
        XCTAssertThrowsError(try Fixture.loadSpec(unknownTop)) { error in
            XCTAssertTrue(error.localizedDescription.contains("unknown field(s) temperature"), error.localizedDescription)
        }

        let inlineKey = Fixture.spec().replacingOccurrences(of: #""api_key_env": "TEST_TEACHER_KEY""#, with: #""api_key": "sk-live-secret""#)
        XCTAssertThrowsError(try Fixture.loadSpec(inlineKey)) { error in
            XCTAssertTrue(error.localizedDescription.contains("unknown field(s) api_key"), error.localizedDescription)
        }

        let wrongVersion = Fixture.spec().replacingOccurrences(of: #""spec_version": 1"#, with: #""spec_version": 2"#)
        XCTAssertThrowsError(try Fixture.loadSpec(wrongVersion))

        let noModel = Fixture.spec().replacingOccurrences(of: #""model": "test-teacher""#, with: #""model": """#)
        XCTAssertThrowsError(try Fixture.loadSpec(noModel)) { error in
            XCTAssertTrue(error.localizedDescription.contains("teacher.model must be set explicitly"))
        }

        let badAbstain = Fixture.spec().replacingOccurrences(of: #""label": "ask", "min_confidence""#, with: #""label": "maybe", "min_confidence""#)
        XCTAssertThrowsError(try Fixture.loadSpec(badAbstain))

        let plainHTTP = Fixture.spec(provider: "openai-compatible").replacingOccurrences(of: "http://127.0.0.1:8080", with: "http://teacher.example.com")
        XCTAssertThrowsError(try Fixture.loadSpec(plainHTTP)) { error in
            XCTAssertTrue(error.localizedDescription.contains("https"))
        }
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
        let directory = try Fixture.temporaryDirectory()
        let file = directory.appendingPathComponent("data.jsonl")

        try #"{"id": "a", "input": {"request": "read"}, "gold": "allow"}"#.appending("\n").write(to: file, atomically: true, encoding: .utf8)
        XCTAssertEqual(try Dataset.load(file, spec: spec).count, 1)

        try #"{"id": "a", "input": {"request": "read"}, "label": "allow"}"#.write(to: file, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try Dataset.load(file, spec: spec), "unknown row field")

        try "{\"id\": \"a\", \"input\": {\"request\": \"x\"}}\n{\"id\": \"a\", \"input\": {\"request\": \"y\"}}\n".write(to: file, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try Dataset.load(file, spec: spec), "duplicate id")

        let tight = try Fixture.loadSpec(Fixture.spec().replacingOccurrences(of: #""holdout_fraction""#, with: #""max_examples": 1, "holdout_fraction""#))
        XCTAssertThrowsError(try Dataset.load(file, spec: tight), "row count bound")
    }

    func testSplitLeakageControls() throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        func example(_ id: String, _ text: String, group: String? = nil) throws -> DatasetExample {
            DatasetExample(id: id, input: try spec.input.parse(.object([("request", .string(text))])), group: group, gold: nil)
        }
        func record(_ example: DatasetExample, _ label: String) -> LabelRecord {
            LabelRecord(id: example.id, contentHash: example.contentHash, label: label, status: .ok, reason: nil, teacher: "t", inputTokens: 0, outputTokens: 0, costUsd: 0)
        }

        var examples = [try example("dup1", "same text"), try example("dup2", "SAME   text"), try example("c1", "conflict"), try example("c2", "Conflict"),
                        try example("fewshot", "Read the changelog")]
        examples += try (0..<40).map { try example("g\($0)", "grouped \($0)", group: "family-\($0 % 4)") }

        var labels = Dictionary(uniqueKeysWithValues: examples.map { ($0.id, record($0, "allow")) })
        labels["c2"] = record(examples[3], "deny")
        labels["g0"] = LabelRecord(id: "g0", contentHash: "stale", label: "allow", status: .ok, reason: nil, teacher: "t", inputTokens: 0, outputTokens: 0, costUsd: 0)

        let split = Splitter.split(examples, labels: labels, spec: spec)
        XCTAssertEqual(split.report.duplicatesRemoved, 1)
        XCTAssertEqual(split.report.conflictsDropped, 2)
        XCTAssertEqual(split.report.staleLabels, 1)
        XCTAssertFalse(split.holdout.contains { $0.example.id == "fewshot" }, "few-shot inputs never enter holdout")

        let trainGroups = Set(split.train.compactMap(\.example.group))
        let holdoutGroups = Set(split.holdout.compactMap(\.example.group))
        XCTAssertTrue(trainGroups.isDisjoint(with: holdoutGroups), "a group never straddles the split")
        XCTAssertTrue(Set(split.train.map(\.example.contentHash)).isDisjoint(with: split.holdout.map(\.example.contentHash)))

        let again = Splitter.split(examples, labels: labels, spec: spec)
        XCTAssertEqual(again.holdout.map(\.example.id), split.holdout.map(\.example.id), "split is deterministic")

        let excluded = Splitter.split(examples, labels: labels, spec: spec, excluding: Set(split.holdout.map(\.example.contentHash)))
        XCTAssertTrue(excluded.holdout.isEmpty)
        XCTAssertEqual(excluded.report.trainOverlapExcluded, split.holdout.count)
    }

    func testRedactor() {
        let text = "mail bob@example.com key sk-ant-abcdefghijklmnop phone +1 (415) 555-0100"
        let redacted = Redactor.redact(text)

        XCTAssertFalse(redacted.contains("bob@example.com"))
        XCTAssertFalse(redacted.contains("sk-ant-abcdefghijklmnop"))
        XCTAssertFalse(redacted.contains("555-0100"))
    }
}
