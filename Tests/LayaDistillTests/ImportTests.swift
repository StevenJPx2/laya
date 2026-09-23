import XCTest
import LayaCore
@testable import LayaDistill

final class ImportTests: XCTestCase {
    func testJevImportGatesLabelsAndNeverTrainsOnFailures() throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let examples = try Fixture.dataset(spec, pool: 9, gold: 0)
        let directory = try Fixture.temporaryDirectory()
        let ledger = directory.appendingPathComponent("jev.jsonl")
        let labelsURL = directory.appendingPathComponent("labels.jsonl")
        let lines = [
            Fixture.jevLine("pool-0", probabilities: ["allow": 1, "deny": 0, "ask": 0]),
            Fixture.jevLine("pool-1", status: "failed", probabilities: nil, error: "timeout"),
            Fixture.jevLine("pool-1", probabilities: ["allow": 0.05, "deny": 0.9, "ask": 0.05]),
            Fixture.jevLine("pool-2", probabilities: ["allow": 1, "deny": 0, "ask": 0]),
            Fixture.jevLine("pool-2", status: "forbidden", probabilities: nil),
            Fixture.jevLine("pool-3", status: "failed", probabilities: nil, error: "rate limited"),
            Fixture.jevLine("pool-4", probabilities: ["allow": 0.5, "deny": 0.3, "ask": 0.2]),
            Fixture.jevLine("pool-5", probabilities: ["allow": 0.3, "deny": 0.1, "ask": 0.1]),
            Fixture.jevLine("ghost", probabilities: ["allow": 1, "deny": 0, "ask": 0]),
        ]
        try (lines.joined(separator: "\n") + "\n").write(to: ledger, atomically: true, encoding: .utf8)

        let (records, summary) = try JevImport.records(ledger, spec: spec, examples: examples)
        let byID = Dictionary(uniqueKeysWithValues: records.map { ($0.id, $0) })

        XCTAssertEqual(summary, ImportSummary(teacher: "jev:jev-1.13.0", ledgerRows: 9, imported: 6, accepted: 2, uncertainToAbstain: 1,
                                              uncertainDropped: 0, forbidden: 1, failed: 1, invalid: 1, unknownIds: 1, missing: 3))
        XCTAssertEqual(byID["pool-1"]?.label, "deny", "the last line per id wins")
        XCTAssertEqual(byID["pool-2"]?.status, .error, "a later forbidden line replaces an earlier label")
        XCTAssertNil(byID["pool-2"]?.label)
        XCTAssertEqual(byID["pool-2"]?.reason, "jev forbidden")
        XCTAssertEqual(byID["pool-2"]?.teacher, "jev:jev-1.13.0", "forbidden rows fall back to the requested model")
        XCTAssertEqual(byID["pool-3"]?.reason, "jev failed: rate limited")
        XCTAssertEqual(byID["pool-4"]?.status, .uncertain, "imported answers pass the same confidence gate")
        XCTAssertEqual(byID["pool-4"]?.layaLabel, "allow")
        XCTAssertEqual(byID["pool-4"]?.label, "ask")
        XCTAssertEqual(byID["pool-5"]?.status, .error)
        XCTAssertTrue(records.allSatisfy { $0.questionSha256 == LayaQuestion.sha256(spec) })
        XCTAssertEqual(byID["pool-0"]?.contentHash, examples[0].contentHash, "the dataset's content hash, not the ledger's")
        XCTAssertNil(byID["pool-0"]?.layaConfidence)

        try LabelStore.merge(records, into: labelsURL, maxRows: 100)
        try LabelStore.merge(records, into: labelsURL, maxRows: 100)
        let labels = try LabelStore.load(labelsURL, maxRows: 100)
        let split = Splitter.split(examples, labels: labels, spec: spec)
        let usable = Set((split.train + split.holdout).map(\.example.id))

        XCTAssertEqual(try String(contentsOf: labelsURL, encoding: .utf8).split(separator: "\n").count, 6, "merging rewrites instead of appending")
        XCTAssertEqual(usable, ["pool-0", "pool-1", "pool-4"])
        XCTAssertEqual(split.report.teacherErrors, 3)
        XCTAssertEqual(split.report.unlabeled, 3)
        XCTAssertEqual(Set(Labeler(spec: spec, examples: examples, existing: labels).pending().map(\.id)),
                       ["pool-2", "pool-3", "pool-5", "pool-6", "pool-7", "pool-8"], "imported labels count as current")
    }

    func testJevImportRejectsLedgersForAnotherTask() throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let examples = try Fixture.dataset(spec, pool: 3, gold: 0)
        let ledger = try Fixture.temporaryDirectory().appendingPathComponent("jev.jsonl")
        let rejected = [
            (Fixture.jevLine("pool-0", probabilities: ["allow": 1, "deny": 0]), "task labels are [allow, deny, ask]"),
            (Fixture.jevLine("pool-0", probabilities: ["allow": 1, "deny": 0, "maybe": 0]), "task labels"),
            (Fixture.jevLine("pool-0", probabilities: ["allow": 1, "deny": 0, "ask": 0]).replacingOccurrences(of: "jev-distill.labels", with: "other"), "expected ledger"),
            (Fixture.jevLine("pool-0", status: "pending", probabilities: nil), "unknown status"),
        ]

        for (line, message) in rejected {
            try (line + "\n").write(to: ledger, atomically: true, encoding: .utf8)
            XCTAssertThrowsError(try JevImport.records(ledger, spec: spec, examples: examples), message) { error in
                XCTAssertTrue(error.localizedDescription.contains(message), "\(message) not in: \(error.localizedDescription)")
            }
        }
    }
}
