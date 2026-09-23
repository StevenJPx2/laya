import XCTest
import LayaCore
@testable import LayaDistill

final class TeacherTests: XCTestCase {
    func testDryRunSendsNothing() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let transport = Fixture.anthropicTeacher()
        let client = TeacherClient(spec: spec, apiKey: "test-key", transport: transport)
        let labeler = Labeler(spec: spec, examples: try Fixture.dataset(spec, count: 10), existing: [:])
        var written = 0

        let summary = try await labeler.run(approve: false, client: client, sink: { _ in written += 1 }, log: { _ in })

        XCTAssertTrue(summary.dryRun)
        XCTAssertEqual(transport.requests.count, 0)
        XCTAssertEqual(written, 0)
        XCTAssertEqual(summary.plan.selected, 10)
        XCTAssertGreaterThan(summary.plan.worstCaseUsd, 0)
    }

    func testAnthropicRequestMatchesDocumentedShape() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let transport = Fixture.anthropicTeacher()
        let client = TeacherClient(spec: spec, apiKey: "test-key", transport: transport)
        let input = try spec.input.parse(.object([("request", .string("delete user secrets"))]))

        let result = try await client.label(input)
        let request = try XCTUnwrap(transport.requests.first)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: request.httpBody!) as? [String: Any])
        let format = try XCTUnwrap((body["output_config"] as? [String: Any])?["format"] as? [String: Any])
        let schema = try XCTUnwrap(format["schema"] as? [String: Any])
        let enumValues = ((schema["properties"] as? [String: Any])?["label"] as? [String: Any])?["enum"] as? [String]

        XCTAssertEqual(request.url?.absoluteString, "https://api.anthropic.com/v1/messages")
        XCTAssertEqual(request.value(forHTTPHeaderField: "x-api-key"), "test-key")
        XCTAssertEqual(request.value(forHTTPHeaderField: "anthropic-version"), "2023-06-01")
        XCTAssertEqual(body["model"] as? String, "test-teacher")
        XCTAssertEqual(body["max_tokens"] as? Int, 16)
        XCTAssertEqual(format["type"] as? String, "json_schema")
        XCTAssertEqual(enumValues, ["allow", "deny", "ask"])
        XCTAssertEqual(schema["additionalProperties"] as? Bool, false)
        XCTAssertTrue((body["system"] as? String)?.contains("answer \"ask\"") == true, "abstain policy is explained to the teacher")

        XCTAssertEqual(result.label, "deny", "reply 'Deny' is normalized to the declared label")
        XCTAssertEqual(result.inputTokens, 120)
        XCTAssertEqual(result.outputTokens, 6)
    }

    func testOpenAICompatibleRequestAndRefusal() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec(provider: "openai-compatible"))
        let transport = StubTransport { _ in
            let reply: [String: Any] = ["choices": [["finish_reason": "stop", "message": ["role": "assistant", "content": NSNull(), "refusal": "no"]]],
                                        "usage": ["prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53]]
            return (200, try! JSONSerialization.data(withJSONObject: reply))
        }
        let client = TeacherClient(spec: spec, apiKey: "local", transport: transport)

        let result = try await client.label(try spec.input.parse(.object([("request", .string("read logs"))])))
        let request = try XCTUnwrap(transport.requests.first)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: request.httpBody!) as? [String: Any])
        let format = try XCTUnwrap(body["response_format"] as? [String: Any])
        let jsonSchema = try XCTUnwrap(format["json_schema"] as? [String: Any])

        XCTAssertEqual(request.url?.absoluteString, "http://127.0.0.1:8080/v1/chat/completions")
        XCTAssertEqual(request.value(forHTTPHeaderField: "authorization"), "Bearer local")
        XCTAssertEqual(format["type"] as? String, "json_schema")
        XCTAssertEqual(jsonSchema["strict"] as? Bool, true)
        XCTAssertEqual(body["max_completion_tokens"] as? Int, 16)
        XCTAssertEqual(result.status, .refused)
        XCTAssertNil(result.label)
    }

    func testBudgetIsCumulativeAndStopsBeforeOverspend() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec(maxUsd: 0.01))
        let examples = try Fixture.dataset(spec, count: 30)
        let transport = Fixture.anthropicTeacher()
        let client = TeacherClient(spec: spec, apiKey: "k", transport: transport)
        var records: [String: LabelRecord] = [:]

        let first = try await Labeler(spec: spec, examples: examples, existing: [:])
            .run(approve: true, client: client, sink: { records[$0.id] = $0 }, log: { _ in })

        XCTAssertTrue(first.plan.truncatedByBudget)
        XCTAssertLessThan(first.requests, 30)
        XCTAssertEqual(transport.requests.count, first.requests)
        XCTAssertLessThanOrEqual(first.spentUsd, 0.01)
        XCTAssertEqual(first.spentUsd, Double(first.requests) * (120 * 10 + 6 * 50) / 1_000_000, accuracy: 1e-12, "billed from reported usage")

        let resumed = try await Labeler(spec: spec, examples: examples, existing: records)
            .run(approve: true, client: client, sink: { records[$0.id] = $0 }, log: { _ in })
        XCTAssertLessThanOrEqual(first.spentUsd + resumed.spentUsd, 0.01, "budget spans runs")
    }

    func testRetriesAreBoundedAndErrorsRedacted() async throws {
        let raw = Fixture.spec().replacingOccurrences(of: #""max_retries": 0"#, with: #""max_retries": 2"#)
        let spec = try Fixture.loadSpec(raw)
        let transport = StubTransport { _ in
            (529, Data(#"{"type":"error","error":{"type":"overloaded_error","message":"busy for bob@example.com"}}"#.utf8))
        }
        let client = TeacherClient(spec: spec, apiKey: "k", transport: transport, sleep: { _ in })

        let result = try await client.label(try spec.input.parse(.object([("request", .string("read"))])))

        XCTAssertEqual(transport.requests.count, 3, "one attempt plus max_retries")
        XCTAssertEqual(result.status, .error)
        XCTAssertTrue(result.reason?.contains("overloaded_error") == true)
        XCTAssertFalse(result.reason?.contains("bob@example.com") == true)
    }

    func testLabelFileNeverContainsInputsOrKeys() async throws {
        let spec = try Fixture.loadSpec(Fixture.spec())
        let directory = try Fixture.temporaryDirectory()
        let file = directory.appendingPathComponent("labels.jsonl")
        let client = TeacherClient(spec: spec, apiKey: "sk-test-SECRET-value", transport: Fixture.anthropicTeacher())

        _ = try await Labeler(spec: spec, examples: try Fixture.dataset(spec, count: 5), existing: [:])
            .run(approve: true, client: client, sink: { try LabelStore.append($0, to: file) }, log: { _ in })
        let text = try String(contentsOf: file, encoding: .utf8)

        XCTAssertEqual(try LabelStore.load(file, maxRows: 10).count, 5)
        XCTAssertFalse(text.contains("SECRET"))
        XCTAssertFalse(text.contains("item 0"), "raw input text is not persisted")
    }
}
