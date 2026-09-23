import XCTest
@testable import LayaCore

/// Hard parity gate: every question in validation.json must match the Python
/// reference's selected answer, with probabilities within tolerance.
///
/// Requires the exported Core ML model and its assets:
///   LAYA_MODEL=build/laya.mlpackage LAYA_ASSETS=build/assets swift test
/// Set LAYA_CASE=<name> to run a single fixture (lighter on the machine).
final class ParityTests: XCTestCase {
    private let tolerance = 0.02

    func testValidationFixtures() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["LAYA_MODEL"] else { throw XCTSkip("set LAYA_MODEL to run Core ML parity") }
        guard let assetsPath = ProcessInfo.processInfo.environment["LAYA_ASSETS"] else { throw XCTSkip("set LAYA_ASSETS to run Core ML parity") }

        let url = try XCTUnwrap(Bundle.module.url(forResource: "validation", withExtension: "json", subdirectory: "Fixtures"))
        let cases = try JSONDecoder().decode([GoldenCase].self, from: Data(contentsOf: url))
        let runtime = try await LayaRuntime(modelURL: URL(fileURLWithPath: modelPath), assetsURL: URL(fileURLWithPath: assetsPath))

        let selectedCase = ProcessInfo.processInfo.environment["LAYA_CASE"]
        var checked = 0

        for item in cases where selectedCase == nil || selectedCase == item.name {
            let actual = try await runtime.predict(PredictRequest(state: item.state, questions: item.questions))

            for (id, expected) in item.expected.answers {
                let answer = try XCTUnwrap(actual.answers[id])

                if expected.type == "choice" {
                    XCTAssertEqual(answer.choice, expected.choice, "\(item.name)/\(id) selected answer")
                }

                if let expectedProbabilities = expected.probabilities, let actualProbabilities = answer.probabilities {
                    for (key, value) in expectedProbabilities {
                        let actualValue = actualProbabilities[key] ?? 0

                        XCTAssertLessThanOrEqual(abs(actualValue - value), tolerance, "\(item.name)/\(id)/\(key): expected \(value), actual \(actualValue)")
                    }
                }

                XCTAssertLessThanOrEqual(
                    abs(answer.action.act_probability - expected.action.act_probability),
                    tolerance,
                    "\(item.name)/\(id)/action probability"
                )

                checked += 1
            }
        }

        let expectedCount = selectedCase == nil ? 63 : questionCount(cases, named: selectedCase!)
        XCTAssertEqual(checked, expectedCount)
    }

    private func questionCount(_ cases: [GoldenCase], named name: String) -> Int {
        cases.first(where: { $0.name == name })?.expected.answers.count ?? 0
    }
}

private struct GoldenCase: Decodable {
    let name: String
    let state: JSONValue
    let questions: [String: Question]
    let expected: PredictResponse
}
