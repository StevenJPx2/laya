import XCTest
@testable import LayaCore

final class PromptTests: XCTestCase {
    func testNoulDefaultsAndConfidence() throws {
        let question = Question(type: "noul", instructions: .string("holds?"), criteria: nil)
        XCTAssertEqual(try Prompt.renderedOptions(question), ["false: no, the statement does not hold", "true: yes, the statement holds"])
        XCTAssertEqual(Prompt.confidence([0.5, 0.5]), 0, accuracy: 0.0001)
    }
}
