import XCTest
import LayaCore
@testable import LayaDistill

/// Trains a head on real frozen Laya representations. Requires the exported model:
///   LAYA_MODEL=build/laya.mlpackage LAYA_ASSETS=build/assets swift test --filter LayaFeatureTests
final class LayaFeatureTests: XCTestCase {
    func testPermissionTemplateOnLayaFeatures() async throws {
        guard let model = ProcessInfo.processInfo.environment["LAYA_MODEL"],
              let assetsPath = ProcessInfo.processInfo.environment["LAYA_ASSETS"] else { throw XCTSkip("set LAYA_MODEL and LAYA_ASSETS") }

        let assets = URL(fileURLWithPath: assetsPath)
        let runtime = try await LayaRuntime(modelURL: URL(fileURLWithPath: model), assetsURL: assets)
        let spec = try TaskSpec.decode(Data(PermissionTemplate.spec.utf8))
        let directory = try Fixture.temporaryDirectory()
        let dataURL = directory.appendingPathComponent("data.jsonl")
        try PermissionTemplate.dataset.write(to: dataURL, atomically: true, encoding: .utf8)

        let examples = try Dataset.load(dataURL, spec: spec)
        var labels: [String: LabelRecord] = [:]
        _ = try await Labeler(spec: spec, examples: examples, existing: [:]).run(approve: false, client: nil, sink: { labels[$0.id] = $0 }, log: { _ in })

        let fingerprint = try LayaFeatures.fingerprint(assets: assets)
        let extractor = try LayaFeatures(runtime: runtime, spec: spec, fingerprint: fingerprint)
        let artifactURL = directory.appendingPathComponent("permission.classifier.json")
        try await Workflow.train(spec: spec, examples: examples, labels: labels, extractor: extractor).save(to: artifactURL)

        let artifact = try ClassifierArtifact.load(artifactURL)
        let classifier = try Classifier(artifact: artifact, runtime: runtime, assets: assets)
        let report = try XCTUnwrap(artifact.evaluation)
        let prediction = try await classifier.predict(.object([("tool", .string("Bash")), ("request", .string("cat ~/.ssh/id_ed25519"))]))

        XCTAssertEqual(artifact.features.kind, .laya)
        XCTAssertEqual(artifact.features.dimensions, 1024 + 3)
        XCTAssertEqual(report.baselines.last?.status, "ok", "zero-shot baseline runs on Laya features")
        XCTAssertEqual(prediction.probabilities.count, 3)
        print("laya-features holdout report:\n\(report.markdown)")

        let other = ClassifierArtifact(spec: artifact.spec, features: FeatureDescriptor(kind: .laya, dimensions: 1027, modelFingerprint: "other"),
                                       model: artifact.model, training: artifact.training, evaluation: nil)
        XCTAssertThrowsError(try Classifier(artifact: other, runtime: runtime, assets: assets), "fingerprint mismatch is refused")
    }
}
