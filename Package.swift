// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "laya",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "LayaCore", targets: ["LayaCore"]),
        .library(name: "LayaDistill", targets: ["LayaDistill"]),
        .executable(name: "laya-daemon", targets: ["laya-daemon"]),
        .executable(name: "laya", targets: ["laya"]),
        .executable(name: "laya-distill", targets: ["laya-distill"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.4"),
    ],
    targets: [
        .target(name: "LayaCore", dependencies: [.product(name: "Tokenizers", package: "swift-transformers")]),
        .target(name: "LayaDistill", dependencies: ["LayaCore"]),
        .executableTarget(name: "laya-daemon", dependencies: ["LayaCore", "LayaDistill"]),
        .executableTarget(name: "laya", dependencies: ["LayaCore"]),
        .executableTarget(name: "laya-distill", dependencies: ["LayaCore", "LayaDistill"]),
        .testTarget(name: "LayaCoreTests", dependencies: ["LayaCore"], resources: [.copy("Fixtures")]),
        .testTarget(name: "LayaDistillTests", dependencies: ["LayaDistill", "LayaCore"]),
    ]
)
