// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "laya",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "LayaCore", targets: ["LayaCore"]),
        .executable(name: "laya-daemon", targets: ["laya-daemon"]),
        .executable(name: "laya", targets: ["laya"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.4"),
    ],
    targets: [
        .target(name: "LayaCore", dependencies: [.product(name: "Tokenizers", package: "swift-transformers")]),
        .executableTarget(name: "laya-daemon", dependencies: ["LayaCore"]),
        .executableTarget(name: "laya", dependencies: ["LayaCore"]),
        .testTarget(name: "LayaCoreTests", dependencies: ["LayaCore"], resources: [.copy("Fixtures")]),
    ]
)
