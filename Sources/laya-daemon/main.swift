import Foundation
import LayaCore
import LayaDistill

@main struct LayaDaemon {
    static func main() async throws {
        // A client that disconnects mid-write must not take the daemon down with SIGPIPE.
        signal(SIGPIPE, SIG_IGN)

        var args = Array(CommandLine.arguments.dropFirst())
        var classifiers = ProcessInfo.processInfo.environment["LAYA_CLASSIFIERS"]

        if let flag = args.firstIndex(of: "--classifiers"), flag + 1 < args.count {
            classifiers = args[flag + 1]
            args.removeSubrange(flag...(flag + 1))
        }

        guard let model = args.first else {
            fputs("usage: laya-daemon /path/to/model.mlpackage [assets-directory] [--classifiers <dir>]\n", stderr)
            return
        }

        let assets = URL(fileURLWithPath: args.dropFirst().first ?? URL(fileURLWithPath: model).deletingLastPathComponent().path)
        let socket = ProcessInfo.processInfo.environment["LAYA_SOCKET"] ?? NSString("~/Library/Application Support/laya/laya.sock").expandingTildeInPath
        let runtime = try await LayaRuntime(modelURL: URL(fileURLWithPath: model), assetsURL: assets)
        var operations: UnixSocketServer.OperationHandler?

        if let classifiers {
            let representations = try RuntimeRepresentations(runtime: runtime, assets: assets)
            let registry = try ClassifierRegistry.load(directory: URL(fileURLWithPath: classifiers), representations: representations)
            fputs("laya-daemon: serving classifiers \(registry.names.joined(separator: ", "))\n", stderr)
            operations = { op, line in try await registry.handle(op: op, line: line) }
        }

        try await UnixSocketServer(path: socket, runtime: runtime, operations: operations).run()
    }
}
