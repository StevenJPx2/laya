import Foundation
import LayaCore

@main struct LayaDaemon {
    static func main() async throws {
        // A client that disconnects mid-write must not take the daemon down with SIGPIPE.
        signal(SIGPIPE, SIG_IGN)

        let args = CommandLine.arguments
        guard let model = args.dropFirst().first else { fputs("usage: laya-daemon /path/to/model.mlmodelc [assets-directory]\n", stderr); return }
        let assets = args.dropFirst(2).first ?? URL(fileURLWithPath: model).deletingLastPathComponent().path
        let socket = NSString("~/Library/Application Support/laya/laya.sock").expandingTildeInPath
        let runtime = try await LayaRuntime(modelURL: URL(fileURLWithPath: model), assetsURL: URL(fileURLWithPath: assets))
        try await UnixSocketServer(path: socket, runtime: runtime).run()
    }
}
