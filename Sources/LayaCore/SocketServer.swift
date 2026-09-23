import Foundation
import Darwin

/// Newline-delimited JSON over a Unix socket. One request per line; the daemon
/// keeps the model warm and serves connections as they arrive.
public final class UnixSocketServer: @unchecked Sendable {
    /// Serves additional `op` values (for example `classify`). Receives the op
    /// name and the raw request line; returns one encoded JSON response.
    public typealias OperationHandler = @Sendable (String, Data) async throws -> Data

    private let path: String
    private let runtime: LayaRuntime
    private let operations: OperationHandler?

    public init(path: String, runtime: LayaRuntime, operations: OperationHandler? = nil) {
        self.path = path
        self.runtime = runtime
        self.operations = operations
    }

    public func run() async throws {
        let directory = (path as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        unlink(path)

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw LayaError.protocolError("socket: \(errno)") }

        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        _ = path.withCString { ptr in
            withUnsafeMutableBytes(of: &address.sun_path) { bytes in
                memcpy(bytes.baseAddress, ptr, min(path.utf8.count, bytes.count - 1))
            }
        }

        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard bound == 0, listen(fd, 16) == 0 else { throw LayaError.protocolError("bind/listen: \(errno)") }

        while true {
            let client = accept(fd, nil, nil)

            if client >= 0 {
                Task { await self.serve(client) }
            }
        }
    }

    private func serve(_ client: Int32) async {
        defer { close(client) }

        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)

        while true {
            let count = read(client, &buffer, buffer.count)
            if count <= 0 { return }

            data.append(contentsOf: buffer[0..<count])

            guard let newline = data.firstIndex(of: 10) else { continue }

            let line = data.subdata(in: data.startIndex..<newline)
            await respond(to: line, on: client)
            data.removeSubrange(data.startIndex...newline)
        }
    }

    private func respond(to line: Data, on client: Int32) async {
        do {
            let request = try JSONDecoder().decode(DaemonRequest.self, from: line)

            if request.op == "health" {
                let health = await runtime.health()
                try writeJSON(health, to: client)
                return
            }

            if let op = request.op {
                guard let operations else {
                    try writeError("unknown op \(op)", to: client)
                    return
                }

                try write(try await operations(op, line) + Data([10]), to: client)
                return
            }

            guard let state = request.state, let questions = request.questions else {
                try writeError("request must include state and questions", to: client)
                return
            }

            let response = try await runtime.predict(PredictRequest(state: state, questions: questions))
            try writeJSON(response, to: client)
        } catch {
            try? writeError(error.localizedDescription, to: client)
        }
    }

    private func writeJSON<T: Encodable>(_ value: T, to client: Int32) throws {
        try write(JSONEncoder().encode(value) + Data([10]), to: client)
    }

    private func writeError(_ message: String, to client: Int32) throws {
        let escaped = message.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        try write(Data("{\"error\":\"\(escaped)\"}\n".utf8), to: client)
    }

    private func write(_ data: Data, to client: Int32) throws {
        try data.withUnsafeBytes {
            guard Darwin.write(client, $0.baseAddress, data.count) == data.count else {
                throw LayaError.protocolError("write: \(errno)")
            }
        }
    }
}
