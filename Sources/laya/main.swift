import Foundation
import Darwin
import LayaCore

/// Small client for the laya daemon: predict, health, and bench over the socket.
@main struct LayaCLI {
    static let socketPath = ProcessInfo.processInfo.environment["LAYA_SOCKET"] ?? NSString("~/Library/Application Support/laya/laya.sock").expandingTildeInPath

    static func main() async throws {
        let arguments = CommandLine.arguments

        switch arguments.dropFirst().first {
        case "health":
            print(try request(Data(#"{"op":"health"}"#.utf8)))

        case "predict":
            guard let path = arguments.dropFirst(2).first else { return usage() }
            print(try request(Data(contentsOf: URL(fileURLWithPath: path))))

        case "classify":
            guard let name = arguments.dropFirst(2).first, let path = arguments.dropFirst(3).first else { return usage() }
            let input = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
            let payload: [String: JSONValue] = ["op": .string("classify"), "classifier": .string(name), "input": input]
            print(try request(JSONEncoder().encode(payload)))

        case "bench":
            guard let path = arguments.dropFirst(2).first else { return usage() }
            let iterations = arguments.dropFirst(3).first.flatMap(Int.init) ?? 1000
            try bench(path: path, iterations: iterations)

        default:
            usage()
        }
    }

    private static func usage() {
        fputs("usage: laya health | laya predict <request.json> | laya classify <classifier> <input.json> | laya bench <request.json> [iterations]\n", stderr)
    }

    /// Send one newline-terminated JSON request and return the daemon's reply line.
    private static func request(_ payload: Data) throws -> String {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw LayaError.protocolError("socket failed") }
        defer { close(fd) }

        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        _ = socketPath.withCString { ptr in
            withUnsafeMutableBytes(of: &address.sun_path) { bytes in
                memcpy(bytes.baseAddress, ptr, min(socketPath.utf8.count, bytes.count - 1))
            }
        }

        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard connected == 0 else { throw LayaError.protocolError("connect failed; is laya-daemon running?") }

        let line = payload.last == 10 ? payload : payload + Data([10])
        line.withUnsafeBytes { _ = Darwin.write(fd, $0.baseAddress, line.count) }

        var response = Data()
        var buffer = [UInt8](repeating: 0, count: 16384)

        while true {
            let count = read(fd, &buffer, buffer.count)
            if count <= 0 { break }

            response.append(contentsOf: buffer[0..<count])
            if response.contains(10) { break }
        }

        return String(decoding: response, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Time `iterations` predictions and report P50/P95 latency at the socket boundary.
    private static func bench(path: String, iterations: Int) throws {
        let payload = try Data(contentsOf: URL(fileURLWithPath: path))
        var samples: [Double] = []
        samples.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try request(payload)
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        }

        samples.sort()
        let percentile: (Double) -> Double = { samples[min(samples.count - 1, Int(Double(samples.count) * $0))] }

        print("iterations \(samples.count)")
        print(String(format: "p50 %.3f ms", percentile(0.50)))
        print(String(format: "p95 %.3f ms", percentile(0.95)))
        print(String(format: "min %.3f ms", samples.first ?? 0))
        print(String(format: "max %.3f ms", samples.last ?? 0))
    }
}
