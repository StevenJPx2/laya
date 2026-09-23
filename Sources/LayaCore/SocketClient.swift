import Darwin
import Foundation

/// Client for the daemon's newline-delimited JSON protocol: one request line
/// in, one reply line out, with explicit timeouts and a response size bound.
public enum SocketClient {
    public static var defaultPath: String {
        ProcessInfo.processInfo.environment["LAYA_SOCKET"] ?? NSString("~/Library/Application Support/laya/laya.sock").expandingTildeInPath
    }

    public static func request(_ payload: Data, path: String = defaultPath, timeoutSeconds: Int = 30, maxResponseBytes: Int = 8 << 20) throws -> Data {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw LayaError.protocolError("socket failed") }
        defer { close(fd) }

        var timeout = timeval(tv_sec: timeoutSeconds, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        try connect(fd, path: path)

        let line = payload.last == 10 ? payload : payload + Data([10])
        let written = line.withUnsafeBytes { Darwin.write(fd, $0.baseAddress, line.count) }
        guard written == line.count else { throw LayaError.protocolError("write to \(path) failed") }

        return try readLine(fd, limit: maxResponseBytes)
    }

    private static func connect(_ fd: Int32, path: String) throws {
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        guard path.utf8.count < MemoryLayout.size(ofValue: address.sun_path) else { throw LayaError.protocolError("socket path too long") }

        _ = path.withCString { pointer in
            withUnsafeMutableBytes(of: &address.sun_path) { bytes in memcpy(bytes.baseAddress, pointer, path.utf8.count) }
        }

        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard connected == 0 else { throw LayaError.protocolError("connect to \(path) failed; is laya-daemon running?") }
    }

    private static func readLine(_ fd: Int32, limit: Int) throws -> Data {
        var response = Data()
        var buffer = [UInt8](repeating: 0, count: 16384)

        while true {
            let count = read(fd, &buffer, buffer.count)
            guard count > 0 else { break }

            response.append(contentsOf: buffer[0..<count])
            guard response.count <= limit else { throw LayaError.protocolError("daemon reply exceeds \(limit) bytes") }

            if let newline = response.firstIndex(of: 10) { return response.subdata(in: response.startIndex..<newline) }
        }

        guard !response.isEmpty else { throw LayaError.protocolError("daemon closed the connection or timed out without replying") }

        return response
    }
}
