import Foundation
import LayaCore

/// One dataset row. `gold` is an optional human label, used by the `dataset`
/// teacher and reported separately from teacher agreement during evaluation.
public struct DatasetExample: Sendable {
    public let id: String
    public let input: TaskInput
    public let group: String?
    public let gold: String?

    public var contentHash: String { input.contentHash }
}

private struct RawExample: Decodable {
    let id: String
    let input: JSONValue
    let group: String?
    let gold: String?

    enum CodingKeys: String, CodingKey, CaseIterable { case id, input, group, gold }

    init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "dataset row")
        id = try c.decode(String.self, forKey: .id)
        input = try c.decode(JSONValue.self, forKey: .input)
        group = try c.decodeIfPresent(String.self, forKey: .group)
        gold = try c.decodeIfPresent(String.self, forKey: .gold)
    }
}

enum JSONLines {
    /// Read a JSONL file within explicit size bounds. Blank lines are skipped.
    static func read(_ url: URL, maxLines: Int, maxLineBytes: Int) throws -> [(line: Int, data: Data)] {
        let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0
        guard size <= maxLines * (maxLineBytes + 1) else {
            throw DistillError.invalidData("\(url.lastPathComponent) is \(size) bytes, above the \(maxLines) x \(maxLineBytes) byte bound")
        }

        let data = try Data(contentsOf: url)
        var rows: [(Int, Data)] = []

        for (index, slice) in data.split(separator: 10, omittingEmptySubsequences: false).enumerated() {
            guard !slice.allSatisfy({ $0 == 32 || $0 == 9 || $0 == 13 }) else { continue }
            guard slice.count <= maxLineBytes else { throw DistillError.invalidData("line \(index + 1) exceeds \(maxLineBytes) bytes") }

            rows.append((index + 1, Data(slice)))
            guard rows.count <= maxLines else { throw DistillError.invalidData("more than \(maxLines) rows; raise the configured bound deliberately") }
        }

        return rows
    }
}

public enum Dataset {
    public static func load(_ url: URL, spec: TaskSpec) throws -> [DatasetExample] {
        let rows = try JSONLines.read(url, maxLines: spec.dataset.maxExamples, maxLineBytes: spec.dataset.maxLineBytes)
        var seen = Set<String>()
        var examples: [DatasetExample] = []

        for row in rows {
            let raw: RawExample
            do {
                raw = try JSONDecoder().decode(RawExample.self, from: row.data)
            } catch let error as DistillError {
                throw DistillError.invalidData("line \(row.line): \(error.localizedDescription)")
            } catch {
                throw DistillError.invalidData("line \(row.line): \(describe(error))")
            }

            examples.append(try validate(raw, line: row.line, spec: spec, seen: &seen))
        }

        return examples
    }

    private static func validate(_ raw: RawExample, line: Int, spec: TaskSpec, seen: inout Set<String>) throws -> DatasetExample {
        guard matches(raw.id, "^[A-Za-z0-9._:-]{1,128}$") else { throw DistillError.invalidData("line \(line): id must match ^[A-Za-z0-9._:-]{1,128}$") }
        guard seen.insert(raw.id).inserted else { throw DistillError.invalidData("line \(line): duplicate id \(raw.id)") }

        if let gold = raw.gold, spec.labelIndex(gold) == nil {
            throw DistillError.invalidData("line \(line): gold '\(gold)' is not a declared label")
        }
        if let group = raw.group, group.isEmpty || group.count > 128 {
            throw DistillError.invalidData("line \(line): group must be 1-128 characters")
        }

        let input = try spec.input.parse(raw.input, context: "line \(line) input")

        return DatasetExample(id: raw.id, input: input, group: raw.group, gold: raw.gold)
    }

    /// Hash of the dataset contents in file order; recorded in artifacts.
    static func fingerprint(_ examples: [DatasetExample]) -> String {
        Canonical.sha256(examples.map { "\($0.id)\t\($0.contentHash)\t\($0.group ?? "")\t\($0.gold ?? "")" }.joined(separator: "\n"))
    }
}

public enum LabelStatus: String, Codable, Sendable { case ok, invalid, refused, error }

/// One teacher decision. Prompts and inputs are never stored, only the
/// example id, its content hash, the label, and usage/cost accounting.
public struct LabelRecord: Codable, Sendable {
    public let id: String
    public let contentHash: String
    public let label: String?
    public let status: LabelStatus
    public let reason: String?
    public let teacher: String
    public let inputTokens: Int
    public let outputTokens: Int
    public let costUsd: Double

    enum CodingKeys: String, CodingKey, CaseIterable {
        case id, contentHash = "content_hash", label, status, reason, teacher
        case inputTokens = "input_tokens", outputTokens = "output_tokens", costUsd = "cost_usd"
    }

    public init(id: String, contentHash: String, label: String?, status: LabelStatus, reason: String?, teacher: String, inputTokens: Int, outputTokens: Int, costUsd: Double) {
        self.id = id
        self.contentHash = contentHash
        self.label = label
        self.status = status
        self.reason = reason
        self.teacher = teacher
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.costUsd = costUsd
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.strictContainer(keyedBy: CodingKeys.self, context: "label row")
        id = try c.decode(String.self, forKey: .id)
        contentHash = try c.decode(String.self, forKey: .contentHash)
        label = try c.decodeIfPresent(String.self, forKey: .label)
        status = try c.decode(LabelStatus.self, forKey: .status)
        reason = try c.decodeIfPresent(String.self, forKey: .reason)
        teacher = try c.decode(String.self, forKey: .teacher)
        inputTokens = try c.decode(Int.self, forKey: .inputTokens)
        outputTokens = try c.decode(Int.self, forKey: .outputTokens)
        costUsd = try c.decode(Double.self, forKey: .costUsd)
    }
}

public enum LabelStore {
    /// Load label records; the last record per id wins, so interrupted runs resume.
    public static func load(_ url: URL, maxRows: Int) throws -> [String: LabelRecord] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [:] }

        var records: [String: LabelRecord] = [:]

        for row in try JSONLines.read(url, maxLines: maxRows * 4, maxLineBytes: 16_384) {
            do {
                let record = try JSONDecoder().decode(LabelRecord.self, from: row.data)
                records[record.id] = record
            } catch {
                throw DistillError.invalidData("\(url.lastPathComponent) line \(row.line): \(describe(error))")
            }
        }

        return records
    }

    public static func append(_ record: LabelRecord, to url: URL) throws {
        let line = try Canonical.encoder().encode(record) + Data([10])

        if !FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }

        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }

        try handle.seekToEnd()
        try handle.write(contentsOf: line)
    }

    static func fingerprint(_ records: [String: LabelRecord]) -> String {
        Canonical.sha256(records.keys.sorted().map { "\($0)\t\(records[$0]!.contentHash)\t\(records[$0]!.label ?? "-")" }.joined(separator: "\n"))
    }
}
