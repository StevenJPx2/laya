import Foundation
import Tokenizers

struct PromptItem {
    let ids: [Int32]
    let markers: [Int32]
    let type: Int32
    let question: Question
    let options: [String]
}

private final class ExactByteLevelTokenizer {
    private let vocab: [String: Int32]
    private let ranks: [String: Int]
    private let addedTokens: [(String, Int32)]
    private let pattern = try! NSRegularExpression(pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)

    init(file: URL) throws {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
        let model = object["model"] as! [String: Any]
        vocab = (model["vocab"] as! [String: Int]).mapValues(Int32.init)
        let merges = model["merges"] as! [[String]]
        let added = object["added_tokens"] as! [[String: Any]]
        var table: [String: Int] = [:]
        for (index, merge) in merges.enumerated() { table[merge.joined(separator: " ")] = index }
        ranks = table
        addedTokens = added.compactMap { value in
            guard let content = value["content"] as? String, let id = value["id"] as? Int else { return nil }

            return (content, Int32(id))
        }.sorted { $0.0.count > $1.0.count }
    }

    func encode(_ text: String) -> [Int32] {
        let normalized = text.precomposedStringWithCanonicalMapping
        var result: [Int32] = []
        var plain = ""
        var index = normalized.startIndex

        func flush() {
            result.append(contentsOf: encodePlain(plain))
            plain.removeAll(keepingCapacity: true)
        }

        while index < normalized.endIndex {
            let suffix = normalized[index...]

            if let token = addedTokens.first(where: { suffix.hasPrefix($0.0) }) {
                flush()
                result.append(token.1)
                index = normalized.index(index, offsetBy: token.0.count)
            } else {
                plain.append(normalized[index])
                index = normalized.index(after: index)
            }
        }

        flush()

        return result
    }

    private func encodePlain(_ text: String) -> [Int32] {
        guard !text.isEmpty else { return [] }

        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        var result: [Int32] = []

        for match in pattern.matches(in: text, range: range) {
            let piece = String(text[Range(match.range, in: text)!])
            let symbols = piece.utf8.map(byteSymbol)
            result.append(contentsOf: bpe(symbols))
        }

        return result
    }

    private func bpe(_ symbols: [String]) -> [Int32] {
        var current = symbols
        while current.count > 1 {
            var best: (Int, Int)?
            for index in 0..<(current.count - 1) {
                guard let rank = ranks["\(current[index]) \(current[index + 1])"] else { continue }
                if best == nil || rank < best!.0 { best = (rank, index) }
            }
            guard let pair = best else { break }
            var merged: [String] = [], index = 0
            while index < current.count {
                if index + 1 < current.count, ranks["\(current[index]) \(current[index + 1])"] == pair.0 {
                    merged.append(current[index] + current[index + 1]); index += 2
                } else { merged.append(current[index]); index += 1 }
            }
            current = merged
        }
        return current.map { vocab[$0] ?? 50280 }
    }

    private func byteSymbol(_ byte: UInt8) -> String {
        var value = Int(byte)
        if !(33...126 ~= value || 161...172 ~= value || 174...255 ~= value) {
            value = 256 + byteMapOffset(value)
        }
        return String(UnicodeScalar(value)!)
    }

    private func byteMapOffset(_ byte: Int) -> Int {
        var offset = 0
        for value in 0..<byte { if !(33...126 ~= value || 161...172 ~= value || 174...255 ~= value) { offset += 1 } }
        return offset
    }
}

final class LayaTokenizer {
    private let tokenizer: ExactByteLevelTokenizer
    let clsID: Int32
    let sepID: Int32
    let padID: Int32
    let maskID: Int32
    let maskToken: String

    init(folder: URL) async throws {
        tokenizer = try ExactByteLevelTokenizer(file: folder.appendingPathComponent("tokenizer.json"))
        let config = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: folder.appendingPathComponent("tokenizer_config.json")))
        func token(_ name: String) throws -> (String, Int32) {
            guard let value = config[name]?.stringValue else { throw LayaError.invalid("tokenizer is missing \(name)") }
            switch name { case "cls_token": return (value, 50281); case "sep_token": return (value, 50282); case "pad_token": return (value, 50283); default: return (value, 50284) }
        }
        let cls = try token("cls_token"), sep = try token("sep_token"), pad = try token("pad_token"), mask = try token("mask_token")
        clsID = cls.1; sepID = sep.1; padID = pad.1; maskID = mask.1; maskToken = mask.0
    }

    func encode(_ text: String) -> [Int32] { tokenizer.encode(text) }

    func prepare(state: JSONValue, questions: [String: Question]) throws -> [String: PromptItem] {
        var result: [String: PromptItem] = [:]
        for (id, question) in questions {
            let options = try Prompt.renderedOptions(question)
            let instruction = instructionText(question.instructions).replacingOccurrences(of: maskToken, with: " ")
            var head = encode("\(question.type) question: \(instruction)")
            var optionIDs: [[Int32]] = options.map { [maskID] + encode(" " + $0.replacingOccurrences(of: maskToken, with: " ")).prefix(48) }
            var budget = 192 - optionIDs.reduce(0) { $0 + $1.count }
            if budget < 16 { let per = max(4, (192 - 16) / max(1, optionIDs.count)); optionIDs = optionIDs.map { Array($0.prefix(per)) }; budget = 192 - optionIDs.reduce(0) { $0 + $1.count } }
            head = Array(head.prefix(max(8, budget)))
            var ids: [Int32] = [clsID] + head + [sepID], markers: [Int32] = []
            for option in optionIDs { markers.append(Int32(ids.count)); ids.append(contentsOf: option) }
            ids.append(sepID)
            let stateIDs = encode(Prompt.serialize(state).replacingOccurrences(of: maskToken, with: " "))
            let room = max(0, 512 - ids.count - 1); ids.append(contentsOf: stateIDs.prefix(room)); ids.append(sepID)
            result[id] = PromptItem(ids: Array(ids.prefix(512)), markers: markers, type: typeID(question.type), question: question, options: options)
        }
        return result
    }
    private func instructionText(_ value: JSONValue) -> String { value.stringValue ?? Prompt.serialize(value) }
    private func typeID(_ type: String) -> Int32 { type == "choice" ? 0 : type == "score" ? 1 : 2 }
}
