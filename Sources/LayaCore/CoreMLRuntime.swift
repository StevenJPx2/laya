import CoreML
import Foundation

/// Warm Core ML runtime. Actor isolation serializes predictions and allows the
/// reusable input buffers to be mutated without locks.
public actor LayaRuntime {
    private let models: [Int: MLModel]
    private let tokenizer: LayaTokenizer
    private let modelName: String
    private let temperatures: [Double]
    private let optionTemperatures: [String: Double]
    private let manifest: RuntimeManifest
    private let embeddings: Data
    private let actionHead: ActionHead
    private var buffers: [Int: ModelBuffers] = [:]

    public init(modelURL: URL, assetsURL: URL, modelName: String = "laya-typed-decisions") async throws {
        let manifest = try Self.decode(RuntimeManifest.self, at: assetsURL.appendingPathComponent("runtime_manifest.json"))
        let config = try Self.decode(AgentConfig.self, at: assetsURL.appendingPathComponent("rl_agent_config.json"))
        let compiledURL = try await Self.compiledModel(at: modelURL)

        self.manifest = manifest
        self.models = try await Self.loadModels(at: compiledURL, lengths: manifest.lengths)
        self.tokenizer = try await LayaTokenizer(folder: assetsURL.appendingPathComponent("tokenizer"))
        self.embeddings = try Data(contentsOf: assetsURL.appendingPathComponent("embeddings.f16.bin"), options: .mappedIfSafe)
        self.actionHead = try ActionHead(
            url: assetsURL.appendingPathComponent("act_head.f32.bin"),
            inputSize: manifest.hidden + 4,
            outputSize: config.actCosts.count + 1
        )
        self.modelName = modelName
        self.temperatures = config.temperature.map(clamp)
        self.optionTemperatures = config.temperatureByOptions.mapValues(clamp)

        try validateAssets()
    }

    public func health() -> HealthResponse {
        HealthResponse(status: "ok", model: modelName, warm: true)
    }

    public func predict(_ request: PredictRequest) throws -> PredictResponse {
        guard !request.questions.isEmpty else {
            return PredictResponse(model: modelName, answers: [:], usage: Usage(input_tokens: 0, output_tokens: 0))
        }

        let prepared = try tokenizer.prepare(state: request.state, questions: request.questions)
        var answers: [String: Answer] = [:]
        var tokenCount = 0

        for (id, item) in prepared {
            let output = try infer(item)
            let calibration = temperature(type: item.type, count: item.markers.count)
            let probabilities = softmax(Array(output.logits.prefix(item.markers.count)), temperature: calibration)

            answers[id] = makeAnswer(item, probabilities: probabilities, action: actionProbability(output.action))
            tokenCount += item.ids.count
        }

        return PredictResponse(model: modelName, answers: answers, usage: Usage(input_tokens: tokenCount, output_tokens: 0))
    }

    /// Frozen encoder representation for one typed question: the pooled decision
    /// vector the checkpoint's own heads consume, plus the uncalibrated option
    /// logits. Task-specific heads train on these without touching checkpoint weights.
    public func representation(state: JSONValue, question: Question) throws -> Representation {
        guard let item = try tokenizer.prepare(state: state, questions: ["q": question])["q"] else {
            throw LayaError.invalid("question could not be prepared")
        }

        let output = try forward(item)

        return Representation(pooled: output.cls, logits: Array(output.logits.prefix(item.markers.count)), options: item.options)
    }

    private func infer(_ item: PromptItem) throws -> (logits: [Double], action: [Double]) {
        let output = try forward(item)
        let action = actionHead.predict(cls: output.cls, logits: output.logits, markerCount: item.markers.count)

        return (output.logits, action)
    }

    private func forward(_ item: PromptItem) throws -> (logits: [Double], cls: [Double]) {
        let target = bucketLength(item.ids.count)

        guard let model = models[target] else {
            throw LayaError.modelUnavailable("Core ML function seq\(target) is unavailable")
        }

        let input = try inputProvider(item: item, target: target)
        let output = try model.prediction(from: input)

        guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue,
              let clsArray = output.featureValue(for: "cls")?.multiArrayValue else {
            throw LayaError.modelUnavailable("Core ML outputs logits/cls are missing")
        }

        let logits = (0..<manifest.markerSlots).map { logitsArray[$0].doubleValue }
        let cls = (0..<manifest.hidden).map { clsArray[$0].doubleValue }

        return (logits, cls)
    }

    private func inputProvider(item: PromptItem, target: Int) throws -> MLFeatureProvider {
        let modelBuffers: ModelBuffers

        if let existing = buffers[target] {
            modelBuffers = existing
        } else {
            modelBuffers = try ModelBuffers(length: target, manifest: manifest)
            buffers[target] = modelBuffers
        }

        try modelBuffers.fill(item: item, padID: tokenizer.padID, embeddings: embeddings, manifest: manifest)

        return try MLDictionaryFeatureProvider(dictionary: [
            "embeds": MLFeatureValue(multiArray: modelBuffers.embeds),
            "global_bias": MLFeatureValue(multiArray: modelBuffers.globalBias),
            "sliding_bias": MLFeatureValue(multiArray: modelBuffers.slidingBias),
            "rope": MLFeatureValue(multiArray: modelBuffers.rope),
            "type_onehot": MLFeatureValue(multiArray: modelBuffers.typeOnehot),
            "marker_onehot": MLFeatureValue(multiArray: modelBuffers.markerOnehot),
            "marker_mask": MLFeatureValue(multiArray: modelBuffers.markerMask),
        ])
    }

    private func validateAssets() throws {
        let expectedEmbeddingBytes = manifest.vocabSize * manifest.hidden * MemoryLayout<UInt16>.size

        guard embeddings.count == expectedEmbeddingBytes else {
            throw LayaError.modelUnavailable("embedding asset has \(embeddings.count) bytes; expected \(expectedEmbeddingBytes)")
        }
        guard manifest.lengths == [128, 256, 512], manifest.markerSlots == 20 else {
            throw LayaError.modelUnavailable("unsupported runtime manifest")
        }
    }

    /// Compile an `.mlpackage` once and cache the `.mlmodelc` beside it.
    private static func compiledModel(at modelURL: URL) async throws -> URL {
        if modelURL.pathExtension == "mlmodelc" { return modelURL }

        let cacheURL = modelURL.deletingPathExtension().appendingPathExtension("mlmodelc")
        let files = FileManager.default

        if let cached = try? cacheURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate,
           let source = try? modelURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate,
           cached >= source {
            return cacheURL
        }

        let compiled = try await MLModel.compileModel(at: modelURL)
        try? files.removeItem(at: cacheURL)
        try files.moveItem(at: compiled, to: cacheURL)

        return cacheURL
    }

    private static func loadModels(at url: URL, lengths: [Int]) async throws -> [Int: MLModel] {
        var result: [Int: MLModel] = [:]

        for length in lengths {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits()
            configuration.functionName = "seq\(length)"
            result[length] = try await MLModel.load(contentsOf: url, configuration: configuration)
        }

        return result
    }

    /// The fixed-shape graph is fully ANE-resident. Override for diagnostics
    /// with `LAYA_COMPUTE=all|ane|gpu|cpu`.
    private static func computeUnits() -> MLComputeUnits {
        switch ProcessInfo.processInfo.environment["LAYA_COMPUTE"]?.lowercased() {
        case "all": return .all
        case "gpu": return .cpuAndGPU
        case "cpu": return .cpuOnly
        default: return .cpuAndNeuralEngine
        }
    }

    private static func decode<T: Decodable>(_ type: T.Type, at url: URL) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }

    private func actionProbability(_ logits: [Double]) -> Double {
        let exps = logits.map { exp($0 - logits.max()!) }

        return exps[0] / exps.reduce(0, +)
    }

    private func makeAnswer(_ item: PromptItem, probabilities: [Double], action: Double) -> Answer {
        let confidence = round4(Prompt.confidence(probabilities))
        var base = Answer(type: item.question.type, confidence: confidence, action: Action(act_probability: round4(action)))

        switch item.question.type {
        case "choice":
            let labels = choiceLabels(item.question)
            base.choice = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }).map { labels[$0] }
            base.probabilities = Dictionary(uniqueKeysWithValues: zip(labels, probabilities.map(round4)))

        case "score":
            base.score = round4(probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element })
            base.legend = item.question.criteria?.arrayValue?.enumerated().reduce(into: [:]) { $0["\($1.offset)"] = $1.element }
            base.probabilities = Dictionary(uniqueKeysWithValues: probabilities.indices.map { ("\($0)", round4(probabilities[$0])) })

        default:
            base.noul = round4(probabilities[1])
            base.confidence = round4(max(probabilities[1], 1 - probabilities[1]))
        }

        return base
    }

    private func choiceLabels(_ question: Question) -> [String] {
        if let labels = question.criteria?.arrayValue { return labels.compactMap(\.stringValue) }

        return Prompt.ordered(question.criteria?.objectValue ?? []).map { $0.0 }
    }

    private func temperature(type: Int32, count: Int) -> Double {
        optionTemperatures["\(typeName(type)):\(bucket(count))"] ?? temperatures[Int(type)]
    }

    private func softmax(_ values: [Double], temperature: Double) -> [Double] {
        let scaled = values.map { $0 / temperature }
        let exps = scaled.map { exp($0 - scaled.max()!) }

        return exps.map { $0 / exps.reduce(0, +) }
    }

    private func bucketLength(_ length: Int) -> Int { length <= 128 ? 128 : length <= 256 ? 256 : 512 }
    private func bucket(_ count: Int) -> String { count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+" }
    private func typeName(_ type: Int32) -> String { ["choice", "score", "noul"][Int(type)] }
    private func round4(_ value: Double) -> Double { (value * 10000).rounded() / 10000 }
}

private final class ModelBuffers {
    let length: Int
    let embeds: MLMultiArray
    let globalBias: MLMultiArray
    let slidingBias: MLMultiArray
    let rope: MLMultiArray
    let typeOnehot: MLMultiArray
    let markerOnehot: MLMultiArray
    let markerMask: MLMultiArray

    init(length: Int, manifest: RuntimeManifest) throws {
        self.length = length
        embeds = try Self.array([1, length, manifest.hidden])
        globalBias = try Self.array([1, 1, 1, length])
        slidingBias = try Self.array([1, 1, length, length])
        rope = try Self.array([1, 2, length, manifest.headDim])
        typeOnehot = try Self.array([1, 3])
        markerOnehot = try Self.array([1, manifest.markerSlots, length])
        markerMask = try Self.array([1, manifest.markerSlots])

        fillRope(manifest: manifest)
    }

    func fill(item: PromptItem, padID: Int32, embeddings: Data, manifest: RuntimeManifest) throws {
        guard item.ids.count <= length, item.markers.count <= manifest.markerSlots else {
            throw LayaError.invalid("input exceeds exported model limits")
        }
        guard item.type >= 0, item.type < 3 else {
            throw LayaError.invalid("unknown question type \(item.type)")
        }

        fillEmbeddings(ids: item.ids, padID: padID, data: embeddings, hidden: manifest.hidden)
        fillAttentionBiases(validLength: item.ids.count, window: manifest.localWindow, maskBias: manifest.maskBias)
        fillOneHot(type: Int(item.type), markers: item.markers, slots: manifest.markerSlots)
    }

    private func fillEmbeddings(ids: [Int32], padID: Int32, data: Data, hidden: Int) {
        let destination = embeds.dataPointer.bindMemory(to: Float16.self, capacity: length * hidden)

        data.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: UInt16.self)

            for position in 0..<length {
                let token = Int(position < ids.count ? ids[position] : padID)
                let sourceOffset = token * hidden
                let destinationOffset = position * hidden

                for feature in 0..<hidden {
                    destination[destinationOffset + feature] = Float16(bitPattern: source[sourceOffset + feature])
                }
            }
        }
    }

    private func fillAttentionBiases(validLength: Int, window: Int, maskBias: Float) {
        let global = globalBias.dataPointer.bindMemory(to: Float16.self, capacity: length)
        let sliding = slidingBias.dataPointer.bindMemory(to: Float16.self, capacity: length * length)
        let blocked = Float16(maskBias)
        let radius = window / 2

        for key in 0..<length {
            global[key] = key < validLength ? 0 : blocked
        }

        for query in 0..<length {
            let paddedQuery = query >= validLength
            let row = query * length

            for key in 0..<length {
                let visible = key < validLength && (paddedQuery || abs(query - key) <= radius)
                sliding[row + key] = visible ? 0 : blocked
            }
        }
    }

    private func fillOneHot(type: Int, markers: [Int32], slots: Int) {
        let typePointer = typeOnehot.dataPointer.bindMemory(to: Float16.self, capacity: 3)
        let markerPointer = markerOnehot.dataPointer.bindMemory(to: Float16.self, capacity: slots * length)
        let maskPointer = markerMask.dataPointer.bindMemory(to: Float16.self, capacity: slots)

        typePointer.initialize(repeating: 0, count: 3)
        markerPointer.initialize(repeating: 0, count: slots * length)
        maskPointer.initialize(repeating: 0, count: slots)
        typePointer[type] = 1

        for (slot, position) in markers.enumerated() {
            markerPointer[slot * length + Int(position)] = 1
            maskPointer[slot] = 1
        }
    }

    private func fillRope(manifest: RuntimeManifest) {
        let pointer = rope.dataPointer.bindMemory(to: Float16.self, capacity: 2 * length * manifest.headDim)
        let half = manifest.headDim / 2
        let bases = [manifest.ropeThetaGlobal, manifest.ropeThetaLocal]

        for (table, base) in bases.enumerated() {
            for position in 0..<length {
                let offset = (table * length + position) * manifest.headDim

                for index in 0..<half {
                    let angle = Double(position) / pow(base, Double(index) / Double(half))
                    pointer[offset + index] = Float16(cos(angle))
                    pointer[offset + half + index] = Float16(sin(angle))
                }
            }
        }
    }

    private static func array(_ shape: [Int]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    }
}

private struct ActionHead {
    private static let hiddenSize = 256

    let inputSize: Int
    let outputSize: Int
    let firstWeights: [Double]
    let firstBias: [Double]
    let secondWeights: [Double]
    let secondBias: [Double]

    init(url: URL, inputSize: Int, outputSize: Int) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let count = Self.hiddenSize * inputSize + Self.hiddenSize + outputSize * Self.hiddenSize + outputSize

        guard data.count == count * MemoryLayout<Float>.size else {
            throw LayaError.modelUnavailable("action-head asset has an invalid size")
        }

        let values = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self)).map(Double.init)
        }
        var offset = 0

        firstWeights = Array(values[offset..<(offset + Self.hiddenSize * inputSize)])
        offset += Self.hiddenSize * inputSize
        firstBias = Array(values[offset..<(offset + Self.hiddenSize)])
        offset += Self.hiddenSize
        secondWeights = Array(values[offset..<(offset + outputSize * Self.hiddenSize)])
        offset += outputSize * Self.hiddenSize
        secondBias = Array(values[offset..<(offset + outputSize)])
        self.inputSize = inputSize
        self.outputSize = outputSize
    }

    func predict(cls: [Double], logits: [Double], markerCount: Int) -> [Double] {
        let probabilities = Self.softmax(logits)
        let sorted = probabilities.sorted()
        let count = max(2, markerCount)
        let entropy = -probabilities.reduce(0) { $0 + $1 * log(max($1, 1e-9)) } / log(Double(count))
        let features = [sorted.last!, sorted.last! - sorted[sorted.count - 2], entropy, Double(count) / 255]
        let input = cls + features

        var hidden = firstBias

        for row in 0..<Self.hiddenSize {
            let offset = row * inputSize
            var value = hidden[row]

            for column in 0..<inputSize {
                value += input[column] * firstWeights[offset + column]
            }

            hidden[row] = 0.5 * value * (1 + erf(value / sqrt(2)))
        }

        var output = secondBias

        for row in 0..<outputSize {
            let offset = row * Self.hiddenSize

            for column in 0..<Self.hiddenSize {
                output[row] += hidden[column] * secondWeights[offset + column]
            }
        }

        return output
    }

    private static func softmax(_ values: [Double]) -> [Double] {
        let exps = values.map { exp($0 - values.max()!) }
        let sum = exps.reduce(0, +)

        return exps.map { $0 / sum }
    }
}

private struct RuntimeManifest: Decodable {
    let hidden: Int
    let headDim: Int
    let vocabSize: Int
    let localWindow: Int
    let ropeThetaGlobal: Double
    let ropeThetaLocal: Double
    let markerSlots: Int
    let maskBias: Float
    let lengths: [Int]

    enum CodingKeys: String, CodingKey {
        case hidden
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case localWindow = "local_window"
        case ropeThetaGlobal = "rope_theta_global"
        case ropeThetaLocal = "rope_theta_local"
        case markerSlots = "marker_slots"
        case maskBias = "mask_bias"
        case lengths
    }
}

private struct AgentConfig: Decodable {
    let temperature: [Double]
    let temperatureByOptions: [String: Double]
    let actCosts: [String: Double]

    enum CodingKeys: String, CodingKey {
        case temperature
        case temperatureByOptions = "temperature_by_options"
        case actCosts = "act_costs"
    }
}

private func clamp(_ value: Double) -> Double { min(5, max(0.5, value)) }

private extension JSONValue {
    var arrayValue: [JSONValue]? { if case .array(let value) = self { return value }; return nil }
    var objectValue: [(String, JSONValue)]? { if case .object(let value) = self { return value }; return nil }
}
