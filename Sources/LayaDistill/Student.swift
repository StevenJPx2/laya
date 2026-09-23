import Foundation

/// Softmax regression head. Weights are row-major `[classes x dimensions]`.
public struct LinearModel: Codable, Sendable {
    public let classes: Int
    public let dimensions: Int
    public let weights: [Double]
    public let bias: [Double]

    public func probabilities(_ x: SparseVector) -> [Double] {
        var logits = bias

        for (position, index) in x.indices.enumerated() {
            for c in 0..<classes { logits[c] += weights[c * dimensions + index] * x.values[position] }
        }

        return softmax(logits)
    }

    func validate() throws {
        guard classes >= 2, dimensions >= 1, weights.count == classes * dimensions, bias.count == classes else {
            throw DistillError.artifact("model shape does not match its weights")
        }
        guard weights.allSatisfy(\.isFinite), bias.allSatisfy(\.isFinite) else { throw DistillError.artifact("model weights are not finite") }
    }
}

func softmax(_ logits: [Double]) -> [Double] {
    let top = logits.max() ?? 0
    let exps = logits.map { exp($0 - top) }
    let sum = exps.reduce(0, +)

    return exps.map { $0 / sum }
}

func argmax(_ values: [Double]) -> Int {
    values.indices.max { values[$0] < values[$1] } ?? 0
}

public struct TrainingResult: Sendable {
    public let model: LinearModel
    public let finalLoss: Double
    public let trainAccuracy: Double
}

public enum Trainer {
    public static func train(_ rows: [SparseVector], labels: [Int], classes: Int, dimensions: Int, config: StudentSpec, l2: Double) throws -> TrainingResult {
        guard !rows.isEmpty, rows.count == labels.count else { throw DistillError.training("no training rows") }

        let weights = classWeights(labels, classes: classes, mode: config.classWeighting)
        var optimizer = Adam(count: classes * dimensions + classes, rate: config.learningRate)
        var parameters = [Double](repeating: 0, count: classes * dimensions + classes)
        var loss = 0.0

        for _ in 0..<config.epochs {
            let step = gradient(rows, labels: labels, weights: weights, parameters: parameters, classes: classes, dimensions: dimensions, l2: l2)
            loss = step.loss
            optimizer.update(&parameters, gradient: step.gradient)
        }

        guard loss.isFinite else { throw DistillError.training("training diverged; lower student.learning_rate") }

        let model = LinearModel(classes: classes, dimensions: dimensions, weights: Array(parameters[0..<(classes * dimensions)]),
                                bias: Array(parameters[(classes * dimensions)...]))
        let correct = zip(rows, labels).filter { row, label in argmax(model.probabilities(row)) == label }.count

        return TrainingResult(model: model, finalLoss: loss, trainAccuracy: Double(correct) / Double(rows.count))
    }

    /// Weighted mean cross-entropy plus L2 on weights (not biases), and its gradient.
    private static func gradient(_ rows: [SparseVector], labels: [Int], weights: [Double], parameters: [Double],
                                 classes: Int, dimensions: Int, l2: Double) -> (loss: Double, gradient: [Double]) {
        let biasOffset = classes * dimensions
        let total = labels.reduce(0.0) { $0 + weights[$1] }
        var grad = [Double](repeating: 0, count: parameters.count)
        var loss = 0.0

        for (row, label) in zip(rows, labels) {
            var logits = Array(parameters[biasOffset...])
            for (position, index) in row.indices.enumerated() {
                for c in 0..<classes { logits[c] += parameters[c * dimensions + index] * row.values[position] }
            }

            let p = softmax(logits)
            let w = weights[label] / total
            loss -= w * log(max(p[label], 1e-12))

            for c in 0..<classes {
                let g = w * (p[c] - (c == label ? 1 : 0))
                grad[biasOffset + c] += g
                for (position, index) in row.indices.enumerated() { grad[c * dimensions + index] += g * row.values[position] }
            }
        }

        for i in 0..<biasOffset {
            loss += 0.5 * l2 * parameters[i] * parameters[i]
            grad[i] += l2 * parameters[i]
        }

        return (loss, grad)
    }

    /// Balanced weights; classes without training rows get weight 0.
    private static func classWeights(_ labels: [Int], classes: Int, mode: ClassWeighting) -> [Double] {
        guard mode == .balanced else { return [Double](repeating: 1, count: classes) }

        let counts = (0..<classes).map { c in labels.filter { $0 == c }.count }
        let present = Double(counts.filter { $0 > 0 }.count)

        return counts.map { $0 == 0 ? 0 : Double(labels.count) / (present * Double($0)) }
    }
}

struct Adam {
    private var m: [Double]
    private var v: [Double]
    private var t = 0
    private let rate: Double

    init(count: Int, rate: Double) {
        m = [Double](repeating: 0, count: count)
        v = [Double](repeating: 0, count: count)
        self.rate = rate
    }

    mutating func update(_ parameters: inout [Double], gradient: [Double]) {
        t += 1
        let correction1 = 1 - pow(0.9, Double(t))
        let correction2 = 1 - pow(0.999, Double(t))

        for i in parameters.indices {
            m[i] = 0.9 * m[i] + 0.1 * gradient[i]
            v[i] = 0.999 * v[i] + 0.001 * gradient[i] * gradient[i]
            parameters[i] -= rate * (m[i] / correction1) / (sqrt(v[i] / correction2) + 1e-8)
        }
    }
}
