import Foundation

public struct SelectionResult: Codable, Sendable {
    public let l2: Double
    public let folds: Int
    /// Mean out-of-fold accuracy per candidate, in `student.l2_grid` order.
    public let candidates: [Candidate]

    public struct Candidate: Codable, Sendable {
        public let l2: Double
        public let accuracy: Double
    }
}

enum ModelSelection {
    static let maxFolds = 5

    /// Choose L2 by k-fold cross-validation over the training split. Folds are
    /// assigned by the same group/content key as the train/holdout split, so a
    /// group never straddles folds. The holdout is never touched.
    static func selectL2(_ items: [LabeledExample], rows: [SparseVector], spec: TaskSpec, dense: Bool, dimensions: Int) throws -> SelectionResult? {
        guard let grid = spec.student.l2Grid else { return nil }

        let folds = min(maxFolds, Set(items.map(key)).count)
        guard folds >= 2 else { throw DistillError.training("l2_grid needs at least 2 distinct training groups") }

        let assignment = items.map { min(folds - 1, Int(Canonical.unitInterval("cv\u{0}\(spec.dataset.splitSeed)\u{0}\(key($0))") * Double(folds))) }
        var candidates: [SelectionResult.Candidate] = []

        for l2 in grid {
            var correct = 0
            var total = 0

            for fold in 0..<folds {
                let train = items.indices.filter { assignment[$0] != fold }
                let test = items.indices.filter { assignment[$0] == fold }
                guard !test.isEmpty, Set(train.map { items[$0].label }).count > 1 else { continue }

                let result = try Trainer.train(train.map { rows[$0] }, labels: train.map { items[$0].label }, classes: spec.labels.count,
                                               dimensions: dimensions, dense: dense, config: spec.student, l2: l2)
                correct += test.filter { argmax(result.model.probabilities(rows[$0])) == items[$0].label }.count
                total += test.count
            }

            candidates.append(.init(l2: l2, accuracy: total == 0 ? 0 : Double(correct) / Double(total)))
        }

        // Highest CV accuracy; ties go to the stronger regularizer.
        let best = candidates.max { ($0.accuracy, $0.l2) < ($1.accuracy, $1.l2) }!

        return SelectionResult(l2: best.l2, folds: folds, candidates: candidates)
    }

    private static func key(_ item: LabeledExample) -> String {
        item.example.group.map { "group:\($0)" } ?? "content:\(item.example.contentHash)"
    }
}
