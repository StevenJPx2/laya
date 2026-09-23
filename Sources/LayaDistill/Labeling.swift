import Foundation

public struct LabelingPlan: Codable, Sendable {
    public let teacher: String
    public let pending: Int
    public let selected: Int
    public let alreadyLabeled: Int
    public let priorRequests: Int
    public let priorSpendUsd: Double
    public let estimatedInputTokens: Int
    public let maxOutputTokens: Int
    public let worstCaseUsd: Double
    public let budgetUsd: Double
    public let maxRequests: Int
    public let truncatedByBudget: Bool

    enum CodingKeys: String, CodingKey {
        case teacher, pending, selected
        case alreadyLabeled = "already_labeled", priorRequests = "prior_requests", priorSpendUsd = "prior_spend_usd"
        case estimatedInputTokens = "estimated_input_tokens", maxOutputTokens = "max_output_tokens", worstCaseUsd = "worst_case_usd"
        case budgetUsd = "budget_usd", maxRequests = "max_requests", truncatedByBudget = "truncated_by_budget"
    }
}

public struct LabelingSummary: Codable, Sendable {
    public var plan: LabelingPlan
    public var dryRun: Bool
    public var requests = 0
    public var ok = 0
    public var invalid = 0
    public var refused = 0
    public var errors = 0
    public var inputTokens = 0
    public var outputTokens = 0
    public var spentUsd = 0.0
    public var stopReason: String?

    enum CodingKeys: String, CodingKey {
        case plan, requests, ok, invalid, refused, errors
        case dryRun = "dry_run", inputTokens = "input_tokens", outputTokens = "output_tokens", spentUsd = "spent_usd", stopReason = "stop_reason"
    }
}

/// Bounded teacher labeling. Nothing is sent unless `approve` is true, and the
/// spec's `budget` is enforced cumulatively across runs using the label file.
public struct Labeler: Sendable {
    public static let maxConsecutiveFailures = 5

    let spec: TaskSpec
    let examples: [DatasetExample]
    let existing: [String: LabelRecord]

    public init(spec: TaskSpec, examples: [DatasetExample], existing: [String: LabelRecord]) {
        self.spec = spec
        self.examples = examples
        self.existing = existing
    }

    public func plan(limit: Int? = nil) -> (plan: LabelingPlan, batch: [DatasetExample]) {
        let pending = examples.filter { example in
            guard let record = existing[example.id] else { return true }

            return record.status != .ok || record.contentHash != example.contentHash
        }
        let prior = existing.values.filter { $0.teacher == spec.teacher.identity }
        let priorSpend = prior.reduce(0) { $0 + $1.costUsd }
        let networked = spec.teacher.provider.isNetworked
        let requestRoom = networked ? max(0, spec.budget.maxRequests - prior.count) : pending.count
        var batch = Array(pending.prefix(min(limit ?? pending.count, requestRoom)))

        var estimated = 0
        var worst = 0.0
        var fitted: [DatasetExample] = []

        for example in batch {
            let tokens = TeacherPrompt.estimatedInputTokens(spec, example.input)
            let cost = networked ? worstCase(tokens) : 0

            if networked, priorSpend + worst + cost > spec.budget.maxUsd { break }

            estimated += tokens
            worst += cost
            fitted.append(example)
        }

        let truncated = fitted.count < batch.count
        batch = fitted

        let plan = LabelingPlan(
            teacher: spec.teacher.identity, pending: pending.count, selected: batch.count, alreadyLabeled: examples.count - pending.count,
            priorRequests: prior.count, priorSpendUsd: priorSpend, estimatedInputTokens: estimated, maxOutputTokens: spec.teacher.maxOutputTokens,
            worstCaseUsd: worst, budgetUsd: spec.budget.maxUsd, maxRequests: spec.budget.maxRequests, truncatedByBudget: truncated
        )

        return (plan, batch)
    }

    func worstCase(_ inputTokens: Int) -> Double {
        spec.teacher.pricing?.cost(input: inputTokens, output: spec.teacher.maxOutputTokens) ?? 0
    }

    public func run(limit: Int? = nil, approve: Bool, client: TeacherClient?,
                    sink: (LabelRecord) throws -> Void, log: (String) -> Void) async throws -> LabelingSummary {
        let (plan, batch) = plan(limit: limit)
        var summary = LabelingSummary(plan: plan, dryRun: false)

        if spec.teacher.provider == .dataset {
            for example in batch { try record(goldRecord(example), into: &summary, sink: sink) }
            return summary
        }

        guard approve else {
            summary.dryRun = true
            summary.stopReason = "dry run: pass --approve to send \(plan.selected) request(s), worst case $\(String(format: "%.4f", plan.worstCaseUsd))"
            return summary
        }
        guard let client else { throw DistillError.teacher("no teacher client configured") }

        try await label(batch, client: client, summary: &summary, sink: sink, log: log)

        return summary
    }

    private func label(_ batch: [DatasetExample], client: TeacherClient, summary: inout LabelingSummary,
                       sink: (LabelRecord) throws -> Void, log: (String) -> Void) async throws {
        var failures = 0

        for (index, example) in batch.enumerated() {
            let worst = worstCase(TeacherPrompt.estimatedInputTokens(spec, example.input))

            guard summary.plan.priorSpendUsd + summary.spentUsd + worst <= spec.budget.maxUsd else {
                summary.stopReason = "budget: next request could exceed max_usd"
                return
            }

            let result = try await client.label(example.input)
            let billed = result.inputTokens + result.outputTokens == 0 ? worst : spec.teacher.pricing!.cost(input: result.inputTokens, output: result.outputTokens)

            summary.requests += 1
            try record(LabelRecord(id: example.id, contentHash: example.contentHash, label: result.label, status: result.status, reason: result.reason,
                                   teacher: spec.teacher.identity, inputTokens: result.inputTokens, outputTokens: result.outputTokens, costUsd: billed),
                       into: &summary, sink: sink)
            log("[label] \(index + 1)/\(batch.count) \(String(example.contentHash.prefix(10))) \(result.status.rawValue)")

            failures = result.status == .ok ? 0 : failures + 1
            if failures >= Self.maxConsecutiveFailures {
                summary.stopReason = "stopped after \(failures) consecutive failures (last: \(result.reason ?? "unknown"))"
                return
            }
        }
    }

    private func goldRecord(_ example: DatasetExample) -> LabelRecord {
        LabelRecord(id: example.id, contentHash: example.contentHash, label: example.gold, status: example.gold == nil ? .invalid : .ok,
                    reason: example.gold == nil ? "row has no gold label" : nil, teacher: spec.teacher.identity, inputTokens: 0, outputTokens: 0, costUsd: 0)
    }

    private func record(_ record: LabelRecord, into summary: inout LabelingSummary, sink: (LabelRecord) throws -> Void) throws {
        try sink(record)

        summary.inputTokens += record.inputTokens
        summary.outputTokens += record.outputTokens
        summary.spentUsd += record.costUsd

        switch record.status {
        case .ok: summary.ok += 1
        case .invalid: summary.invalid += 1
        case .refused: summary.refused += 1
        case .error: summary.errors += 1
        }
    }
}
