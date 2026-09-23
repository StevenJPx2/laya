# laya

A compiled, Python-free runtime for the Laya typed-decision model on Apple
Silicon. It runs the ModernBERT-large encoder and Laya decision heads through
**Core ML**, exposed two ways:

- **`LayaCore`** — a Swift library you embed directly.
- **`laya-daemon`** — a long-running daemon that keeps the model warm and serves
  newline-delimited JSON over a Unix socket at
  `~/Library/Application Support/laya/laya.sock` (override with `LAYA_SOCKET`).
- **`laya-distill`** — define your own classification task, label it with a
  configurable teacher, train a task-specific head, evaluate it, and serve it
  from the same daemon ([Task-specific classifiers](#task-specific-classifiers)).

The prompt formatting, marker extraction, softmax/entropy/act features, and
temperature calibration match the Python MLX reference exactly. All **63
validation fixtures match the reference's selected answer**, with probabilities
within 0.02 (`Tests/LayaCoreTests/ParityTests.swift`).

Swift 6, macOS 15+.

## Layout

| Path | What |
|---|---|
| `Sources/LayaCore` | Library: tokenizer, prompt build, Core ML runtime, socket server, schema |
| `Sources/laya-daemon` | Warm daemon over the Unix socket |
| `Sources/laya` | CLI client: `predict`, `classify`, `health`, `bench` |
| `Sources/LayaDistill` | Task spec, teacher labeling, split, student training, evaluation, artifacts |
| `Sources/laya-distill` | CLI: `init`, `validate`, `label`, `train`, `eval`, `predict` |
| `tools/convert` | One-time Core ML export (its own `uv` project) |
| `tools/generate_golden.py` | Regenerate parity fixtures from the Python reference |
| `tools/benchmark.sh` | Latency + peak RSS/CPU over N predictions |
| `tools/install-daemon.sh` | Build, install, and manage the user-level launchd daemon |
| `launchd/com.laya.daemon.plist` | Per-user launch agent |

## Build

```sh
swift build -c release
swift test            # unit tests; the parity gate is skipped unless LAYA_MODEL/LAYA_ASSETS are set
```

## 1. Export the model (build-time, Python)

Conversion is isolated in `tools/convert` and never runs at runtime. It loads
the cached MLX `model.safetensors` into a faithful PyTorch mirror, checks it
against the MLX reference, then exports one multifunction `.mlpackage` with
fixed-shape `seq128`, `seq256`, and `seq512` functions and shared fp16 weights.
Fixed shapes are required for this graph to remain on the Neural Engine.

```sh
cd tools/convert
uv sync
CKPT=~/.cache/huggingface/hub/models--aac6fef--laya-mlx/snapshots/*/

# Needs the reference venv on PYTHONPATH for `mlx` + `laya_mlx` (used by --verify):
uv run python export.py --checkpoint $CKPT --verify --report \
    --output ../../build/laya.mlpackage --assets ../../build/assets
```

`--verify` reports the Torch-vs-MLX error (logits ~4e-6, action logits ~3e-3).
The exporter also writes the mmap-friendly token embedding table, action-head
weights, and runtime manifest. Copy the tokenizer and config next to them:

```sh
cd ../..
mkdir -p build/assets
cp -RL "$CKPT/tokenizer" build/assets/
cp "$CKPT/rl_agent_config.json" build/assets/
```

## 2. Run the daemon

The installer is the recommended path. It builds the native release binary,
copies the model and assets into `~/Library/Application Support/laya`, and
loads a per-user launchd agent. No sudo or Python runtime is needed:

```sh
tools/install-daemon.sh
tools/install-daemon.sh --status
```

When `build/laya.mlmodelc` is available and newer than the package, the
installer uses it directly so launchd does not repeat the first-run Core ML
compilation. Otherwise it installs the `.mlpackage` and waits for Core ML to
compile it; set `LAYA_WAIT_SECONDS` to change the warm-up limit.

The service is kept alive across OpenCode restarts and macOS login sessions.
To remove the installed service and payload:

```sh
tools/install-daemon.sh --uninstall
```

For development, the daemon can still be run directly:

```sh
swift run -c release laya-daemon build/laya.mlpackage build/assets
```

On first start an `.mlpackage` is compiled once to a cached `.mlmodelc` beside
it, so restarts are fast. The daemon keeps the three shared-weight functions
warm and serves one prediction at a time (lowest steady-state footprint).

### Socket protocol

Newline-delimited JSON, one request per line.

- **Predict:** `{"state": <text|object|array>, "questions": {<id>: <question>}}`
  → the same result schema as the Python reference.
- **Health:** `{"op":"health"}` → `{"status":"ok","model":...,"warm":true}`

```sh
printf '%s\n' '{"op":"health"}' | nc -U "$HOME/Library/Application Support/laya/laya.sock"
```

## 3. Use it

### CLI

```sh
laya health
laya predict build/request.json
laya bench   build/request.json 1000
```

### From Swift

```swift
import LayaCore

let runtime = try await LayaRuntime(
    modelURL: URL(fileURLWithPath: "build/laya.mlpackage"),
    assetsURL: URL(fileURLWithPath: "build/assets")
)

let request = PredictRequest(
    state: .string("I was billed twice, please refund the duplicate."),
    questions: ["department": Question(
        type: "choice",
        instructions: .string("Which team should handle this?"),
        criteria: .array([.string("billing"), .string("technical"), .string("sales")])
    )]
)

let result = try await runtime.predict(request)
print(result.answers["department"]!.choice!)   // "billing"
```

### From any language (over the socket)

Send one JSON line, read one JSON line back. Python, no dependencies:

```python
import json, socket

def predict(state, questions):
    path = f"{__import__('os').path.expanduser('~')}/Library/Application Support/laya/laya.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(path)
        s.sendall((json.dumps({"state": state, "questions": questions}) + "\n").encode())
        return json.loads(s.makefile().readline())

print(predict("refund me please", {
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"}
}))
```

The result schema matches the reference: each answer has `type`, `confidence`,
`action.act_probability`, and per-type fields (`choice` + `probabilities`,
`score` + `legend` + `probabilities`, or `noul`).

## Task-specific classifiers

`laya-distill` turns a task you define — input schema, labels, abstain policy,
examples — into a small **trained** classifier, evaluates it on held-out data,
and serves it from `laya-daemon`.

### What is actually trained

The student is a **softmax-regression head with learned, task-specific
weights**, trained natively in Swift (full-batch Adam, L2, optional balanced
class weights). No new dependencies, no Python at training or serving time. It
trains on one of two feature sets:

| `student.features` | Input to the head | Needs the Laya model |
|---|---|---|
| `laya` | Frozen Laya representation: the 1,024-d pooled decision vector plus Laya's raw logit for each of your labels, from one ANE forward pass | yes |
| `hashed` | Hashed word unigrams/bigrams (global and per field), log-TF, L2-normalized | no |

What is **not** trained: the 421M-parameter encoder and Laya's own heads stay
frozen. Neither this repo nor `laya-mlx` contains a training path for them
(`laya-mlx` states RLCD training and fine-tuning live upstream), so encoder
fine-tuning is out of scope. The **Laya zero-shot** row in every report is the
stock checkpoint answering your task as a generic `choice` question; it is an
untrained baseline, not a distilled model.

```text
task.json ──► label (teacher, bounded) ──► labels.jsonl
data.jsonl ─┘                                   │
                  split (dedup, groups, leakage) ▼
          features (Laya ANE or hashed) ──► train head ──► evaluate on holdout
                                                               │
                        <name>.classifier.json (weights + spec + provenance + report)
                                                               │
                         laya-distill predict  /  laya-daemon {"op":"classify"}
```

### Workflow

```sh
swift build -c release
B=.build/release/laya-distill

$B init tasks/permission --template permission     # task.json + 60 example rows
$B validate tasks/permission/task.json
$B label tasks/permission/task.json --data tasks/permission/data.jsonl --labels tasks/permission/labels.jsonl
$B train tasks/permission/task.json --data tasks/permission/data.jsonl --labels tasks/permission/labels.jsonl \
    --out tasks/permission/permission.classifier.json --report tasks/permission/report.md \
    --model build/laya.mlpackage --assets build/assets
$B eval tasks/permission/permission.classifier.json --data tasks/permission/data.jsonl --labels tasks/permission/labels.jsonl \
    --model build/laya.mlpackage --assets build/assets
$B predict tasks/permission/permission.classifier.json --model build/laya.mlpackage --assets build/assets \
    --input '{"tool":"Bash","request":"cat ~/.aws/credentials"}'
```

Serve every `*.classifier.json` in a directory from the warm daemon:

```sh
laya-daemon build/laya.mlpackage build/assets --classifiers tasks/permission   # or LAYA_CLASSIFIERS=<dir>
laya classify permission input.json
printf '%s\n' '{"op":"classify","classifier":"permission","input":{"tool":"Bash","request":"ls"}}' \
  | nc -U "$HOME/Library/Application Support/laya/laya.sock"
# {"abstained":false,"argmax":"…","classifier":"permission","confidence":…,"label":"…","probabilities":{"allow":…,"ask":…,"deny":…},"version":"0.1.0"}
```

`{"op":"classifiers"}` lists what is loaded. The daemon refuses to start if
any artifact fails validation, and existing `predict`/`health` requests are
unchanged.

### The task spec (`spec_version: 1`)

Strict JSON: every object rejects unknown fields, so a typo fails instead of
silently falling back to a default. The permission template, abridged:

```json
{
  "spec_version": 1, "name": "permission", "version": "0.1.0",
  "instructions": "An autonomous coding agent wants to run the tool call below…",
  "input": {"fields": [
    {"name": "tool", "type": "string", "max_chars": 64},
    {"name": "request", "type": "string", "max_chars": 2000},
    {"name": "context", "type": "string", "required": false, "max_chars": 2000}
  ]},
  "labels": [
    {"name": "allow", "description": "read-only or clearly scoped, reversible work inside the project workspace"},
    {"name": "deny", "description": "destructive, credential-exposing, privilege-escalating, or exfiltrating actions"},
    {"name": "ask", "description": "plausibly legitimate but consequential or ambiguous; needs confirmation"}
  ],
  "abstain": {"label": "ask", "min_confidence": 0.55},
  "examples": [{"input": {"tool": "Bash", "request": "git push origin main"}, "label": "ask"}],
  "teacher": {"provider": "dataset", "model": "gold"},
  "budget": {"max_requests": 0, "max_usd": 0},
  "dataset": {"max_examples": 5000, "holdout_fraction": 0.3, "split_seed": "permission-v1"},
  "student": {"features": "laya", "epochs": 300, "learning_rate": 0.05,
              "l2_grid": [0.001, 0.01, 0.1, 1, 3], "class_weighting": "balanced"}
}
```

- **Input schema** — `string`, `number`, `boolean`, or `json` fields with
  `required` and `max_chars`. Undeclared, missing, mistyped, or oversized
  fields are rejected at training, prediction, and daemon time alike.
- **Abstain** — when the student's top probability is below `min_confidence`
  it answers the abstain label (here `ask`); the teacher is told to use the
  same label when an input is ambiguous. Reports show both raw argmax and the
  served (abstain-applied) result.
- **Dataset rows** (`data.jsonl`) — `{"id", "input", "group"?, "gold"?}`.
  `group` keeps related rows (paraphrases, one conversation) on one side of the
  split. `gold` is an optional human label.

A routing task is the same shape — for example labels `billing` / `technical` /
`sales` over `{"subject", "body"}` fields, with `abstain` pointing at an
`other` or `human_review` label.

### Teachers, budgets, and keys

The teacher is configured per task; nothing is assumed. `teacher.model` must be
set explicitly (this build agent's own model is not a default).

| `provider` | Request | Notes |
|---|---|---|
| `anthropic` | Messages API with `output_config.format` JSON schema (label `enum`) | [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) |
| `openai-compatible` | Chat Completions with `response_format: json_schema, strict: true` at `base_url` | OpenAI, or local servers such as vLLM/Ollama; loopback may use `http` |
| `dataset` | none — uses each row's `gold` label | free, offline |

```json
"teacher": {"provider": "anthropic", "model": "<model id you choose>", "api_key_env": "ANTHROPIC_API_KEY",
            "max_output_tokens": 16, "pricing": {"input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0}},
"budget": {"max_requests": 500, "max_usd": 2.0}
```

- **Keys are environment-only.** The spec names the variable
  (`api_key_env`); an inline `api_key` is rejected as an unknown field. The key
  is read only when `--approve` is given.
- **Nothing is sent without `--approve`.** Without it `label` prints a plan:
  pending rows, selected rows, estimated input tokens, and a worst-case cost.
- **Budgets are cumulative.** `max_requests` and `max_usd` cover every run
  against the same label file, so resuming cannot exceed them. Before each
  request the worst case (estimated input + `max_output_tokens`) must fit;
  actual usage reported by the provider is billed, and the worst case is
  charged when usage is missing. Five consecutive failures stop the run;
  429/5xx/transport errors retry at most `max_retries` times with backoff.
- **You supply `pricing`.** Prices change; copy current per-MTok prices from
  your provider. Cost ≈ `requests × (input_tokens × input_price +
  output_tokens × output_price) / 1e6`. The input estimate is conservative
  (~3 bytes/token) and is used only to refuse or stop before spending. On the
  60-row template, a dry run with the illustrative prices above planned 25
  requests (the `max_requests` it was given) at a worst case of $0.0324.
- **Replies are validated.** Only a declared label is accepted; casing is
  normalized because Anthropic's docs note enum casing isn't guaranteed.
  Refusals, `max_tokens`, and malformed replies are recorded as
  `refused`/`invalid` and excluded from training.
- **No prompts or inputs are persisted or logged.** Label rows store id,
  content hash, label, status, token counts, and cost. Logs show only a hash
  prefix and status; provider error text is shortened and redacted (emails,
  long tokens, phone numbers).

### Split and leakage controls

- Inputs are hashed after rendering in schema order, lowercasing, and
  collapsing whitespace; duplicates collapse, and duplicates with conflicting
  teacher labels are dropped.
- Train/holdout assignment is a deterministic hash of `split_seed` and the
  row's `group` (or content hash), so a group never straddles the split.
- Rows identical to a few-shot example shown to the teacher never enter holdout.
- Labels whose content hash no longer matches the row are treated as stale.
- `l2_grid` selects L2 by group-aware 5-fold cross-validation on the **training
  split only**; the holdout is never used for model selection.
- Artifacts store the training rows' content hashes. `eval` re-derives the split
  and excludes, and reports, any holdout row that was trained on.

### Artifacts

`<name>.classifier.json` (`format: laya.classifier`, `format_version: 1`)
contains the normalized spec and its SHA-256, the feature descriptor, weights
and standardization statistics (fitted on train only), provenance (teacher,
dataset and label fingerprints, training hashes, chosen L2 and CV scores), the
holdout report, and an integrity hash verified on load. `laya` artifacts also
record an asset fingerprint (runtime manifest, action-head weights, first 4 MiB
of embeddings) and refuse to load against different Laya assets.

### Measured result: the permission template

These numbers come from running the commands above on this machine
(`laya` features, `dataset` teacher = the template's hand-written labels). They
show the pipeline works; **they are not evidence of a useful permission gate.**
The 60 rows are synthetic and the holdout has 19 rows, so one row moves
accuracy by 5.3 points.

| Holdout, n = 19 | Accuracy | Macro-F1 |
|---|---:|---:|
| Student, argmax | 36.8% | 0.363 |
| Student, served (abstain rate 10.5%) | 47.4% | 0.469 |
| Laya zero-shot choice (untrained) | 47.4% | 0.287 |
| Majority class (`allow`) | 15.8% | 0.091 |

Cross-validation on the 41 training rows chose `l2 = 0.01` at 43.9% accuracy,
consistent with the holdout. With a fixed `l2 = 0.001` and no selection, the
same data gave 31.6% on holdout — below zero-shot — which is why the template
uses `l2_grid`. Hashed features on the same split scored 52.6% served, but only
by abstaining to `ask` on 52.6% of rows. All of these differences are within
noise at this size. In one daemon check the model answered `ask` for
`sudo rm -rf / --no-preserve-root`, which should be `deny`.

To get a usable classifier, label hundreds to thousands of representative rows
with a strong teacher (or humans), keep `gold` labels on a subset to measure the
teacher itself (`teacher_vs_gold`), and ship only if the student beats both
baselines on held-out data by more than the noise.

### Limitations

- Only a linear head is trained; the encoder is frozen. Laya's 512-token context
  applies to `laya` features.
- Hard teacher labels only; no soft-label or logit distillation.
- Training is full-batch and in memory. It has been exercised only at the test
  and template sizes here (≤ 60 rows); larger datasets are untested. Laya
  features cost one ANE forward pass per row (about 20 ms at the 128-token
  bucket; see Benchmarks).
- Rows are bounded by `dataset.max_examples` and `max_line_bytes`, and teacher
  output by `max_output_tokens`.
- The Laya asset fingerprint is a cheap compatibility check, not a hash of every
  weight.

### Tests

`swift test` runs the distillation suite offline: strict spec parsing, input
validation, bounded loading, the split and leakage controls, Anthropic and
OpenAI-compatible request/response shapes, dry-run and cumulative budget
enforcement, bounded retries, redaction, and an end-to-end tiny classifier
(teacher labels through a stub transport, training, artifact save/load,
prediction, re-evaluation, daemon `classify`, tamper rejection). No test makes a
network call. `LayaFeatureTests` trains on real Laya representations and runs
only when `LAYA_MODEL` and `LAYA_ASSETS` are set.

## Parity gate

The 63 golden fixtures live in `Tests/LayaCoreTests/Fixtures/validation.json`.
The gate verifies selected answers, class probabilities, and action
probabilities. Regenerate them from the Python reference and run it against the
exported model:

```sh
uv run --project ~/Documents/Projects/laya-mlx --extra reference \
    python tools/generate_golden.py

LAYA_MODEL=build/laya.mlpackage LAYA_ASSETS=build/assets swift test
```

Set `LAYA_CASE=<name>` (e.g. `email`) to run a single fixture — lighter on the
machine while iterating.

## Benchmarks

End-to-end over the socket, default compute (`.cpuAndNeuralEngine`), warm model,
on this machine. Model loading is excluded; `laya bench` times connect → send →
inference → reply.

```sh
tools/benchmark.sh build/laya.mlpackage build/assets build/request-one.json 100
```

| Workload | P50 | P95 | Peak RSS | Python MLX baseline |
|---|---:|---:|---:|---|
| One short choice question | **22.9 ms** | **23.3 ms** | **98.5 MiB** | ~13 ms P50, ~944 MiB |

Two notes on those numbers:

- **RAM is much lower than MLX.** Core ML and the runtime memory-map their fp16
  assets, so process RSS stays under 100 MiB in this test versus MLX's ~944 MiB
  active allocation.
- **Latency is close to MLX while using the ANE.** The 22.9 ms result includes
  tokenization, host-side preprocessing, socket I/O, and response encoding.

## Neural Engine placement

The model executes fully on the Apple Neural Engine. `MLComputePlan`, Core ML's
authoritative placement API, reports **1,271 / 1,271 placed operations on ANE**
for `seq128`, with no CPU fallback. `export.py --report` times each fixed-shape
function (this machine):

```
             CPU_AND_NE     CPU_ONLY       ALL
seq128       19.7 ms        75.2 ms        19.6 ms
seq256       36.6 ms       128.7 ms        36.7 ms
seq512      107.2 ms       253.6 ms       110.4 ms
```

The export keeps unsupported and shape-hostile work outside the Core ML graph:
Swift performs token embedding lookup from an mmap'd fp16 table, builds RoPE and
additive attention biases, supplies type/marker one-hot tensors, and evaluates
the small entropy/top-2 action head. The Core ML graph contains the encoder and
decision layers as ANE-supported arithmetic.

An enumerated-shape version of this graph still placed every operation on CPU.
The exporter therefore creates three fixed-shape functions and merges them into
one macOS 15 multifunction package. Core ML deduplicates their constants, so the
package is **706 MiB**, smaller than the prior 804 MiB package rather than three
times larger. The runtime selects the matching function for each request.

Override the compute path for experiments:

```sh
LAYA_COMPUTE=all|ane|gpu|cpu   # default: ane
```

The complete 63-question parity gate passes on this ANE runtime with probability
error at or below 0.02.

## Attribution

Laya and its pretrained weights are by Convai Innovations and upstream
contributors. Prompt construction, calibration, and output schema follow the
`laya-mlx` reference port. This is an independent Core ML runtime, not an
official release.
