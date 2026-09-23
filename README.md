# laya

A compiled, Python-free runtime for the Laya typed-decision model on Apple
Silicon. It runs the ModernBERT-large encoder and Laya decision heads through
**Core ML**, exposed two ways:

- **`LayaCore`** — a Swift library you embed directly.
- **`laya-daemon`** — a long-running daemon that keeps the model warm and serves
  newline-delimited JSON over a Unix socket at
  `~/Library/Application Support/laya/laya.sock` (override with `LAYA_SOCKET`).
- **`laya-distill`** — define a classification task, have Laya label it as the
  teacher, train a small student that runs without Laya, evaluate it against
  Laya and human gold labels, and serve it from the same daemon
  ([Task-specific classifiers](#task-specific-classifiers)).

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
| `Sources/LayaDistill` | Task spec, Laya teacher and confidence gate, split, student training, evaluation, artifacts |
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

`laya-distill` uses **Laya as the teacher** for small, task-specific students:

```text
unlabeled rows ──► ask Laya a task question (choice / noul / score) ──► confidence gate ──► labels.jsonl
                     (installed daemon or local runtime)                 uncertain → abstain label or drop
labels.jsonl ──► train a hashed n-gram softmax student ──► evaluate vs Laya (holdout) and vs human gold
                                                        ──► <name>.classifier.json  (runs without Laya)
```

The student is a softmax-regression head with learned weights over hashed word
unigrams and bigrams. It is trained natively in Swift with no new
dependencies. It does **not** use Laya embeddings, and serving it loads no Laya
model: `laya-distill predict` and the daemon's `classify` op read only the
artifact. Laya is needed only by `label`.

Nothing here fine-tunes Laya itself; the checkpoint and its heads are unchanged.

### Workflow

```sh
swift build -c release
B=.build/release/laya-distill

$B init tasks/routing --template routing            # task.json + 60 unlabeled pool rows + 48 gold rows
$B validate tasks/routing/task.json

# Label with the installed daemon (default), or in-process with --teacher runtime.
$B label tasks/routing/task.json --data tasks/routing/data.jsonl --labels tasks/routing/labels.jsonl
$B label tasks/routing/task.json --data tasks/routing/data.jsonl --labels tasks/routing/labels.jsonl \
    --teacher runtime --model build/laya.mlpackage --assets build/assets

# Everything below runs without Laya.
$B train tasks/routing/task.json --data tasks/routing/data.jsonl --labels tasks/routing/labels.jsonl \
    --out tasks/routing/routing.classifier.json --report tasks/routing/report.md
$B eval tasks/routing/routing.classifier.json --data tasks/routing/data.jsonl --labels tasks/routing/labels.jsonl
$B predict tasks/routing/routing.classifier.json --input '{"subject":"Refund","body":"I was charged twice."}'
```

`label` is resumable and bounded: it asks Laya only about rows without a
current record, `--limit N` caps a run, and five consecutive teacher errors stop
it. It labels gold rows too, so Laya itself can be scored against humans. These
labels are never trained on.

Serve every `*.classifier.json` in a directory from the daemon:

```sh
laya-daemon build/laya.mlpackage build/assets --classifiers tasks/routing   # or LAYA_CLASSIFIERS=<dir>
laya classify routing input.json
printf '%s\n' '{"op":"classify","classifier":"routing","input":{"body":"The app crashes on launch."}}' \
  | nc -U "$HOME/Library/Application Support/laya/laya.sock"
```

`{"op":"classifiers"}` lists loaded students. The daemon refuses to start if
any artifact fails validation. Existing `predict` and `health` requests are
unchanged. `LAYA_SOCKET` overrides the socket path for the daemon, `laya`, and
`laya-distill`.

### The task spec (`spec_version: 2`)

Strict JSON: every object rejects unknown fields. The routing template:

```json
{
  "spec_version": 2, "name": "routing", "version": "0.1.0",
  "instructions": "A customer sent the support message below. Which team should handle it?",
  "input": {"fields": [
    {"name": "subject", "type": "string", "required": false, "max_chars": 200},
    {"name": "body", "type": "string", "max_chars": 4000}
  ]},
  "labels": [
    {"name": "billing", "description": "invoices, payments, charges, refunds, and existing subscriptions"},
    {"name": "technical", "description": "bugs, errors, outages, login problems, and how to use the product"},
    {"name": "sales", "description": "new purchases, quotes, demos, and pricing for prospective customers"},
    {"name": "other", "description": "anything else, or no clear team"}
  ],
  "abstain": {"label": "other", "min_confidence": 0.5},
  "teacher": {"question_type": "choice", "min_confidence": 0.6, "min_margin": 0.2, "uncertain": "abstain"},
  "dataset": {"max_examples": 5000, "holdout_fraction": 0.3, "split_seed": "routing-v1"},
  "student": {"hash_dimensions": 4096, "epochs": 300, "learning_rate": 0.05,
              "l2_grid": [0.0001, 0.001, 0.01, 0.1], "class_weighting": "balanced"}
}
```

- **`teacher.question_type`**: how the task is asked of Laya.
  - `choice`: one option per label, `name: description`.
  - `noul`: exactly two labels, false first and true second; Laya's `P(true)`
    becomes the second label's probability.
  - `score`: labels are ordered rubric levels, lowest first.
- **Confidence gate**: a Laya answer becomes a hard training label only if its
  top probability is at least `min_confidence` and it leads the runner-up by at
  least `min_margin`. Otherwise it is `uncertain`, and `uncertain` decides what
  happens to it:
  - `abstain` trains the row as the abstain label (`other` here, `ask` for a
    permission task), never as Laya's argmax;
  - `drop` excludes the row from training.
- **`abstain`** is also the student's serving policy: below its
  `min_confidence`, the student answers the abstain label.
- **Input schema**: `string`, `number`, `boolean`, or `json` fields with
  `required` and `max_chars`. Undeclared, missing, mistyped, or oversized fields
  are rejected at labeling, training, prediction, and daemon time alike.
- **Dataset rows** (`data.jsonl`): `{"id", "input", "group"?, "gold"?}`.
  - Rows with `gold` (a human label) are **evaluation-only**.
  - `group` keeps related rows (paraphrases, one conversation) together.

### Label records and provenance

Each `labels.jsonl` row records the id, the content hash, a hash of the exact
Laya question, and the teacher identity. For example:

- `laya-daemon:laya-typed-decisions`, or
- `laya-runtime:laya-typed-decisions@<asset fingerprint>`.

It also records:

- the status (`accepted`, `uncertain`, or `error`);
- Laya's raw top label and the gated training label;
- every per-label probability, the top probability, and the margin;
- Laya's own confidence and action probability.

Inputs are never persisted. Changing the instructions, labels, or question type
changes the question hash, so earlier labels become stale and are asked again.

### Split and leakage controls

- **Gold is evaluation-only.** Pool rows that share a gold row's content or
  `group` are also kept out of training, so student-vs-human accuracy is
  measured on inputs nothing was trained on.
- **Duplicates.** Inputs are hashed after rendering in schema order,
  lowercasing, and collapsing whitespace. Duplicates collapse, and duplicates
  with conflicting labels are dropped.
- **Deterministic holdout.** Train and the Laya holdout are assigned by a hash
  of `split_seed` and the row's `group` (or content hash), so a group never
  straddles the split.
- **Model selection uses train only.** `l2_grid` picks L2 by group-aware 5-fold
  cross-validation on the training split.
- **Re-evaluation stays clean.** Artifacts store the training rows' content
  hashes. `eval` re-derives the split, then excludes and reports any evaluation
  row that was trained on.

### Artifacts

`<name>.classifier.json` (`format: laya.classifier`, `format_version: 2`)
contains:

- the normalized spec and its SHA-256;
- the feature scheme (`hashed-ngram-v1`), weights, and biases;
- provenance: teacher identity, question hash, dataset and label fingerprints,
  training content hashes, and the chosen L2 with its CV scores;
- the evaluation report;
- an integrity hash, verified on load.

### Measured results

Both templates were run end to end on this machine with the real Laya runtime
as teacher, using the commands above. All rows are synthetic, and the sets are
tiny: one row is about 2 points on gold (n = 48) and 5.6 points on the routing
holdout (n = 18).

**Routing** (60 pool rows, 48 gold rows):

| Reference | Model | n | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Human gold | Laya teacher, raw argmax | 48 | 85.4% | 0.782 |
| Human gold | Laya teacher, confidence-gated | 48 | 70.8% | 0.674 |
| Human gold | Student, argmax | 48 | 43.8% | 0.409 |
| Human gold | Student, served (abstain to `other`) | 48 | 12.5% | 0.056 |
| Human gold | Majority class | 48 | 12.5% | 0.056 |
| Laya labels | Student, argmax | 18 | 55.6% | 0.317 |
| Laya labels | Majority class | 18 | 38.9% | 0.140 |

- **Laya is a strong teacher for routing.** Every gold row that passed the gate
  was labeled correctly (28 of 28).
- **The gate costs coverage.** It sent 46 of 108 answers to `other`, and only
  4 pool rows kept a confident `sales` label.
- **The student is not usable.** Trained on 42 rows, it gives diffuse
  probabilities. Its serving threshold (`other` below 0.5) fired on every gold
  row, so the served student equals the majority baseline.

The fix is more unlabeled data, which Laya labels locally, at about 20 ms for a
short row (see [Benchmarks](#benchmarks)).
Thresholds were not tuned against these evaluation sets.

**Permission** (`--template permission`: 60 pool rows, 60 gold allow/deny/ask
rows): this template is a negative result. Laya answered `ask` for 111 of 120
rows and matched human gold on 20 of 60 (33%), including `ask` for every `allow`
row. `train` then refused to build a student, because the confident labels
covered only one class. **Do not use either template's student as a permission
gate, router, or any other safety or production control.**

The takeaway is to trust a student only after the gold report shows Laya
itself is accurate for your question, and only if the student beats the
majority baseline on both references by more than noise.

### Limitations

- The student is a linear model over hashed n-grams; it is only as good as the
  volume and quality of Laya's confident labels.
- Training uses hard labels only; Laya's probabilities are recorded but not used
  as soft targets.
- Laya reads at most 512 tokens per question. Longer inputs are truncated by
  Laya's own prompt builder when labeling.
- Training is full-batch and in memory, and has been run only at these template
  and test sizes (≤ 120 rows).
- The runtime teacher's asset fingerprint is a cheap compatibility check
  (runtime manifest, action-head weights, first 4 MiB of embeddings), not a hash
  of every weight. The daemon teacher reports only the model name.

### Tests

`swift test` runs 15 offline tests with a deterministic fake Laya teacher and
no model:

- strict spec parsing and input validation;
- bounded loading;
- gold-only evaluation and the leakage controls;
- exact gate boundaries, including the floating-point margin edge;
- `choice`, `noul`, and `score` question mapping;
- labeler resume, staleness, limits, and the error stop;
- an end-to-end student (label, train, save, reload, predict without Laya,
  re-evaluate), daemon `classify`, tamper rejection, and CV selection.

Two tests need real Laya:

- `LayaTeacherTests/testRealRuntimeLabelsAndStudentRunsWithoutLaya` runs with
  `LAYA_MODEL` and `LAYA_ASSETS`. The real runtime labels the routing template,
  then the test trains, saves, reloads, and predicts with no Laya object in
  scope.
- `testDaemonTeacherWhenAvailable` runs with `LAYA_TEST_SOCKET` pointing at a
  running daemon.

These tests need Apple Silicon, macOS 15, and the exported `build/laya.mlpackage`
and `build/assets`. The first load of a new package spends about a minute
compiling the Core ML model; later loads use the cached `.mlmodelc`. Without the
model they skip rather than fail.

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
