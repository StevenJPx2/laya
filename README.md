# laya

A compiled, Python-free runtime for the Laya typed-decision model on Apple
Silicon. It runs the ModernBERT-large encoder and Laya decision heads through
**Core ML**, exposed two ways:

- **`LayaCore`** — a Swift library you embed directly.
- **`laya-daemon`** — a long-running daemon that keeps the model warm and serves
  newline-delimited JSON over a Unix socket at
  `~/Library/Application Support/laya/laya.sock` (override with `LAYA_SOCKET`).
- **`laya-distill`** — define a classification task, have Laya label it as the
  teacher (or import labels from an external teacher), train a small student,
  evaluate it against the teacher, Laya zero-shot, and human gold labels, and
  serve it from the same daemon
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
| `Sources/LayaDistill` | Task spec, Laya teacher, Jev label import, confidence gate, split, student features and training, evaluation, artifacts |
| `Sources/laya-distill` | CLI: `init`, `validate`, `label`, `import`, `train`, `eval`, `predict` |
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

`laya-distill` trains small, task-specific students from a teacher's labels.
The teacher is **Laya** by default, or an external teacher whose labels are
imported (currently a [jev-distill](#importing-jev-labels) ledger):

```text
unlabeled rows ──► ask Laya a task question (choice / noul / score) ──┐
                     (installed daemon or local runtime)              ├─► confidence gate ──► labels.jsonl
jev-distill ledger ──► laya-distill import ───────────────────────────┘   uncertain → abstain label or drop
labels.jsonl ──► train a softmax student ──► evaluate vs teacher (holdout), Laya zero-shot, and human gold
                                          ──► <name>.classifier.json
```

The student is a softmax-regression head trained natively in Swift with no new
dependencies. `student.features` picks its input:

- **`hashed-ngram-v1`** (default): hashed word unigrams and bigrams. Serving
  loads no Laya model; `predict` and the daemon's `classify` op read only the
  artifact.
- **`laya-logits-v1`**: Laya's raw (uncalibrated) option logits for the task
  question, one per label, standardized with the training split's mean and
  standard deviation. Training, evaluation, and serving each run one Laya
  forward pass per row (cached within a run), so they need the Laya runtime.
- **`laya-embedding-v1`**: those logits plus Laya's 1024-d pooled decision
  vector, standardized the same way. Same runtime needs; wants hundreds of
  labeled rows per label, a lower `learning_rate`, and more `epochs`.
- **`laya-hybrid-v1`**: the embedding features followed by `hash_dimensions`
  hashed n-grams, in one head. Laya contributes semantics, the n-grams
  lexical cues.

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

# For hashed-ngram students (the default), everything below runs without Laya.
$B train tasks/routing/task.json --data tasks/routing/data.jsonl --labels tasks/routing/labels.jsonl \
    --out tasks/routing/routing.classifier.json --report tasks/routing/report.md
$B eval tasks/routing/routing.classifier.json --data tasks/routing/data.jsonl --labels tasks/routing/labels.jsonl
$B predict tasks/routing/routing.classifier.json --input '{"subject":"Refund","body":"I was charged twice."}'
```

`label` is resumable and bounded: it asks Laya only about rows without a
current record, `--limit N` caps a run, and five consecutive teacher errors stop
it. It labels gold rows too, so Laya itself can be scored against humans. These
labels are never trained on.

### Importing Jev labels

```sh
$B import tasks/routing/task.json --data tasks/routing/data.jsonl     --jev jev-ledger.jsonl --labels tasks/routing/labels.jsonl
```

`import` reads a `jev-distill.labels` v1 ledger and prints a JSON summary
(`imported`, `accepted`, `uncertain_*`, `forbidden`, `failed`, `invalid`,
`unknown_ids`, `missing`). The rules:

- Ledger rows map to dataset rows by `id`; the last line per id wins, and
  unknown ids are counted and skipped.
- A labeled row's `probabilities` must cover exactly the task's label names,
  or the whole import fails.
- Labeled rows pass the same confidence gate as Laya answers.
- `forbidden` and `failed` rows become `error` records. They are never trained
  on, and no label is synthesized for them.
- Records carry teacher `jev:<answered_model>` and the task's current question
  hash, so staleness and the split treat them like Laya labels. `import`
  rewrites the labels file, replacing records by id.

Set `"teacher": {"source": "import", …}` so `label` refuses to overwrite
imported labels.

### Logit students (`laya-logits-v1`)

```sh
# in task.json: "student": {"features": "laya-logits-v1", "l2_grid": [...], ...}
$B train tasks/routing/task.json --data tasks/routing/data.jsonl --labels tasks/routing/labels.jsonl     --out tasks/routing/routing.classifier.json --report tasks/routing/report.md     --model "$HOME/Library/Application Support/laya/laya.mlmodelc" --assets "$HOME/Library/Application Support/laya/assets"
```

`train`, `eval`, and `predict` load Laya in-process from `--model`/`--assets`.
Without them they use `$LAYA_MODEL`/`$LAYA_ASSETS`, then
`~/Library/Application Support/laya/{laya.mlmodelc,assets}`. The artifact
records the standardization statistics, the question hash, and the Laya asset
fingerprint. Loading it against a runtime with a different fingerprint fails.
Passing `--model`/`--assets` to a hashed student's `train` or `eval` adds the
Laya zero-shot row to its report.

The gold report scores the student (served and argmax), the teacher by
identity, **Laya zero-shot** (argmax of the raw logits), and the majority
class. `beats_laya_zero_shot_on_gold` records whether the served student beats
zero-shot.

A logit head can only reweight what Laya's options already carry, and on a
10-label task it plateaus early (see [Measured results](#measured-results)).
Embedding and hybrid heads need hundreds of labels per label to beat it; on
tiny sets (≈75 rows) they overfit. On permissions, where Laya's options carry
little signal, no Laya scheme improved. Heads on rounded `predict`
probabilities did worse than heads on raw logits.

Serve every `*.classifier.json` in a directory from the daemon:

```sh
laya-daemon build/laya.mlpackage build/assets --classifiers tasks/routing   # or LAYA_CLASSIFIERS=<dir>
laya classify routing input.json
printf '%s\n' '{"op":"classify","classifier":"routing","input":{"body":"The app crashes on launch."}}' \
  | nc -U "$HOME/Library/Application Support/laya/laya.sock"
```

`{"op":"classifiers"}` lists loaded students. The daemon refuses to start if
any artifact fails validation, including a logit student trained on different
Laya assets. It serves logit students with its own warm runtime: one forward
pass, then the head and the abstain policy. Existing `predict` and `health` requests are
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
- **`teacher.source`** (optional): `laya` (default) or `import`.
- **`student.features`** (optional): `hashed-ngram-v1` (default),
  `laya-logits-v1`, `laya-embedding-v1`, or `laya-hybrid-v1`; see
  [Logit students](#logit-students-laya-logits-v1).
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

`<name>.classifier.json` (`format: laya.classifier`, `format_version: 3`)
contains:

- the normalized spec and its SHA-256;
- the feature descriptor, weights, and biases. For Laya schemes the
  descriptor also holds the per-dimension mean and standard deviation of the
  Laya values (floor 1e-6), the question hash, and the Laya asset fingerprint;
- provenance: teacher identity, question hash, dataset and label fingerprints,
  training content hashes, and the chosen L2 with its CV scores;
- the evaluation report;
- an integrity hash, verified on load.

Version 2 artifacts are rejected with a message to retrain.

### Measured results

**Distilling Jev into Laya: CLINC150 domain routing.** The questions and
human gold labels come from [CLINC150](https://github.com/clinc/oos-eval)
(crowd-written queries; the official 150-intent → 10-domain map, no
out-of-scope rows). Pool: 30 train-split queries per intent (4,500), labeled
by Jev `jev-1.13.0` via jev-distill for $0.113 total with no refusals, then
`import`ed. Gold: 2 test-split queries per intent (300), evaluation-only. All
students used the same settings (`learning_rate` 0.01, 1000 epochs, `l2_grid`
`[0.0001, 0.001, 0.01]`, 4096 hash dimensions) and the same 3,400 / 1,100
train / teacher-holdout split.

| Model | Gold accuracy (n = 300) | Agreement with Jev (n = 1,100) | CV on train |
|---|---:|---:|---:|
| Jev (teacher) | 83.3% | — | — |
| Student, `laya-hybrid-v1` | **79.7%** (±4.6) | **86.5%** | 87.6% |
| Student, `hashed-ngram-v1` | 78.3% (±4.7) | 84.8% | 84.6% |
| Student, `laya-embedding-v1` | 75.3% | 77.5% | 79.4% |
| Student, `laya-logits-v1` | 66.0% | 67.9% | 71.0% |
| Laya zero-shot | 54.7% | — | — |
| Majority class | 10.0% | — | — |

- **Distillation works.** The hybrid student retains about 96% of Jev's gold
  accuracy and gains 25 points over Laya zero-shot. Served from the daemon it
  answers in about 80–110 ms, including Laya's forward pass.
- **Laya's part is small.** Hybrid beats hashed-only by 1.4 points on gold, which is
  within noise, and by 1.7 points of agreement with Jev. The gain is
  consistent across CV, teacher holdout, and gold, but a hashed student is
  nearly as accurate and needs no model at serving time.
- **Labels matter most.** Going from 798 to 3,400 Jev labels moved hashed from
  73.0% to 78.3% and hybrid from 74.3% to 79.7%. Logit heads stayed at about 65%.
- Training embedding or hybrid heads on 3,400 rows took about 40 minutes
  (full-batch, in-process Laya features); hashed took about 1 minute.

The templates below were run end to end on this machine with the real Laya
runtime as teacher, using the commands above. All rows are synthetic, and the sets are
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

- The student is a linear model over hashed n-grams and/or frozen Laya
  features; it is only as good as the volume and quality of the teacher's
  confident labels.
- Laya features are recomputed on every `train`/`eval` run (about 0.1 s per
  row); they are not persisted between runs.
- Training uses hard labels only; Laya's probabilities are recorded but not used
  as soft targets.
- Laya reads at most 512 tokens per question. Longer inputs are truncated by
  Laya's own prompt builder when labeling.
- Training is full-batch and in memory. It has been run up to 4,800 rows,
  where dense Laya schemes are slow (see above).
- The runtime teacher's asset fingerprint is a cheap compatibility check
  (runtime manifest, action-head weights, first 4 MiB of embeddings), not a hash
  of every weight. The daemon teacher reports only the model name.

### Tests

`swift test` runs 23 offline tests with a deterministic fake Laya teacher, a
fake Laya representation provider, and no model:

- strict spec parsing and input validation;
- bounded loading;
- gold-only evaluation and the leakage controls;
- exact gate boundaries, including the floating-point margin edge;
- `choice`, `noul`, and `score` question mapping;
- labeler resume, staleness, limits, and the error stop;
- an end-to-end student (label, train, save, reload, predict without Laya,
  re-evaluate), daemon `classify`, tamper rejection, and CV selection;
- Jev import (gate, last line wins, forbidden and failed rows, unknown ids,
  mismatched labels);
- a logit student end to end (import, train with one forward pass per row,
  beat zero-shot, reload), the v3 fingerprint check, and daemon `classify`
  for a logit student;
- embedding and hybrid students end to end (standardized logits plus pooled
  values, hashed n-grams appended for hybrid, zero-shot from logits only,
  serving, and the runtime requirement).

Three tests need real Laya:

- `LayaTeacherTests/testRealRuntimeLabelsAndStudentRunsWithoutLaya` runs with
  `LAYA_MODEL` and `LAYA_ASSETS`. The real runtime labels the routing template,
  then the test trains, saves, reloads, and predicts with no Laya object in
  scope.
- `testRealRepresentationLogitsMatchPredict` (same variables) checks that the
  raw-logit argmax, mapped to labels, agrees with Laya's `predict` choice.
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
