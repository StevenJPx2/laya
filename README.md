# laya

A compiled, Python-free runtime for the Laya typed-decision model on Apple
Silicon. It runs the ModernBERT-large encoder and Laya decision heads through
**Core ML**, exposed two ways:

- **`LayaCore`** — a Swift library you embed directly.
- **`laya-daemon`** — a long-running daemon that keeps the model warm and serves
  newline-delimited JSON over a Unix socket at
  `~/Library/Application Support/laya/laya.sock`.

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
| `Sources/laya` | CLI client: `predict`, `health`, `bench` |
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
