"""Build-time Core ML export for the Laya typed-decision model (ANE-first).

Loads the cached MLX ``model.safetensors`` into the Torch mirror (``model.py``),
verifies it against the MLX reference, exports an ``.mlpackage`` whose graph is
entirely ANE-eligible, and writes the host-side assets the Swift runtime needs:

    <assets>/embeddings.f16.bin    token embedding table, fp16 [vocab, hidden]
    <assets>/act_head.f32.bin      act_head weights, fp32 (W1, b1, W2, b2)
    <assets>/runtime_manifest.json shapes and constants shared with Swift

Every per-token tensor is its own enumerated-shape input (L in 128/256/512).
Core ML permits several enumerated inputs from macOS 15; packing them into one
tensor (the macOS 14 workaround) forces static-width slices off a symbolic axis,
which the on-device execution planner rejects (error -14).

Run from ``tools/convert`` with the reference venv (provides ``mlx`` + ``laya_mlx``):

    python export.py --checkpoint <snapshot> --assets ../../build/assets \
        --output ../../build/laya.mlpackage --verify --report
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from model import MASK_BIAS, Config, attention_biases, load_model, rope_table

REFERENCE = Path.home() / "Documents/Projects/laya-mlx"
LENGTHS = (128, 256, 512)
MARKER_SLOTS = 20


def input_shapes(cfg: Config, length: int) -> dict[str, list[int]]:
    """Input name -> shape for one sequence length. Order matches LayaModel.forward."""
    return {
        "embeds": [1, length, cfg.hidden_size],
        "global_bias": [1, 1, 1, length],
        "sliding_bias": [1, 1, length, length],
        "rope": [1, 2, length, cfg.head_dim],
        "type_onehot": [1, 3],
        "marker_onehot": [1, MARKER_SLOTS, length],
        "marker_mask": [1, MARKER_SLOTS],
    }


def manifest(cfg: Config) -> dict:
    return {
        "hidden": cfg.hidden_size,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "local_window": cfg.local_attention,
        "rope_theta_global": cfg.rope_base("full_attention"),
        "rope_theta_local": cfg.rope_base("sliding_attention"),
        "marker_slots": MARKER_SLOTS,
        "mask_bias": MASK_BIAS,
        "lengths": list(LENGTHS),
        "inputs": list(input_shapes(cfg, LENGTHS[0]).keys()),
    }


def host_inputs(cfg: Config, embeds: torch.Tensor, valid: torch.Tensor, markers: list[int], qtype: int) -> dict[str, torch.Tensor]:
    """Host-side input construction (mirrored in Swift). embeds: [L, H]; valid: [L] bool."""
    length = valid.shape[0]
    global_bias, sliding_bias = attention_biases(valid[None], cfg.local_attention)

    rope = torch.zeros((1, 2, length, cfg.head_dim))
    half = cfg.head_dim // 2
    for index, kind in enumerate(("full_attention", "sliding_attention")):
        cos, sin = rope_table(length, cfg.rope_base(kind), cfg.head_dim)
        rope[0, index, :, :half], rope[0, index, :, half:] = cos, sin

    marker_onehot = torch.zeros((1, MARKER_SLOTS, length))
    marker_mask = torch.zeros((1, MARKER_SLOTS))
    for slot, position in enumerate(markers[:MARKER_SLOTS]):
        marker_onehot[0, slot, position] = 1.0
        marker_mask[0, slot] = 1.0

    return {
        "embeds": embeds[None],
        "global_bias": global_bias,
        "sliding_bias": sliding_bias,
        "rope": rope,
        "type_onehot": torch.nn.functional.one_hot(torch.tensor([qtype]), 3).float(),
        "marker_onehot": marker_onehot,
        "marker_mask": marker_mask,
    }


def host_action(logits: np.ndarray, marker_mask: np.ndarray, cls: np.ndarray, act_head) -> np.ndarray:
    """Host post-processing (mirrored in Swift): act features + act_head MLP."""
    p = np.exp(logits - logits.max())
    p /= p.sum()
    count = max(2.0, float(marker_mask.sum()))
    entropy = -(p * np.log(np.maximum(p, 1e-9))).sum() / math.log(count)
    top = np.sort(p)[-2:]
    features = np.array([top[1], top[1] - top[0], entropy, count / 255.0], dtype=np.float32)

    w1, b1, w2, b2 = (t.detach().numpy() for t in (act_head[0].weight, act_head[0].bias, act_head[2].weight, act_head[2].bias))
    pooled = np.concatenate([cls.astype(np.float32), features])
    hidden = pooled @ w1.T + b1
    hidden = 0.5 * hidden * (1.0 + np.vectorize(math.erf)(hidden / math.sqrt(2.0)))

    return hidden @ w2.T + b2


def write_assets(model, cfg: Config, assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    model.encoder.tok.weight.detach().to(torch.float16).numpy().tofile(assets / "embeddings.f16.bin")

    act = model.act_head
    with open(assets / "act_head.f32.bin", "wb") as handle:
        for tensor in (act[0].weight, act[0].bias, act[2].weight, act[2].bias):
            tensor.detach().to(torch.float32).numpy().tofile(handle)

    (assets / "runtime_manifest.json").write_text(json.dumps(manifest(cfg), indent=2) + "\n")
    print(f"assets written to {assets}")


def convert_fixed(model, cfg: Config, length: int, output: Path) -> None:
    """One fixed-shape function. Fixed shapes are what lets the ANE compiler take the graph."""
    shapes = input_shapes(cfg, length)
    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(torch.zeros(shape) for shape in shapes.values()), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=shape, dtype=np.float16) for name, shape in shapes.items()],
        outputs=[ct.TensorType(name="logits"), ct.TensorType(name="cls")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(str(output))


def convert(model, cfg: Config, output: Path) -> None:
    """Export one multifunction package: seq128/seq256/seq512 sharing a single weight blob.

    Enumerated shapes would be simpler, but Core ML refuses to place this graph on
    the Neural Engine unless every shape is static. Merging fixed-shape functions
    keeps one copy of the 800 MB weights while giving each length an ANE-resident
    specialization.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    descriptor = ct.utils.MultiFunctionDescriptor()
    for length in LENGTHS:
        part = staging / f"seq{length}.mlpackage"
        convert_fixed(model, cfg, length, part)
        descriptor.add_function(str(part), "main", f"seq{length}")
        print(f"converted seq{length}")

    descriptor.default_function_name = f"seq{LENGTHS[0]}"
    if output.exists():
        shutil.rmtree(output)
    ct.utils.save_multifunction(descriptor, str(output))
    shutil.rmtree(staging)
    print(json.dumps({"output": str(output), "functions": [f"seq{n}" for n in LENGTHS], "inputs": list(input_shapes(cfg, LENGTHS[0]))}, indent=2))


def verify(model, cfg: Config, root: Path) -> None:
    """Compare Torch mirror + host post-processing against MLX on one prepared batch."""
    import sys

    sys.path.insert(0, str(REFERENCE))
    from laya_mlx.agent import Agent, collate_items

    agent = Agent(root, dtype="float32")
    state = "charged twice; refund requested"
    questions = {"q": {"type": "choice", "instructions": "Who handles this?", "criteria": ["billing", "technical", "sales"]}}
    items, _ = agent.prepare(state, questions)
    batch = collate_items(items, agent.tok.pad_token_id)

    ids = torch.from_numpy(batch["input_ids"])[0]
    valid = torch.from_numpy(batch["attention_mask"])[0]
    inputs = host_inputs(cfg, model.encoder.tok.weight.detach()[ids], valid, items[0]["markers"], items[0]["qtype"])

    with torch.inference_mode():
        expected_logits, expected_act = (np.asarray(t) for t in agent.forward(batch))
        logits, cls = model(*inputs.values())

    logits, cls = logits.numpy()[0], cls.numpy()[0]
    act = host_action(logits, inputs["marker_mask"].numpy()[0], cls, model.act_head)

    width = expected_logits.shape[1]
    errors = {
        "logits_max_abs": float(np.max(np.abs(logits[:width] - expected_logits[0]))),
        "action_max_abs": float(np.max(np.abs(act - expected_act[0]))),
    }
    print(json.dumps({"torch_vs_mlx": errors}, indent=2))

    if errors["logits_max_abs"] > 1e-2 or errors["action_max_abs"] > 5e-2:
        raise SystemExit("Torch mirror verification failed")


def report(output: Path, cfg: Config) -> None:
    """Time each function per compute unit. ANE residency shows as CPU_AND_NE << CPU_ONLY."""
    for length in LENGTHS:
        inputs = {name: np.zeros(shape, dtype=np.float16) for name, shape in input_shapes(cfg, length).items()}
        inputs["type_onehot"][0, 0] = 1
        inputs["marker_mask"][:] = 1

        for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.ALL):
            model = ct.models.MLModel(str(output), compute_units=units, function_name=f"seq{length}")
            for _ in range(3):
                model.predict(inputs)

            samples = []
            for _ in range(20):
                start = time.perf_counter_ns()
                model.predict(inputs)
                samples.append((time.perf_counter_ns() - start) / 1e6)

            samples.sort()
            print("seq%-4d %-24s p50 %6.1f ms  p95 %6.1f ms" % (length, units, samples[len(samples) // 2], samples[int(len(samples) * 0.95)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/laya.mlpackage"))
    parser.add_argument("--assets", type=Path, default=Path("build/assets"))
    parser.add_argument("--verify", action="store_true", help="check the Torch mirror against MLX before export")
    parser.add_argument("--report", action="store_true", help="time the exported model per compute unit")
    args = parser.parse_args()

    model = load_model(args.checkpoint).requires_grad_(False)
    cfg = Config.load(args.checkpoint / "encoder/config.json")

    if args.verify:
        verify(model, cfg, args.checkpoint)

    write_assets(model, cfg, args.assets)
    convert(model, cfg, args.output)

    if args.report:
        report(args.output, cfg)


if __name__ == "__main__":
    main()
