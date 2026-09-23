"""Per-operation compute-device placement for an exported Core ML package.

Uses Core ML's own compute plan (``MLComputePlan``) — the same placement the
runtime performs — so the report is authoritative rather than inferred from
timing. Run it against a package to see which ops the Neural Engine takes and
which fall back, grouped by op type and by first blocking op in program order.

    uv run python ane_report.py ../../build/laya.mlpackage
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def device_name(device) -> str:
    name = type(device).__name__
    return {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}.get(name, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--units", default="CPU_AND_NE", choices=["CPU_AND_NE", "ALL", "CPU_AND_GPU"])
    parser.add_argument("--list-fallbacks", type=int, default=25, help="print the first N non-ANE ops in program order")
    args = parser.parse_args()

    compiled = ct.utils.compile_model(str(args.package)) if args.package.suffix == ".mlpackage" else str(args.package)
    plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=getattr(ct.ComputeUnit, args.units))

    program = plan.model_structure.program
    if program is None:
        sys.exit("not an ML Program")

    by_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    fallbacks: list[tuple[int, str, str, str]] = []
    total = collections.Counter()

    for function in program.functions.values():
        for index, op in enumerate(function.block.operations):
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue

            preferred = device_name(usage.preferred_compute_device)
            supported = ",".join(sorted({device_name(d) for d in usage.supported_compute_devices}))
            by_type[op.operator_name][preferred] += 1
            total[preferred] += 1

            if preferred != "ANE":
                outputs = ",".join(o.name for o in op.outputs)[:60]
                fallbacks.append((index, op.operator_name, supported, outputs))

    print(f"compute units: {args.units}")
    print(f"placed ops: {sum(total.values())}   " + "  ".join(f"{k}={v}" for k, v in sorted(total.items())))
    print()
    print(f"{'op type':<28}{'ANE':>6}{'GPU':>6}{'CPU':>6}")
    for name, counts in sorted(by_type.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
        print(f"{name:<28}{counts.get('ANE', 0):>6}{counts.get('GPU', 0):>6}{counts.get('CPU', 0):>6}")

    if fallbacks:
        print()
        print(f"first {min(len(fallbacks), args.list_fallbacks)} of {len(fallbacks)} non-ANE ops (program order):")
        for index, name, supported, outputs in fallbacks[: args.list_fallbacks]:
            print(f"  #{index:<5}{name:<24}supported={supported:<12}{outputs}")


if __name__ == "__main__":
    main()
