#!/bin/bash
# Benchmark the laya daemon: P50/P95 latency plus peak RSS and CPU over N
# predictions. Starts a private daemon, warms it, samples the process while
# `laya bench` drives the socket, then shuts the daemon down.
#
# Usage: tools/benchmark.sh <model.mlpackage> <assets-dir> <request.json> [iterations]
set -euo pipefail

MODEL="${1:?model .mlpackage or .mlmodelc}"
ASSETS="${2:?assets directory}"
REQUEST="${3:?request json}"
ITERATIONS="${4:-1000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

swift build -c release --package-path "$ROOT" >/dev/null

"$ROOT/.build/release/laya-daemon" "$MODEL" "$ASSETS" &
DAEMON=$!
trap 'kill "$DAEMON" 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
    "$ROOT/.build/release/laya" health >/dev/null 2>&1 && break
    sleep 1
done

"$ROOT/.build/release/laya" predict "$REQUEST" >/dev/null

USAGE="$(mktemp)"
( while kill -0 "$DAEMON" 2>/dev/null; do ps -o rss=,%cpu= -p "$DAEMON" 2>/dev/null; sleep 0.05; done > "$USAGE" ) &
SAMPLER=$!

"$ROOT/.build/release/laya" bench "$REQUEST" "$ITERATIONS"

kill "$SAMPLER" 2>/dev/null || true
awk 'BEGIN{r=0;c=0} {if($1>r)r=$1; if($2>c)c=$2} END{printf "peak RSS %.1f MiB, peak CPU %s%%\n", r/1024, c}' "$USAGE"
rm -f "$USAGE"
