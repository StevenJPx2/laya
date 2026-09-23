import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path.home() / "Documents/Projects/laya-mlx"
sys.path.insert(0, str(REFERENCE))
from benchmarks.common import parity_cases
from laya_mlx import load

checkpoint = Path.home() / ".cache/huggingface/hub/models--aac6fef--laya-mlx/snapshots/20aed815fc6acde75733882e7ec0e3f28aeb9717"
agent = load(checkpoint, dtype="float32", device="cpu")
cases = [{"name": name, "state": state, "questions": questions, "expected": agent.predict(state, questions)} for name, state, questions in parity_cases()]
out = ROOT / "Tests/LayaCoreTests/Fixtures/validation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {len(cases)} cases to {out}")
