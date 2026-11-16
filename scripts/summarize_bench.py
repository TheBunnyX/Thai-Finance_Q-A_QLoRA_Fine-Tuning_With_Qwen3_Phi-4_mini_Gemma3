"""Print a speed/VRAM table for completed benchmark runs: python scripts/summarize_bench.py outputs/bench/linux-batch"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/bench/linux-batch")
print(f"{'run':<22}{'steps':>6}{'train s':>9}{'s/step':>8}{'ans tok/s':>11}{'pad eff':>9}"
      f"{'peak alloc':>12}{'peak rsv':>10}{'eval loss':>11}")
for path in sorted(root.glob("*/run.json")):
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        print(f"{path.parent.name:<22}  {run.get('status')}")
        continue
    m = run["metrics"]
    steps = run["optimizer_steps"]
    print(f"{path.parent.name:<22}{steps:>6}{run['training_seconds']:>9.1f}"
          f"{run['training_seconds'] / steps:>8.2f}{m['training_answer_tokens_per_second_including_eval']:>11.0f}"
          f"{(m.get('training_padding_efficiency') or 0):>9.2f}{run['peak_allocated_gib']:>11.2f}G"
          f"{run['peak_reserved_gib']:>9.2f}G{m.get('eval_loss', float('nan')):>11.4f}")
