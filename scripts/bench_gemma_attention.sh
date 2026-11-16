#!/bin/bash
# Gemma attention comparison: forced SDPA (UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM=0) at the same
# micro-batches as the Flex sweep (gemma-b*), 40 steps, effective batch 8. Waits for a running sweep.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=${D:-data/processed/finance-1250-split-v1}
OUT=outputs/bench/linux-batch
# Anchor on the script process itself; a plain pattern also matches any parent shell whose
# command line mentions the script, which would wait forever.
while pgrep -f "^/bin/bash (\./)?scripts/bench_batch_linux\.sh" > /dev/null; do sleep 20; done
for b in ${*:-2 4 8}; do
  run="$OUT/gemma-sdpa-b$b"
  if [ -e "$run" ]; then echo "SKIP $run (exists)"; continue; fi
  echo "TRAIN gemma sdpa b$b $(date +%T)"
  UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM=0 "$PY" -X utf8 -m finance_qlora train --model gemma --data "$D" \
    --run-dir "$run" --max-steps 40 --batch-size "$b" --no-progress > "$run.log" 2>&1 \
    || echo "FAILED gemma sdpa b$b (see $run.log)"
done
for d in "$OUT"/gemma*/checkpoints; do [ -d "$d" ] && rm -r -- "$d"; done
"$PY" scripts/summarize_bench.py "$OUT" | grep -E "run|gemma"
"$PY" - <<'EOF'
import json, pathlib
for p in sorted(pathlib.Path("outputs/bench/linux-batch").glob("gemma*/run.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    print(p.parent.name, r.get("attention"))
EOF
