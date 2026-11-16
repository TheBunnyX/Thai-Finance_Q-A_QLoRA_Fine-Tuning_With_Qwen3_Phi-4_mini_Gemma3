#!/bin/bash
# Micro-batch sweep on Linux: same data, effective batch 8 (accumulation adjusts), 40 optimizer steps.
# Usage: scripts/bench_batch_linux.sh [model] [batch sizes...]   e.g. scripts/bench_batch_linux.sh qwen 1 2 4 8
# Existing run directories are skipped, never overwritten. Summarize with scripts/summarize_bench.py.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=${1:-qwen}; shift
SIZES=${*:-1 2 4 8}
D=${D:-data/processed/finance-1250-split-v1}
STEPS=${STEPS:-40}
OUT=outputs/bench/linux-batch
mkdir -p "$OUT"
for b in $SIZES; do
  run="$OUT/$MODEL-b$b"
  if [ -e "$run" ]; then echo "SKIP $run (exists)"; continue; fi
  echo "TRAIN $MODEL b$b $(date +%T)"
  "$PY" -X utf8 -m finance_qlora train --model "$MODEL" --data "$D" --run-dir "$run" \
    --max-steps "$STEPS" --batch-size "$b" --no-progress > "$run.log" 2>&1 || echo "FAILED $MODEL b$b (see $run.log)"
done
