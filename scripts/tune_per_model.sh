#!/bin/bash
# Per-model tuning with the current config (no checkpointing, seq 512): train micro-batch 1/2/4 for Qwen and Phi
# (20 steps), then evaluation batch 4/8 for Phi and Gemma (48 rows, f1250 adapters).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=.venv/Scripts/python.exe; D=data/processed/finance-1250-split-v1; B=outputs/bench/tune
rm -rf $B; mkdir -p $B
for m in qwen phi; do for b in 1 2 4; do
  echo "TRAIN $m b$b $(date +%T)"
  $PY -X utf8 -m finance_qlora train --model $m --data $D --run-dir $B/train-$m-b$b --batch-size $b --max-steps 20 --no-progress > $B/train-$m-b$b.log 2>&1 || echo FAILED
done; done
for m in phi gemma; do for b in 4 8; do
  echo "EVAL $m b$b $(date +%T)"
  $PY -X utf8 -m finance_qlora evaluate --model $m --data $D --split validation --limit 48 --max-new-tokens 256 --warmup-batches 1 \
    --batch-size $b --adapter outputs/f1250-$m/adapter --output $B/eval-$m-b$b --no-progress > $B/eval-$m-b$b.log 2>&1 || echo FAILED
done; done
echo "ALL DONE $(date +%T)"
