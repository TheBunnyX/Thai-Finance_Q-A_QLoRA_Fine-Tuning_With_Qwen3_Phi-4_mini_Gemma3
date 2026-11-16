#!/bin/bash
# Train 4 models on finance-1250-split-v1 (2 epochs), then score each on the test split once.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=.venv/Scripts/python.exe
D=${D:-data/processed/finance-1250-split-v1}
TAG=${TAG:-f1250}
for m in ${MODELS:-phi qwen gemma}; do
  echo "== TRAIN $m $(date +%T)"
  $PY -X utf8 -m finance_qlora train --model $m --data $D --run-dir outputs/$TAG-$m --epochs 2 --no-progress > outputs/$TAG-$m.train.log 2>&1
  echo "== EVAL $m $(date +%T)"
  $PY -X utf8 -m finance_qlora evaluate --model $m --data $D --split test --adapter outputs/$TAG-$m/adapter \
    --max-new-tokens 256 --batch-size 4 --output outputs/eval-$TAG-$m-test --no-progress > outputs/$TAG-$m.eval.log 2>&1
done
echo "== ALL DONE $(date +%T)"
