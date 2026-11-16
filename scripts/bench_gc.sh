#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=.venv/Scripts/python.exe
for m in qwen phi gemma; do for mode in on off; do
  d=outputs/bench/gc-$m-$mode; rm -rf $d; extra=""; [ $mode = off ] && extra="--no-gradient-checkpointing"
  "$PY" -X utf8 -m finance_qlora train --model $m --data data/processed/finance-1250-split-v1 --run-dir $d --max-steps 20 --no-progress $extra > $d.log 2>&1
done; done; echo done > outputs/bench/gc-done.txt
