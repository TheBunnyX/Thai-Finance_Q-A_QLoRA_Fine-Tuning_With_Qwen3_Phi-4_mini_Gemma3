#!/bin/bash
cd "$(dirname "$0")/.."
KPY=.venv-kernel/bin/python; [ -x "$KPY" ] || KPY=.venv-kernel/Scripts/python.exe
for m in qwen phi gemma; do for mode in base liger compile liger_compile; do
  echo "== $m $mode $(date +%T)"
  timeout 900 "$KPY" -X utf8 scripts/bench_kernels.py --model $m --mode $mode --steps 20 > outputs/bench/kernels-$m-$mode.log 2>&1 || echo "FAILED $m $mode"
done; done; echo "== ALL DONE $(date +%T)"
