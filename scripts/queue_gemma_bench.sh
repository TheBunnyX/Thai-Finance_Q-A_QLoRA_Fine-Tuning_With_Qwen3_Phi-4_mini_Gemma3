#!/bin/bash
# Wait for the running Phi sweep, free benchmark-only checkpoints, then sweep Gemma if disk allows.
cd "$(dirname "$0")/.."
while pgrep -f "bench_batch_linux.sh phi" > /dev/null; do sleep 20; done
# Bench checkpoints are model-only copies of the saved adapter; run.json and adapter/ are kept.
for d in outputs/bench/linux-batch/*/checkpoints; do [ -d "$d" ] && rm -r -- "$d"; done
free_gb=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb} GB"
if [ "$free_gb" -lt 11 ]; then
  echo "STOP: Gemma needs ~8.6 GB download plus run outputs; free disk first"; exit 1
fi
scripts/bench_batch_linux.sh gemma 2 4 8
