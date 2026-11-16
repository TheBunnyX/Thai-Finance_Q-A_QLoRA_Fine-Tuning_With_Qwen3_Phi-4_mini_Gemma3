#!/bin/bash
# Evaluate Qwen adapters trained on finance-5000-v2 variants, then the untuned baseline, on
#   (a) finance-5000-v2 test (templates seen in training, new numbers; numeric annotations)
#   (b) finance-5000-v2-heldout test (answer templates never seen in training)
# Usage: scripts/eval_qwen_5000_v2.sh [file-to-wait-for "done" line]
# Existing outputs are skipped, never overwritten. Greedy, 256 new tokens, 1 warmup batch.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WAIT_FILE=$1
if [ -n "$WAIT_FILE" ]; then
  until grep -q "^done " "$WAIT_FILE" 2>/dev/null; do sleep 30; done
fi
BATCH=${BATCH:-8}
export HF_HUB_OFFLINE=1

run_eval() {  # name data adapter-or-empty
  local name=$1 data=$2 adapter=$3 out="outputs/eval-$1"
  if [ -e "$out" ]; then echo "SKIP $name (exists)"; return; fi
  if [ -n "$adapter" ] && [ ! -f "$adapter/adapter_config.json" ]; then echo "SKIP $name (no adapter)"; return; fi
  echo "EVAL $name $(date +%T)"
  local extra=(); [ -n "$adapter" ] && extra=(--adapter "$adapter")
  if ! "$PY" -X utf8 -m finance_qlora evaluate --model qwen --data "$data" --split test \
      --max-new-tokens 256 --warmup-batches 1 --batch-size "$BATCH" "${extra[@]}" --output "$out" \
      --no-progress --progress-log "reports/eval-$name-progress.jsonl" > "$out.log" 2>&1; then
    if grep -q "out of memory" "$out.log" && [ "$BATCH" -gt 4 ]; then
      echo "  OOM at batch $BATCH; retrying $name with batch 4"
      mv "$out" "$out-oom" 2>/dev/null; mv "$out.log" "$out-oom.log"
      "$PY" -X utf8 -m finance_qlora evaluate --model qwen --data "$data" --split test \
        --max-new-tokens 256 --warmup-batches 1 --batch-size 4 "${extra[@]}" --output "$out" \
        --no-progress --progress-log "reports/eval-$name-progress.jsonl" > "$out.log" 2>&1 \
        || echo "  FAILED $name (see $out.log)"
    else
      echo "  FAILED $name (see $out.log)"
    fi
  fi
}

V2=data/processed/finance-5000-v2
HO=data/processed/finance-5000-v2-heldout
for v in v2 v2-steps v2-cap30; do run_eval "qwen-5000-$v-test" $V2 "outputs/qwen-5000-$v/adapter"; done
for v in v2 v2-steps v2-cap30; do run_eval "qwen-5000-$v-heldout" $HO "outputs/qwen-5000-$v/adapter"; done
run_eval qwen-base-5000-v2-test $V2 ""
run_eval qwen-base-5000-v2-heldout $HO ""

# Matched comparisons against the baseline (same test set, generation config and revision).
for set in test heldout; do
  for v in v2 v2-steps v2-cap30; do
    a=outputs/eval-qwen-base-5000-v2-$set/metrics.json; b=outputs/eval-qwen-5000-$v-$set/metrics.json
    [ -f "$a" ] && [ -f "$b" ] && "$PY" -m finance_qlora compare "$a" "$b" \
      --output "reports/qwen-5000-$v-$set-comparison.json" --no-progress > /dev/null 2>&1 \
      && echo "compared $v $set"
  done
done
echo "eval done $(date +%T)"
