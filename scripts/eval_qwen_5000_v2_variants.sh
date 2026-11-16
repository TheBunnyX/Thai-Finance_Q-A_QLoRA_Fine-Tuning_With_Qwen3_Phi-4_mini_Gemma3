#!/bin/bash
# Variant evaluations: each adapter is evaluated with --data whose train split equals its own
# training split (evaluate's leakage guard). Test files are byte-identical across
#   finance-5000-v2{,-steps,-cap30}            (v2 test, 393 rows)
#   finance-5000-v2{,-steps,-cap30}-heldout    (held-out templates, 238 rows)
# so all results are directly comparable. Then compares every adapter with the baseline.
# Usage: scripts/eval_qwen_5000_v2_variants.sh [file to wait for an "eval done" line]
cd "$(dirname "$0")/.."
PY=.venv/bin/python
if [ -n "$1" ]; then until grep -q "^eval done" "$1" 2>/dev/null; do sleep 30; done; fi
export HF_HUB_OFFLINE=1
BATCH=${BATCH:-8}

run_eval() {  # name data adapter
  local name=$1 data=$2 adapter=$3 out="outputs/eval-$1"
  if [ -e "$out" ]; then echo "SKIP $name (exists)"; return; fi
  echo "EVAL $name $(date +%T)"
  "$PY" -X utf8 -m finance_qlora evaluate --model qwen --data "$data" --split test \
    --max-new-tokens 256 --warmup-batches 1 --batch-size "$BATCH" --adapter "$adapter" \
    --output "$out" --no-progress --progress-log "reports/eval-$name-progress.jsonl" > "$out.log" 2>&1 \
    || echo "  FAILED $name (see $out.log)"
}

D=data/processed
run_eval qwen-5000-v2-steps-test   $D/finance-5000-v2-steps         outputs/qwen-5000-v2-steps/adapter
run_eval qwen-5000-v2-cap30-test   $D/finance-5000-v2-cap30         outputs/qwen-5000-v2-cap30/adapter
run_eval qwen-5000-v2-heldout      $D/finance-5000-v2-heldout       outputs/qwen-5000-v2/adapter
run_eval qwen-5000-v2-steps-heldout $D/finance-5000-v2-steps-heldout outputs/qwen-5000-v2-steps/adapter
run_eval qwen-5000-v2-cap30-heldout $D/finance-5000-v2-cap30-heldout outputs/qwen-5000-v2-cap30/adapter

for set in test heldout; do
  for v in v2 v2-steps v2-cap30; do
    a=outputs/eval-qwen-base-5000-v2-$set/metrics.json; b=outputs/eval-qwen-5000-$v-$set/metrics.json
    if [ -f "$a" ] && [ -f "$b" ]; then
      "$PY" -m finance_qlora compare "$a" "$b" --output "reports/qwen-5000-$v-$set-comparison.json" \
        --no-progress > "reports/qwen-5000-$v-$set-comparison.log" 2>&1 \
        && echo "compared $v $set" || echo "  COMPARE FAILED $v $set (see reports/qwen-5000-$v-$set-comparison.log)"
    fi
  done
done
echo "variants done $(date +%T)"
