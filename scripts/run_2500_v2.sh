#!/bin/bash
# Full experiment on finance_best_available_2500 (prepared by scripts/build_5000_v2.py):
#   3 models (qwen, phi, gemma) x 3 training variants (v2, v2-steps, v2-cap15), then evaluation
#   of every adapter and each untuned baseline on
#     test     data/processed/finance-2500-v2*/test.jsonl          (211 rows, identical across variants)
#     heldout  data/processed/finance-2500-v2*-heldout/test.jsonl  (78 rows, unseen answer templates)
#   and baseline-vs-adapter comparisons. Existing outputs are skipped, never overwritten.
# Model-only checkpoints are deleted after each run (the best adapter is kept in adapter/).
# Phi/Gemma use micro-batch 4 for the longer step-by-step answers (effective batch stays 8).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=data/processed
export HF_HUB_OFFLINE=1
mkdir -p outputs reports
MODELS=${MODELS:-"qwen phi gemma"}
VARIANTS="v2 v2-steps v2-cap15"

for m in $MODELS; do
  for v in $VARIANTS; do
    run="outputs/$m-2500-$v"
    if [ -e "$run" ]; then echo "SKIP train $m $v (exists)"; continue; fi
    extra=()
    if [ "$v" = v2-steps ] && [ "$m" != qwen ]; then extra=(--batch-size 4); fi
    echo "TRAIN $m $v $(date +%T)"
    "$PY" -X utf8 -m finance_qlora train --model "$m" --data "$D/finance-2500-$v" --run-dir "$run" \
      "${extra[@]}" --no-progress --progress-log "reports/$m-2500-$v-progress.jsonl" > "$run.log" 2>&1 \
      || { echo "  FAILED train $m $v (see $run.log)"; continue; }
    [ -f "$run/adapter/adapter_config.json" ] && rm -r -- "$run/checkpoints"
    "$PY" -c "import json; r=json.load(open('$run/run.json')); m=r['metrics']; print('  done', r['optimizer_steps'], 'steps', round(r['training_seconds']), 's, eval_loss', round(m.get('eval_loss', 0), 4), ', peak', round(r['peak_reserved_gib'], 2), 'GiB')"
  done
done
echo "train done $(date +%T)"; df -h . | tail -1

evaluate() {  # model name data adapter-or-empty
  local m=$1 name=$2 data=$3 adapter=$4 out="outputs/eval-$2"
  if [ -e "$out" ]; then echo "SKIP eval $name (exists)"; return; fi
  if [ -n "$adapter" ] && [ ! -f "$adapter/adapter_config.json" ]; then echo "SKIP eval $name (no adapter)"; return; fi
  local extra=(); [ -n "$adapter" ] && extra=(--adapter "$adapter")
  for batch in 8 4; do
    echo "EVAL $name batch $batch $(date +%T)"
    if "$PY" -X utf8 -m finance_qlora evaluate --model "$m" --data "$data" --split test \
        --max-new-tokens 256 --warmup-batches 1 --batch-size $batch "${extra[@]}" --output "$out" \
        --no-progress --progress-log "reports/eval-$name-progress.jsonl" > "$out.log" 2>&1; then
      return
    fi
    if grep -q "out of memory" "$out.log" && [ $batch = 8 ]; then
      echo "  OOM at batch 8; retrying with 4"; rm -r -- "$out"; continue
    fi
    echo "  FAILED eval $name (see $out.log)"; return
  done
}

for m in $MODELS; do
  for v in $VARIANTS; do
    evaluate "$m" "$m-2500-$v-test" "$D/finance-2500-$v" "outputs/$m-2500-$v/adapter"
    evaluate "$m" "$m-2500-$v-heldout" "$D/finance-2500-$v-heldout" "outputs/$m-2500-$v/adapter"
  done
  evaluate "$m" "$m-base-2500-test" "$D/finance-2500-v2" ""
  evaluate "$m" "$m-base-2500-heldout" "$D/finance-2500-v2-heldout" ""
  for set in test heldout; do
    for v in $VARIANTS; do
      a="outputs/eval-$m-base-2500-$set/metrics.json"; b="outputs/eval-$m-2500-$v-$set/metrics.json"
      [ -f "$a" ] && [ -f "$b" ] && { "$PY" -m finance_qlora compare "$a" "$b" \
        --output "reports/$m-2500-$v-$set-comparison.json" --no-progress > /dev/null 2>&1 \
        || echo "  COMPARE FAILED $m $v $set"; }
    done
  done
done
echo "all done $(date +%T)"; df -h . | tail -1
