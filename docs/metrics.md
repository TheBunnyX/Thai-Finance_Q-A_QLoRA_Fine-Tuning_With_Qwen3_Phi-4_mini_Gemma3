# Metrics and how to use them

Automatic results are kept separate from human review of the content. Text similarity is never
used as a substitute for financial correctness.

## Automatic quality

| Metric | Meaning / rows counted |
|---|---|
| exact_match | The answer equals the reference after Unicode / whitespace / case normalization |
| character_f1 | F1 over the multiset of characters; not a semantic or factual score |
| numeric_accuracy | The number captured by the regex equals `value` within atol/rtol; `numeric_rows` is reported |
| unit_accuracy | The unit captured by the regex is in `accepted`; `unit_rows` is reported |
| numeric_and_unit_accuracy | Both number and unit are correct in the same row; `numeric_and_unit_rows` is reported |

Every score is a 0–1 proportion, also reported per `language` and per `topic`.
Rows without an annotation are not in the denominator, and the metric is `null` when no row
can be measured. In an annotated row, a regex that does not match, or matches more than once,
counts as wrong.

Add an `evaluation` object to a JSONL row (see `data/examples/finance_sample.jsonl`):

```json
{
  "type": "numeric",
  "value": 2.0,
  "pattern": "=\\s*([0-9.]+)\\s*[^0-9\\s]+\\s*$",
  "atol": 0.01,
  "rtol": 0.0,
  "unit": {
    "pattern": "=\\s*[0-9.]+\\s*(เท่า|times|%)\\s*$",
    "accepted": ["เท่า", "times"]
  },
  "answerable": true
}
```

(`เท่า` is the Thai unit "times" used by the ratio answers in the dataset, so it stays Thai.)

If the model answers `= 2 %` the number is right but the unit is wrong, so the joint accuracy is wrong.
This regex expects answers that end in `= value unit`, so it may not match a correct answer written
in another form. Design the regex to accept every acceptable form and check with a person; do not force
every question to use this example. Make sure the number and the unit come from the same result, and
never use a pattern that captures a percentage inside the formula instead of the final answer.
A regex checks the format against the annotation; it does not prove that the formula, or every unit
in the sentence, is correct.

For questions with insufficient information:

```json
{"evaluation":{"answerable":false}}
```

This label marks the rows a person should judge for "does the model say what is missing". Keywords are
never used to guess and count the answer as correct. Questions without the label are not part of this metric.

## Inference time and memory

A normal evaluation:

```bash
.venv/bin/python -m finance_qlora evaluate --model qwen --data data/processed/finance-2500-v2 --batch-size 1 --output outputs/eval-base
```

A separate run for TTFT:

```bash
.venv/bin/python -m finance_qlora evaluate --model qwen --data data/processed/finance-2500-v2 --batch-size 1 --measure-ttft --output outputs/eval-base-ttft
```

- `request_latency_median_seconds`, `request_latency_p95_seconds`: the summed time of the tokenize, generate and text-decode stages of each row; batch size 1 only
- This time excludes model load, dataset I/O, scoring, network and queueing, so it is not the latency of a deployed API
- `batch_latency_median_seconds`, `batch_latency_p95_seconds`: generate + decode per batch, not divided by the number of rows
- `mean_generation_seconds`: amortized generation time per row when batching
- `ttft_median_seconds`, `ttft_p95_seconds`: tokenize + the time until the first generated token is available on the host; needs batch size 1 and `--measure-ttft`
- TTFT is the first token, not the first renderable word; it can be a special token / EOS. Without `--measure-ttft` it is `null`
- `batch_ttft_*`: time to the first generated tokens of a batch, excluding the prepared tokenization time
- `subsequent_tokens_per_second`: tokens after the first, over the time between the first and last token, when TTFT is on (includes the terminal EOS, excludes padding after EOS)
- `tokens_per_second_including_first_run`: throughput including prefill / first-run overhead and the time to send tokens back to the CPU
- `peak_allocated_gib`, `peak_reserved_gib`: peak of the CUDA allocator during evaluation, including the model memory still in use; not the VRAM of every process on the machine
- `peak_process_ram_gib`: the OS process-lifetime peak RSS / working set, including model loading; excludes child processes and system-wide RAM

Measuring TTFT through a streamer adds the cost of sending tokens back to the CPU and can slow generation down.
Use `--warmup-batches 1` to run generation before timing starts; the warmup is not counted in scores or latency.
`warmup_seconds` is reported separately, and `compare` requires the same warmup count.
The default is 0, and warming up one batch does not guarantee that other shapes will have no compile overhead.
For a short profile, add `--limit 10 --selection-seed 42` to both the baseline and the adapter run,
with batch 1 and identical generation settings. The selection hash is recorded and `compare` rejects
results that selected different rows. A normal command without `--limit` still evaluates the whole split.
A 10-row profile is only a small sample, so the p95 is not an accurate latency benchmark.
Do not compare throughput between a run with TTFT on and one with it off.
Current timings include cold / first-generation overhead; there is no separate warmed benchmark.
Percentiles use linear interpolation and report the sample count.
With a batch larger than 1, request latency / TTFT summaries are not shown, so they are not mixed with batch latency.

## Training results and file sizes

In `run.json`:

- `training_seconds`: the `trainer.train` span, including validation / checkpoints and loading the best checkpoint
- `total_seconds`: includes model load, tokenization, training, validation and saving the adapter
- `metrics.training_nonpadding_tokens`, `training_answer_tokens`, `training_padded_tokens`: counted from the training batches that actually contribute to the loss; validation is excluded
- `metrics.training_nonpadding_tokens_per_second_including_eval`, `training_answer_tokens_per_second_including_eval`: token counts divided by `training_seconds`, so validation / checkpoint overhead is in the denominator
- `metrics.training_padding_efficiency`: non-padding / padded tokens; close to 1 means little padding
- `adapter_size_bytes`, `checkpoint_size_bytes`: logical file sizes, not physical disk usage
- `peak_process_ram_gib`, `peak_allocated_gib`, `peak_reserved_gib`: CPU / GPU memory with the scopes described above (the GPU peak spans training until saving finishes)
- `log_history`: train loss every logging step and the validation loss at each evaluation, used for the loss curves

Validation loss is used to choose a checkpoint within a model; it is not used to rank models across tokenizers.

## Blind human review

Once you have real predictions from the baseline and an adapter:

```bash
.venv/bin/python -m finance_qlora review-export outputs/eval-base/predictions.jsonl outputs/eval-sft/predictions.jsonl --limit 100 --output outputs/review-round1
```

It samples the same 100 questions per experiment and shuffles the answer order. The files must contain
matching IDs, questions, references and annotations. Without `--limit` every row is exported.

- Send `blind_review.csv` and `RUBRIC.md` to the reviewers yourself; the tool does not send any text or file to anyone
- Keep `review_key.json` with whoever collects the results, because it maps items back to experiments
- Before reviewing starts, copy the blank CSV separately for each reviewer and fill in a constant `reviewer_id`
- Do not edit item_id, question, reference, prediction or expected_answerable; the importer checks the hashes of these texts
- The CSV is UTF-8 with a BOM so Thai displays correctly in Windows Excel; text that looks like a formula gets an apostrophe prefix, so do not remove it
- Hiding the model name does not remove a self-introduction the model writes in its own answer

| Score column | Range / criteria |
|---|---|
| financial_correctness | 0 wrong, 1 partly correct, 2 fully correct |
| completeness | 0 misses the main substance, 1 partly complete, 2 complete |
| clarity | 0 hard to understand, 1 understandable, 2 clear |
| unit_correct | 0 wrong unit / missing required unit, 1 correct; leave blank if not applicable |
| unsupported_claim | 1 fabricated or unsupported information, 0 none found; if 1, `evidence` is required |
| insufficient_information_handling | only when expected_answerable=0: 1 states what is missing and does not guess, 0 fails to |

Check the reference against evidence too, because a reference can be wrong. If unsure, leave the score
blank and add notes. A blank score is missing, not zero, so read the coverage alongside it.

```bash
.venv/bin/python -m finance_qlora review-score outputs/review-round1/reviewer1.csv outputs/review-round1/reviewer2.csv --key outputs/review-round1/review_key.json --output reports/human_metrics.json
```

The output is split by experiment with the number of ratings / items per dimension:

- Financial correctness accuracy = the share of score-2 answers, averaged over the reviewers within an item, then averaged equally across items
- correctness / completeness / clarity have both a 0–2 mean and a normalized 0–1 mean
- Unsupported claim rate: lower is better; for the other metrics higher is better
- Insufficient-information handling accuracy only covers labelled rows and reviewer scores
- With more than one reviewer per item there is pairwise exact agreement with the number of compared pairs;
  this is descriptive agreement, not Cohen's kappa, and it does not automatically adjudicate disagreeing reviewers
- Duplicate ratings for the same reviewer / item are rejected, and `metrics.json` is never edited automatically

The automatic comparison (`compare`) and the human results (`human_metrics.json`) are stored separately,
so an item that has no reviewer yet never looks like it already has a correctness score.

## Implementation references

- [Transformers generation streamer](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextStreamer)
- [Windows PROCESS_MEMORY_COUNTERS](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters)
