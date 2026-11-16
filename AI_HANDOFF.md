# AI handoff — Fine_Tune_LLM_Finance

Last updated **17 November 2026** (Asia/Bangkok). Check the real state of processes and files before starting work.

## Current state (Linux, 2500 experiment)

Training moved from Windows to **Linux** (Ubuntu 26.04, RTX 5070 12 GB, driver 595.99). The latest, complete
experiment is **3 models × 3 data variants on `finance_best_available_2500`**. Results, tables and charts are in
[README.md](README.md), `docs/results-2500-v2.json` and `docs/images/`.

- **Python:** 3.12.10 in `.venv` (created with `uv`; there is no `pip` inside, so use `uv pip install --python .venv/bin/python …`).
  Always run `.venv/bin/python`. Dependencies are pinned in `requirements-training-linux.txt`
  (torch 2.11.0+cu128, Unsloth 2026.11.3, transformers 5.17.0, TRL 1.15.0, PEFT 0.21.2, bitsandbytes 0.50.2, Triton 3.6.0);
  `matplotlib==3.10.7` was added for the charts. `gcc` is required (Triton compiles kernels).
- **Node.js:** 22 from apt, only for `web/`. `npm test` cannot run on it (the tests import `.ts` files and this Node build has no
  TypeScript support); `npm run typecheck` and `npm run dev` work.
- **Data:** `Dataset/finance_best_available_2500.jsonl` is prepared by `scripts/build_5000_v2.py` into `data/processed/finance-2500-v2`,
  `-v2-steps`, `-v2-cap15` and their `-heldout` twins (the held-out test is 78 rows of answer templates never seen in training).
  Test files are byte-identical across variants. `finance-1250-*` and `finance-5000-*` are older sets and are not used for the current results.
- **Runs:** `outputs/<qwen|phi|gemma>-2500-<v2|v2-steps|v2-cap15>/` (adapter, `run.json` with config / metrics / loss history) and
  `outputs/eval-*/` (`metrics.json`, `predictions.jsonl`). Comparisons against the base model are in `reports/*-2500-*-comparison.json`.
  Re-run everything with `scripts/run_2500_v2.sh` (it skips outputs that already exist).
- **Training config:** `configs/training.json` (max length 512, LoRA r16/alpha32, LR 1e-4, AdamW 8-bit, 1 epoch, effective batch 8) and
  per-model `train_batch_size` in `configs/models.json` (Qwen 4, Phi 8, Gemma 8; `v2-steps` for Phi/Gemma uses 4). Chosen from
  a micro-batch benchmark (see README). Gemma runs with eager attention on this GPU.
- **Playground:** `finance_qlora/playground.py` (Python API, port 8000) and `web/` (Next.js, port 3000), both loopback only.
  A Python API and a dev server may still be running from an earlier session: check ports 8000 / 3000 and `nvidia-smi` before starting anything.
  The API keeps one model on the GPU; release it (UI button or `POST /unload`) before training or evaluating.

### Rules that still apply

- Reply to the user in **Thai**. Do the authorized work without asking again unnecessarily.
- Never overwrite the original datasets, adapters or earlier evaluation results; use new output directories.
- Never print or send the Hugging Face token; the user is already logged in (`hf auth login`).
- Only one GPU model job at a time (training, evaluation or playground inference). Do not start one while another is running.
- Thai text inside `finance_qlora/calculator.py`, `calc_verify.py`, `scripts/build_scale_steps_dataset.py`, the tests and the datasets is
  **data** (keywords, units, training targets); do not translate it.
- Pin Hugging Face downloads to the revisions in `configs/models.json`. Pre-download with `HF_XET_HIGH_PERFORMANCE=1 .venv/bin/hf download …`;
  Gemma is gated and needs the licence accepted on the website.

### Known gaps

- The earlier `reports/` directory (Windows results) was emptied by mistake and cannot be recovered; the archive below refers to files that no longer exist.
  If the Windows machine still has them, copy them back.
- Test sets are small (72 numeric rows, 3 per topic; 78 held-out rows), each setting ran once (seed 42), and the data is template-generated.
  No human review has been done.
- All nine adapters fail the exponentiation topics (`future_value_annuity`, `inflation_adjustment` at 0%); rely on the calculator cross-check for those.
- `v2-steps` answers are sometimes cut at the 256-token evaluation budget; re-evaluate with a larger budget (for example 384) before drawing conclusions.

### Next

- Re-evaluate `v2-steps` with `--max-new-tokens 384`.
- Human review of a blind sample; add source / group metadata and an independent holdout before judging real financial correctness.
- Check the UI visually in a browser, and run `npm test` on a Node with TypeScript support.

---

# Archive: earlier Windows round (10 November 2026)

> Everything below was written for the **Windows** round (`finance-1250-v1`, `finance-dev-1250-v3`) and is kept for history,
> translated from the original Thai. Its paths, commands and report files do not apply to the current Linux setup.

Snapshot **10 November 2026, about 08:29 Asia/Bangkok**; process and file status were checked again before work started.

## Latest update — work in progress (at that time)

- Fixed "Invalid origin" by checking the real Host of the loopback; localhost / 127.0.0.1 pass and external origins are rejected
- The web page supports Markdown / KaTeX and Thai formulas; output budget starts at 512, maximum 768; build and tests pass
- Phi/Gemma tokenizers passed on the whole dataset, a 20-step smoke run and an adapter reload succeeded, but the smoke answers were still factually wrong
- An AI-run blinded review of 2 questions / 8 answers plus targeted checks; see `reports/blind-review-ai-sample-1250-v1.md`; this is not a human review of all 500
- Arithmetic in 50 rows of the original train set matched the question / reference and annotations were added; the other 950 rows were not fact-checked
- `data/processed/finance-dev-1250-v3` has train 806 / validation 100 / test 100, with 6 AI-written examples added to train only.
  Validation/test come from the original train set that Qwen had already seen, so they are not an independent external holdout; the original test is kept aside
- `scripts/train_dev_v3.ps1` trained all four models for 1 epoch / 101 steps; see `outputs/*-dev-v3/run.json`
- `scripts/evaluate_dev_v3.ps1` evaluated validation before and after for all four models; the new test split was not used.
  Combined report `reports/dev-v3-validation.md` / `.json`; per-model comparisons in `reports/*-dev-v3-validation-comparison.json`
- A closer check of 4 numeric answers is in `reports/phi-dev-v3-numeric-spotcheck.md`: Phi SFT was wrong on 3/4 and Gemma SFT on 4/4.
  The sample is tiny and does not summarize overall accuracy; the regex has false negatives when an answer is correct but formatted differently
- The web UI could select all four dev-v3 adapters; Phi/Gemma were tested through the real Next.js API and the GPU was released
- The playground showed a waiting state during tracked training/evaluation so it would not use the GPU at the same time

## Goals and requirements (Windows round)

- Fine-tune Thai/English Finance Q&A on an **RTX 5070 12 GB**, Windows, **Python 3.12.10 in `.venv`**
- Unsloth + TRL SFT + QLoRA 4-bit NF4 / double quantization / BF16, with progress and ETA
- Target models: Qwen3 4B Instruct 2507, Phi-4 mini, Gemma 3 4B
- The user wanted speed and low resource use, before/after measurement, and a Next.js page for trying the trained models
- Reply to the user in Thai; carry on with authorized work without asking again unnecessarily
- Older text mentioning Python 3.10 was replaced by Python 3.12.10

Workspace: `<project root on the old Windows machine>`
Always use `.\.venv\Scripts\python.exe`; never overwrite the original dataset, adapters or earlier evaluation results.
Never print the Hugging Face token; the user is logged in.
Do not start a GPU model job alongside another training/evaluation, and do not alter Windows Application Control.

## Status (Windows round)

1. Qwen passed its smoke test and trained fully for **1 epoch / 125 optimizer steps** on 1,000 train and 125 validation rows
2. **Baseline and SFT** of each model were evaluated on the full 125-row test set, plus a separate 10-row timing profile
3. Reports `reports/results-1250-v1.md` / `.json`, comparison files and a **500-answer** blind review were created
4. The Next.js playground and Python API were built; `npm run build` and **56** Python tests passed
5. A question went through the Next.js proxy to Qwen SFT on a real GPU: stage/token/done streaming succeeded
7. Checking the UI with Computer Use failed because the system could not confirm the browser's current URL, so the layout was not checked visually
8. Human review had not been scored; Phi/Gemma and the other models finished development-round training and validation evaluation

## Dataset (Windows round)

Original `Dataset/finance_best_available_1250.jsonl`, 1,250 rows in JSONL `question` / `answer` form.
Prepared `data/processed/finance-best-1250-v1`: train 1,000 / validation 125 / test 125, seed 42.
Grouping uses normalized Q/A, source/group when present, and the answer prefix to reduce leakage.
Schema/heuristics and checksums passed, but the financial facts of every row and semantic/source leakage were not confirmed.

```text
source SHA256   a47536db5fed62b8f7269c9dabd67d82a272e0caab039b7a3dae2d7b8146fba8
manifest SHA256 7ebe9ef57e443e26a7a78c04fee6e934761708a9e68852af058b44bcb8bcf5e1
test SHA256     3033e14128ad54dad73da8b225faa88e1d9cea1cda54dff83138234e09128c3a
```

There was no source/group/language/topic metadata and no numeric/unit/answerability annotations, so metrics that depend on annotations had to be N/A.
See `reports/finance_best_available_1250_audit.json` and the manifest in the prepared directory.

## Environment and training (Windows round)

PyTorch 2.11.0+cu128, Unsloth/unsloth_zoo 2026.11.3, transformers 5.17.0, trl 1.15.0,
peft 0.21.2, bitsandbytes 0.50.2, triton-windows 3.6.0.post26, pandas 2.3.3, pyarrow 21.0.0.
NF4 forward/backward, the Triton kernel, CUDA/BF16 and `pip check` passed on the real machine.
See `requirements-training-windows.txt`, `reports/requirements-runtime-windows.txt`,
`reports/environment-training.json`, `reports/training-kernels.json`.
Some DLLs had been blocked by Windows Application Control, so official compatible wheels were pinned instead of changing the OS policy.

`configs/training.json` then: max sequence 1024, LoRA r16/alpha32/dropout 0, LR 1e-4, AdamW 8-bit,
1 epoch, default batch 1 / accumulation 8; full runs used batch 2 / accumulation 4, effective 8, gradient checkpointing,
a token cache, length grouping, dynamic padding and model-only checkpoints (no optimizer state, so training cannot be resumed exactly).
Model revisions pinned in `configs/models.json`:

```text
Qwen  cdbee75f17c01a7cc42f958dc650907174af0554
```

| Model | Run | Train incl. validation/checkpoint | Total incl. load/save | Peak CUDA allocated | Adapter |
|---|---|---:|---:|---:|---:|
| Qwen | `outputs/qwen-1250-v1` | 239.792 s | 271.514 s | 3.009 GiB | 143.626 MB |

`run.json` had status complete; validation loss is not used to rank models across tokenizers.
Unsloth printed a `num_items_in_batch` warning during validation, so gradient-accumulation equivalence is not certified.

## Evaluation and what a person must check (Windows round)

Main test: 125 rows per variant, greedy, batch 4, max_new_tokens 256, 1 warmup batch.
Profile: 10 rows chosen with seed 42 and identical across variants, batch 1, TTFT streamer, 1 warmup.

| Model | Character F1 before → after | Exact match before → after | Baseline answers hitting the 256-token ceiling |
|---|---:|---:|---:|
| Qwen | 0.595 → 0.913 | 0% → 38.4% | 110/125 |

After SFT, 0/125 answers hit the ceiling; the baseline may be cut. Text scores are not financial correctness.
F1/EM do not confirm formulas, units, financial correctness or the absence of fabricated content.
See `reports/results-1250-v1.md` and `.json` for details, speed, TTFT and VRAM;
comparisons are in `reports/qwen-1250-v1-comparison{,-profile}.json`.

`outputs/review-1250-v1/blind_review.csv` holds 125 × 4 = 500 answers.
Reviewers should rate independently following `RUBRIC.md` and `docs/metrics.md`; keep `review_key.json` away from the reviewers.
There were no real reviewer scores: do not invent any, and do not tune hyperparameters on the test set and then call it an untouched test.

## Next.js playground (Windows round)

`web/` offered the original Qwen run and the dev-v3 Qwen/Phi/Gemma runs, baseline/SFT, a 32–768-token budget, example questions,
streaming, metrics, stop, JSON export and unloading the model; one model on the GPU at a time.
The local Python API is `finance_qlora/playground.py`, using completed runs in `outputs/*-1250-v1` and `outputs/*-dev-v3`.
Each question is a single independent turn, with the native chat template, greedy decoding and a 1024 context.
If prompt + budget exceeded the context an error was shown; the question was never truncated silently.

```powershell
# From the workspace; install dependencies the first time
powershell -ExecutionPolicy Bypass -File scripts/start_playground.ps1 -Install
# Afterwards
powershell -ExecutionPolicy Bypass -File scripts/start_playground.ps1
```

Open `http://127.0.0.1:3000` and press Ctrl+C in the launcher to stop the services.
If a port is in use, inspect the process first and never stop a user's process without knowing where it came from.
Starting the API and Next.js separately, limits and troubleshooting are in `docs/playground.md`.
The services bind to loopback only, have no production authentication and must not be exposed to a network.
The first model load may contact Hugging Face; questions go to the Python API on the same machine.

A GPU lock rejected overlapping requests and checked the evaluation-runner heartbeat before inference;
it is not a general scheduler, so it was still forbidden to start another train/evaluate job at the same time.
First text from the streamer is not the same as the evaluation TTFT; if the budget was reached, the page warned that the answer might be incomplete.
Test services that might still have been running from the earlier session: the Python API on port 8000 and Next.js on port 3000;
check their status before starting a new set.

## Tests and code map (Windows round)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd web
npm.cmd run build
```

56 tests passed and the production build of Next 16.4.6 / React 19.3.1 passed at that snapshot.
In another terminal at the root, with the web UI and the API running, a real GPU smoke test could be run:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/check_playground.py --model qwen --variant sft
```

This is a functional smoke test and does not certify financial correctness.
Qwen SFT answered the Current Ratio question for real; the smoke test set a budget of 64 and the answer was cut at the limit, which the UI must report.
Computer Use stopped because it could not confirm the browser URL, so never claim the UI was checked visually.

| Path | Role |
|---|---|
| `finance_qlora/data.py` | audit, grouped split, checksum/leakage checks |
| `finance_qlora/modeling.py` | Unsloth NF4, native template / tokenization / masking |
| `finance_qlora/train.py`, `evaluate.py` | SFT and matched evaluation |
| `finance_qlora/performance.py`, `telemetry.py`, `progress.py` | caching, batching, metrics, progress/ETA |
| `finance_qlora/review.py`, `scoring.py` | blind review and text / annotated numeric / unit metrics |
| `finance_qlora/playground.py` | local streaming inference API, GPU lock, single-model cache |
| `web/app/page.tsx`, `globals.css`, `api/` | Next.js UI and local proxy |
| `scripts/start_playground.ps1`, `scripts/check_playground.py` | start the web UI/API and the GPU smoke test |
| `scripts/evaluate_models.ps1`, `scripts/summarize_results.py` | evaluation runner and combined report |
| `docs/metrics.md`, `docs/playground.md`, `reports/readiness.md` | metric definitions, how to use the web UI, status |

## Next steps (Windows round)

- Test the UI for real once a browser surface that can verify the URL is available
- Check stopping during generation, switching model variant, unloading and the responsive UI as needed
- Have an expert review factual correctness, formulas, numbers, units and unsupported claims from the blind review
- Add annotations / source metadata and an independent holdout before using this to judge real financial correctness
- Try different batch sizes separately from this benchmark to measure throughput / VRAM without changing the earlier comparison
- If the user wants Phi/Gemma, check access / compatibility and run the tokenizer + smoke test before full training
