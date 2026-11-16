"""Build a concise report from completed training/evaluation artifacts."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finance_qlora.data import write_json
from finance_qlora.evaluate import compare


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def display(value, scale=1):
    return "N/A" if value is None else f"{value * scale:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="1250-v1")
    args = parser.parse_args()
    if not args.round or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in args.round):
        raise ValueError("Invalid round name")
    results = {"round": args.round, "financial_correctness_status": "Requires independent human review", "models": {}}
    lines = [f"# Finance Q&A results — {args.round}", "",
             "One epoch, NF4 + double quantization, BF16, LoRA r=16/alpha=32, effective batch 8.",
             "Text-match scores are not financial correctness, and loss is not used to rank models across tokenizers.", "",
             "## Test set results", "",
             "| Model | Variant | Rows | Character F1 | Exact match | Mean output tokens | Hit token limit | Generation tokens/s | Peak CUDA GiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    profile_lines = ["", "## Timing profile (separate from the full test scores)", "",
                     "Batch 1 with a streamer; the row count per line is shown in the table and the same rows are used for baseline and SFT.",
                     "p95 from a small sample is indicative only, not a production latency SLA.", "",
                     "| Model | Variant | Rows | TTFT median s | TTFT p95 s | Request latency median s | Request latency p95 s |",
                     "|---|---|---:|---:|---:|---:|---:|"]
    training_lines = ["", "## Training resources", "",
                      "Train time includes compile, validation and checkpoints; total also includes model load and saving.", "",
                      "| Model | Steps | Train s | Total s | Peak CUDA GiB | Process peak RAM GiB | Adapter MB |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("qwen",):
        train = read(ROOT / f"outputs/{model}-{args.round}/run.json")
        if train["status"] != "complete":
            raise ValueError(f"Training incomplete: {model}")
        item = {"training": train, "evaluation": {}, "profiles": {}}
        for variant in ("base", "sft"):
            folder = ROOT / f"outputs/eval-{model}-{args.round}-{variant}"
            report = read(folder / "metrics.json")
            predictions = [json.loads(s) for s in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            m = report["metrics"]
            if len(predictions) != m["rows"]:
                raise ValueError(f"Prediction count mismatch: {folder}")
            report["mean_generated_tokens"] = sum(p["generated_tokens"] for p in predictions)/len(predictions)
            item["evaluation"][variant] = report
            lines.append(f"| {model} | {variant} | {m['rows']} | {display(m['character_f1'])} | "
                         f"{display(m['exact_match'])} | {display(report['mean_generated_tokens'])} | {m['hit_generation_limit']} | "
                         f"{display(m['tokens_per_second_including_first_run'])} | {display(m['peak_allocated_gib'])} |")
            profile = read(ROOT / f"outputs/eval-{model}-{args.round}-{variant}-profile/metrics.json")
            item["profiles"][variant] = profile
            pm = profile["metrics"]
            profile_lines.append(f"| {model} | {variant} | {pm['rows']} | {display(pm['ttft_median_seconds'])} | "
                                 f"{display(pm['ttft_p95_seconds'])} | {display(pm['request_latency_median_seconds'])} | "
                                 f"{display(pm['request_latency_p95_seconds'])} |")
        for suffix in ("", "-profile"):
            delta = compare(ROOT / f"outputs/eval-{model}-{args.round}-base{suffix}/metrics.json",
                            ROOT / f"outputs/eval-{model}-{args.round}-sft{suffix}/metrics.json")
            item["comparison" + suffix] = delta
        training_lines.append(f"| {model} | {train['optimizer_steps']} | {display(train['training_seconds'])} | "
                              f"{display(train['total_seconds'])} | {display(train['peak_allocated_gib'])} | "
                              f"{display(train['peak_process_ram_gib'])} | {display(train['adapter_size_bytes'],1e-6)} |")
        results["models"][model] = item
    lines += profile_lines + training_lines + ["", "## Limitations", "",
        "- The main evaluation uses batching; the per-row time is the batch average, not per-request latency.",
        "- The excluded warmup is recorded in the generation config; new shapes can still incur compile overhead.",
        "- The dataset has no numeric/unit/answerability annotations, so those metrics are shown as N/A, not zero.",
        "- Rows without language/topic metadata fall under \"unspecified\".",
        "- Facts, formulas, units, fabricated content and semantic/source leakage still need human review.",
        "- Character F1 / exact match can benefit from length and phrasing close to the reference without proving the answer is more correct.",
        "- Read \"Hit token limit\" together with the scores: a long-answering baseline may be cut at the 256-token ceiling; this compares models under an equal answer budget.",
        "- The human review CSV/RUBRIC is at outputs/review-" + args.round + "; keep review_key.json away from the reviewers.",
        "- These results cover Qwen only; Phi/Gemma were not trained in this round.", ""]
    output = ROOT / f"reports/results-{args.round}"
    write_json(output.with_suffix(".json"), results)
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
