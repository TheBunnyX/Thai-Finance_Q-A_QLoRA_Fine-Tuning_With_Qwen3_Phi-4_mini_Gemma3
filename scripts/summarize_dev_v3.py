"""Summarize matched validation experiments without claiming independent test accuracy."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen", "phi", "gemma")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def main():
    result = {"round": "dev-v3", "scope": "development validation, not independent test", "models": {}}
    lines = ["# Development v3 — validation results", "",
             "The validation set was used for development and checkpoint selection, so it is not an independent test result; the earlier Qwen run had seen this data during training.", "",
             "| Model | Variant | Rows | Character F1 | Exact match | Numeric accuracy (rows) | Hit limit | Tokens/s | Peak CUDA GiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    training = ["", "## Training", "", "| Model | Steps | Train s | Total s | Peak CUDA GiB |",
                "|---|---:|---:|---:|---:|"]
    from finance_qlora.evaluate import compare
    for model in MODELS:
        run = read(ROOT / f"outputs/{model}-dev-v3/run.json")
        if run["status"] != "complete":
            raise ValueError(f"Training incomplete: {model}")
        item = {"training": run, "variants": {}}
        for variant in ("base", "sft"):
            metrics = read(ROOT / f"outputs/eval-{model}-dev-v3-validation-{variant}/metrics.json")
            if metrics["split"] != "validation" or metrics["metrics"]["rows"] != 100:
                raise ValueError(f"Unexpected validation set: {model}/{variant}")
            item["variants"][variant] = metrics
            m = metrics["metrics"]
            lines.append(f"| {model} | {variant} | {m['rows']} | {fmt(m['character_f1'])} | {fmt(m['exact_match'])} | "
                         f"{fmt(m['numeric_accuracy'])} ({m['numeric_rows']}) | {m['hit_generation_limit']} | "
                         f"{fmt(m['tokens_per_second_including_first_run'])} | {fmt(m['peak_allocated_gib'])} |")
        item["comparison"] = compare(ROOT / f"outputs/eval-{model}-dev-v3-validation-base/metrics.json",
                                     ROOT / f"outputs/eval-{model}-dev-v3-validation-sft/metrics.json")
        training.append(f"| {model} | {run['optimizer_steps']} | {fmt(run['training_seconds'])} | "
                        f"{fmt(run['total_seconds'])} | {fmt(run['peak_allocated_gib'])} |")
        result["models"][model] = item
    lines += training + ["", "## Limitations", "",
        "- Character F1 / exact match measure closeness to the reference text, not financial correctness.",
        "- Numeric metrics cover only annotated rows and use a strict regex; a correct answer in a different or repeated format may be miscounted.",
        "- The 256-token budget can cut answers, especially for the baseline; read \"Hit limit\" together with the scores.",
        "- Batch size was 2 for Phi/Gemma and 4 for Qwen, so cross-model throughput was not measured at equal batch.",
        "- Humans still need to check formulas, numbers, units, facts, sources and an independent holdout.", ""]
    target = ROOT / "reports/dev-v3-validation"
    target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    target.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target.with_suffix(".md"))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
