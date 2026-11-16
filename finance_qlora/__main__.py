import argparse
import json
import sys
from pathlib import Path

from .data import audit, prepare, write_json
from .progress import Progress, configure


def main():
    parser = argparse.ArgumentParser(description="Finance Q&A dataset and QLoRA workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("audit", help="Validate JSONL and report quality heuristics")
    check.add_argument("input", type=Path)
    check.add_argument("--output", type=Path, default=Path("reports/dataset_audit.json"))
    prep = sub.add_parser("prepare", help="Group and split reviewed data without editing input")
    prep.add_argument("input", type=Path)
    prep.add_argument("--output", type=Path, default=Path("data/processed/finance-v1"))
    prep.add_argument("--seed", type=int, default=42)
    env = sub.add_parser("doctor", help="Check Python, dependencies and GPU")
    env.add_argument("--training", action="store_true")
    env.add_argument("--output", type=Path, default=Path("reports/environment.json"))
    tok = sub.add_parser("tokens", help="Check native chat templates/lengths; tokenizer download only")
    tok.add_argument("--model", choices=["qwen", "phi", "gemma"], required=True)
    tok.add_argument("--data", type=Path, required=True)
    tok.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("train", help="Train one model; first use --max-steps 20")
    fit.add_argument("--model", choices=["qwen", "phi", "gemma"], required=True)
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--run-dir", type=Path, required=True)
    fit.add_argument("--max-steps", type=int)
    fit.add_argument("--batch-size", type=int, help="Micro batch size; keeps effective batch unchanged")
    fit.add_argument("--cache-dir", type=Path, default=Path(".cache/tokens"))
    fit.add_argument("--epochs", type=float, help="Override num_train_epochs for an experiment")
    fit.add_argument("--learning-rate", type=float, help="Override learning_rate for an experiment")
    fit.add_argument("--no-gradient-checkpointing", action="store_true", help="Trade VRAM for speed (experiment)")
    fit.add_argument("--gradient-checkpointing", action="store_true",
                     help="Enable Unsloth gradient checkpointing: trade speed for VRAM (experiment)")
    fit.add_argument("--lora-rank", type=int, help="Override LoRA rank; alpha is set to 2x rank")
    ev = sub.add_parser("evaluate", help="Evaluate baseline or a locally trained adapter")
    ev.add_argument("--model", choices=["qwen", "phi", "gemma"], required=True)
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--split", choices=["validation", "test"], default="test")
    ev.add_argument("--adapter", type=Path)
    ev.add_argument("--max-new-tokens", type=int)
    ev.add_argument("--batch-size", type=int, default=1, help="Try 2 after checking free VRAM")
    ev.add_argument("--measure-ttft", action="store_true", help="Profile first-token timing; adds streamer overhead")
    ev.add_argument("--limit", type=int, help="Optional deterministic profiling subset; omit for full evaluation")
    ev.add_argument("--selection-seed", type=int, default=42)
    ev.add_argument("--warmup-batches", type=int, default=0, help="Generation batches to run before timing (default: 0)")
    ev.add_argument("--output", type=Path, required=True)
    comp = sub.add_parser("compare", help="Compare matched baseline/adapter metrics")
    comp.add_argument("before", type=Path)
    comp.add_argument("after", type=Path)
    comp.add_argument("--output", type=Path, default=Path("reports/comparison.json"))
    export = sub.add_parser("review-export", help="Create blinded CSV and private mapping for human review")
    export.add_argument("predictions", nargs="+", type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--limit", type=int, help="Review the same random question subset for every experiment")
    review = sub.add_parser("review-score", help="Aggregate one or more completed reviewer CSVs")
    review.add_argument("ratings", nargs="+", type=Path)
    review.add_argument("--key", required=True, type=Path)
    review.add_argument("--output", type=Path, default=Path("reports/human_metrics.json"))
    for command in sub.choices.values():
        command.add_argument("--no-progress", action="store_true", help="Hide terminal progress; JSON output is unchanged")
        command.add_argument("--progress-log", type=Path, help="Append structured status/ETA events to a JSONL file")
        command.add_argument("--progress-interval", type=float, default=1.0, help="Minimum seconds between updates (default: 1)")
    args = parser.parse_args()
    try:
        configure(console=not args.no_progress, log=args.progress_log, interval=args.progress_interval)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        with Progress(f"Command {args.command}") as operation:
            code = dispatch(args)
            if code:
                operation.finish("failed", "See errors above")
            return code
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130


def dispatch(args):
    try:
        if args.command == "audit":
            result = audit(args.input)
            write_json(args.output, result)
            print(json.dumps({k: v for k, v in result.items() if k != "flagged_rows"},
                             ensure_ascii=False, indent=2))
            return 1 if result["errors"] else 0
        if args.command == "prepare":
            result = prepare(args.input, args.output, args.seed)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "doctor":
            from .modeling import doctor
            result = doctor(args.training)
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result["errors"] else 0
        elif args.command == "tokens":
            from .modeling import inspect_tokens
            result = inspect_tokens(args.model, args.data)
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if any(s["failures"] for s in result["splits"].values()) else 0
        elif args.command == "train":
            from .train import train
            train(args)
        elif args.command == "evaluate":
            from .evaluate import evaluate
            evaluate(args)
        elif args.command == "compare":
            from .evaluate import compare
            result = compare(args.before, args.after)
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "review-export":
            from .review import export_review
            result = export_review(args.predictions, args.output, args.seed, args.limit)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "review-score":
            from .review import score_review
            result = score_review(args.ratings, args.key)
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
