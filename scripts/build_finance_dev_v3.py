"""Add explicitly synthetic, traceable examples to the v2 training split only."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finance_qlora.data import digest, group_indices, load_prepared, normalize, read_rows, write_json

PREPARED = ROOT / "data/processed/finance-dev-1250-v2"
EXTRAS = ROOT / "data/examples/finance_rounding_v2.jsonl"
OUTPUT = ROOT / "data/processed/finance-dev-1250-v3"


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError("Refusing to overwrite the development dataset")
    splits, old_manifest = load_prepared(PREPARED)
    indexed, errors = read_rows(EXTRAS)
    if errors or not indexed:
        raise ValueError(f"Invalid synthetic examples: {errors}")
    examples = []
    for line, original in indexed:
        row = dict(original)
        pair = (normalize(row["question"]), normalize(row["answer"]))
        row["id"] = hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
        row["source_id"] = f"synthetic-rounding-v2-{line:03d}"
        row["split_group"] = f"synthetic-rounding-v2-{line:03d}"
        examples.append(row)
    old_splits, _ = load_prepared(ROOT / "data/processed/finance-best-1250-v1")
    protected_ids = {row["id"] for name in ("validation", "test") for row in old_splits[name]}
    if any(row["id"] in protected_ids for row in examples):
        raise ValueError("Synthetic example duplicates a held-out original row")
    splits["train"].extend(examples)
    all_rows = [row for name in ("train", "validation", "test") for row in splits[name]]
    owners = [name for name in ("train", "validation", "test") for row in splits[name]]
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate row ID")
    groups = group_indices(all_rows)
    if any(len({owners[i] for i in group}) != 1 for group in groups):
        raise ValueError("Content/source grouping would leak across development splits")
    # Check against original held-out content using the same exact grouping rules.
    protected = [r for name in ("validation", "test") for r in old_splits[name]]
    combined = all_rows + protected
    if any(min(group) < len(all_rows) <= max(group) for group in group_indices(combined)):
        raise ValueError("Content/source grouping would leak into the original held-out splits")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in splits.items():
        path = OUTPUT / f"{name}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = {"file": path.name, "rows": len(rows), "sha256": digest(path)}
    manifest = {"format_version": 1, "source": str(PREPARED), "source_sha256": digest(PREPARED / "manifest.json"),
                "synthetic_source": str(EXTRAS), "synthetic_source_sha256": digest(EXTRAS),
                "seed": old_manifest["seed"], "independent_groups": len(groups),
                "duplicate_pairs_removed": 0, "splits": files,
                "limitations": "Development split derives from old training data previously seen by Qwen. "
                               "Not an independent external holdout; semantic/source leakage remains possible. "
                               "Six AI-created training examples require expert review."}
    write_json(OUTPUT / "manifest.json", manifest)
    load_prepared(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "splits": {n: len(rows) for n, rows in splits.items()},
                      "synthetic_training_rows": len(examples)}, indent=2))


if __name__ == "__main__":
    main()
