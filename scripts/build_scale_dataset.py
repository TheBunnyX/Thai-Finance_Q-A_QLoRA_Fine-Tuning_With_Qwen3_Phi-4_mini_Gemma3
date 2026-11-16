"""Build finance-scale-2500-v1 from the nested finance_best_available bundle.

The bundle sets are nested (1250 < 2500 < 5000 < 10000), so every older
evaluation split lives inside the 2500 training pool. To keep new evaluations
clean by construction:

    train       = the 2500 set
    validation  = sample of (5000 minus 2500)   -> choose epochs / rank
    test        = sample of (10000 minus 5000)  -> score each final variant once

Related rows (shared normalized Q/A or answer prefix) are merged into groups and a
group is owned by the earliest pool in train > validation > test, so no group spans
splits. Calculation rows in validation/test get numeric annotations only after the
independent recomputation in finance_qlora.calc_verify agrees with the reference.

Usage: python scripts/build_scale_dataset.py --bundle <dir> --output <new empty dir>
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finance_qlora import calc_verify  # noqa: E402
from finance_qlora.data import digest, group_indices, load_prepared, normalize, write_json  # noqa: E402
from finance_qlora.scoring import score_answer, validate_evaluation  # noqa: E402

VALIDATION = {"calc_per_topic": 8, "scenario": 100}
TEST = {"calc_per_topic": 10, "scenario": 160}


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def template(text):
    return re.sub(r"\d[\d,]*(?:\.\d+)?", "#", text)


def stratified(rows, per_topic, scenario_total, rng):
    """Equal rows per calculation topic; scenario rows spread evenly over topics."""
    calc = collections.defaultdict(list)
    scenario = collections.defaultdict(list)
    for row in rows:
        (calc if row["category"] == "calculation" else scenario)[row["topic"]].append(row)
    picked = []
    for topic in sorted(calc):
        pool = [r for r in calc[topic] if r["verification"] == "match"]
        picked += rng.sample(pool, min(per_topic, len(pool)))
    topics = sorted(scenario)
    for pool in scenario.values():
        rng.shuffle(pool)
    while len([r for r in picked if r["category"] != "calculation"]) < scenario_total and any(scenario.values()):
        for topic in topics:
            if scenario[topic] and len([r for r in picked if r["category"] != "calculation"]) < scenario_total:
                picked.append(scenario[topic].pop())
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("reports/scale-v1-audit.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory must be empty: {args.output}")

    b = args.bundle
    full, meta = read(b / "finance_best_available_10000.jsonl"), read(b / "finance_best_available_10000_sources.jsonl")
    keys = {n: {r["question"] for r in read(b / f"finance_best_available_{n}.jsonl")} for n in (2500, 5000)}
    assert len(full) == len(meta) == 10000 and all(m["id"] == i + 1 for i, m in enumerate(meta))
    assert keys[2500] < keys[5000] < {r["question"] for r in full}, "bundle sets are expected to be nested"

    rows, verification = [], collections.defaultdict(collections.Counter)
    for item, info in zip(full, meta):
        pool = "train" if item["question"] in keys[2500] else (
            "validation" if item["question"] in keys[5000] else "test")
        row = {"id": f"b{info['id']:05d}", "question": item["question"], "answer": item["answer"],
               "topic": info["topic"], "category": info["category"], "pool": pool, "verification": "n/a"}
        if info["category"] == "calculation":
            status, expected, reference = calc_verify.verify_row(info["topic"], item["question"], item["answer"])
            row["verification"] = status
            verification[info["topic"]][status] += 1
            if status == "match":
                row["evaluation"] = calc_verify.annotation(info["topic"], reference)
        rows.append(row)

    # Merge related rows, then give each group to the earliest pool that touches it.
    rank = {"train": 0, "validation": 1, "test": 2}
    owner_of, dropped = {}, collections.Counter()
    for group in group_indices(rows):
        owner = min((rows[i]["pool"] for i in group), key=rank.get)
        for i in group:
            owner_of[i] = owner
            if rows[i]["pool"] != owner:
                dropped[f"{rows[i]['pool']}_row_dropped_for_{owner}_group"] += 1
    eligible = {name: [r for i, r in enumerate(rows) if owner_of[i] == name and r["pool"] == name]
                for name in rank}
    rng = random.Random(args.seed)
    splits = {"train": eligible["train"],
              "validation": stratified(eligible["validation"], VALIDATION["calc_per_topic"], VALIDATION["scenario"], rng),
              "test": stratified(eligible["test"], TEST["calc_per_topic"], TEST["scenario"], rng)}

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_splits = {}
    for name, records in splits.items():
        rng.shuffle(records)
        path = args.output / f"{name}.jsonl"
        clean = [{k: v for k, v in r.items() if k not in ("pool", "verification")} for r in records]
        for row in clean:
            if "evaluation" in row:
                validate_evaluation(row["evaluation"])
                # Self-check: the reference must pass its own annotation.
                assert score_answer(row["answer"], row)["numeric_correct"], row["id"]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean), encoding="utf-8")
        manifest_splits[name] = {"file": path.name, "rows": len(clean), "sha256": digest(path)}

    train_templates = {template(r["question"]) for r in splits["train"]}
    audit = {
        "calculation_verification_by_topic": {t: dict(c) for t, c in sorted(verification.items())},
        "calculation_verification_total": dict(sum((c for c in verification.values()), collections.Counter())),
        "groups_dropped_to_prevent_leakage": dict(dropped),
        "eligible_rows": {k: len(v) for k, v in eligible.items()},
        "split_composition": {n: dict(collections.Counter(r["category"] == "calculation" and "calculation" or "scenario/other"
                                                           for r in rs)) for n, rs in splits.items()},
        "annotated_rows": {n: sum("evaluation" in r for r in rs) for n, rs in splits.items()},
        "question_template_overlap_with_train": {
            n: {"rows": len(rs), "rows_with_template_seen_in_train": sum(template(r["question"]) in train_templates for r in rs)}
            for n, rs in splits.items() if n != "train"},
        "tolerances": {"atol": calc_verify.ATOL, "rtol": calc_verify.RTOL, "select": "last"},
        "limitations": [
            "Recomputation proves reference numbers follow the generator formulas, not that the finance advice is correct.",
            "All rows except 70 page-referenced seeds are synthetic; the test split contains none of those 70.",
            "Test/validation numbers are new but question and answer templates are the same as in train.",
            "Numeric scoring takes the last number+unit in an answer; an answer that appends another number can be misscored.",
        ],
    }
    manifest = {"format_version": 1, "source": str(b), "source_sha256": digest(b / "finance_best_available_10000.jsonl"),
                "seed": args.seed, "splits": manifest_splits,
                "design": "train=2500 set; validation from 5000 minus 2500; test from 10000 minus 5000; groups never span splits",
                "limitations": audit["limitations"]}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.audit, audit)
    load_prepared(args.output)  # re-checks checksums and cross-split leakage
    print(json.dumps({k: audit[k] for k in ("eligible_rows", "split_composition", "annotated_rows",
                                             "groups_dropped_to_prevent_leakage",
                                             "question_template_overlap_with_train")}, ensure_ascii=False, indent=1))
    print({n: v["rows"] for n, v in manifest_splits.items()})


if __name__ == "__main__":
    main()
