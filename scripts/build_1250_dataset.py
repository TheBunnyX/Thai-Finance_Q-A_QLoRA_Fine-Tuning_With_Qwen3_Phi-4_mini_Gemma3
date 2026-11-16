"""Build finance-1250-split-v1 from Dataset/finance_best_available_1250.jsonl.

80/10/10 split stratified by topic. Related rows (shared normalized Q/A, answer prefix)
are merged into groups that never span splits. Topic/category come from the bundle's
10000 sources file (the 1250 set is a subset). Calculation rows in validation/test get a
numeric annotation only when the independent recomputation agrees with the reference.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finance_qlora import calc_verify  # noqa: E402
from finance_qlora.data import digest, group_indices, load_prepared, write_json  # noqa: E402
from finance_qlora.scoring import score_answer, validate_evaluation  # noqa: E402


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("Dataset/finance_best_available_1250.jsonl"))
    p.add_argument("--bundle", type=Path, default=Path.home() / "Downloads/finance_best_available_bundle")
    p.add_argument("--output", type=Path, default=Path("data/processed/finance-1250-split-v1"))
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit(f"Output directory must be empty: {a.output}")
    full = read(a.bundle / "finance_best_available_10000.jsonl")
    meta = read(a.bundle / "finance_best_available_10000_sources.jsonl")
    info = {f["question"]: m for f, m in zip(full, meta)}
    rows = []
    for i, item in enumerate(read(a.input)):
        m = info[item["question"]]
        row = {"id": f"d{i:04d}", "question": item["question"], "answer": item["answer"],
               "topic": m["topic"], "category": m["category"]}
        if m["category"] == "calculation":
            status, _, ref = calc_verify.verify_row(m["topic"], item["question"], item["answer"])
            row["_status"] = status
            if status == "match":
                row["evaluation"] = calc_verify.annotation(m["topic"], ref)
        rows.append(row)

    rng = random.Random(a.seed)
    by_topic = collections.defaultdict(list)
    for g in group_indices(rows):
        by_topic[rows[g[0]]["topic"]].append(g)
    ratio = {"train": .8, "validation": .1, "test": .1}
    splits = {k: [] for k in ratio}
    for topic in sorted(by_topic):
        groups = by_topic[topic]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        total = sum(len(g) for g in groups)
        have = dict.fromkeys(ratio, 0)
        for g in groups:
            name = max(ratio, key=lambda k: ratio[k] * total - have[k])
            have[name] += len(g)
            splits[name] += [rows[i] for i in g]

    a.output.mkdir(parents=True, exist_ok=True)
    files, annotated = {}, {}
    for name, recs in splits.items():
        rng.shuffle(recs)
        clean = [{k: v for k, v in r.items() if k != "_status"} for r in recs]
        for r in clean:
            if "evaluation" in r:
                validate_evaluation(r["evaluation"])
                assert score_answer(r["answer"], r)["numeric_correct"], r["id"]
        path = a.output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean), encoding="utf-8")
        files[name] = {"file": path.name, "rows": len(clean), "sha256": digest(path)}
        annotated[name] = sum("evaluation" in r for r in clean)
    write_json(a.output / "manifest.json", {
        "format_version": 1, "source": str(a.input), "source_sha256": digest(a.input), "seed": a.seed,
        "splits": files, "design": "80/10/10 stratified by topic; related rows grouped, never split across",
        "limitations": "Template questions repeat across splits (numbers differ); small validation/test."})
    load_prepared(a.output)
    print({n: v["rows"] for n, v in files.items()}, "annotated:", annotated)
    print({n: dict(collections.Counter(r["category"] for r in recs)) for n, recs in splits.items()})


if __name__ == "__main__":
    main()
