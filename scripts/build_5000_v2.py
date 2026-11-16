"""Build finance-5000-v2 (+ held-out-template test and a per-template capped train) from
Dataset/finance_best_available_5000.jsonl. The source file is only read.

Why not plain `prepare`: the bundle is template-generated (24 calculation topics x 65 rows, and
~148 answer templates once numbers are masked), so a random split gives no numeric annotations,
no topic balance, and validation/test that mostly repeat training templates.

Outputs (all pass load_prepared's checksum and cross-split leakage checks):
  finance-5000-v2          80/10/10 split stratified by stratum (calculation topic, else answer
                           template), related rows grouped; verified calculation rows carry a
                           numeric `evaluation` annotation; rows in finance-1250-split-v1 train are
                           forced into train so 1250-trained models can be compared fairly.
  finance-5000-v2-heldout  same train/validation; test = rows from answer templates that are absent
                           from every v2 split (generalisation to unseen templates).
  finance-5000-v2-cap<N>   same validation/test; train keeps at most N rows per stratum.
The step-by-step variant is built from finance-5000-v2 by scripts/build_scale_steps_dataset.py.
"""
import argparse
import collections
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finance_qlora import calc_verify, calculator  # noqa: E402
from finance_qlora.data import digest, group_indices, load_prepared, write_json  # noqa: E402
from finance_qlora.scoring import score_answer, validate_evaluation  # noqa: E402

MASK = re.compile(r"[-−]?\d[\d,]*(?:\.\d+)?")
RATIO = {"train": .8, "validation": .1, "test": .1}


def template(text):
    return MASK.sub("#", text.strip())


def calc_topic(question):
    """Calculation topic via the playground's keyword + shape rules, requiring the exact arity."""
    low = question.casefold()
    numbers = calc_verify.question_numbers(question)
    for topic, _label, _unit, _places, required, forbidden in calculator.RULES:
        if not all(k in low for k in required) or any(k in low for k in forbidden):
            continue
        shape = calculator.SHAPES.get(topic)
        if shape and not shape.search(question):
            continue
        if len(numbers) == calc_verify.TOPICS[topic][2]:
            return topic
    return None


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_split(directory, name, rows):
    path = directory / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"file": path.name, "rows": len(rows), "sha256": digest(path)}


def check_annotations(rows):
    for r in rows:
        if "evaluation" in r:
            validate_evaluation(r["evaluation"])
            assert score_answer(r["answer"], r)["numeric_correct"], r["id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("Dataset/finance_best_available_5000.jsonl"))
    p.add_argument("--exclude-train-of", type=Path, default=Path("data/processed/finance-1250-split-v1"),
                   help="Rows in this prepared set's train split are kept out of v2 validation/test")
    p.add_argument("--output", type=Path, default=Path("data/processed/finance-5000-v2"))
    p.add_argument("--heldout-templates", type=int, default=6)
    p.add_argument("--heldout-rows-per-template", type=int, default=40)
    p.add_argument("--cap", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    heldout_dir = a.output.with_name(a.output.name + "-heldout")
    cap_dir = a.output.with_name(f"{a.output.name}-cap{a.cap}")
    for d in (a.output, heldout_dir, cap_dir):
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"Output directory must be empty: {d}")
    rng = random.Random(a.seed)
    source_sha = digest(a.input)

    # 1) Label every row: calculation topic (verified) or answer template.
    rows, status_counts = [], collections.Counter()
    answer_templates = {}
    for i, item in enumerate(read_jsonl(a.input)):
        row = {"id": f"f5k-{i:04d}", "question": item["question"], "answer": item["answer"]}
        topic = calc_topic(item["question"])
        if topic:
            status, _, ref = calc_verify.verify_row(topic, item["question"], item["answer"])
            status_counts[status] += 1
            row.update(topic=topic, category="calculation", stratum=f"calc:{topic}", calc_status=status)
            if status == "match":
                row["evaluation"] = calc_verify.annotation(topic, ref)
        else:
            key = template(item["answer"])
            answer_templates.setdefault(key, f"tpl-{len(answer_templates):03d}")
            row.update(topic=answer_templates[key], category="scenario",
                       stratum=f"tpl:{answer_templates[key]}")
        rows.append(row)

    # 1b) Answer templates that share a related-row group (same question, or same 80-char answer
    # prefix with a different ending) are merged into one stratum; otherwise holding out one
    # template would leave its linked rows in train and leak.
    parent = {}

    def find(s):
        parent.setdefault(s, s)
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for g in group_indices(rows):
        first = find(rows[g[0]]["stratum"])
        for i in g[1:]:
            other = find(rows[i]["stratum"])
            if other != first:
                parent[other] = first
    merged_templates = sum(1 for s in {r["stratum"] for r in rows} if find(s) != s)
    for r in rows:
        r["cap_key"] = r["stratum"]  # original template/topic: the train cap applies per template
        r["stratum"] = find(r["stratum"])
        if r["category"] == "scenario":
            r["topic"] = r["stratum"].split(":", 1)[1]

    # 2) Hold out whole answer templates (non-calculation only, so every formula stays trainable).
    excluded_q = {r["question"].strip() for r in read_jsonl(a.exclude_train_of / "train.jsonl")}
    by_stratum = collections.defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    candidates = sorted(s for s, rs in by_stratum.items()
                        if all(r["category"] == "scenario" for r in rs)
                        and len(rs) >= a.heldout_rows_per_template)
    heldout_strata = set(rng.sample(candidates, a.heldout_templates))
    heldout = []
    for s in sorted(heldout_strata):
        pool = [r for r in by_stratum[s] if r["question"].strip() not in excluded_q]
        rng.shuffle(pool)
        heldout += pool[:a.heldout_rows_per_template]
    kept = [r for r in rows if r["stratum"] not in heldout_strata]

    # 3) Grouped split stratified by stratum; groups touching the 1250 train are forced to train.
    groups_by_stratum = collections.defaultdict(list)
    for g in group_indices(kept):
        groups_by_stratum[kept[g[0]]["stratum"]].append(g)
    splits = {k: [] for k in RATIO}
    forced_rows = 0
    for stratum in sorted(groups_by_stratum):
        groups = groups_by_stratum[stratum]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        total = sum(len(g) for g in groups)
        have = dict.fromkeys(RATIO, 0)
        forced = [g for g in groups if any(kept[i]["question"].strip() in excluded_q for i in g)]
        free = [g for g in groups if g not in forced]
        for g in forced:
            have["train"] += len(g)
            splits["train"] += [kept[i] for i in g]
            forced_rows += len(g)
        for g in free:
            name = max(RATIO, key=lambda k: RATIO[k] * total - have[k])
            have[name] += len(g)
            splits[name] += [kept[i] for i in g]

    def clean(r):
        return {k: v for k, v in r.items() if k not in ("stratum", "calc_status", "cap_key")}

    design = ("80/10/10 grouped split stratified by calculation topic / answer template; "
              f"{a.exclude_train_of.name} train rows forced into train; "
              f"{a.heldout_templates} answer templates held out entirely")
    base_manifest = {"format_version": 1, "source": str(a.input), "source_sha256": source_sha,
                     "seed": a.seed, "design": design}

    # finance-5000-v2
    a.output.mkdir(parents=True, exist_ok=True)
    files, cleaned = {}, {}
    for name, recs in splits.items():
        rng.shuffle(recs)
        cleaned[name] = [clean(r) for r in recs]
        check_annotations(cleaned[name])
        files[name] = write_split(a.output, name, cleaned[name])
    stats = {name: {"rows": len(recs),
                    "calculation": sum(r["category"] == "calculation" for r in recs),
                    "numeric_annotated": sum("evaluation" in r for r in recs),
                    "strata": len({r["topic"] for r in recs})} for name, recs in cleaned.items()}
    write_json(a.output / "manifest.json", dict(
        base_manifest, splits=files, split_stats=stats,
        calculation_verification=dict(status_counts),
        answer_templates=len(answer_templates), templates_merged_into_linked_strata=merged_templates,
        heldout_templates=sorted(s.split(":", 1)[1] for s in heldout_strata),
        rows_dropped_with_heldout_templates=len(rows) - len(kept) - len(heldout),
        rows_forced_to_train_from_exclusion=forced_rows,
        limitations="Template-generated data: validation/test share templates with train (numbers "
                    "differ); use the -heldout test for unseen templates. Topic labels for "
                    "non-calculation rows are answer templates, not human labels."))
    load_prepared(a.output)

    # finance-5000-v2-heldout: same train/validation, test = held-out templates.
    heldout_dir.mkdir(parents=True, exist_ok=True)
    h_files = {}
    for name in ("train", "validation"):
        shutil.copyfile(a.output / f"{name}.jsonl", heldout_dir / f"{name}.jsonl")
        h_files[name] = dict(files[name])
    rng.shuffle(heldout)
    h_test = [clean(r) for r in heldout]
    h_files["test"] = write_split(heldout_dir, "test", h_test)
    write_json(heldout_dir / "manifest.json", dict(
        base_manifest, splits=h_files, design=design + "; test = held-out answer templates only",
        heldout_templates=sorted(s.split(":", 1)[1] for s in heldout_strata),
        note=f"Evaluate with --data {heldout_dir} --split test; train/validation equal {a.output.name}."))
    load_prepared(heldout_dir)

    # finance-5000-v2-cap<N>: train capped per stratum, validation/test unchanged.
    cap_dir.mkdir(parents=True, exist_ok=True)
    per = collections.defaultdict(list)
    for r in splits["train"]:
        per[r["cap_key"]].append(clean(r))
    capped = []
    for topic in sorted(per):
        group = per[topic][:]
        rng.shuffle(group)
        capped += group[:a.cap]
    rng.shuffle(capped)
    c_files = {"train": write_split(cap_dir, "train", capped)}
    for name in ("validation", "test"):
        shutil.copyfile(a.output / f"{name}.jsonl", cap_dir / f"{name}.jsonl")
        c_files[name] = dict(files[name])
    write_json(cap_dir / "manifest.json", dict(
        base_manifest, splits=c_files, design=design + f"; train capped at {a.cap} rows per stratum",
        train_rows_before_cap=len(cleaned["train"])))
    load_prepared(cap_dir)

    print(json.dumps({"v2": stats, "heldout_test_rows": len(h_test), "cap_train_rows": len(capped),
                      "calculation_verification": dict(status_counts),
                      "forced_to_train": forced_rows,
                      "dropped_with_heldout_templates": len(rows) - len(kept) - len(heldout)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
