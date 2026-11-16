"""Blind human review export/import. No model-based factual scores are invented."""
import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from .data import digest, write_json
from .performance import json_hash
from .progress import Progress, track

SCORES = {"financial_correctness": 2, "completeness": 2, "clarity": 2,
          "unit_correct": 1, "unsupported_claim": 1, "insufficient_information_handling": 1}
TEXT_FIELDS = ("question", "reference", "prediction")
CSV_FIELDS = ["item_id", "reviewer_id", *TEXT_FIELDS, "expected_answerable",
              *SCORES, "evidence", "notes"]


def _write_csv(path, rows):
    # utf-8-sig allows opening Thai text correctly in Windows Excel.
    # JSONL is authoritative: spreadsheet formula escaping is handled reversibly below.
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _csv_text(value):
    # Do not execute dataset/model output as a formula when the CSV is opened in Excel.
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else value


def export_review(prediction_paths, output, seed=42, limit=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty review output directory")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    experiments, expected = [], None
    for path in track(prediction_paths, "Load predictions for blind review", unit="experiments"):
        path = Path(path).resolve()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_id = {}
        for row in rows:
            for field in ("id", *TEXT_FIELDS):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Invalid {field} in {path}")
            if row["id"] in by_id:
                raise ValueError(f"Duplicate prediction id in {path}: {row['id']}")
            by_id[row["id"]] = row
        if not by_id:
            raise ValueError(f"No predictions: {path}")
        identity = {key: {field: row.get(field) for field in ("question", "reference", "evaluation")}
                    for key, row in by_id.items()}
        if expected is not None and identity != expected:
            raise ValueError("All experiments must contain the same questions, references and annotations")
        expected = identity
        if any(item["path"] == str(path) for item in experiments):
            raise ValueError("Duplicate prediction file")
        experiments.append({"path": str(path), "sha256": digest(path), "rows": by_id})
    if not experiments:
        raise ValueError("At least one predictions file is required")
    rng = random.Random(seed)
    ids = sorted(expected)
    rng.shuffle(ids)
    if limit is not None:
        ids = ids[:limit]
    items = [(experiment, item_id) for experiment in range(len(experiments)) for item_id in ids]
    rng.shuffle(items)
    key_items, csv_rows = {}, []
    for index, (experiment, source_id) in enumerate(track(items, "Prepare blinded review items", unit="answers"), 1):
        row = experiments[experiment]["rows"][source_id]
        item_id = f"item-{index:06d}"
        text = {field: _csv_text(row[field]) for field in TEXT_FIELDS}
        answerable = (row.get("evaluation") or {}).get("answerable")
        answerable_text = "" if answerable is None else str(int(answerable))
        exported = dict(text, expected_answerable=answerable_text)
        key_items[item_id] = {"experiment": f"experiment-{experiment+1}", "source_id": source_id,
                              "text_sha256": json_hash(exported), "answerable": answerable}
        csv_rows.append(dict.fromkeys(CSV_FIELDS, "") | {"item_id": item_id, **exported})
    output.mkdir(parents=True, exist_ok=True)
    with Progress("Write blind review CSV"):
        _write_csv(output / "blind_review.csv", csv_rows)
    key = {"format_version": 1, "seed": seed, "items": key_items,
           "experiments": {f"experiment-{i+1}": {"path": item["path"], "sha256": item["sha256"]}
                           for i, item in enumerate(experiments)},
           "note": "Keep this mapping hidden from reviewers until scoring is complete."}
    write_json(output / "review_key.json", key)
    (output / "RUBRIC.md").write_text(
        "# Blind review rubric\n\n"
        "Fill reviewer_id with a stable pseudonymous ID. Leave unknown/not-applicable scores blank.\n"
        "Do not edit item_id, question, reference, prediction or expected_answerable.\n"
        "For another reviewer copy the blank CSV before filling scores. Do not share review_key.json.\n\n"
        "- financial_correctness: 0 wrong, 1 partly correct, 2 fully correct. Verify facts independently; reference may contain errors.\n"
        "- completeness: 0 missing essential answer, 1 partly complete, 2 complete.\n"
        "- clarity: 0 difficult to understand, 1 understandable, 2 clear.\n"
        "- unit_correct: 0 wrong/missing required unit, 1 correct; blank when no unit is required.\n"
        "- unsupported_claim: 1 contains an invented/unsupported claim, 0 none identified. If 1, cite the claim in evidence.\n"
        "- insufficient_information_handling: only when expected_answerable=0; 1 correctly identifies missing input without guessing, 0 fails.\n"
        "- evidence/notes: explain errors or disagreements and record authoritative sources where needed.\n\n"
        "Higher scores are better except unsupported_claim. expected_answerable=1 means answerable; 0 means insufficient inputs; blank means unlabeled.\n"
        "Some text may start with an apostrophe to prevent spreadsheet formulas; preserve it.\n"
        "Blindness removes file/model labels, not self-identification within the answer.\n",
        encoding="utf-8")
    return {"experiments": len(experiments), "questions_per_experiment": len(ids),
            "review_items": len(items), "csv": str(output / "blind_review.csv"),
            "private_key": str(output / "review_key.json")}


def score_review(rating_paths, key_path):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    votes, seen = defaultdict(lambda: defaultdict(list)), set()
    reviewers = set()
    for path in track(rating_paths, "Read reviewer ratings", unit="files"):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(CSV_FIELDS).issubset(reader.fieldnames or []):
                raise ValueError("Review CSV is missing required columns")
            for row in reader:
                if any(row[field] is None for field in CSV_FIELDS):
                    raise ValueError("Review CSV row is missing cells")
                item_id = row["item_id"]
                if item_id not in key["items"]:
                    raise ValueError(f"Unknown review item: {item_id}")
                item = key["items"][item_id]
                exported = {field: row[field] for field in (*TEXT_FIELDS, "expected_answerable")}
                if json_hash(exported) != item["text_sha256"]:
                    raise ValueError(f"Review text/annotation was modified: {item_id}")
                supplied = {field: row[field].strip() for field in SCORES if row[field].strip()}
                if not supplied:
                    continue
                reviewer = row["reviewer_id"].strip()
                if not reviewer:
                    raise ValueError(f"reviewer_id is required: {item_id}")
                if (reviewer, item_id) in seen:
                    raise ValueError(f"Duplicate reviewer/item rating: {reviewer}/{item_id}")
                seen.add((reviewer, item_id))
                reviewers.add(reviewer)
                if supplied.get("unsupported_claim") == "1" and not row["evidence"].strip():
                    raise ValueError(f"Unsupported claim requires evidence: {item_id}")
                if "insufficient_information_handling" in supplied and item["answerable"] is not False:
                    raise ValueError(f"Insufficient-information score needs answerable=false: {item_id}")
                for field, value in supplied.items():
                    if value not in {str(i) for i in range(SCORES[field] + 1)}:
                        raise ValueError(f"Invalid {field} score {value!r}: {item_id}")
                    votes[item_id][field].append(int(value))
    results = {}
    for experiment, metadata in key["experiments"].items():
        items = [item_id for item_id, item in key["items"].items() if item["experiment"] == experiment]
        dimensions = {}
        for field, maximum in SCORES.items():
            item_votes = [votes[i][field] for i in items if votes[i][field]]
            averages = [sum(v) / len(v) for v in item_votes]
            pairs = [pair for values in item_votes for pair in itertools.combinations(values, 2)]
            dimensions[field] = {
                "reviewed_items": len(averages), "ratings": sum(map(len, item_votes)),
                "mean": sum(averages) / len(averages) if averages else None,
                "normalized_mean": sum(averages) / len(averages) / maximum if averages else None,
                "reviewer_pairs": len(pairs),
                "exact_agreement": sum(a == b for a, b in pairs)/len(pairs) if pairs else None,
            }
        correctness = [votes[i]["financial_correctness"] for i in items if votes[i]["financial_correctness"]]
        results[experiment] = {**metadata, "total_items": len(items),
            "financial_correctness_accuracy": sum(sum(v == 2 for v in values)/len(values)
                                                   for values in correctness)/len(correctness)
            if correctness else None,
            "unsupported_claim_rate": dimensions["unsupported_claim"]["mean"],
            "insufficient_information_handling_accuracy": dimensions["insufficient_information_handling"]["mean"],
            "dimensions": dimensions}
    return {"experiments": results, "reviewers": len(reviewers),
            "key_sha256": digest(key_path),
            "aggregation": "Average reviewer votes within each item, then average items equally per dimension. "
                           "Missing ratings are excluded, not zero. Accuracy means fully correct (score 2). "
                           "Agreement is descriptive pairwise exact agreement, not chance-adjusted kappa."}
