"""Dataset operations require only the Python standard library."""
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from .progress import Progress, track


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def digest(path):
    checksum = hashlib.sha256()
    with Progress(f"Checksum {Path(path).name}", total=Path(path).stat().st_size, unit="bytes") as progress, Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
            progress.advance(len(block))
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iter_lines(path):
    with Progress(f"Read JSONL {Path(path).name}", total=Path(path).stat().st_size, unit="bytes") as progress:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            for line in stream:
                yield line
                progress.advance(len(line.encode("utf-8")))


def read_rows(path):
    rows, errors = [], []
    for line_number, line in enumerate(_iter_lines(path), 1):
        try:
            def unique_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"Duplicate JSON key: {key}")
                    result[key] = value
                return result

            row = json.loads(line, object_pairs_hook=unique_keys,
                             parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            if not isinstance(row, dict):
                raise ValueError("Each line must be a JSON object")
            for key in ("question", "answer"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{key} must be a non-empty string")
            for key in ("id", "source_id", "group_id", "topic", "language"):
                if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                    raise ValueError(f"Optional {key} must be a non-empty string")
            from .scoring import validate_evaluation
            validate_evaluation(row.get("evaluation"))
            rows.append((line_number, row))
        except (ValueError, TypeError) as exc:
            errors.append({"line": line_number, "error": str(exc)})
    return rows, errors


def quality_flags(row):
    q, a = row["question"], row["answer"]
    flags = []
    quote = re.search("“(.*)”", q)
    if quote and normalize(a).startswith(normalize(quote[1].removesuffix("…").removesuffix("..."))):
        flags.append("question_contains_answer_prefix")
    if "…" in q or "..." in q:
        flags.append("question_ellipsis")
    if any(0xE000 <= ord(c) <= 0xF8FF or c == "\ufffd" for c in q + a):
        flags.append("extraction_character")
    if re.search(r"(จะช่วย|ได้แก่|และ|หรือ|เพื่อ|โดย|ของ|กับ)$", a.strip()):
        flags.append("possibly_incomplete_answer")
    if a.count("(") != a.count(")"):
        flags.append("unbalanced_parentheses")
    if len(a.strip()) < 20:
        flags.append("very_short_answer")
    return flags


def audit(path):
    rows, errors = read_rows(path)
    answers = Counter(normalize(r["answer"]) for _, r in rows)
    questions = Counter(normalize(r["question"]) for _, r in rows)
    pairs = Counter((normalize(r["question"]), normalize(r["answer"])) for _, r in rows)
    question_answers = defaultdict(set)
    for _, row in rows:
        question_answers[normalize(row["question"])].add(normalize(row["answer"]))
    flagged, flag_counts = [], Counter()
    for line, row in track(rows, "Audit dataset quality", unit="rows"):
        flags = quality_flags(row)
        flag_counts.update(flags)
        if flags:
            flagged.append({"line": line, "flags": flags})
    return {
        "source": str(path), "sha256": digest(path), "valid_rows": len(rows),
        "errors": errors, "unique_questions": len(questions), "unique_answers": len(answers),
        "duplicate_pairs_extra": sum(n - 1 for n in pairs.values()),
        "questions_with_multiple_answers": sum(len(a) > 1 for a in question_answers.values()),
        "languages": dict(Counter(r.get("language", "unspecified") for _, r in rows)),
        "source_ids_present": sum(bool(r.get("source_id")) for _, r in rows),
        "group_ids_present": sum(bool(r.get("group_id")) for _, r in rows),
        "answer_group_sizes": dict(sorted(Counter(answers.values()).items())),
        "quality_flag_counts": dict(flag_counts), "flagged_rows": flagged,
        "note": "Heuristics require human review; not a financial fact check or tokenizer check.",
    }


def group_indices(rows):
    """Union all shared normalized Q/A, source/group IDs, and long answer prefixes."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(track(rows, "Group related questions and sources", unit="rows")):
        keys = [("q", normalize(row["question"])), ("a", normalize(row["answer"]))]
        for key in ("source_id", "group_id"):
            if row.get(key):
                keys.append((key, row[key]))
        answer = normalize(row["answer"])
        if len(answer) >= 80:
            keys.append(("answer_prefix80", answer[:80]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return list(groups.values())


def prepare(source, output, seed=42):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    indexed, errors = read_rows(source)
    if errors:
        raise ValueError(f"Fix {len(errors)} invalid rows first; run audit for line numbers")
    rows, used_ids = [], {}
    for _, original in track(indexed, "Normalize dataset and assign IDs", unit="rows"):
        pair = (normalize(original["question"]), normalize(original["answer"]))
        # Preserve all provenance while grouping; deduplicate only after assigning groups.
        row = dict(original)
        row["id"] = row.get("id") or hashlib.sha256(
            json.dumps(pair, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
        # Same id is fine only for an exact duplicate of the pair that first used it.
        if used_ids.setdefault(row["id"], pair) != pair:
            raise ValueError(f"Duplicate id with different contents: {row['id']}")
        rows.append(row)
    groups = group_indices(rows)
    if len(groups) < 3:
        raise ValueError("Need at least 3 independent content/source groups for three splits")
    random.Random(seed).shuffle(groups)
    # Keep largest groups first to avoid leaving an oversized group to the end.
    groups.sort(key=len, reverse=True)
    splits = {"train": [], "validation": [], "test": []}
    targets = {"train": len(rows) * .8, "validation": len(rows) * .1, "test": len(rows) * .1}
    for group_number, group in enumerate(track(groups, "Assign independent groups to splits", unit="groups")):
        remaining_groups = len(groups) - group_number
        empty = [key for key, values in splits.items() if not values]
        candidates = empty if remaining_groups == len(empty) else list(splits)
        split = max(candidates, key=lambda key: targets[key] - len(splits[key]))
        for i in group:
            row = dict(rows[i], split_group=f"g{group_number:06d}")
            splits[split].append(row)
    removed = 0
    for name, records in track(splits.items(), "Deduplicate splits", unit="splits"):
        unique, seen_pairs = [], set()
        for row in records:
            key = (normalize(row["question"]), normalize(row["answer"]))
            if key not in seen_pairs:
                seen_pairs.add(key)
                unique.append(row)
            else:
                removed += 1
        random.Random(seed).shuffle(unique)
        splits[name] = unique
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, records in track(splits.items(), "Write prepared splits", unit="files"):
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                        encoding="utf-8")
        files[name] = {"file": path.name, "rows": len(records), "sha256": digest(path)}
    manifest = {
        "format_version": 1, "source": str(source), "source_sha256": digest(source),
        "seed": seed, "independent_groups": len(groups), "duplicate_pairs_removed": removed,
        "splits": files, "quality_flags": dict(Counter(
            flag for _, row in indexed for flag in quality_flags(row))),
        "limitations": "Grouped exact normalized Q/A, answer prefix80 and supplied source/group IDs. "
                        "Semantic duplicates need manual review. Ratios approximate; no stratification.",
    }
    with Progress("Save dataset manifest"):
        write_json(output / "manifest.json", manifest)
    return manifest


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    splits = {}
    for name in ("train", "validation", "test"):
        info = manifest["splits"][name]
        path = directory / info["file"]
        if path.resolve().parent != directory.resolve() or digest(path) != info["sha256"]:
            raise ValueError(f"Split checksum/path mismatch: {name}")
        indexed, errors = read_rows(path)
        if errors or not indexed or len(indexed) != info["rows"]:
            raise ValueError(f"Invalid or empty split: {name}")
        splits[name] = [r for _, r in indexed]
    all_rows = [row for records in splits.values() for row in records]
    owners = [name for name, records in splits.items() for _ in records]
    for group in group_indices(all_rows):
        if len({owners[i] for i in group}) > 1:
            raise ValueError("Content/source leakage across splits")
    return splits, manifest
