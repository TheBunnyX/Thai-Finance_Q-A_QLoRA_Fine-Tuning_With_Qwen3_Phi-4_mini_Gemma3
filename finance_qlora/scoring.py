"""Reference-based checks. These are not a factual finance judge."""
import math
import re
from collections import Counter

from .data import normalize


def _pattern(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty regex")
    try:
        result = re.compile(value)
    except re.error as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc
    if result.groups != 1:
        raise ValueError(f"{name} requires exactly one capturing group")
    return result


def validate_evaluation(annotation):
    if annotation is None:
        return
    if not isinstance(annotation, dict):
        raise ValueError("evaluation must be an object")
    if "type" in annotation and annotation["type"] != "numeric":
        raise ValueError("Supported evaluation.type is numeric")
    if annotation.get("type") == "numeric":
        for field, default in (("value", None), ("atol", .01), ("rtol", 0.0)):
            value = annotation.get(field, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"evaluation.{field} must be a finite number")
            if field != "value" and value < 0:
                raise ValueError("Numeric tolerances must be nonnegative")
        _pattern(annotation.get("pattern"), "evaluation.pattern")
        if annotation.get("select", "only") not in ("only", "last"):
            raise ValueError("evaluation.select must be 'only' or 'last'")
    if "unit" in annotation:
        unit = annotation["unit"]
        if not isinstance(unit, dict):
            raise ValueError("evaluation.unit must be an object")
        _pattern(unit.get("pattern"), "evaluation.unit.pattern")
        accepted = unit.get("accepted")
        if not isinstance(accepted, list) or not accepted or any(
                not isinstance(s, str) or not s.strip() for s in accepted):
            raise ValueError("evaluation.unit.accepted must be a non-empty string list")
    if "answerable" in annotation and not isinstance(annotation["answerable"], bool):
        raise ValueError("evaluation.answerable must be boolean")


def score_answer(prediction, row):
    expected = row["answer"]
    left, right = Counter(normalize(prediction)), Counter(normalize(expected))
    overlap = sum((left & right).values())
    denominator = sum(left.values()) + sum(right.values())
    result = {"exact_match": normalize(prediction) == normalize(expected),
              "character_f1": 2 * overlap / denominator if denominator else 1.0}
    annotation = row.get("evaluation") or {}
    validate_evaluation(annotation)
    if annotation.get("type") == "numeric":
        matches = re.findall(annotation["pattern"], prediction)
        # "only" (default) needs exactly one match; "last" takes the final match, which
        # suits step-by-step answers whose last number+unit is the stated result.
        chosen = matches[-1] if matches and annotation.get("select") == "last" else (
            matches[0] if len(matches) == 1 else None)
        try:
            parsed = float(chosen.replace(",", "").replace("−", "-")) if chosen is not None else None
        except ValueError:
            parsed = None
        # Do not write NaN/Infinity into the result JSON.
        parsed = parsed if parsed is not None and math.isfinite(parsed) else None
        result["numeric_correct"] = parsed is not None and math.isclose(
            parsed, annotation["value"], abs_tol=annotation.get("atol", .01),
            rel_tol=annotation.get("rtol", 0.0))
        result["numeric_extracted"] = parsed
    if "unit" in annotation:
        unit = annotation["unit"]
        matches = re.findall(unit["pattern"], prediction)
        extracted = matches[0] if len(matches) == 1 else None
        result["unit_correct"] = extracted is not None and normalize(extracted) in {
            normalize(s) for s in unit["accepted"]}
        result["unit_extracted"] = extracted
    if "numeric_correct" in result and "unit_correct" in result:
        result["numeric_and_unit_correct"] = result["numeric_correct"] and result["unit_correct"]
    return result


def summarize(records):
    result = {"rows": len(records),
              "exact_match": sum(r["scores"]["exact_match"] for r in records) / len(records),
              "character_f1": sum(r["scores"]["character_f1"] for r in records) / len(records)}
    for field, name in (("numeric_correct", "numeric"), ("unit_correct", "unit"),
                        ("numeric_and_unit_correct", "numeric_and_unit")):
        values = [r["scores"][field] for r in records if field in r["scores"]]
        result[name + "_rows"] = len(values)
        result[name + "_accuracy"] = sum(values) / len(values) if values else None
    return result
