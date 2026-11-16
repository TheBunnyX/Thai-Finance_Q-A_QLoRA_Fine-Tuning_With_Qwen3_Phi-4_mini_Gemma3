"""Audit narrowly specified numeric templates in the OLD train split only.

Creates a derived development input, never edits the original. No labels are
inferred for rows outside the two fully recognized calculation templates.
"""
import json
import re
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finance_qlora.data import load_prepared, write_json

SOURCE = ROOT / "data/processed/finance-best-1250-v1"
DESTINATION = ROOT / "data/curated/finance-train-1250-v2.jsonl"
REPORT = ROOT / "reports/finance-train-1250-v2-curation.json"
TEMPLATES = (
    ("current_ratio", re.compile(r"^บริษัทมีสินทรัพย์หมุนเวียน ([\d,]+) บาท และหนี้สินหมุนเวียน ([\d,]+) บาท Current Ratio เป็นเท่าไร\?$"),
     Decimal(1), "เท่า", r"(?<![\d,.])(\d+(?:\.\d+)?)\s*เท่า", "https://corporatefinanceinstitute.com/resources/accounting/current-ratio-vs-quick-ratio/"),
    ("roe", re.compile(r"^บริษัทมีกำไรสุทธิ ([\d,]+) บาท และส่วนของผู้ถือหุ้นเฉลี่ย ([\d,]+) บาท ROE เท่าไร\?$"),
     Decimal(100), "%", r"(?<![\d,.])(\d+(?:\.\d+)?)\s*%", "https://corporatefinanceinstitute.com/resources/accounting/what-is-return-on-equity-roe/"),
)


def curate(row):
    result = {k: v for k, v in row.items() if k != "split_group"}
    for topic, question_re, multiplier, unit, pattern, reference_url in TEMPLATES:
        match = question_re.fullmatch(row["question"])
        if not match:
            continue
        numerator, denominator = (Decimal(value.replace(",", "")) for value in match.groups())
        if not denominator:
            raise ValueError(f"Zero denominator in {row['id']}")
        rounded = (numerator / denominator * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        # Only edit the single explicitly marked approximate result, never source numbers.
        expected_result = re.search(r"≈\s*([\d.]+)\s*(เท่า|%)", row["answer"])
        printed_inputs = re.search(r"=\s*([\d,]+)\s*÷\s*([\d,]+)", row["answer"])
        if (not expected_result or expected_result.group(2) != unit or not printed_inputs
                or tuple(Decimal(value.replace(",", "")) for value in printed_inputs.groups()) != (numerator, denominator)):
            return result, {"id": row["id"], "topic": topic, "status": "needs_manual_review"}
        original = Decimal(expected_result.group(1))
        if original != rounded:
            start, end = expected_result.span(1)
            result["answer"] = row["answer"][:start] + str(rounded) + row["answer"][end:]
        # Evaluation regex intentionally accepts a single explicitly unit-marked result only.
        result["evaluation"] = {"type": "numeric", "value": float(rounded), "atol": 0.0049,
                                "rtol": 0.0, "pattern": pattern,
                                "unit": {"pattern": r"(เท่า)" if unit == "เท่า" else r"(%)",
                                         "accepted": [unit]}}
        result["language"] = "th"
        result["topic"] = topic
        result["reference_url"] = reference_url  # Formula reference, not original row provenance.
        return result, {"id": row["id"], "topic": topic,
                        "status": "corrected" if original != rounded else "verified_arithmetic",
                        "original_value": str(original), "verified_value": str(rounded)}
    return result, None


def main():
    if DESTINATION.exists() or REPORT.exists():
        raise ValueError("Refusing to overwrite a previous curation output")
    splits, manifest = load_prepared(SOURCE)
    curated, checks = [], []
    for row in splits["train"]:
        edited, check = curate(row)
        curated.append(edited)
        if check:
            checks.append(check)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("x", encoding="utf-8", newline="\n") as stream:
        for row in curated:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(REPORT, {"original_dataset_sha256": manifest["source_sha256"],
                        "source_split": "original train only", "rows": len(curated),
                        "counts": dict(Counter(c["status"] for c in checks)), "checks": checks,
                        "limitations": "Only two exact question templates; formulas sourced, row provenance unknown. "
                                       "Other rows have not been financially verified. The derived split is not an independent external holdout."})
    print(json.dumps({"output": str(DESTINATION), "report": str(REPORT), "rows": len(curated),
                      "counts": dict(Counter(c["status"] for c in checks))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
