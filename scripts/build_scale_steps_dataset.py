"""Build finance-scale-steps-v1: scale-2500-v1 with step-by-step working in train calculation answers.

Validation and test files are copied unchanged (same checksums), so scores stay comparable
with the plain-answer models. Only train answers for multi-step topics are rewritten:

    คำนวณทีละขั้น:
    1) ... intermediate value ...
    2) ...
    สรุป: <original verified answer, unchanged>

The original verified answer stays last, so the existing numeric annotation (last
number + unit) still applies, and every rewritten answer is re-checked against it.
Intermediate values are computed in code, not by a model.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finance_qlora import calc_verify  # noqa: E402
from finance_qlora.data import digest, load_prepared, write_json  # noqa: E402
from finance_qlora.scoring import score_answer  # noqa: E402


def f(x, places=2):
    return f"{x:,.{places}f}"


def g(x):
    """Intermediate value with enough digits that the final rounding is reproducible."""
    return f"{x:.6f}".rstrip("0").rstrip(".")


# topic -> lines built from the numbers in the question (same order as calc_verify.TOPICS)
STEPS = {
    "compound_interest": lambda pv, r, n: [
        f"อัตราต่อปี r = {r:g}% = {g(r / 100)}",
        f"ตัวคูณทบต้น (1+r)^n = (1+{g(r / 100)})^{n:g} = {g((1 + r / 100) ** n)}",
        f"มูลค่าอนาคต = {f(pv, 0)} × {g((1 + r / 100) ** n)} = {f(pv * (1 + r / 100) ** n)} บาท"],
    "inflation_adjustment": lambda p, r, n: [
        f"อัตราเงินเฟ้อ r = {r:g}% = {g(r / 100)}",
        f"ตัวคูณ (1+r)^n = (1+{g(r / 100)})^{n:g} = {g((1 + r / 100) ** n)}",
        f"ราคาในอนาคต = {f(p, 0)} × {g((1 + r / 100) ** n)} = {f(p * (1 + r / 100) ** n)} บาท"],
    "future_value_annuity": lambda pmt, r, n: [
        f"อัตราต่อปี r = {r:g}% = {g(r / 100)}",
        f"(1+r)^n = (1+{g(r / 100)})^{n:g} = {g((1 + r / 100) ** n)}",
        f"((1+r)^n − 1) ÷ r = ({g((1 + r / 100) ** n)} − 1) ÷ {g(r / 100)} = {g(((1 + r / 100) ** n - 1) / (r / 100))}",
        f"มูลค่าอนาคต = {f(pmt, 0)} × {g(((1 + r / 100) ** n - 1) / (r / 100))} = {f(pmt * ((1 + r / 100) ** n - 1) / (r / 100))} บาท"],
    "real_return": lambda nom, inf: [
        f"1 + ผลตอบแทน = 1 + {nom:g}/100 = {g(1 + nom / 100)}; 1 + เงินเฟ้อ = 1 + {inf:g}/100 = {g(1 + inf / 100)}",
        f"อัตราส่วน = {g(1 + nom / 100)} ÷ {g(1 + inf / 100)} = {g((1 + nom / 100) / (1 + inf / 100))}",
        f"ผลตอบแทนที่แท้จริง = ({g((1 + nom / 100) / (1 + inf / 100))} − 1) × 100 = {f(((1 + nom / 100) / (1 + inf / 100) - 1) * 100)}%"],
    "dca_cost": lambda a1, n1, a2, n2: [
        f"หน่วยครั้งแรก = {f(a1, 0)} ÷ {n1:g} = {a1 / n1:,.3f}; หน่วยครั้งที่สอง = {f(a2, 0)} ÷ {n2:g} = {a2 / n2:,.3f}",
        f"เงินลงทุนรวม = {f(a1, 0)} + {f(a2, 0)} = {f(a1 + a2, 0)} บาท; หน่วยรวม = {a1 / n1 + a2 / n2:,.3f}",
        f"ต้นทุนเฉลี่ย = {f(a1 + a2, 0)} ÷ {a1 / n1 + a2 / n2:,.3f} = {f((a1 + a2) / (a1 / n1 + a2 / n2))} บาทต่อหน่วย"],
    "gross_margin": lambda s, c: [
        f"กำไรขั้นต้น = {f(s, 0)} − {f(c, 0)} = {f(s - c, 0)} บาท",
        f"สัดส่วน = {f(s - c, 0)} ÷ {f(s, 0)} = {g((s - c) / s)}",
        f"อัตรากำไรขั้นต้น = {g((s - c) / s)} × 100 = {f((s - c) / s * 100)}%"],
    "quick_ratio": lambda ca, inv, cl: [
        f"สินทรัพย์หมุนเวียนหักสินค้าคงเหลือ = {f(ca, 0)} − {f(inv, 0)} = {f(ca - inv, 0)} บาท",
        f"Quick Ratio = {f(ca - inv, 0)} ÷ {f(cl, 0)} = {f((ca - inv) / cl)} เท่า"],
    "mutual_fund_gain": lambda u, b, s: [
        f"ส่วนต่าง NAV = {s:.2f} − {b:.2f} = {s - b:.2f} บาทต่อหน่วย",
        f"กำไร = {f(u, 0)} × {s - b:.2f} = {f(u * (s - b))} บาท"],
}
# Single-division percentage/ratio topics: show the unrounded quotient first.
RATIOS = {
    "capital_gain": (lambda b, s: [f"ส่วนต่างราคา = {s:.2f} − {b:.2f} = {s - b:.2f} บาท",
                                   f"สัดส่วน = {s - b:.2f} ÷ {b:.2f} = {g((s - b) / b)}",
                                   f"ผลตอบแทน = {g((s - b) / b)} × 100 = {f((s - b) / b * 100)}%"]),
    "savings_rate": (lambda inc, sv: [f"สัดส่วน = {f(sv, 0)} ÷ {f(inc, 0)} = {g(sv / inc)}",
                                      f"อัตราการออม = {g(sv / inc)} × 100 = {f(sv / inc * 100)}%"]),
    "debt_service_ratio": (lambda inc, pay: [f"สัดส่วน = {f(pay, 0)} ÷ {f(inc, 0)} = {g(pay / inc)}",
                                             f"ภาระผ่อนหนี้ = {g(pay / inc)} × 100 = {f(pay / inc * 100)}%"]),
    "net_margin": (lambda sales, np_: [f"สัดส่วน = {f(np_, 0)} ÷ {f(sales, 0)} = {g(np_ / sales)}",
                                       f"อัตรากำไรสุทธิ = {g(np_ / sales)} × 100 = {f(np_ / sales * 100)}%"]),
    "roa": (lambda np_, a: [f"สัดส่วน = {f(np_, 0)} ÷ {f(a, 0)} = {g(np_ / a)}",
                            f"ROA = {g(np_ / a)} × 100 = {f(np_ / a * 100)}%"]),
    "roe": (lambda np_, e: [f"สัดส่วน = {f(np_, 0)} ÷ {f(e, 0)} = {g(np_ / e)}",
                            f"ROE = {g(np_ / e)} × 100 = {f(np_ / e * 100)}%"]),
    "dividend_yield": (lambda price, d: [f"สัดส่วน = {d:.2f} ÷ {price:.2f} = {g(d / price)}",
                                         f"Dividend Yield = {g(d / price)} × 100 = {f(d / price * 100)}%"]),
    "pbv_ratio": (lambda p, b: [f"P/BV = {p:.2f} ÷ {b:.2f} = {f(p / b)} เท่า"]),
    "pe_ratio": (lambda p, e: [f"P/E = {p:.2f} ÷ {e:.2f} = {f(p / e)} เท่า"]),
}
STEPS.update(RATIOS)


def rewrite(row):
    topic = row.get("topic")
    if topic not in STEPS or row.get("category") != "calculation":
        return None
    status, _, _ = calc_verify.verify_row(topic, row["question"], row["answer"])
    if status not in ("match", "rounding_noise"):
        return None
    steps = STEPS[topic](*calc_verify.question_numbers(row["question"]))
    body = "\n".join(f"{i}) {line}" for i, line in enumerate(steps, 1))
    return f"คำนวณทีละขั้น:\n{body}\nสรุป: {row['answer']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/processed/finance-scale-2500-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/finance-scale-steps-v1"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory must be empty: {args.output}")
    splits, manifest = load_prepared(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    changed = 0
    train = []
    for row in splits["train"]:
        new = rewrite(row)
        if new:
            if "evaluation" in row:  # the original final statement is kept, so it must still score
                assert score_answer(new, row)["numeric_correct"], row["id"]
            row = dict(row, answer=new)
            changed += 1
        train.append(row)
    path = args.output / "train.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
    files = {"train": {"file": "train.jsonl", "rows": len(train), "sha256": digest(path)}}
    for name in ("validation", "test"):
        shutil.copyfile(args.source / f"{name}.jsonl", args.output / f"{name}.jsonl")
        files[name] = dict(manifest["splits"][name])
        assert digest(args.output / f"{name}.jsonl") == files[name]["sha256"]
    new_manifest = dict(manifest, source=str(args.source), splits=files,
                        design=manifest["design"] + "; train calculation answers rewritten with step-by-step working",
                        steps_rewritten_rows=changed, steps_topics=sorted(STEPS))
    write_json(args.output / "manifest.json", new_manifest)
    load_prepared(args.output)
    print({"rewritten_train_rows": changed, "train_rows": len(train)})
    sample = next(r for r in train if r["answer"].startswith("คำนวณทีละขั้น"))
    print(sample["question"]); print(sample["answer"])


if __name__ == "__main__":
    main()
