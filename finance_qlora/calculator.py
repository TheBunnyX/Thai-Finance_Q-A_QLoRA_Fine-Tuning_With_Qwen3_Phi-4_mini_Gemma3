"""Calculator cross-check for the playground.

If a question looks like one of the 24 supported calculation templates, recompute the
result in code and compare it with the model's final number. The answer text is never
changed. Numbers are read in the order they appear in the question, so a differently
ordered rewording can give a wrong "expected" value; the UI says so.
"""
import re

from . import calc_verify

# (topic, label, unit text, decimals, required keywords [all], forbidden keywords [any])
# Keywords, unit text and the regexes in calc_verify stay Thai on purpose: they match the Thai
# questions and answers the models are trained on. Do not translate them.
RULES = [
    ("real_return", "Real return", "%", 2, ("ผลตอบแทนที่แท้จริง",), ()),
    ("inflation_adjustment", "Future price", " บาท", 2, ("เงินเฟ้อ",), ("ผลตอบแทนที่แท้จริง",)),
    ("capital_gain", "Price return", "%", 2, ("ซื้อหุ้น", "ขาย", "เปอร์เซ็นต์"), ()),
    ("compound_interest", "Compound future value", " บาท", 2, ("ทบต้น", "เงินต้น"), ()),
    ("future_value_annuity", "Annuity future value", " บาท", 2, ("ออมปลายปี",), ()),
    ("quick_ratio", "Quick Ratio", " เท่า", 2, ("quick ratio",), ()),
    ("current_ratio", "Current Ratio", " เท่า", 2, ("current ratio",), ("quick",)),
    ("de_ratio", "D/E Ratio", " เท่า", 2, ("d/e",), ()),
    ("debt_service_ratio", "Debt service ratio", "%", 2, ("ผ่อนหนี้", "เปอร์เซ็นต์"), ()),
    ("dividend_yield", "Dividend Yield", "%", 2, ("dividend yield",), ()),
    ("emergency_fund", "Emergency fund target", " บาท", 2, ("เงินสำรอง", "ค่าใช้จ่ายจำเป็น"), ()),
    ("gross_margin", "Gross margin", "%", 2, ("กำไรขั้นต้น", "ต้นทุนขาย"), ()),
    ("net_margin", "Net margin", "%", 2, ("net profit margin",), ()),
    ("monthly_goal", "Minimum monthly saving", " บาท", 2, ("อยากเก็บเงิน", "ไม่คิดดอกเบี้ย"), ()),
    ("monthly_saving", "Monthly saving", " บาท", 2, ("เหลือเงินออม",), ()),
    ("mutual_fund_gain", "NAV gain", " บาท", 2, ("กำไรจากส่วนต่าง nav",), ()),
    ("mutual_fund_units", "Fund units", " หน่วย", 4, ("กี่หน่วย",), ()),
    ("dca_cost", "Average cost per unit", " บาทต่อหน่วย", 2, ("ต้นทุนเฉลี่ย", "ครั้งที่สอง"), ()),
    ("net_worth", "Net worth", " บาท", 2, ("ความมั่งคั่งสุทธิ",), ()),
    ("pbv_ratio", "P/BV", " เท่า", 2, ("p/bv",), ()),
    ("pe_ratio", "P/E", " เท่า", 2, ("p/e",), ("p/bv",)),
    ("roa", "ROA", "%", 2, ("roa",), ()),
    ("roe", "ROE", "%", 2, ("roe",), ()),
    ("savings_rate", "Savings rate", "%", 2, ("อัตราการออม",), ()),
]


# Keyword matching is loose, so templates whose numbers carry different roles must also
# match the template's order: amount in baht, then a % rate, then a number of years.
_AMOUNT_RATE_YEARS = re.compile(r"\d[\d,.]*\s*บาท.*?[-−]?\d[\d,.]*\s*%.*?\d+\s*ปี", re.S)
SHAPES = {
    "real_return": re.compile(r"%.*%", re.S),
    "inflation_adjustment": _AMOUNT_RATE_YEARS,
    "compound_interest": _AMOUNT_RATE_YEARS,
    "future_value_annuity": _AMOUNT_RATE_YEARS,
}


def check(question, answer):
    """Return a cross-check dict, or None when the question is not a supported template."""
    lowered = question.casefold()
    numbers = calc_verify.question_numbers(question)
    for topic, label, unit, places, required, forbidden in RULES:
        if not all(k in lowered for k in required) or any(k in lowered for k in forbidden):
            continue
        shape = SHAPES.get(topic)
        if shape and not shape.search(question):
            continue
        formula, _, arity = calc_verify.TOPICS[topic]
        # Keywords alone are not enough: try later rules when the numbers do not fit.
        if len(numbers) != arity:
            continue
        try:
            expected = formula(*numbers)
        except (ZeroDivisionError, OverflowError, ValueError):
            continue
        model_value = calc_verify.last_value(answer, topic)
        agrees = None if model_value is None else (
            abs(model_value - expected) <= max(calc_verify.ATOL, calc_verify.RTOL * abs(expected)))
        return {"topic": topic, "label": label, "expected": round(expected, places),
                "expected_text": f"{expected:,.{places}f}{unit}", "model_value": model_value,
                "agrees": agrees}
    return None
