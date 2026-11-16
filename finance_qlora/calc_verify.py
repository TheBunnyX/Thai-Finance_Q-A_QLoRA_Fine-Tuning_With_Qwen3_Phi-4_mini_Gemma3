"""Recompute template-generated calculation answers from their questions.

Only the 25 calculation topics of the finance_best_available bundle are covered.
A passing row means the final number in the reference answer agrees with an
independent recomputation from the question; it does not validate the finance
concept, the wording or the surrounding advice.
"""
import re

# A leading minus counts only when it is not a hyphen inside a word or range (e.g. 3-5).
NUMBER = re.compile(r"(?:(?<![\w.,])[-−])?\d[\d,]*(?:\.\d+)?")
PERCENT = r"([-−]?\d[\d,]*(?:\.\d+)?)\s*%"
TIMES = r"(\d[\d,]*(?:\.\d+)?)\s*เท่า"
BAHT = r"(\d[\d,]*(?:\.\d+)?)\s*บาท"
UNITS = r"(\d[\d,]*(?:\.\d+)?)\s*หน่วย"
PER_UNIT = r"(\d[\d,]*(?:\.\d+)?)\s*บาทต่อหน่วย"

# Tolerances are fixed here, before any model output is scored: references are
# rounded to 2 decimals (0.011 covers half a unit in the last place plus float
# noise) and baht amounts may differ by 0.1% from the rounded reference.
ATOL, RTOL = 0.011, 0.001

# topic -> (expected value from question numbers, final-value regex, #numbers in question)
TOPICS = {
    "capital_gain": (lambda b, s: (s - b) / b * 100, PERCENT, 2),
    "compound_interest": (lambda pv, r, n: pv * (1 + r / 100) ** n, BAHT, 3),
    "current_ratio": (lambda a, b: a / b, TIMES, 2),
    "dca_cost": (lambda a1, n1, a2, n2: (a1 + a2) / (a1 / n1 + a2 / n2), PER_UNIT, 4),
    "de_ratio": (lambda a, b: a / b, TIMES, 2),
    "debt_service_ratio": (lambda income, pay: pay / income * 100, PERCENT, 2),
    "dividend_yield": (lambda price, dividend: dividend / price * 100, PERCENT, 2),
    "emergency_fund": (lambda monthly, months: monthly * months, BAHT, 2),
    "future_value_annuity": (lambda pmt, r, n: pmt * ((1 + r / 100) ** n - 1) / (r / 100), BAHT, 3),
    "gross_margin": (lambda sales, cost: (sales - cost) / sales * 100, PERCENT, 2),
    "inflation_adjustment": (lambda price, r, n: price * (1 + r / 100) ** n, BAHT, 3),
    "monthly_goal": (lambda goal, months: goal / months, BAHT, 2),
    "monthly_saving": (lambda income, spend: income - spend, BAHT, 2),
    "mutual_fund_gain": (lambda units, buy, sell: units * (sell - buy), BAHT, 3),
    "mutual_fund_units": (lambda amount, nav: amount / nav, UNITS, 2),
    "net_margin": (lambda sales, profit: profit / sales * 100, PERCENT, 2),
    "net_worth": (lambda assets, debts: assets - debts, BAHT, 2),
    "pbv_ratio": (lambda price, book: price / book, TIMES, 2),
    "pe_ratio": (lambda price, eps: price / eps, TIMES, 2),
    "quick_ratio": (lambda ca, inventory, cl: (ca - inventory) / cl, TIMES, 3),
    "real_return": (lambda nominal, inflation: ((1 + nominal / 100) / (1 + inflation / 100) - 1) * 100, PERCENT, 2),
    "roa": (lambda profit, assets: profit / assets * 100, PERCENT, 2),
    "roe": (lambda profit, equity: profit / equity * 100, PERCENT, 2),
    "savings_rate": (lambda income, saved: saved / income * 100, PERCENT, 2),
}


def _float(text):
    return float(text.replace(",", "").replace("−", "-"))


def question_numbers(question):
    return [_float(m) for m in NUMBER.findall(question)]


def final_pattern(topic):
    return TOPICS[topic][1]


def last_value(answer, topic):
    matches = re.findall(final_pattern(topic), answer)
    return _float(matches[-1]) if matches else None


def verify_row(topic, question, answer):
    """Return (status, expected, reference). status: match | rounding_noise | mismatch | unparsed | unsupported."""
    if topic not in TOPICS:
        return "unsupported", None, None
    formula, _, arity = TOPICS[topic]
    numbers = question_numbers(question)
    reference = last_value(answer, topic)
    if len(numbers) != arity or reference is None:
        return "unparsed", None, reference
    try:
        expected = formula(*numbers)
    except (ZeroDivisionError, OverflowError, ValueError):
        return "unparsed", None, reference
    gap = abs(reference - expected)
    if gap <= max(ATOL, RTOL * abs(expected)):
        return "match", expected, reference
    # real_return answers were generated from unrounded inputs, so the rounded
    # figures in the question can disagree by a hundredth or two.
    return ("rounding_noise" if gap <= 0.05 else "mismatch"), expected, reference


def annotation(topic, reference_value):
    """Numeric evaluation annotation using the reference's own (verified) value."""
    return {"type": "numeric", "value": reference_value, "pattern": final_pattern(topic),
            "select": "last", "atol": ATOL, "rtol": RTOL}
