import unittest

from finance_qlora import calc_verify
from finance_qlora.scoring import score_answer, validate_evaluation


class CalcVerifyTests(unittest.TestCase):
    def test_match_mismatch_and_noise(self):
        q = "รายได้ต่อเดือน 38,740 บาท และใช้จ่าย 21,370 บาท เหลือเท่าไร?"
        self.assertEqual(calc_verify.verify_row("monthly_saving", q, "38,740 − 21,370 = 17,370 บาท")[0], "match")
        self.assertEqual(calc_verify.verify_row("monthly_saving", q, "= 17,000 บาท")[0], "mismatch")
        self.assertEqual(calc_verify.verify_row("monthly_saving", q, "ไม่มีตัวเลข")[0], "unparsed")
        self.assertEqual(calc_verify.verify_row("nope", q, "1 บาท")[0], "unsupported")

    def test_negative_percent_is_signed(self):
        q = "ซื้อหุ้นที่ราคา 212.00 บาทและขายที่ 201.40 บาทต่อหุ้น"
        self.assertEqual(calc_verify.verify_row("capital_gain", q, "× 100 = -5.00% ยังไม่หัก")[0], "match")
        self.assertEqual(calc_verify.verify_row("capital_gain", q, "× 100 = 5.00%")[0], "mismatch")

    def test_annotation_uses_last_match_and_passes_own_reference(self):
        answer = "รายได้ 38,740 บาท − 21,370 บาท = 17,370 บาท"
        row = {"answer": answer, "evaluation": calc_verify.annotation("monthly_saving", 17370.0)}
        validate_evaluation(row["evaluation"])
        self.assertTrue(score_answer(answer, row)["numeric_correct"])
        self.assertFalse(score_answer("ได้ 17,000 บาท", row)["numeric_correct"])
        self.assertTrue(score_answer("ผลลัพธ์ −5.00%", {"answer": "x", "evaluation": calc_verify.annotation("capital_gain", -5.0)})["numeric_correct"])

    def test_select_default_still_needs_exactly_one_match(self):
        row = {"answer": "x", "evaluation": {"type": "numeric", "value": 2, "pattern": r"(\d+) เท่า"}}
        self.assertFalse(score_answer("1 เท่า และ 2 เท่า", row)["numeric_correct"])
        with self.assertRaises(ValueError):
            validate_evaluation({"type": "numeric", "value": 2, "pattern": "(x)", "select": "first"})


if __name__ == "__main__":
    unittest.main()


class CalculatorCheckTests(unittest.TestCase):
    def test_flags_wrong_model_number_and_confirms_right_one(self):
        from finance_qlora import calculator
        q = "ยอดขายบริษัท 5,987,600 บาท และกำไรสุทธิ 384,190 บาท Net Profit Margin เท่าไร?"
        bad = calculator.check(q, "384,190 ÷ 5,987,600 × 100 ≈ 6.48%")
        self.assertEqual((bad["topic"], bad["agrees"], bad["expected_text"]), ("net_margin", False, "6.42%"))
        self.assertTrue(calculator.check(q, "≈ 6.42%")["agrees"])
        self.assertIsNone(calculator.check(q, "ไม่มีตัวเลข")["agrees"])

    def test_unsupported_or_ambiguous_questions_return_none(self):
        from finance_qlora import calculator
        self.assertIsNone(calculator.check("NPV เป็นบวกหมายความว่าอย่างไร", "x"))
        self.assertIsNone(calculator.check("บริษัทมีกำไรสุทธิ 100,000 บาท ช่วยคำนวณ ROE ให้หน่อย", "x"))
        self.assertEqual(calculator.check("Quick Ratio: สินทรัพย์หมุนเวียน 100 สินค้าคงเหลือ 20 หนี้สินหมุนเวียน 40", "x")["topic"], "quick_ratio")

    def test_negative_inputs_and_loose_keyword_matches(self):
        from finance_qlora import calc_verify, calculator
        self.assertEqual(calc_verify.question_numbers("ผลตอบแทน 5% เงินเฟ้อ -1% ช่วง 3-5 ปี"), [5, -1, 3, 5])
        real = calculator.check("ผลตอบแทน 5% เงินเฟ้อ -1% ผลตอบแทนที่แท้จริงเท่าไร", "≈ 6.06%")
        self.assertEqual((real["expected_text"], real["agrees"]), ("6.06%", True))
        # Mentions inflation with three numbers but is not the future-price template.
        self.assertIsNone(calculator.check("อายุ 30 ปี เก็บเงิน 10 ปี มี 500,000 บาท กลัวเงินเฟ้อ", "x"))
        q = "ของราคา 383,500 บาทวันนี้ ถ้าราคาเพิ่มขึ้นตามเงินเฟ้อปีละ 1.7% อีก 3 ปีจะมีราคาประมาณเท่าไร?"
        self.assertTrue(calculator.check(q, "≈ 403,392.88 บาท")["agrees"])
