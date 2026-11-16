import importlib.util
import unittest
from pathlib import Path

from finance_qlora.scoring import score_answer, validate_evaluation

spec = importlib.util.spec_from_file_location("curation", Path(__file__).resolve().parents[1] / "scripts/curate_finance_train_v2.py")
curation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curation)


class CurationTests(unittest.TestCase):
    def test_corrects_only_result_and_preserves_reference_inputs(self):
        row = {"id": "x", "question": "บริษัทมีสินทรัพย์หมุนเวียน 500,000 บาท และหนี้สินหมุนเวียน 250,000 บาท Current Ratio เป็นเท่าไร?",
               "answer": "Current Ratio = สินทรัพย์หมุนเวียน ÷ หนี้สินหมุนเวียน = 500,000 ÷ 250,000 ≈ 2.10 เท่า"}
        result, check = curation.curate(row)
        self.assertEqual(check["status"], "corrected")
        self.assertIn("500,000 ÷ 250,000 ≈ 2.00 เท่า", result["answer"])
        self.assertIn("2.10", row["answer"])
        validate_evaluation(result["evaluation"])
        self.assertTrue(score_answer("ได้ 2.00 เท่า", result)["numeric_and_unit_correct"])

    def test_ambiguous_source_or_template_is_not_silently_annotated(self):
        row = {"id": "x", "question": "บริษัทมีสินทรัพย์หมุนเวียน 500,000 บาท และหนี้สินหมุนเวียน 250,000 บาท Current Ratio เป็นเท่าไร?",
               "answer": "Current Ratio = 501,000 ÷ 250,000 ≈ 2.00 เท่า"}
        result, check = curation.curate(row)
        self.assertEqual(check["status"], "needs_manual_review")
        self.assertNotIn("evaluation", result)
        other, check = curation.curate({"question": "What is ROE?", "answer": "It is a profitability ratio."})
        self.assertIsNone(check)
        self.assertNotIn("evaluation", other)


if __name__ == "__main__":
    unittest.main()
