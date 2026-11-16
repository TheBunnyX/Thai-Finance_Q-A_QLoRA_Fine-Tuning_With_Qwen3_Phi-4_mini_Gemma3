import json
import tempfile
import unittest
from pathlib import Path

from finance_qlora.data import audit, group_indices, load_prepared, prepare, read_rows
from finance_qlora.evaluate import compare, score_answer
from finance_qlora.modeling import encode_pair


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, rows):
        path = self.root / 'input.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        return path

    def test_invalid_rows_and_duplicate_json_keys(self):
        path = self.root / 'bad.jsonl'
        path.write_text('{"question":"x","question":"y","answer":"z"}\n\n'
                        '{"question":"x","answer":""}\n', encoding='utf-8')
        rows, errors = read_rows(path)
        self.assertEqual(rows, [])
        self.assertEqual([e['line'] for e in errors], [1, 2, 3])

    def test_audit_detects_answer_copy_and_extraction_character(self):
        path = self.source([{'question':'อธิบาย “ตัวอย่าง…”?', 'answer':'ตัวอย่างข้อความ\ue001'}])
        result = audit(path)
        self.assertEqual(result['quality_flag_counts']['question_contains_answer_prefix'], 1)
        self.assertEqual(result['quality_flag_counts']['extraction_character'], 1)

    def test_transitive_sources_and_duplicates_stay_together(self):
        rows = [
            {'question':'Q1', 'answer':'A1', 'source_id':'doc-a'},
            {'question':'Q2', 'answer':'A2', 'source_id':'doc-a'},
            {'question':'Q3', 'answer':'A2', 'source_id':'doc-b'},
            {'question':'Q4', 'answer':'A4', 'source_id':'doc-b'},
            {'question':'independent', 'answer':'other'},
        ]
        self.assertEqual(sorted(map(len, group_indices(rows))), [1, 4])

    def test_prepare_determinism_no_leakage_and_source_unchanged(self):
        rows = [{'question':f'Question {i}', 'answer':f'Answer {i}'} for i in range(30)]
        rows.extend([{'question':'paraphrase', 'answer':'Answer 0'}, dict(rows[1])])
        path = self.source(rows)
        original = path.read_bytes()
        first = prepare(path, self.root/'a')
        second = prepare(path, self.root/'b')
        self.assertEqual(first['splits'], second['splits'])
        self.assertEqual(first['duplicate_pairs_removed'], 1)
        splits, _ = load_prepared(self.root/'a')
        self.assertEqual(sum(map(len, splits.values())), 31)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(ValueError):
            prepare(path, self.root/'a')

    def test_detect_checksum_change(self):
        path = self.source([{'question':f'Q{i}', 'answer':f'A{i}'} for i in range(10)])
        prepare(path, self.root/'prepared')
        (self.root/'prepared/test.jsonl').write_text('', encoding='utf-8')
        with self.assertRaises(ValueError):
            load_prepared(self.root/'prepared')

    def test_fewer_than_three_groups_fails(self):
        path = self.source([{'question':'q1','answer':'same'}, {'question':'q2','answer':'same'}])
        with self.assertRaises(ValueError):
            prepare(path, self.root/'prepared')

    def test_conflicting_ids_fail(self):
        path = self.source([{'id':'same','question':f'Q{i}','answer':f'A{i}'} for i in range(4)])
        with self.assertRaises(ValueError):
            prepare(path, self.root/'prepared')

    def test_id_reused_for_previously_seen_pair_fails(self):
        rows = [{'id':'x','question':'Q1','answer':'A1'}, {'id':'y','question':'Q2','answer':'A2'},
                {'id':'x','question':'Q2','answer':'A2'}, {'question':'Q3','answer':'A3'},
                {'question':'Q4','answer':'A4'}]
        with self.assertRaisesRegex(ValueError, 'Duplicate id'):
            prepare(self.source(rows), self.root/'prepared')


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 7, 8, 9]


class TrainingBoundaryTests(unittest.TestCase):
    def test_mapping_tokenizer_result_has_correct_answer_mask(self):
        class MappingTokenizer(FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                ids = super().apply_chat_template(*args, **kwargs)
                return {'input_ids': ids, 'attention_mask': [1]*len(ids)}
        row = encode_pair(MappingTokenizer(), {'question':'q','answer':'a'}, 20)
        self.assertEqual(row['input_ids'], [1,2,3,7,8,9])
        self.assertEqual(row['labels'], [-100,-100,-100,7,8,9])

    def test_only_answer_tokens_have_labels(self):
        row = encode_pair(FakeTokenizer(), {'question':'q','answer':'a'}, 20)
        self.assertEqual(row['labels'], [-100, -100, -100, 7, 8, 9])

    def test_never_silently_truncate(self):
        with self.assertRaises(ValueError):
            encode_pair(FakeTokenizer(), {'question':'q','answer':'a'}, 5)

    def test_template_prefix_mismatch_fails(self):
        class BadTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [99] if add_generation_prompt else [1, 2]
        with self.assertRaises(ValueError):
            encode_pair(BadTokenizer(), {'question':'q','answer':'a'}, 20)

    def test_numeric_requires_explicit_final_answer_and_tolerance(self):
        row = {'answer':'2.0 เท่า', 'evaluation':{'type':'numeric','value':2,
               'pattern':r'ผลลัพธ์ ([\d.]+) เท่า', 'atol':.01}}
        self.assertTrue(score_answer('500 / 250 ผลลัพธ์ 2.005 เท่า', row)['numeric_correct'])
        self.assertFalse(score_answer('ผลลัพธ์ 3 เท่า', row)['numeric_correct'])
        self.assertFalse(score_answer('ผลลัพธ์ 2 เท่า หรือ ผลลัพธ์ 3 เท่า', row)['numeric_correct'])
        self.assertNotIn('numeric_correct', score_answer('2', {'answer':'2'}))

    def test_compare_rejects_different_test_set(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a.json', Path(temp)/'b.json'
            a.write_text(json.dumps({'model':'m','split':'test','split_sha256':'a'}))
            b.write_text(json.dumps({'model':'m','split':'test','split_sha256':'b'}))
            with self.assertRaises(ValueError):
                compare(a, b)


class AdapterDatasetCheckTests(unittest.TestCase):
    def test_same_set_or_same_train_split_only(self):
        from finance_qlora.data import digest
        from finance_qlora.evaluate import adapter_dataset_compatible
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"splits": {"train": {"sha256": "aaa"}, "test": {"sha256": "ttt"}}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            exact = {"dataset_manifest_sha256": digest(root / "manifest.json"), "dataset_manifest": {}}
            self.assertTrue(adapter_dataset_compatible(exact, root, manifest))
            same_train = {"dataset_manifest_sha256": "other",
                          "dataset_manifest": {"splits": {"train": {"sha256": "aaa"}}}}
            self.assertTrue(adapter_dataset_compatible(same_train, root, manifest))
            other_train = {"dataset_manifest_sha256": "other",
                           "dataset_manifest": {"splits": {"train": {"sha256": "bbb"}}}}
            self.assertFalse(adapter_dataset_compatible(other_train, root, manifest))
            self.assertFalse(adapter_dataset_compatible({"dataset_manifest_sha256": "other"}, root, manifest))


if __name__ == '__main__':
    unittest.main()
