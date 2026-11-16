import csv
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from finance_qlora.data import read_rows
from finance_qlora.modeling import AnswerCollator
from finance_qlora.evaluate import generate_batch
from finance_qlora.evaluate import evaluate
from finance_qlora.review import export_review, score_review
from finance_qlora.scoring import score_answer, summarize, validate_evaluation
from finance_qlora.telemetry import (
    TokenTimingStreamer, directory_size, measured_trainer_class, peak_process_ram, percentile,
)


class MetricTests(unittest.TestCase):
    def numeric_row(self):
        return {'answer': 'Result = 2 times', 'evaluation': {
            'type': 'numeric', 'value': 2, 'pattern': r'Result = ([\d.]+)',
            'unit': {'pattern': r'Result = [\d.]+ (times|%)', 'accepted': ['times']}}}

    def test_numeric_and_unit_are_separate_and_joint(self):
        row = self.numeric_row()
        score = score_answer('Result = 2 %', row)
        self.assertTrue(score['numeric_correct'])
        self.assertFalse(score['unit_correct'])
        self.assertFalse(score['numeric_and_unit_correct'])
        self.assertTrue(score_answer('Result = 2 times', row)['numeric_and_unit_correct'])
        self.assertFalse(score_answer('Result = 3 times', row)['numeric_and_unit_correct'])

    def test_unannotated_rows_do_not_lower_unit_coverage(self):
        scores = [score_answer('Result = 2 times', self.numeric_row()),
                  score_answer('definition', {'answer': 'definition'})]
        result = summarize([{'scores': s} for s in scores])
        self.assertEqual(result['rows'], 2)
        self.assertEqual(result['unit_rows'], 1)
        self.assertEqual(result['unit_accuracy'], 1)

    def test_annotation_validation(self):
        for annotation in ({'type': 'numeric', 'value': True, 'pattern': '(x)'},
                           {'type': 'numeric', 'value': 1, 'atol': -1, 'pattern': '(x)'},
                           {'unit': {'pattern': '[', 'accepted': ['%']}},
                           {'unit': {'pattern': '(x)(y)', 'accepted': ['%']}},
                           {'answerable': 'false'}):
            with self.assertRaises(ValueError):
                validate_evaluation(annotation)

    def test_annotation_schema_errors_report_line(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'input.jsonl'
            path.write_text(json.dumps({'question':'q','answer':'a','evaluation':{'answerable':1}}))
            rows, errors = read_rows(path)
            self.assertFalse(rows)
            self.assertEqual(errors[0]['line'], 1)

    def test_streamer_skips_prompt_and_measures_generated_tokens(self):
        times = iter([11.5, 12.5])
        timing = TokenTimingStreamer(started=10, clock=lambda: next(times))
        timing.put('prompt')
        self.assertIsNone(timing.metrics()['batch_ttft_seconds'])
        timing.put('first token')
        timing.put('second token')
        timing.end()
        self.assertEqual(timing.metrics(), {'batch_ttft_seconds':1.5,'subsequent_token_seconds':1.0})

    def test_single_token_has_no_subsequent_throughput_interval(self):
        timing = TokenTimingStreamer(started=0, clock=lambda: 1)
        timing.put('prompt')
        timing.put('eos')
        self.assertIsNone(timing.metrics()['subsequent_token_seconds'])

    def test_percentile_interpolation_and_empty(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)
        self.assertIsNone(percentile([], 95))

    def test_ram_and_artifact_measurement(self):
        peak = peak_process_ram()['peak_process_ram_gib']
        self.assertTrue(math.isfinite(peak) and peak > 0)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'weights.bin').write_bytes(b'12345')
            self.assertEqual(directory_size(temp), 5)

    def test_training_counters_exclude_validation_and_remove_metadata(self):
        class Base:
            def compute_loss(self, model, inputs, **kwargs):
                if '_token_counts' in inputs:
                    raise AssertionError('metadata leaked to model')
                return 7
        trainer = measured_trainer_class(Base)()
        self.assertEqual(trainer.compute_loss(SimpleNamespace(training=True),
                         {'_token_counts':(10,4,16)}), 7)
        trainer.compute_loss(SimpleNamespace(training=False), {'_token_counts':(20,8,32)})
        self.assertEqual(trainer.training_token_counts, {'nonpadding':10,'answer':4,'padded':16})

    def test_collator_counts_real_and_answer_tokens_before_padding(self):
        collator = AnswerCollator(0, track_tokens=True)
        output = collator([{'input_ids':[1,2,3], 'attention_mask':[1,1,1], 'labels':[-100,2,3]}])
        self.assertEqual(output['_token_counts'], (3,2,8))

    def test_generate_passes_timing_streamer_to_model(self):
        import torch
        class Model:
            generation_config = SimpleNamespace(eos_token_id=2)
            def generate(self, input_ids, streamer, **kwargs):
                streamer.put(input_ids)
                streamer.put(torch.tensor([7]))
                streamer.put(torch.tensor([2]))
                streamer.end()
                return torch.cat((input_ids,torch.tensor([[7,2]])),dim=1)
        clock = iter([1.0,1.5])
        timing = TokenTimingStreamer(started=0,clock=lambda:next(clock))
        generated, _ = generate_batch(Model(), SimpleNamespace(pad_token_id=0,eos_token_id=2),
                                      [[4]], 2, device='cpu', streamer=timing)
        self.assertEqual(generated, [[7,2]])
        self.assertEqual(timing.metrics()['batch_ttft_seconds'],1)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        for name in ('base', 'adapter'):
            path = self.root/(name+'.jsonl')
            rows = [{'id':str(i),'question':f'q{i}','reference':'reference',
                     'prediction':'=2' if i == 0 else name + ' answer',
                     'evaluation':{'answerable': False}} for i in range(3)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.paths.append(path)
        self.output = self.root/'review'
        export_review(self.paths, self.output, limit=2)
        self.key = self.output/'review_key.json'
        with (self.output/'blind_review.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.rows = list(csv.DictReader(stream))

    def save(self, name, rows):
        path = self.root/name
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_blinded_export_uses_same_subset_and_excludes_model_paths(self):
        self.assertEqual(len(self.rows), 4)
        self.assertNotIn('experiment', self.rows[0])
        key = json.loads(self.key.read_text())
        groups = {}
        for item in key['items'].values():
            groups.setdefault(item['experiment'],set()).add(item['source_id'])
        self.assertEqual(groups['experiment-1'],groups['experiment-2'])

    def test_blank_scores_are_missing_not_zero(self):
        result = score_review([self.output/'blind_review.csv'],self.key)
        self.assertEqual(result['reviewers'],0)
        self.assertIsNone(result['experiments']['experiment-1']['unsupported_claim_rate'])

    def test_multiple_reviewers_are_averaged_with_agreement(self):
        a, b = [dict(r) for r in self.rows], [dict(r) for r in self.rows]
        for rows, reviewer, score in ((a,'r1','2'),(b,'r2','0')):
            for row in rows:
                row.update(reviewer_id=reviewer,financial_correctness=score,unsupported_claim='0',
                           insufficient_information_handling='1')
        result = score_review([self.save('a.csv',a),self.save('b.csv',b)],self.key)
        experiment = result['experiments']['experiment-1']
        self.assertEqual(experiment['financial_correctness_accuracy'],.5)
        self.assertEqual(experiment['dimensions']['financial_correctness']['mean'],1)
        self.assertEqual(experiment['dimensions']['financial_correctness']['exact_agreement'],0)
        self.assertEqual(experiment['insufficient_information_handling_accuracy'],1)

    def test_duplicate_reviewer_and_changed_text_are_rejected(self):
        rows = [dict(r,reviewer_id='r1',financial_correctness='2') for r in self.rows]
        path = self.save('rated.csv',rows)
        with self.assertRaises(ValueError):
            score_review([path,path],self.key)
        rows[0]['prediction'] = 'edited'
        with self.assertRaises(ValueError):
            score_review([self.save('edited.csv',rows)],self.key)

    def test_invalid_score_and_unsupported_claim_without_evidence(self):
        for values in ({'financial_correctness':'3'},{'unsupported_claim':'1'}):
            rows = [dict(r) for r in self.rows]
            rows[0].update(reviewer_id='r1',**values)
            with self.assertRaises(ValueError):
                score_review([self.save('invalid.csv',rows)],self.key)


class EvaluationIntegrationTests(unittest.TestCase):
    def test_metrics_report_for_single_and_batched_inference(self):
        # Exercise report creation without downloading models or starting a CUDA workload.
        rows = [{'id':str(i),'question':f'q{i}','answer':'Result = 2 times',
                 'evaluation':{'type':'numeric','value':2,'pattern':r'= ([\d.]+)',
                               'unit':{'pattern':r'2 (times|%)','accepted':['times']}}}
                for i in range(2)]
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):
                return [5,6]
            def batch_decode(self,tokens,**kwargs):
                return ['Result = 2 times']*len(tokens)
        class Model:
            config = SimpleNamespace(use_cache=False)
            def eval(self):
                pass
        def fake_generate(model,tokenizer,prompts,max_new_tokens,streamer=None):
            if streamer:
                streamer.put('prompt')
                streamer.put('first')
                streamer.put('last')
            return [[7,2] for _ in prompts], {2}
        with tempfile.TemporaryDirectory() as temp:
            for batch in (1,2):
                args = SimpleNamespace(data=Path(temp),model='qwen',split='test',adapter=None,batch_size=batch,
                                       max_new_tokens=4,measure_ttft=True,warmup_batches=1,
                                       output=Path(temp)/f'eval-{batch}')
                with patch('finance_qlora.evaluate.load_prepared',return_value=(
                        {'test':rows},{'splits':{'test':{'sha256':'fixed'}}})), \
                     patch('finance_qlora.evaluate.settings',return_value=(
                        {'model_id':'model','revision':'commit'},{'max_seq_length':32})), \
                     patch('finance_qlora.evaluate.load_model',return_value=(
                        Model(),None,Tokenizer(),SimpleNamespace(for_inference=lambda m:None),'commit')), \
                     patch('finance_qlora.evaluate.generate_batch',side_effect=fake_generate), \
                     patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), \
                     patch('torch.cuda.max_memory_allocated',return_value=1024), \
                     patch('torch.cuda.max_memory_reserved',return_value=2048), \
                     contextlib.redirect_stdout(io.StringIO()):
                    evaluate(args)
                metrics = json.loads((args.output/'metrics.json').read_text())['metrics']
                self.assertEqual(metrics['unit_accuracy'],1)
                self.assertEqual(metrics['numeric_and_unit_rows'],2)
                self.assertEqual(metrics['warmup_batches'],1)
                self.assertEqual(metrics['rows'],2)
                self.assertEqual(metrics['request_latency_rows'],2 if batch == 1 else 0)
                if batch == 1:
                    self.assertGreaterEqual(metrics['ttft_p95_seconds'],0)
                    self.assertGreaterEqual(metrics['request_latency_p95_seconds'],0)
                else:
                    self.assertIsNone(metrics['ttft_p95_seconds'])
                    self.assertIsNone(metrics['request_latency_p95_seconds'])



class ReviewNullEvaluationTests(unittest.TestCase):
    def test_export_accepts_null_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'p.jsonl'
            path.write_text(json.dumps({'id':'1','question':'q','reference':'r','prediction':'p',
                                        'evaluation':None})+'\n')
            export_review([path], root/'review')
            self.assertTrue((root/'review'/'blind_review.csv').is_file())


if __name__ == '__main__':
    unittest.main()
