import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from finance_qlora.evaluate import generate_batch
from finance_qlora.modeling import AnswerCollator
from finance_qlora.performance import (
    cached_encode, evaluation_subset, generation_batches, length_sampler_options, training_plan, trim_generated,
)


class CountingTokenizer:
    chat_template = 'template-v1'
    special_tokens_map = {'eos_token': '</s>'}
    calls = 0

    def get_vocab(self):
        return {'a': 1, 'b': 2}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls += 1
        prefix = [1, len(messages[0]['content']), 3]
        return prefix if add_generation_prompt else prefix + [7, 8, 9]


class PerformanceTests(unittest.TestCase):
    def test_profile_subset_is_reproducible_and_does_not_modify_input(self):
        rows = [{'id':str(i)} for i in range(125)]
        a, metadata = evaluation_subset(rows,10,42)
        b, same = evaluation_subset(rows,10,42)
        _, other = evaluation_subset(rows,10,43)
        self.assertEqual(a,b)
        self.assertEqual(metadata,same)
        self.assertNotEqual(metadata['ids_sha256'],other['ids_sha256'])
        self.assertEqual(len(rows),125)
        self.assertEqual(len(a),10)
        self.assertEqual(evaluation_subset(rows)[0],rows)
        with self.assertRaises(ValueError):
            evaluation_subset(rows,0)

    def test_cache_avoids_tokenization_and_invalidates_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = CountingTokenizer()
            rows = [{'question': 'question', 'answer': 'answer'}]
            spec = {'model_id': 'model', 'revision': 'commit-a'}
            first, stats = cached_encode(tokenizer, rows, 32, spec, temp)
            self.assertFalse(stats['hit'])
            self.assertEqual(tokenizer.calls, 2)
            second, stats = cached_encode(tokenizer, rows, 32, spec, temp)
            self.assertTrue(stats['hit'])
            self.assertEqual(first, second)
            self.assertEqual(tokenizer.calls, 2)
            variants = [(rows, 64, spec),
                        (rows, 32, dict(spec, revision='commit-b')),
                        ([{'question': 'new question', 'answer': 'answer'}], 32, spec)]
            for records, length, model in variants:
                _, stats = cached_encode(tokenizer, records, length, model, temp)
                self.assertFalse(stats['hit'])
            tokenizer.chat_template = 'template-v2'
            _, stats = cached_encode(tokenizer, rows, 32, spec, temp)
            self.assertFalse(stats['hit'])

    def test_corrupted_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = CountingTokenizer()
            rows = [{'question': 'q', 'answer': 'a'}]
            _, stats = cached_encode(tokenizer, rows, 32, {'revision': 'a'}, temp)
            path = Path(temp) / (stats['key'] + '.json')
            payload = json.loads(path.read_text())
            payload['rows'][0]['labels'] = [99]
            path.write_text(json.dumps(payload))
            encoded, stats = cached_encode(tokenizer, rows, 32, {'revision': 'a'}, temp)
            self.assertFalse(stats['hit'])
            self.assertEqual(encoded[0]['labels'], [-100, -100, -100, 7, 8, 9])

    def test_batch_override_preserves_effective_batch_and_short_run_evaluates(self):
        config = {'batch_size': 1, 'gradient_accumulation_steps': 8,
                  'num_train_epochs': 1, 'eval_steps': 200}
        changed, interval, steps = training_plan(config, 17, batch_size=2)
        self.assertEqual(changed['gradient_accumulation_steps'], 4)
        self.assertEqual((interval, steps), (3, 3))
        self.assertEqual(config['batch_size'], 1)
        with self.assertRaises(ValueError):
            training_plan(config, 17, batch_size=3)

    def test_sampler_api_compatibility(self):
        @dataclass
        class Modern:
            train_sampling_strategy: str = 'random'
        @dataclass
        class Legacy:
            group_by_length: bool = False
        self.assertEqual(length_sampler_options(Modern), {'train_sampling_strategy': 'group_by_length'})
        self.assertEqual(length_sampler_options(Legacy), {'group_by_length': True})

    def test_generation_buckets_cover_every_row_and_trim_eos(self):
        batches = generation_batches([[1]*n for n in (9, 2, 8, 3)], 2)
        self.assertEqual(batches, [[1, 3], [2, 0]])
        self.assertEqual(trim_generated([7, 2, 0, 0], {2}), [7, 2])
        self.assertEqual(trim_generated([7, 8], {2}), [7, 8])

    def test_collator_masks_padding_and_keeps_context_limit(self):
        examples = [{'input_ids': [1]*n, 'attention_mask': [1]*n,
                     'labels': [-100]*(n-1)+[9]} for n in (7, 9)]
        output = AnswerCollator(0, pad_to_multiple_of=8, max_length=10)(examples)
        self.assertEqual(tuple(output['input_ids'].shape), (2, 10))
        self.assertEqual(output['labels'][0].tolist(), [-100]*6+[9]+[-100]*3)
        self.assertEqual(output['attention_mask'][1].tolist(), [1]*9+[0])

    def test_batched_generation_left_padding_and_attention_mask(self):
        import torch
        class FakeModel:
            generation_config = SimpleNamespace(eos_token_id=[2, 3])
            def generate(self, input_ids, attention_mask, **kwargs):
                self.inputs = input_ids.tolist()
                self.mask = attention_mask.tolist()
                return torch.cat([input_ids, torch.tensor([[7, 2, 0], [8, 9, 3]])], dim=1)
        model = FakeModel()
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        generated, eos = generate_batch(model, tokenizer, [[11], [12, 13]], 3, device='cpu')
        self.assertEqual(model.inputs, [[0, 11], [12, 13]])
        self.assertEqual(model.mask, [[0, 1], [1, 1]])
        self.assertEqual(generated, [[7, 2], [8, 9, 3]])
        self.assertEqual(eos, {2, 3})


if __name__ == '__main__':
    unittest.main()
