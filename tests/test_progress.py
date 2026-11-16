import contextlib
import io
import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from finance_qlora.__main__ import main
from finance_qlora.progress import Progress, Reporter, duration, track, trainer_progress_callback


class ProgressTests(unittest.TestCase):
    def test_eta_and_finish_timestamp_use_bangkok(self):
        current = [0.0]
        stage = Progress('work', total=10, clock=lambda:current[0], heartbeat=0,
                         wall_clock=lambda:datetime(2026,11,10,tzinfo=timezone.utc)+timedelta(seconds=current[0]),
                         reporter=Reporter(console=True,stream=io.StringIO()))
        with stage:
            self.assertIsNone(stage.snapshot()['eta_seconds'])
            current[0] = 1
            stage.update(1)
            current[0] = 2
            stage.update(2)
            snapshot = stage.snapshot()
            self.assertEqual(snapshot['eta_seconds'],8)
            self.assertEqual(snapshot['estimated_finish_at'],'2026-11-10T07:00:10+07:00')
        self.assertEqual(stage.snapshot()['status'],'done')

    def test_fast_updates_do_not_evict_all_eta_history(self):
        current = [0.0]
        stage = Progress('fast', total=10000, clock=lambda:current[0], heartbeat=0,
                         reporter=Reporter(console=True,stream=io.StringIO(),interval=100))
        for i in range(1,2001):
            current[0] = i/1000
            stage.update(i)
        self.assertIsNotNone(stage.snapshot()['eta_seconds'])
        self.assertLess(len(stage.samples),10)

    def test_warmup_reset_keeps_elapsed_but_excludes_compile_from_eta(self):
        current = [0.0]
        stage = Progress('training',total=10,clock=lambda:current[0],heartbeat=0,
                         reporter=Reporter(console=True,stream=io.StringIO()))
        current[0] = 100
        stage.update(1)
        stage.reset_rate()
        current[0] = 101
        stage.update(2)
        current[0] = 102
        stage.update(3)
        self.assertEqual(stage.snapshot()['elapsed_seconds'],102)
        self.assertEqual(stage.snapshot()['eta_seconds'],7)

    def test_throttled_console_and_structured_log(self):
        with tempfile.TemporaryDirectory() as temp:
            log, output = Path(temp)/'events.jsonl', io.StringIO()
            clock = [0.0]
            with Progress('example', total=10, heartbeat=0, clock=lambda:clock[0],
                          reporter=Reporter(console=True,log=log,interval=10,stream=output)) as stage:
                clock[0] = 1
                stage.update(3)
            events = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(events),2)
            self.assertEqual(events[-1]['completed'],10)
            self.assertIn('ETA',output.getvalue())

    def test_failure_and_interrupt_never_report_success(self):
        for exception, expected in ((ValueError('bad data'),'failed'),(KeyboardInterrupt(),'cancelled')):
            output = io.StringIO()
            stage = Progress('work', total=20, heartbeat=0,reporter=Reporter(console=True,stream=output))
            with self.assertRaises(type(exception)):
                with stage:
                    stage.update(4)
                    raise exception
            self.assertEqual(stage.snapshot()['status'],expected)
            self.assertEqual(stage.snapshot()['completed'],4)
            self.assertIsNone(stage.snapshot()['eta_seconds'])
            self.assertNotIn('[DONE]',output.getvalue())

    def test_unknown_work_and_disabled_mode(self):
        silent = io.StringIO()
        with Progress('disabled',reporter=Reporter(console=False,stream=silent)) as stage:
            stage.advance()
        self.assertEqual(silent.getvalue(),'')
        self.assertIsNone(stage.thread)
        with Progress('loading',reporter=Reporter(console=True,stream=io.StringIO()),heartbeat=0) as stage:
            self.assertIsNone(stage.snapshot()['eta_seconds'])
            self.assertIsNone(stage.snapshot()['total'])
        self.assertEqual(duration(.1),'00:00:00')
        self.assertEqual(duration(.1,round_up=True),'00:00:01')

    def test_unknown_phase_heartbeat_stops_cleanly(self):
        seen = threading.Event()
        class HeartbeatReporter(Reporter):
            def __init__(self):
                super().__init__(console=True,stream=io.StringIO(),interval=.001)
                self.calls = 0
            def emit(self,event):
                self.calls += 1
                if self.calls >= 2:
                    seen.set()
        with Progress('loading',heartbeat=.01,reporter=HeartbeatReporter()) as stage:
            self.assertTrue(seen.wait(timeout=1))
        self.assertFalse(stage.thread.is_alive())

    def test_trainer_callback_tracks_steps_and_validation(self):
        output = io.StringIO()
        with patch('finance_qlora.progress._reporter',Reporter(console=True,stream=output)):
            callback = trainer_progress_callback(object)
            state = SimpleNamespace(max_steps=4,global_step=0,epoch=0)
            callback.on_train_begin(None,state,None)
            state.global_step = 1
            callback.on_step_end(None,state,None)
            callback.on_log(None,state,None,logs={'loss':.5})
            callback.on_prediction_step(None,state,None,eval_dataloader=[1,2])
            callback.on_prediction_step(None,state,None,eval_dataloader=[1,2])
            callback.on_evaluate(None,state,None,metrics={'eval_loss':.4})
            state.global_step = 4
            callback.on_step_end(None,state,None)
            callback.on_train_end(None,state,None)
            callback.close()
        self.assertIn('Training optimizer steps',output.getvalue())
        self.assertIn('Validation forward passes',output.getvalue())
        self.assertEqual(callback.training.status,'done')

    def test_early_stop_does_not_claim_all_steps_complete(self):
        with patch('finance_qlora.progress._reporter',Reporter(console=True,stream=io.StringIO())):
            callback = trainer_progress_callback(object)
            state = SimpleNamespace(max_steps=10,global_step=2,epoch=.2)
            callback.on_train_begin(None,state,None)
            callback.on_step_end(None,state,None)
            callback.on_train_end(None,state,None)
            self.assertEqual(callback.training.status,'stopped')
            self.assertEqual(callback.training.completed,2)

    def test_checkpoint_has_a_distinct_start_and_finish(self):
        output = io.StringIO()
        with patch('finance_qlora.progress._reporter',Reporter(console=True,stream=output)):
            callback = trainer_progress_callback(object)
            state = SimpleNamespace(max_steps=10,global_step=2,epoch=.2)
            control = SimpleNamespace(should_save=True,should_evaluate=True)
            callback.on_train_begin(None,state,control)
            callback.on_step_end(None,state,control)
            self.assertIsNone(callback.checkpoint)
            callback.on_evaluate(None,state,control,metrics={'eval_loss':.3})
            checkpoint = callback.checkpoint
            self.assertEqual(checkpoint.status,'running')
            callback.on_save(None,state,control)
            self.assertEqual(checkpoint.status,'done')
            callback.close('cancelled')
        self.assertIn('Save checkpoint at step 2',output.getvalue())

    def test_cli_json_stdout_stays_clean_and_errors_mark_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'data.jsonl'
            source.write_text('{"question":"q","answer":"a"}\n')
            for quiet in (False,True):
                stdout, stderr = io.StringIO(),io.StringIO()
                argv = ['finance_qlora','audit',str(source),'--output',str(root/'report.json'),
                        '--progress-log',str(root/f'log-{quiet}.jsonl')]
                if quiet:
                    argv.append('--no-progress')
                with patch('finance_qlora.progress._reporter'), patch('sys.argv',argv), \
                     contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(),0)
                self.assertEqual(json.loads(stdout.getvalue())['valid_rows'],1)
                self.assertEqual(bool(stderr.getvalue()),not quiet)
            argv = ['finance_qlora','audit',str(root/'missing.jsonl'),'--output',str(root/'r.json'),
                    '--progress-log',str(root/'failed.jsonl')]
            with patch('finance_qlora.progress._reporter'),patch('sys.argv',argv), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(),1)
            events = [json.loads(line) for line in (root/'failed.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['status'],'failed')


if __name__ == '__main__':
    unittest.main()
