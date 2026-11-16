"""Throttled CLI progress on stderr; JSON results on stdout stay machine readable."""
import json
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

BANGKOK = timezone(timedelta(hours=7), "Asia/Bangkok")


def duration(seconds, round_up=False):
    seconds = max(0, math.ceil(seconds) if round_up else math.floor(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Reporter:
    def __init__(self, console=False, log=None, interval=1.0, stream=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Progress interval must be finite and positive")
        self.console, self.interval = console, interval
        self.stream = stream if stream is not None else sys.stderr
        self.log = Path(log) if log else None
        self.lock = threading.RLock()
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self):
        return self.console or self.log is not None

    def emit(self, event):
        with self.lock:
            if self.console:
                total, completed = event["total"], event["completed"]
                if total is None:
                    counter = f"{completed:g} {event['unit']}" if completed else (
                        "working" if event['status'] == 'running' else event['status'])
                else:
                    fraction = completed / total if total else 1
                    bar = "=" * int(fraction * 16) + "." * (16 - int(fraction * 16))
                    counter = f"[{bar}] {completed:g}/{total:g} {event['unit']} ({fraction:.0%})"
                eta = event["eta_seconds"]
                eta_text = duration(eta, round_up=True) if eta is not None else "estimating"
                finish = f" | finish ~{event['estimated_finish_at']}" if event['estimated_finish_at'] else ""
                detail = f" | {event['detail']}" if event.get('detail') else ""
                self.stream.write(f"[{event['status'].upper()}] {event['stage']}: {counter}"
                                  f" | elapsed {duration(event['elapsed_seconds'])} | ETA {eta_text}"
                                  f"{finish}{detail}\n")
                self.stream.flush()
            if self.log:
                with self.log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")


_reporter = Reporter()


def configure(console=True, log=None, interval=1.0):
    global _reporter
    _reporter = Reporter(console, log, interval)
    return _reporter


class Progress:
    def __init__(self, label, total=None, unit="items", reporter=None,
                 clock=time.perf_counter, wall_clock=None, heartbeat=5.0):
        if total is not None and (not math.isfinite(total) or total < 0):
            raise ValueError("Progress total must be nonnegative and finite")
        self.label, self.total, self.unit = label, total, unit
        self.reporter = reporter if reporter is not None else _reporter
        self.clock = clock
        self.wall_clock = wall_clock or (lambda: datetime.now(BANGKOK))
        self.started = clock()
        self.completed = 0
        self.samples = deque([(self.started, 0)], maxlen=50)
        self.last_emit = -math.inf
        self.status, self.detail = "running", None
        self.heartbeat = heartbeat
        self.stop = threading.Event()
        self.thread = None
        self.lock = threading.RLock()

    def snapshot(self):
        with self.lock:
            now = self.clock()
            elapsed = max(0, now - self.started)
            eta = None
            if self.status == "done":
                eta = 0.0
            elif self.status == "running" and self.total is not None and len(self.samples) >= 3:
                first_time, first_count = self.samples[0]
                span = now - first_time
                if span >= 1 and self.completed > first_count:
                    eta = max(0, (self.total - self.completed) * span / (self.completed - first_count))
            wall = self.wall_clock().astimezone(BANGKOK)
            # Guard against nonsensical ETA overflow after extremely slow/early samples.
            finish = (wall + timedelta(seconds=eta)).isoformat(timespec="seconds") if eta is not None and eta < 365*86400 else None
            return {"time": wall.isoformat(timespec="seconds"), "stage": self.label,
                    "status": self.status, "completed": self.completed, "total": self.total,
                    "unit": self.unit, "elapsed_seconds": elapsed, "eta_seconds": eta,
                    "estimated_finish_at": finish, "detail": self.detail}

    def _emit(self, force=False):
        if not self.reporter.enabled:
            return
        with self.lock:
            now = self.clock()
            if force or now - self.last_emit >= self.reporter.interval:
                self.reporter.emit(self.snapshot())
                self.last_emit = now

    def __enter__(self):
        self._emit(force=True)
        if self.reporter.enabled and self.heartbeat:
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return self

    def _heartbeat(self):
        while not self.stop.wait(self.heartbeat):
            self._emit()

    def update(self, completed=None, detail=None):
        if not self.reporter.enabled:
            return
        with self.lock:
            if self.status != "running":
                return
            if completed is not None:
                if completed < self.completed or not math.isfinite(completed):
                    raise ValueError("Progress cannot go backwards or be non-finite")
                self.completed = min(completed, self.total) if self.total is not None else completed
                now = self.clock()
                # Bound sampling overhead and keep a useful time window for fast loops.
                if now - self.samples[-1][0] >= .5:
                    self.samples.append((now, self.completed))
            if detail is not None:
                self.detail = str(detail)[:160]
            self._emit()

    def advance(self, amount=1, detail=None):
        self.update(self.completed + amount, detail)

    def reset_rate(self):
        """Discard one-time warmup from ETA without resetting elapsed/completed work."""
        with self.lock:
            self.samples.clear()
            self.samples.append((self.clock(), self.completed))

    def finish(self, status="done", detail=None):
        self.stop.set()
        if self.thread is not None and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1)
        with self.lock:
            if self.status != "running":
                return
            self.status = status
            if status == "done" and self.total is not None:
                self.completed = self.total
            if detail is not None:
                self.detail = str(detail)[:160]
            self._emit(force=True)

    def __exit__(self, exc_type, exc, traceback):
        status = "done" if exc_type is None else (
            "cancelled" if issubclass(exc_type, (KeyboardInterrupt, GeneratorExit)) else "failed")
        self.finish(status, str(exc) if exc is not None else None)
        return False


def track(iterable, label, total=None, unit="items"):
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            pass
    with Progress(label, total=total, unit=unit) as progress:
        for item in iterable:
            yield item
            progress.advance()


def trainer_progress_callback(base):
    """Build a TrainerCallback only after Unsloth/Transformers have been imported."""
    class Callback(base):
        def __init__(self):
            self.training = None
            self.validation = None
            self.checkpoint = None

        def _start_checkpoint(self, step):
            if self.checkpoint is None:
                self.checkpoint = Progress(f"Save checkpoint at step {step}")
                self.checkpoint.__enter__()

        def on_train_begin(self, args, state, control, **kwargs):
            self.training = Progress("Training optimizer steps", total=state.max_steps, unit="steps")
            self.training.__enter__()

        def on_step_end(self, args, state, control, **kwargs):
            if self.training:
                self.training.update(state.global_step, f"epoch={state.epoch}")
                if state.global_step == 1:
                    self.training.reset_rate()
            if getattr(control, "should_save", False) and not getattr(control, "should_evaluate", False):
                self._start_checkpoint(state.global_step)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.training and logs and "loss" in logs:
                self.training.update(detail=f"loss={logs['loss']:.4f}")

        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
            if self.validation is None:
                try:
                    total = len(eval_dataloader)
                except TypeError:
                    total = None
                self.validation = Progress("Validation forward passes", total=total, unit="batches")
                self.validation.__enter__()
            self.validation.advance()

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if self.validation:
                self.validation.finish(detail=f"eval_loss={(metrics or {}).get('eval_loss', 'unknown')}")
                self.validation = None
            if getattr(control, "should_save", False):
                self._start_checkpoint(state.global_step)

        def on_save(self, args, state, control, **kwargs):
            if self.checkpoint:
                self.checkpoint.finish()
                self.checkpoint = None
            if self.training:
                self.training.update(detail=f"checkpoint saved at step {state.global_step}")

        def on_train_end(self, args, state, control, **kwargs):
            if self.training:
                if state.global_step >= state.max_steps:
                    self.training.finish()
                else:
                    self.training.finish("stopped", "Training stopped before planned steps")

        def close(self, status="failed"):
            for stage in (self.validation, self.checkpoint, self.training):
                if stage:
                    stage.finish(status)
    return Callback()
