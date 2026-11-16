import json
import threading
import unittest
import tempfile
from datetime import datetime, timezone, timedelta
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

from finance_qlora import playground


class PlaygroundTests(unittest.TestCase):
    def test_tracked_training_heartbeat_blocks_only_while_fresh_and_running(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(playground, "ROOT", Path(tmp)):
            reports = Path(tmp) / "reports"
            reports.mkdir()
            path = reports / "gemma-progress.jsonl"
            event = {"stage": "Command train", "status": "running", "time": datetime.now(timezone.utc).isoformat()}
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            self.assertTrue(playground.evaluation_active())
            event["status"] = "done"
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            self.assertFalse(playground.evaluation_active())
            event.update(status="running", time=(datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat())
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            self.assertFalse(playground.evaluation_active())

    def test_request_limits_and_allowlist(self):
        with patch.object(playground, "catalog", return_value=[{"id": "qwen", "ready": True}]):
            value = playground.validate_request({"model": "qwen", "question": "  ROE?  "})
            self.assertEqual(value["question"], "ROE?")
            for change in ({"model": "../../secret"}, {"variant": "other"}, {"question": " "},
                           {"max_new_tokens": True}, {"max_new_tokens": 10000}):
                with self.assertRaises(playground.RequestError):
                    playground.validate_request({"model": "qwen", "question": "ROE?", **change})

    def setUp(self):
        self.server = playground.ThreadingHTTPServer(("127.0.0.1", 0), playground.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, payload, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request("POST", path, json.dumps(payload), headers or {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def test_http_rejects_busy_gpu_without_loading(self):
        with patch.object(playground, "catalog", return_value=[{"id": "qwen", "ready": True}]), \
                patch.object(playground, "evaluation_active", return_value=False), \
                patch.object(playground.ENGINE, "load") as load:
            playground.ENGINE.lock.acquire()
            try:
                status, _ = self.request("/chat", {"model": "qwen", "question": "ROE?"})
            finally:
                playground.ENGINE.lock.release()
            self.assertEqual(status, 409)
            load.assert_not_called()

    def test_http_stream_and_lock_release_on_failure(self):
        def fake(request, emit):
            emit({"type": "token", "text": "ภาษาไทย"})
            raise RuntimeError("test failure")

        with patch.object(playground, "catalog", return_value=[{"id": "qwen", "ready": True}]), \
                patch.object(playground, "evaluation_active", return_value=False), \
                patch.object(playground.ENGINE, "generate", side_effect=fake), \
                patch.object(playground.ENGINE, "unload"):
            status, body = self.request("/chat", {"model": "qwen", "question": "ROE?"})
            messages = [json.loads(line) for line in body.splitlines()]
            self.assertEqual(status, 200)
            self.assertEqual(messages[0]["text"], "ภาษาไทย")
            self.assertEqual(messages[-1]["type"], "error")
            self.assertFalse(playground.ENGINE.lock.locked())

    def test_http_blocks_cross_origin_and_active_evaluation(self):
        status, _ = self.request("/unload", {}, {"Content-Type": "application/json", "Origin": "https://example.org"})
        self.assertEqual(status, 403)
        with patch.object(playground, "evaluation_active", return_value=True), \
                patch.object(playground, "catalog", return_value=[{"id": "qwen", "ready": True}]), \
                patch.object(playground.ENGINE, "unload") as unload:
            status, _ = self.request("/chat", {"model": "qwen", "question": "ROE?"})
            self.assertEqual(status, 409)
            # Releasing VRAM must stay possible while a training job is running.
            status, _ = self.request("/unload", {})
            self.assertEqual(status, 200)
            unload.assert_called_once()

    def test_request_error_keeps_loaded_model(self):
        def fake(request, emit):
            raise playground.RequestError("too long")

        with patch.object(playground, "catalog", return_value=[{"id": "qwen", "ready": True}]), \
                patch.object(playground, "evaluation_active", return_value=False), \
                patch.object(playground.ENGINE, "generate", side_effect=fake), \
                patch.object(playground.ENGINE, "unload") as unload:
            status, body = self.request("/chat", {"model": "qwen", "question": "ROE?"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body.splitlines()[-1])["type"], "error")
            unload.assert_not_called()

    def test_partial_heartbeat_line_does_not_hide_running_job(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(playground, "ROOT", Path(tmp)):
            reports = Path(tmp) / "reports"
            reports.mkdir()
            event = {"stage": "Command train", "status": "running", "time": datetime.now(timezone.utc).isoformat()}
            (reports / "qwen-progress.jsonl").write_text(
                json.dumps(event) + '\n{"stage": "Command train", "sta', encoding="utf-8")
            self.assertTrue(playground.evaluation_active())


    def test_answer_budget_clamps_instead_of_rejecting(self):
        self.assertEqual(playground.answer_budget(100, 320, 512), 320)
        self.assertEqual(playground.answer_budget(300, 320, 512), 212)  # shrinks to fit
        with self.assertRaises(playground.RequestError):
            playground.answer_budget(490, 320, 512)  # <32 tokens left for the answer
        self.assertEqual(playground.context_for(512, "sft"), 512)
        self.assertEqual(playground.context_for(512, "base"), 1024)

    def test_context_precheck_rejects_before_loading(self):
        class Tok:
            chat_template = "x"

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
                return list(range(500))

        engine = playground.Engine()
        engine.light_tokenizers["qwen"] = (Tok(), 512)
        request = {"model": "qwen", "variant": "sft", "question": "q", "max_new_tokens": 256}
        with patch.object(engine, "load") as load:
            with self.assertRaises(playground.RequestError):
                engine.generate(request, lambda event: None)
            load.assert_not_called()
        # A long-but-fitting question is clamped, not rejected: precheck passes for the same prompt
        # on the untuned model's larger window.
        engine.check_context(dict(request, variant="base"))
        # The adapter's 512 window leaves only 12 tokens after a 500-token prompt: still rejected.
        with self.assertRaises(playground.RequestError):
            engine.check_context(dict(request, max_new_tokens=999))

if __name__ == "__main__":
    unittest.main()
