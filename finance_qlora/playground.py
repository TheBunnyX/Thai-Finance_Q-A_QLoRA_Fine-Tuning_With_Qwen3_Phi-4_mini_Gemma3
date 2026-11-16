"""Loopback-only inference API for the Next.js playground (no extra Python deps)."""
import argparse
import gc
import json
import queue
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import calculator
from .modeling import ROOT, chat_token_ids, load_model


class RequestError(ValueError):
    pass


# The adapters were trained with 512-token sequences; the untuned model is not bound by that, and it
# answers much longer (most baseline answers exceed 256 tokens), so it gets the earlier 1,024 window.
BASE_CONTEXT = 1024
MIN_ANSWER_TOKENS = 32


def context_for(adapter_context, variant):
    return adapter_context if variant == "sft" else max(adapter_context, BASE_CONTEXT)


def answer_budget(prompt_tokens, requested, context):
    """Tokens the answer may use: the request, clamped to what is left of the context window.

    A question that is merely long no longer fails the whole request; it just shrinks the budget.
    Only a question that leaves almost no room is rejected.
    """
    available = context - prompt_tokens
    if available < MIN_ANSWER_TOKENS:
        raise RequestError(f"Question too long: only {max(available, 0)} tokens are left for the answer "
                           f"(context {context:,} tokens). Please shorten the question.")
    return min(requested, available)


def catalog():
    # Runs from scripts/run_2500_v2.sh (outputs/<model>-2500-<variant>); best numeric result first.
    labels = {"qwen": "Qwen3 · 4B", "phi": "Phi-4 mini · 3.8B", "gemma": "Gemma 3 · 4B"}
    variants = ("v2-steps", "v2", "v2-cap15")
    names = {f"{model}-2500-{variant}": (model, f"{labels[model]} · 2500 {variant}", f"{model}-2500-{variant}")
             for model in ("qwen", "gemma", "phi") for variant in variants}
    result = []
    for key, (model_key, name, run_name) in names.items():
        path = ROOT / "outputs" / run_name
        try:
            run = json.loads((path / "run.json").read_text(encoding="utf-8"))
            ready = (run.get("status") == "complete" and run.get("model_key") == model_key
                     and (path / "adapter/adapter_config.json").is_file())
        except (OSError, ValueError):
            run, ready = {}, False
        result.append({"id": key, "name": name, "ready": ready,
                       "model_id": run.get("model", {}).get("model_id", ""),
                       "run": path.name, "epochs": run.get("training", {}).get("num_train_epochs"),
                       "quantization": "NF4 · BF16", "context": run.get("training", {}).get("max_seq_length", 1024),
                       "base_context": context_for(run.get("training", {}).get("max_seq_length", 1024), "base")})
    return result


def validate_request(payload):
    if not isinstance(payload, dict):
        raise RequestError("Request must be a JSON object")
    model = payload.get("model")
    if model not in {item["id"] for item in catalog() if item["ready"]}:
        raise RequestError("This model has no completed adapter yet")
    variant = payload.get("variant", "sft")
    if variant not in ("base", "sft"):
        raise RequestError("variant must be base or sft")
    question = payload.get("question")
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 6000:
        raise RequestError("Enter a question of 1–6,000 characters")
    tokens = payload.get("max_new_tokens", 256)
    if type(tokens) is not int or not 32 <= tokens <= 768:
        raise RequestError("max_new_tokens must be an integer between 32 and 768")
    return {"model": model, "variant": variant, "question": question.strip(), "max_new_tokens": tokens}


def evaluation_active():
    # Avoid starting playground inference during a recently active tracked GPU job.
    for path in (ROOT / "reports").glob("*-progress.jsonl"):
        try:
            with path.open("rb") as stream:
                offset = max(0, path.stat().st_size - 16384)
                stream.seek(offset)
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            continue
        # Parse line by line: a half-written tail line must not hide earlier valid heartbeats.
        last = None
        for line in (lines[1:] if offset else lines):
            if '"stage": "Command evaluate"' not in line and '"stage": "Command train"' not in line:
                continue
            try:
                event = json.loads(line)
                event_time = datetime.fromisoformat(event["time"])
                status = event["status"]
            except (ValueError, KeyError, TypeError):
                continue
            if event_time.tzinfo is None:
                continue
            last = (status, event_time)
        if last:
            age = (datetime.now(timezone.utc) - last[1]).total_seconds()
            if last[0] == "running" and 0 <= age < 60:
                return True
    return False


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.loaded = None
        self.model = self.tokenizer = None
        self.config = None
        self.phase = "idle"
        self.light_tokenizers = {}

    def status(self):
        return {"models": catalog(), "loaded": self.loaded, "phase": self.phase,
                "busy": self.lock.locked(), "evaluation_active": evaluation_active()}

    def unload(self):
        self.model = self.tokenizer = self.config = None
        self.loaded = None
        gc.collect()
        # Importing torch before Unsloth would prevent some Unsloth patches.
        import sys
        if "torch" in sys.modules:
            import torch
            torch.cuda.empty_cache()

    def load(self, key, variant):
        if self.loaded == {"model": key, "variant": variant}:
            return
        self.unload()
        item = next((entry for entry in catalog() if entry["id"] == key and entry["ready"]), None)
        if not item:
            raise RequestError("This model has no completed adapter yet")
        path = ROOT / "outputs" / item["run"]
        run = json.loads((path / "run.json").read_text(encoding="utf-8"))
        config = dict(run["training"], max_seq_length=context_for(run["training"]["max_seq_length"], variant))
        model, _, tokenizer, fast, _ = load_model(run["model"], config)
        if variant == "sft":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(path / "adapter"), is_trainable=False)
        fast.for_inference(model)
        model.eval()
        model.config.use_cache = True
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.loaded = {"model": key, "variant": variant}

    def light_tokenizer(self, key):
        """Tokenizer saved with the adapter at training time; loads without GPU weights."""
        if key not in self.light_tokenizers:
            item = next((entry for entry in catalog() if entry["id"] == key and entry["ready"]), None)
            if not item:
                raise RequestError("This model has no completed adapter yet")
            import unsloth  # noqa: F401  Must precede transformers so Unsloth patches still apply.
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(ROOT / "outputs" / item["run"] / "adapter"), trust_remote_code=False)
            self.light_tokenizers[key] = (tokenizer if tokenizer.chat_template else None,
                                          item["context"])
        return self.light_tokenizers[key]

    def check_context(self, request):
        """Reject over-long requests before spending time loading model weights."""
        if self.loaded and self.loaded["model"] == request["model"]:
            return  # The loaded tokenizer is checked right after load() anyway.
        try:
            tokenizer, context = self.light_tokenizer(request["model"])
        except RequestError:
            raise
        except Exception:
            return  # Best effort only: the authoritative check runs after load().
        if tokenizer is None:
            return
        ids = chat_token_ids(tokenizer, [{"role": "user", "content": request["question"]}], True)
        answer_budget(len(ids), request["max_new_tokens"], context_for(context, request["variant"]))

    def generate(self, request, emit):
        started = time.perf_counter()
        self.check_context(request)
        self.phase = "loading"
        emit({"type": "stage", "stage": "loading", "message": "Preparing the model on the GPU…"})
        self.load(request["model"], request["variant"])
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        ids = chat_token_ids(self.tokenizer, [{"role": "user", "content": request["question"]}], True)
        budget = answer_budget(len(ids), request["max_new_tokens"], self.config["max_seq_length"])
        inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
        stop = threading.Event()

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.5)
        result = {}
        self.phase = "generating"
        emit({"type": "stage", "stage": "generating", "message": "Generating the answer…", "prompt_tokens": len(ids), "budget": budget})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        generation_started = time.perf_counter()

        def work():
            try:
                with torch.inference_mode():
                    result["output"] = self.model.generate(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        do_sample=False, num_beams=1, max_new_tokens=budget,
                        pad_token_id=self.tokenizer.pad_token_id, streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([Cancelled()]))
            except Exception as exc:
                result["error"] = exc
                streamer.end()

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        text, first_text = "", None
        last_heartbeat = time.perf_counter()
        try:
            while True:
                try:
                    chunk = next(streamer)
                except queue.Empty:
                    if not worker.is_alive():
                        break
                    if time.perf_counter() - last_heartbeat >= 3:
                        emit({"type": "heartbeat"})
                        last_heartbeat = time.perf_counter()
                    continue
                except StopIteration:
                    break
                if chunk:
                    first_text = first_text or time.perf_counter()
                    text += chunk
                    emit({"type": "token", "text": chunk})
        finally:
            # A disconnected browser must not release the GPU lock while generation still runs.
            stop.set()
            worker.join()
        if "error" in result:
            raise result["error"]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - generation_started
        output = result["output"][0, len(ids):].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        answer = text.strip()
        emit({"type": "done", "answer": answer, "check": calculator.check(request["question"], answer),
              "model": request["model"], "variant": request["variant"],
              "metrics": {"generated_tokens": len(output), "prompt_tokens": len(ids),
                          "generation_seconds": seconds, "total_seconds": time.perf_counter() - started,
                          "first_text_seconds": first_text - generation_started if first_text else None,
                          "tokens_per_second": len(output) / seconds,
                          "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "budget_used": budget, "hit_limit": len(output) == budget and output[-1] not in eos}})


ENGINE = Engine()


class Handler(BaseHTTPRequestHandler):
    def json_response(self, code, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/status":
            return self.json_response(404, {"error": "Not found"})
        self.json_response(200, ENGINE.status())

    def do_POST(self):
        if self.path not in ("/chat", "/unload"):
            return self.json_response(404, {"error": "Not found"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.json_response(415, {"error": "Use application/json"})
        # No CORS: API is used through the same-origin Next.js server proxy.
        if self.headers.get("Origin"):
            return self.json_response(403, {"error": "Use the Next.js proxy"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise RequestError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            request = validate_request(payload) if self.path == "/chat" else None
        except (ValueError, UnicodeError) as exc:
            return self.json_response(400, {"error": str(exc)})
        # /unload must stay available during training so the job can reclaim VRAM.
        if self.path == "/chat" and evaluation_active():
            return self.json_response(409, {"error": "Training or evaluation is running. Please wait for it to finish."})
        if not ENGINE.lock.acquire(blocking=False):
            return self.json_response(409, {"error": "The GPU is busy with another request. Please wait a moment."})
        try:
            if self.path == "/unload":
                ENGINE.unload()
                return self.json_response(200, {"ok": True})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def emit(event):
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()

            try:
                ENGINE.generate(request, emit)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except RequestError as exc:
                # Invalid input (e.g. too long for the context): keep the loaded model.
                try:
                    emit({"type": "error", "message": str(exc)})
                except OSError:
                    pass
            except Exception as exc:
                try:
                    emit({"type": "error", "message": str(exc)})
                except OSError:
                    pass
                # Clear a failed/partially loaded model (including after CUDA OOM).
                ENGINE.unload()
        finally:
            ENGINE.phase = "idle"
            ENGINE.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"Finance playground API: http://127.0.0.1:{args.port} (GPU loads on first question)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
