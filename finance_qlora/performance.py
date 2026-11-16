"""Reusable CPU token cache and resource settings, with no GPU imports."""
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from .progress import Progress, track


def json_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_encode(tokenizer, rows, max_length, model_spec, cache_dir):
    from .modeling import encode_pair
    started = time.perf_counter()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    vocab = backend.to_str() if backend is not None else tokenizer.get_vocab()
    identity = {
        "format": 2, "model": model_spec, "max_length": max_length,
        "template": tokenizer.chat_template,
        "tokenizer": vocab, "special_tokens": tokenizer.special_tokens_map,
        "class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
        # Metadata changes do not alter tokens; the caller still verifies the full split manifest.
        "pairs": [[r["question"], r["answer"]] for r in rows],
    }
    key = json_hash(identity)
    path = Path(cache_dir) / f"{key}.json"
    encoded = None
    if path.exists():
        try:
            with Progress("Load and verify token cache") as progress:
                entry = json.loads(path.read_text(encoding="utf-8"))
                if entry["key"] == key and json_hash(entry["rows"]) == entry["rows_sha256"]:
                    encoded = entry["rows"]
                    progress.update(detail=f"cache hit: {len(encoded)} rows; skipping tokenization")
        except (ValueError, KeyError, TypeError):
            pass  # Rebuild interrupted/corrupted local caches, never silently use them.
    hit = encoded is not None
    if not hit:
        encoded = [encode_pair(tokenizer, row, max_length)
                   for row in track(rows, "Tokenize Q&A and check response labels", unit="rows")]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "rows": encoded, "rows_sha256": json_hash(encoded)}
        # Atomic replacement also allows independent runs to share this cache safely.
        with Progress("Write token cache"), tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return encoded, {"hit": hit, "key": key, "seconds": time.perf_counter() - started,
                     "rows": len(encoded)}


def training_plan(config, rows, max_steps=None, batch_size=None):
    config = dict(config)
    effective = config["batch_size"] * config["gradient_accumulation_steps"]
    if batch_size is not None:
        if batch_size < 1 or effective % batch_size:
            raise ValueError(f"--batch-size must divide effective batch size {effective}")
        config["batch_size"] = batch_size
        config["gradient_accumulation_steps"] = effective // batch_size
    if rows < 1:
        raise ValueError("Training split is empty")
    steps = max_steps if max_steps is not None else math.ceil(
        math.ceil(math.ceil(rows / config["batch_size"]) /
                  config["gradient_accumulation_steps"]) * config["num_train_epochs"])
    if steps < 1:
        raise ValueError("Training requires at least one optimizer step")
    interval = min(config["eval_steps"], steps)
    return config, interval, steps


def length_sampler_options(config_class):
    """Transformers 5 renamed this setting; support installed 4.x/5.x APIs explicitly."""
    fields = config_class.__dataclass_fields__
    if "train_sampling_strategy" in fields:
        return {"train_sampling_strategy": "group_by_length"}
    if "group_by_length" in fields:
        return {"group_by_length": True}
    raise RuntimeError("Installed Trainer lacks length grouping; check pinned dependencies")


def generation_batches(prompts, batch_size):
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    # Similar-length prompts share a batch to limit KV-cache memory and left padding.
    indices = sorted(range(len(prompts)), key=lambda i: len(prompts[i])) if batch_size > 1 else list(range(len(prompts)))
    return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


def trim_generated(tokens, eos_ids):
    for i, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:i + 1]
    return tokens


def evaluation_subset(rows, limit=None, seed=42):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if limit is None or limit >= len(rows):
        selected = list(rows)
    else:
        indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
        selected = [rows[i] for i in indices]
    return selected, {"limit": limit, "seed": seed if limit is not None else None,
                      "rows": len(selected), "ids_sha256": json_hash([r["id"] for r in selected])}
