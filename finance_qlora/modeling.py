"""GPU dependencies are imported lazily; dataset work needs none of them."""
import importlib.metadata
import json
import os
import platform
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from .progress import Progress, track

ROOT = Path(__file__).resolve().parents[1]


def settings(model_key):
    catalog = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    if model_key not in catalog:
        raise ValueError(f"Unknown model {model_key!r}; choose {', '.join(catalog)}")
    config = json.loads((ROOT / "configs/training.json").read_text(encoding="utf-8"))
    return dict(catalog[model_key]), config


def environment():
    names = ["torch", "unsloth", "unsloth_zoo", "trl", "transformers", "peft",
             "datasets", "accelerate", "bitsandbytes", "triton", "triton-windows", "numpy"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions}


def doctor(training=False):
    result = environment()
    errors = []
    if sys.version_info[:3] != (3, 12, 10):
        errors.append("Use Python 3.12.10 from this project's .venv")
    if sys.prefix == sys.base_prefix:
        errors.append("Python is not running inside a virtual environment")
    if training:
        try:
            import unsloth  # Must precede torch/transformers imports.
            result["unsloth_import"] = True
        except Exception as exc:
            errors.append(f"Unsloth import failed: {type(exc).__name__}: {exc}")
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_runtime"] = torch.version.cuda
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable")
        else:
            result["gpu"] = torch.cuda.get_device_name(0)
            result["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            result["bf16_supported"] = torch.cuda.is_bf16_supported()
            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            loss = (x @ x.T).square().mean()
            loss.backward()
            torch.cuda.synchronize()
            assert torch.isfinite(x.grad).all().item()
            result["cuda_backward"] = "PASS"
    except Exception as exc:
        errors.append(f"GPU check failed: {type(exc).__name__}: {exc}")
    if training:
        for name in ("unsloth", "trl", "transformers", "peft", "datasets", "accelerate", "bitsandbytes"):
            if result["packages"][name] is None:
                errors.append(f"Missing package: {name}")
    result["errors"] = errors
    return result


def resolve_revision(spec):
    """Return an immutable commit SHA; skip the Hub call when it is already pinned."""
    revision = str(spec.get("revision") or "")
    # Avoid network access for pinned runs so cached weights load offline.
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi
    return HfApi().model_info(spec["model_id"], revision=spec.get("revision")).sha


def load_model(spec, config):
    # Original checkpoints + exact name prevent silent switching to a dynamic quantized model.
    from unsloth import FastModel
    import torch
    from transformers import BitsAndBytesConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires CUDA with BF16 support")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    revision = resolve_revision(spec)
    if not revision:
        raise RuntimeError("Could not resolve immutable model revision")
    # Experiment only: an explicit attention backend (e.g. flex_attention) bypasses Unsloth's
    # automatic choice. Unset keeps Unsloth's default, which is the supported path.
    attention_kwargs = {}
    if os.environ.get("FINANCE_QLORA_ATTN_IMPL"):
        attention_kwargs["attn_implementation"] = os.environ["FINANCE_QLORA_ATTN_IMPL"]
    model, processor = FastModel.from_pretrained(
        model_name=spec["model_id"], revision=revision,
        max_seq_length=config["max_seq_length"], dtype=torch.bfloat16,
        load_in_4bit=True, full_finetuning=False, quantization_config=quant,
        use_exact_model_name=True, trust_remote_code=False,
        use_gradient_checkpointing="unsloth", device_map={"": 0}, **attention_kwargs,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if not tokenizer.chat_template:
        raise RuntimeError("Model has no native chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Check actual modules, not just the requested config: silent FP4 is unacceptable.
    from bitsandbytes.nn import Linear4bit
    modules = [m for m in model.modules() if isinstance(m, Linear4bit)]
    if not modules or any(getattr(m.weight, "quant_type", None) != "nf4" for m in modules):
        raise RuntimeError("Loaded model is not NF4; verify this Unsloth/bitsandbytes combination")
    if any(not getattr(m.weight, "compress_statistics", False) for m in modules):
        raise RuntimeError("Double quantization is not active")
    return model, processor, tokenizer, FastModel, revision


def chat_token_ids(tokenizer, messages, add_generation_prompt=False):
    """Normalize list, BatchEncoding and single-batch tensor tokenizer APIs."""
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    ids = result["input_ids"] if isinstance(result, Mapping) else result
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("Expected exactly one conversation from tokenizer")
        ids = ids[0]
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError("Tokenizer did not return a non-empty token ID sequence")
    return list(ids)


def encode_pair(tokenizer, row, max_length):
    """Native template, exact prefix check, answer-only labels, no silent truncation."""
    user = [{"role": "user", "content": row["question"]}]
    full = user + [{"role": "assistant", "content": row["answer"]}]
    prompt_ids = chat_token_ids(tokenizer, user, add_generation_prompt=True)
    input_ids = chat_token_ids(tokenizer, full)
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not an exact token prefix; review assistant masking")
    if len(input_ids) > max_length:
        raise ValueError(f"Row {row.get('id', '?')} has {len(input_ids)} tokens > {max_length}; "
                         "revise data or raise max_seq_length; no automatic truncation")
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("No assistant tokens remain")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]}


def inspect_tokens(model_key, data_directory):
    """Download tokenizer/config only; do not allocate model weights or GPU memory."""
    from transformers import AutoTokenizer
    from .data import load_prepared
    spec, config = settings(model_key)
    splits, _ = load_prepared(data_directory)
    with Progress(f"Load tokenizer {spec['model_id']}"):
        revision = resolve_revision(spec)
        tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=revision,
                                                  trust_remote_code=False)
    result = {"model": dict(spec, revision=revision), "max_seq_length": config["max_seq_length"],
              "splits": {}}
    for name, rows in splits.items():
        lengths, failures = [], []
        for row in track(rows, f"Check {name} token lengths and labels", unit="rows"):
            try:
                encoded = encode_pair(tokenizer, row, config["max_seq_length"])
                lengths.append(len(encoded["input_ids"]))
            except ValueError as exc:
                failures.append({"id": row["id"], "error": str(exc)})
        lengths.sort()
        result["splits"][name] = {
            "rows": len(rows), "passed": len(lengths), "failures": failures,
            "p95_passed_tokens": lengths[int((len(lengths)-1)*.95)] if lengths else None,
            "max_passed_tokens": max(lengths) if lengths else None,
        }
    return result


class AnswerCollator:
    def __init__(self, pad_id, pad_to_multiple_of=8, max_length=None, track_tokens=False):
        self.pad_id = pad_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length
        self.track_tokens = track_tokens

    def __call__(self, examples):
        import torch
        length = max(len(e["input_ids"]) for e in examples)
        if self.max_length is not None and length > self.max_length:
            raise ValueError("Collator input exceeds context length")
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            length = ((length + multiple - 1) // multiple) * multiple
            if self.max_length is not None:
                length = min(length, self.max_length)
        result = {}
        for name, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
            result[name] = torch.tensor([list(e[name]) + [pad] * (length - len(e[name]))
                                         for e in examples], dtype=torch.long)
        if self.track_tokens:
            result["_token_counts"] = (
                sum(len(e["input_ids"]) for e in examples),
                sum(sum(t != -100 for t in e["labels"]) for e in examples),
                length * len(examples),
            )
        return result
