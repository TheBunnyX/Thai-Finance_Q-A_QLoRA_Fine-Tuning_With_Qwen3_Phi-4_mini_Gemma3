"""Kernel experiment in the separate .venv-kernel (no Unsloth): HF + PEFT QLoRA, optional Liger / torch.compile.

    .venv-kernel\Scripts\python.exe -X utf8 scripts/bench_kernels.py --model qwen --mode base|liger|compile|liger_compile

Same data, NF4+double quant, BF16, LoRA r16/a32 on language-model projections, answer-only loss,
effective batch 8, gradient checkpointing OFF. Times optimizer steps after warmup (compile overhead reported apart).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mode", default="base", choices=["base", "liger", "compile", "liger_compile"])
ap.add_argument("--data", default="data/processed/finance-1250-split-v1")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--micro-batch", type=int)
ap.add_argument("--out", default="outputs/bench/kernels")
a = ap.parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from finance_qlora.data import load_prepared
from finance_qlora.modeling import AnswerCollator, encode_pair, settings

spec, cfg = settings(a.model)
micro = a.micro_batch or spec.get("train_batch_size", cfg["batch_size"])
accum = (cfg["batch_size"] * cfg["gradient_accumulation_steps"]) // micro
if "liger" in a.mode:
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    from transformers import AutoConfig
    mt = AutoConfig.from_pretrained(spec["model_id"], revision=spec["revision"]).model_type
    fn = MODEL_TYPE_TO_APPLY_LIGER_FN[mt]
    fn()
    print("liger patched", mt, fn.__name__)
tok = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
splits, _ = load_prepared(a.data)
rows = [encode_pair(tok, r, cfg["max_seq_length"]) for r in splits["train"]]
rows.sort(key=lambda r: len(r["input_ids"]))
collate = AnswerCollator(tok.pad_token_id, cfg["pad_to_multiple_of"], cfg["max_seq_length"])
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(spec["model_id"], revision=spec["revision"], quantization_config=bnb,
                                             dtype=torch.bfloat16, device_map={"": 0})
mods = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)" if a.model == "gemma" else \
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj"]
if a.model != "gemma":  # Phi fuses qkv / gate_up; keep only names that exist
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    mods = [m for m in mods if m in present]
model = get_peft_model(model, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                                         target_modules=mods, bias="none", task_type="CAUSAL_LM"))
model.config.use_cache = False
if "compile" in a.mode:
    model = torch.compile(model)
import bitsandbytes as bnb_lib
opt = bnb_lib.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])
model.train()
# length-grouped micro-batches, shuffled order (same idea as group_by_length)
import random
random.seed(42)
batches = [rows[i:i + micro] for i in range(0, len(rows), micro)]
random.shuffle(batches)
it = iter(batches * 4)
times, losses = [], []
torch.cuda.reset_peak_memory_stats()
for step in range(a.steps):
    torch.cuda.synchronize(); t0 = time.perf_counter(); total = 0.0
    for _ in range(accum):
        batch = {k: v.cuda() for k, v in collate(next(it)).items() if k in ("input_ids", "attention_mask", "labels")}
        out = model(**batch)
        (out.loss / accum).backward(); total += out.loss.item() / accum
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t0); losses.append(total)
    print(f"step {step+1} {times[-1]:.2f}s loss {total:.3f}", flush=True)
warm = 5
res = {"model": a.model, "mode": a.mode, "micro_batch": micro, "first_step_s": round(times[0], 1),
       "median_step_s_after_warmup": round(statistics.median(times[warm:]), 3),
       "mean_step_s_after_warmup": round(statistics.mean(times[warm:]), 3),
       "total_s": round(sum(times), 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "loss_first": round(losses[0], 3), "loss_last": round(losses[-1], 3), "loss_mean": round(statistics.mean(losses), 3)}
print(json.dumps(res))
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / f"{a.model}-{a.mode}.json").write_text(json.dumps(res), encoding="utf-8")
