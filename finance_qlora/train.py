import json
import os
import time

from .data import digest, load_prepared, write_json
from .modeling import AnswerCollator, environment, load_model, settings
from .performance import cached_encode, length_sampler_options, training_plan
from .telemetry import directory_size, measured_trainer_class, peak_process_ram
from .progress import Progress, trainer_progress_callback


def resolved_requirements():
    """`pip freeze`-style list of installed distributions without requiring pip."""
    import importlib.metadata
    pins = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pins[name.lower()] = f"{name}=={dist.version}"
    return "".join(pins[key] + "\n" for key in sorted(pins))


def attention_implementation(model):
    """Attention backend per config (decoder and any sub-configs), e.g. sdpa or flex_attention."""
    found = {}
    for name, module in model.named_modules():
        config = getattr(module, "config", None)
        impl = getattr(config, "_attn_implementation", None)
        if impl is not None:
            found.setdefault(type(config).__name__, impl)
    return found


def train(args):
    splits, manifest = load_prepared(args.data)
    spec, config = settings(args.model)
    # Optional experiment overrides; omitted flags keep configs/training.json unchanged.
    if getattr(args, "epochs", None) is not None:
        config = dict(config, num_train_epochs=args.epochs)
    if getattr(args, "learning_rate", None) is not None:
        config = dict(config, learning_rate=args.learning_rate)
    if getattr(args, "lora_rank", None) is not None:
        config = dict(config, lora_rank=args.lora_rank, lora_alpha=2 * args.lora_rank)
    if getattr(args, "no_gradient_checkpointing", False) and getattr(args, "gradient_checkpointing", False):
        raise ValueError("Choose either --gradient-checkpointing or --no-gradient-checkpointing")
    if getattr(args, "no_gradient_checkpointing", False):
        config = dict(config, gradient_checkpointing=False)
    if getattr(args, "gradient_checkpointing", False):
        config = dict(config, gradient_checkpointing="unsloth")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    # Per-model speed settings measured on this GPU (Gemma: head_dim 256 needs forced Flex
    # Attention; micro-batch 4 keeps effective batch 8). Must be set before Unsloth imports.
    if spec.get("flex_attention_large_head_dim"):
        os.environ.setdefault("UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM", "1")
    batch_size = args.batch_size if args.batch_size is not None else spec.get("train_batch_size")
    config, interval, estimated_steps = training_plan(
        config, len(splits["train"]), args.max_steps, batch_size)
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("Use a new/empty run directory to preserve previous results")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Progress(f"Load model {spec['model_id']} with NF4"):
        model, processor, tokenizer, fast, commit = load_model(spec, config)
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import set_seed, TrainerCallback
    set_seed(config["seed"])
    resolved = dict(spec, revision=commit)
    tokenized = {}
    token_lengths = {}
    token_cache = {}
    for name in ("train", "validation"):
        encoded, token_cache[name] = cached_encode(
            tokenizer, splits[name], config["max_seq_length"], resolved, args.cache_dir)
        tokenized[name] = Dataset.from_list(encoded).add_column(
            "length", [len(row["input_ids"]) for row in encoded])
        lengths = sorted(len(row["input_ids"]) for row in encoded)
        token_lengths[name] = {"rows": len(lengths), "max": max(lengths),
                               "p95": lengths[int((len(lengths)-1)*.95)],
                               "total_tokens": sum(lengths),
                               "answer_tokens": sum(sum(t != -100 for t in e["labels"]) for e in encoded)}
    with Progress("Attach language-only LoRA adapters"):
        model = fast.get_peft_model(
            model, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], bias="none",
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            use_gradient_checkpointing=config.get("gradient_checkpointing", "unsloth"), random_state=config["seed"],
        )
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Unexpected trainable parameters; expected only LoRA adapters")
    write_json(args.run_dir / "trainable_parameters.json", trainable)
    model.print_trainable_parameters()
    model.config.use_cache = False
    sampler_options = length_sampler_options(SFTConfig) if config["group_by_length"] else {}
    # Experiment path (FINANCE_QLORA_ATTN_IMPL): Inductor's flex lowering reads the legacy TF32
    # flags and refuses a mix with transformers' new fp32_precision API (torch 2.11), so set
    # TF32 through the legacy flags only. The default path keeps transformers' tf32=True.
    use_tf32 = True
    if os.environ.get("FINANCE_QLORA_ATTN_IMPL"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_tf32 = None
    training_args = SFTConfig(
        output_dir=str(args.run_dir / "checkpoints"),
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], num_train_epochs=config["num_train_epochs"],
        max_steps=args.max_steps or -1, bf16=True, fp16=False, tf32=use_tf32, optim=config["optim"],
        lr_scheduler_type=config["lr_scheduler_type"], warmup_ratio=config["warmup_ratio"],
        logging_steps=min(config["logging_steps"], interval),
        eval_strategy="steps", eval_steps=interval,
        save_strategy="steps", save_steps=interval,
        save_total_limit=2, load_best_model_at_end=True,
        save_only_model=config["save_only_model"],
        metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, eval_accumulation_steps=1,
        max_length=config["max_seq_length"], packing=False,
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=config["dataloader_pin_memory"],
        report_to="none", seed=config["seed"], disable_tqdm=True, **sampler_options,
    )
    progress_callback = trainer_progress_callback(TrainerCallback)
    trainer = measured_trainer_class(SFTTrainer)(
        model=model, processing_class=tokenizer, args=training_args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        data_collator=AnswerCollator(tokenizer.pad_token_id, config["pad_to_multiple_of"],
                                    config["max_seq_length"], track_tokens=True),
        callbacks=[progress_callback],
    )
    run = {"status": "initialized", "model_key": args.model, "model": resolved,
           "requested_model": spec, "training": config, "max_steps": args.max_steps,
           "dataset_manifest_sha256": digest(args.data / "manifest.json"),
           "dataset_manifest": manifest, "environment": environment(),
           "token_lengths": token_lengths, "token_cache": token_cache, "packing": False,
           "estimated_optimizer_steps": estimated_steps,
           "effective_batch_size": config["batch_size"] * config["gradient_accumulation_steps"],
           "eval_interval": interval, "sampler_options": sampler_options,
           # Record what actually ran, so attention benchmarks can be verified per run.
           "attention": {"implementation": attention_implementation(model),
                         "flex_large_head_dim_env": os.environ.get("UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM"),
                         "requested": os.environ.get("FINANCE_QLORA_ATTN_IMPL")},
           "quantization": "bitsandbytes NF4 + double quantization; BF16 compute"}
    write_json(args.run_dir / "run.json", run)
    # importlib.metadata works in uv-created venvs, which do not ship pip.
    (args.run_dir / "requirements-resolved.txt").write_text(resolved_requirements(), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    try:
        with Progress("Train, validate and save checkpoints"):
            result = trainer.train()
    finally:
        progress_callback.close()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_started
    metrics = dict(result.metrics)
    # load_best_model_at_end restores the checkpoint already evaluated during training.
    # Reuse its recorded validation metrics rather than forwarding the full split again.
    eval_logs = [item for item in trainer.state.log_history if "eval_loss" in item]
    if eval_logs and trainer.state.best_model_checkpoint:
        best_log = min(eval_logs, key=lambda item: item["eval_loss"])
        metrics.update({key: value for key, value in best_log.items() if key.startswith("eval_")})
        metrics["best_validation_step"] = best_log["step"]
    else:
        try:
            with Progress("Evaluate final validation loss"):
                metrics.update(trainer.evaluate())
        finally:
            progress_callback.close()
    adapter_path = args.run_dir / "adapter"
    with Progress("Save best adapter and tokenizer"):
        trainer.model.save_pretrained(adapter_path)
        processor.save_pretrained(adapter_path)
    counts = trainer.training_token_counts
    metrics.update(training_nonpadding_tokens=counts["nonpadding"],
                   training_answer_tokens=counts["answer"], training_padded_tokens=counts["padded"],
                   training_nonpadding_tokens_per_second_including_eval=counts["nonpadding"] / training_seconds,
                   training_answer_tokens_per_second_including_eval=counts["answer"] / training_seconds,
                   training_padding_efficiency=counts["nonpadding"] / counts["padded"] if counts["padded"] else None)
    run.update(status="complete", metrics=metrics,
               training_seconds=training_seconds,
               total_seconds=time.perf_counter() - started,
               peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
               **peak_process_ram(),
               adapter_size_bytes=directory_size(adapter_path),
               checkpoint_size_bytes=directory_size(args.run_dir / "checkpoints"),
               best_checkpoint=trainer.state.best_model_checkpoint,
               optimizer_steps=trainer.state.global_step,
               # Per-step train loss / eval loss for loss-curve reports.
               log_history=[{k: v for k, v in item.items()
                             if k in ("step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm")}
                            for item in trainer.state.log_history])
    with Progress("Save completed training metrics"):
        write_json(args.run_dir / "run.json", run)
    print(json.dumps(run, ensure_ascii=False, indent=2))
