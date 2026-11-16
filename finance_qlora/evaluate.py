import json
import time
from pathlib import Path

from .data import digest, load_prepared, write_json
from .modeling import chat_token_ids, environment, load_model, settings
from .performance import evaluation_subset, generation_batches, trim_generated
from .scoring import score_answer, summarize
from .telemetry import TokenTimingStreamer, peak_process_ram, percentile
from .progress import Progress, track


def generate_batch(model, tokenizer, prompts, max_new_tokens, device="cuda", streamer=None):
    import torch
    width = max(map(len, prompts))
    # Decoder-only generation needs left padding; do not change tokenizer global state.
    inputs = torch.tensor([[tokenizer.pad_token_id] * (width - len(ids)) + ids
                           for ids in prompts], dtype=torch.long, device=device)
    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids)
                         for ids in prompts], dtype=torch.long, device=device)
    extras = {"streamer": streamer} if streamer is not None else {}
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=mask, do_sample=False, num_beams=1,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id, **extras)
    eos = getattr(model.generation_config, "eos_token_id", None)
    eos = eos if eos is not None else tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    result = [trim_generated(tokens, eos_ids) for tokens in output[:, width:].cpu().tolist()]
    return result, eos_ids


def adapter_dataset_compatible(run, data_dir, manifest):
    """True when evaluating `data_dir` cannot leak the adapter's training rows.

    Either it is the exact prepared set used for training, or its train split has the same
    checksum as the adapter's training split: load_prepared already rejected any content overlap
    between that train split and this set's validation/test, so those rows were never trained on.
    """
    if run["dataset_manifest_sha256"] == digest(Path(data_dir) / "manifest.json"):
        return True
    trained = run.get("dataset_manifest", {}).get("splits", {}).get("train", {}).get("sha256")
    return bool(trained) and trained == manifest["splits"]["train"]["sha256"]


def evaluate(args):
    splits, manifest = load_prepared(args.data)
    spec, config = settings(args.model)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    warmup_batches = getattr(args, "warmup_batches", 0)
    if warmup_batches < 0:
        raise ValueError("--warmup-batches must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new/empty evaluation output directory")
    if args.adapter:
        run = json.loads((args.adapter.parent / "run.json").read_text(encoding="utf-8"))
        if run["status"] != "complete" or run["model_key"] != args.model:
            raise ValueError("Adapter run is incomplete or belongs to a different model")
        if not adapter_dataset_compatible(run, args.data, manifest):
            raise ValueError("Dataset changed since adapter training")
        spec, config = run["model"], run["training"]
    args.max_new_tokens = (config["max_new_tokens"] if args.max_new_tokens is None
                           else args.max_new_tokens)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    rows, selection = evaluation_subset(splits[args.split], getattr(args, "limit", None),
                                        getattr(args, "selection_seed", 42))
    with Progress(f"Load evaluation model {spec['model_id']}"):
        model, _, tokenizer, fast, commit = load_model(spec, config)
    import torch
    if args.adapter:
        from peft import PeftModel
        with Progress("Load trained adapter"):
            model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    fast.for_inference(model)
    model.eval()
    model.config.use_cache = True
    args.output.mkdir(parents=True, exist_ok=True)
    # Token lengths are checked before inference, without modifying held-out rows.
    prompts, tokenize_seconds = [], []
    for row in track(rows, "Prepare evaluation prompts", unit="questions"):
        token_started = time.perf_counter()
        ids = chat_token_ids(tokenizer, [{"role": "user", "content": row["question"]}],
                             add_generation_prompt=True)
        tokenize_seconds.append(time.perf_counter() - token_started)
        if len(ids) + args.max_new_tokens > config["max_seq_length"]:
            raise ValueError(f"Prompt + generation budget exceeds context: {row['id']}")
        prompts.append(ids)
    batches = generation_batches(prompts, args.batch_size)
    warmup_started = time.perf_counter()
    for i in track(range(warmup_batches), "Warm up generation (excluded from timing)", unit="batches"):
        indices = batches[i % len(batches)]
        timing = TokenTimingStreamer() if args.measure_ttft else None
        generate_batch(model, tokenizer, [prompts[j] for j in indices], args.max_new_tokens,
                       streamer=timing)
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    records, batch_times, ttft_times, decode_intervals = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for indices in track(batches,
                             "Generate and score answers", unit="batches"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            timing = TokenTimingStreamer(started) if args.measure_ttft else None
            generated, eos_ids = generate_batch(model, tokenizer, [prompts[i] for i in indices],
                                                args.max_new_tokens, streamer=timing)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            decode_started = time.perf_counter()
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            decode_elapsed = time.perf_counter() - decode_started
            batch_times.append(elapsed + decode_elapsed)
            timing_metrics = timing.metrics() if timing is not None else {
                "batch_ttft_seconds": None, "subsequent_token_seconds": None}
            if timing_metrics["batch_ttft_seconds"] is not None:
                ttft_times.append(timing_metrics["batch_ttft_seconds"])
            if timing_metrics["subsequent_token_seconds"]:
                decode_intervals.append((sum(max(0, len(t) - 1) for t in generated),
                                         timing_metrics["subsequent_token_seconds"]))
            for i, tokens, prediction in zip(indices, generated, predictions):
                row = rows[i]
                prediction = prediction.strip()
                record = {"id": row["id"], "row_index": i, "question": row["question"],
                          "reference": row["answer"], "prediction": prediction,
                          "scores": score_answer(prediction, row),
                          "language": row.get("language", "unspecified"),
                          "topic": row.get("topic", "unspecified"),
                          "evaluation": row.get("evaluation", {}),
                          "generation_seconds": elapsed / len(indices),
                          "batch_generation_seconds": elapsed, "batch_size": len(indices),
                          "tokenization_seconds": tokenize_seconds[i],
                          "request_latency_seconds": tokenize_seconds[i] + elapsed + decode_elapsed
                          if len(indices) == 1 else None,
                          "ttft_seconds": tokenize_seconds[i] + timing_metrics["batch_ttft_seconds"]
                          if len(indices) == 1 and timing_metrics["batch_ttft_seconds"] is not None else None,
                          **timing_metrics,
                          "generated_tokens": len(tokens),
                          "hit_generation_limit": len(tokens) == args.max_new_tokens
                          and (not tokens or tokens[-1] not in eos_ids)}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Buffered disk writes; preserve partial progress at bounded intervals.
            if len(records) % 32 < len(indices):
                stream.flush()
    metrics = summarize(records)
    for field in ("language", "topic"):
        metrics[f"by_{field}"] = {key: summarize([r for r in records if r[field] == key])
                                   for key in sorted({r[field] for r in records})}
    seconds = [r["generation_seconds"] for r in records]
    # Do not mix a short final batch into single-request latency statistics for batched runs.
    latencies = [r["request_latency_seconds"] for r in records] if args.batch_size == 1 else []
    request_ttfts = [r["ttft_seconds"] for r in records if r["ttft_seconds"] is not None] if args.batch_size == 1 else []
    metrics.update(mean_generation_seconds=sum(seconds)/len(seconds),
                   warmup_batches=warmup_batches, warmup_seconds=warmup_seconds,
                   tokens_per_second_including_first_run=sum(r["generated_tokens"] for r in records)/sum(seconds),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   **peak_process_ram(),
                   request_latency_rows=len(latencies),
                   request_latency_median_seconds=percentile(latencies, 50),
                   request_latency_p95_seconds=percentile(latencies, 95),
                   batch_latency_median_seconds=percentile(batch_times, 50),
                   batch_latency_p95_seconds=percentile(batch_times, 95),
                   ttft_rows=len(request_ttfts),
                   ttft_median_seconds=percentile(request_ttfts, 50),
                   ttft_p95_seconds=percentile(request_ttfts, 95),
                   batch_ttft_median_seconds=percentile(ttft_times, 50),
                   batch_ttft_p95_seconds=percentile(ttft_times, 95),
                   subsequent_tokens_per_second=(sum(n for n, _ in decode_intervals) /
                                                 sum(s for _, s in decode_intervals))
                   if decode_intervals else None,
                   hit_generation_limit=sum(r["hit_generation_limit"] for r in records))
    report = {"model": dict(spec, revision=commit or spec["revision"]),
              "metric_schema_version": 3, "selection": selection,
              "adapter": str(args.adapter) if args.adapter else None,
              "split": args.split, "split_sha256": manifest["splits"][args.split]["sha256"],
              "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                             "max_seq_length": config["max_seq_length"],
                             "batch_size": args.batch_size, "measure_ttft": args.measure_ttft,
                             "num_beams": 1, "warmup_batches": warmup_batches},
              "quantization": "NF4 double quantization BF16", "metrics": metrics,
              "environment": environment(),
              "limitations": "Character F1/EM are text overlap only. Numeric checks use explicit "
                              "annotations. Human review is required for factual correctness. "
                              f"Timing excludes {warmup_batches} warmup batches; additional shapes can still compile. "
                              "With zero warmup, first-generation overhead is included. "
                              "Request latency sums tokenization, generation and text decode stages; "
                              "excludes model load, dataset I/O, scoring, queue/network time. "
                              "TTFT is generated-token host availability, not first rendered word. "
                              "Streamer instrumentation may slow generation. "
                              "Per-row time is amortized batch time, not individual request latency. "
                              "Batched prediction rows are length-ordered; row_index retains source order."}
    with Progress("Save evaluation metrics"):
        write_json(args.output / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def compare(before_path, after_path):
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))
    for key in ("model", "split", "split_sha256", "generation", "quantization"):
        if before[key] != after[key]:
            raise ValueError(f"Unmatched evaluation settings: {key}")
    if before.get("selection") != after.get("selection"):
        raise ValueError("Different evaluation question selection")
    if before["adapter"] is not None or after["adapter"] is None:
        raise ValueError("Expected baseline first and adapter result second")
    if before["environment"]["packages"] != after["environment"]["packages"]:
        raise ValueError("Package versions differ between baseline and adapter evaluation")
    keys = ("exact_match", "character_f1", "numeric_accuracy", "unit_accuracy",
            "numeric_and_unit_accuracy", "mean_generation_seconds", "peak_allocated_gib",
            "peak_reserved_gib", "peak_process_ram_gib", "request_latency_median_seconds",
            "request_latency_p95_seconds", "ttft_median_seconds", "ttft_p95_seconds",
            "subsequent_tokens_per_second")
    for key in ("numeric_rows", "unit_rows", "numeric_and_unit_rows"):
        if before["metrics"].get(key) != after["metrics"].get(key):
            raise ValueError(f"Different annotation coverage: {key}")
    return {key: {"before": before["metrics"].get(key), "after": after["metrics"].get(key),
                  "delta": after["metrics"][key] - before["metrics"][key]
                  if after["metrics"].get(key) is not None and before["metrics"].get(key) is not None else None}
            for key in keys}
