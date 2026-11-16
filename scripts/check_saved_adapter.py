"""Reload a completed local adapter and generate one non-benchmark probe."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finance_qlora.data import write_json
from finance_qlora.modeling import chat_token_ids, load_model
from finance_qlora.progress import Progress, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    configure()
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "complete":
        raise ValueError("Only completed training runs can be probed")
    with Progress("Reload base model and saved LoRA adapter"):
        model, _, tokenizer, fast, _ = load_model(run["model"], run["training"])
        import torch
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.run_dir / "adapter"), is_trainable=False)
        fast.for_inference(model)
        model.eval()
        model.config.use_cache = True
    prompt = "อธิบาย Current Ratio และสูตรคำนวณเป็นภาษาไทยแบบสั้น"
    ids = chat_token_ids(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
    with Progress("Generate one Thai finance probe after adapter reload"):
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    do_sample=False, num_beams=1, max_new_tokens=128,
                                    pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(output[0, len(ids):], skip_special_tokens=True).strip()
        if not response:
            raise RuntimeError("Empty generated response")
    result = {"status": "PASS", "model": run["model"], "prompt": prompt, "response": response,
              "generated_tokens": output.shape[1] - len(ids),
              "probe": "Adapter reload + non-empty generation only; not a factual correctness score"}
    write_json(args.run_dir / "reload-probe.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
