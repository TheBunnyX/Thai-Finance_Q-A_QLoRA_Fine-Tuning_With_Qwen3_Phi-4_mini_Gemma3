"""Small real-GPU probes; no model downloads or training dataset required."""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finance_qlora.data import write_json
from finance_qlora.progress import Progress, configure


def main():
    configure()
    report = {"checks": {}, "errors": {}}
    with Progress("Import Unsloth training stack"):
        try:
            import unsloth
            report["checks"]["unsloth_import"] = "PASS"
        except Exception:
            report["errors"]["unsloth_import"] = traceback.format_exc()
    import torch
    report.update(torch=torch.__version__, cuda=torch.version.cuda,
                  cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        report["errors"]["cuda"] = "CUDA unavailable"
    else:
        report["gpu"] = torch.cuda.get_device_name(0)
        with Progress("Test bitsandbytes NF4 forward and backward"):
            try:
                from bitsandbytes.nn import Linear4bit
                layer = Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                                   compress_statistics=True, quant_type="nf4")
                layer.requires_grad_(False).to("cuda")
                x = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
                layer(x).float().square().mean().backward()
                torch.cuda.synchronize()
                assert torch.isfinite(x.grad).all().item()
                assert layer.weight.quant_type == "nf4"
                report["checks"]["nf4_forward_backward"] = "PASS"
            except Exception:
                report["errors"]["nf4_forward_backward"] = traceback.format_exc()
        with Progress("Compile and execute Triton CUDA kernel"):
            try:
                import triton
                import triton.language as tl
                # Globals are required for Triton to resolve symbols in the JIT function.
                globals()["triton"] = triton
                globals()["tl"] = tl
                kernel = triton.jit(add_one)
                x = torch.arange(1024, device="cuda", dtype=torch.float32)
                result = torch.empty_like(x)
                kernel[(4,)](x, result, 1024, BLOCK=256)
                torch.cuda.synchronize()
                assert torch.equal(result, x + 1)
                report["checks"]["triton_cuda_kernel"] = "PASS"
            except Exception:
                report["errors"]["triton_cuda_kernel"] = traceback.format_exc()
    write_json(ROOT / "reports/training-kernels.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


def add_one(source, target, size, BLOCK: "tl.constexpr"):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, mask=offsets < size, other=0)
    tl.store(target + offsets, values + 1, mask=offsets < size)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
