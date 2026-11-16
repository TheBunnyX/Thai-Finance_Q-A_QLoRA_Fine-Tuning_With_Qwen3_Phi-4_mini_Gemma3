"""Pair a training variant with the held-out-template test set.

    python scripts/make_heldout_variant.py data/processed/finance-5000-v2-steps

writes <variant>-heldout with the variant's train split, the heldout set's validation/test,
so `evaluate --adapter <variant adapter>` accepts it (same train checksum) and load_prepared
re-checks leakage between that train split and the held-out test.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finance_qlora.data import load_prepared, write_json  # noqa: E402


def main():
    variant = Path(sys.argv[1])
    heldout = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/finance-5000-v2-heldout")
    out = variant.with_name(variant.name + "-heldout")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory must be empty: {out}")
    _, v_manifest = load_prepared(variant)
    _, h_manifest = load_prepared(heldout)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(variant / "train.jsonl", out / "train.jsonl")
    for name in ("validation", "test"):
        shutil.copyfile(heldout / f"{name}.jsonl", out / f"{name}.jsonl")
    splits = {"train": dict(v_manifest["splits"]["train"]),
              "validation": dict(h_manifest["splits"]["validation"]),
              "test": dict(h_manifest["splits"]["test"])}
    write_json(out / "manifest.json", dict(
        h_manifest, splits=splits, train_from=str(variant),
        design=h_manifest["design"] + f"; train split from {variant.name}"))
    load_prepared(out)
    print(json.dumps({"output": str(out), **{k: v["rows"] for k, v in splits.items()}}))


if __name__ == "__main__":
    main()
