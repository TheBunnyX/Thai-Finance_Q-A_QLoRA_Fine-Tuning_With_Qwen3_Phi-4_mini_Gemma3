"""Charts + results table for the finance-2500-v2 experiment (scripts/run_2500_v2.sh).

    .venv/bin/python scripts/plot_2500_results.py

Reads outputs/<model>-2500-<variant>/run.json and outputs/eval-*/metrics.json, writes PNGs to
docs/images/ and a machine-readable summary to docs/results-2500-v2.json. Missing runs are
skipped, so it can be re-run while the pipeline is still going.

Palette (reference data-viz palette, light surface): baseline = muted neutral (context), then
categorical slots 1-3 in fixed order for the three training variants; one y-axis per chart;
the per-topic heatmap uses the single-hue blue sequential ramp.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
MODELS = [("qwen", "Qwen3 4B"), ("phi", "Phi-4 mini"), ("gemma", "Gemma 3 4B")]
VARIANTS = [("base", "Base (no fine-tune)"), ("v2", "SFT v2"), ("v2-steps", "SFT v2-steps"),
            ("v2-cap15", "SFT v2-cap15")]
COLORS = {"base": "#898781", "v2": "#2a78d6", "v2-steps": "#eb6834", "v2-cap15": "#1baf7a"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def eval_name(model, variant, split):
    return f"{model}-base-2500-{split}" if variant == "base" else f"{model}-2500-{variant}-{split}"


def subset(metrics, numeric):
    """Character F1 / EM over calculation (numeric) or non-calculation topics only."""
    topics = [v for v in metrics["by_topic"].values() if bool(v["numeric_rows"]) == numeric]
    rows = sum(v["rows"] for v in topics)
    if not rows:
        return None, None
    f1 = sum(v["character_f1"] * v["rows"] for v in topics) / rows
    em = sum(v["exact_match"] * v["rows"] for v in topics) / rows
    return f1, em


def collect():
    data = {}
    for m, _ in MODELS:
        for v, _ in VARIANTS:
            entry = {}
            run = load(ROOT / f"outputs/{m}-2500-{v}/run.json") if v != "base" else None
            if run and run.get("status") == "complete":
                entry["train"] = {
                    "steps": run["optimizer_steps"], "train_seconds": run["training_seconds"],
                    "total_seconds": run["total_seconds"], "peak_reserved_gib": run["peak_reserved_gib"],
                    "micro_batch": run["training"]["batch_size"],
                    "grad_accum": run["training"]["gradient_accumulation_steps"],
                    "eval_loss": run["metrics"].get("eval_loss"),
                    "train_loss": run["metrics"].get("train_loss"),
                    "log_history": run.get("log_history", [])}
            for split in ("test", "heldout"):
                ev = load(ROOT / f"outputs/eval-{eval_name(m, v, split)}/metrics.json")
                if ev:
                    mt = ev["metrics"]
                    calc_f1, _ = subset(mt, True)
                    text_f1, text_em = subset(mt, False)
                    entry[split] = {
                        "rows": mt["rows"], "exact_match": mt["exact_match"],
                        "character_f1": mt["character_f1"], "numeric_rows": mt["numeric_rows"],
                        "numeric_accuracy": mt["numeric_accuracy"], "text_f1": text_f1,
                        "text_exact_match": text_em, "hit_generation_limit": mt.get("hit_generation_limit"),
                        "mean_generation_seconds": mt.get("mean_generation_seconds"),
                        "by_topic_numeric": {t: x["numeric_accuracy"] for t, x in mt["by_topic"].items()
                                             if x["numeric_rows"]}}
            if entry:
                data.setdefault(m, {})[v] = entry
    return data


def bar_label(ax, rects, fmt):
    for r in rects:
        h = r.get_height()
        if h == h:  # skip NaN
            ax.annotate(fmt(h), (r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)


def grouped(ax, data, getter, fmt, ylabel, ylim=None, variants=VARIANTS):
    n = len(variants)
    width = 0.8 / n
    for i, (v, label) in enumerate(variants):
        vals = []
        for m, _ in MODELS:
            val = getter(data.get(m, {}).get(v, {}))
            vals.append(float("nan") if val is None else val)
        xs = [k + (i - (n - 1) / 2) * width for k in range(len(MODELS))]
        # 2px surface gap between adjacent bars via an edge in the surface color.
        rects = ax.bar(xs, vals, width, color=COLORS[v], label=label,
                       edgecolor=SURFACE, linewidth=1.5)
        bar_label(ax, rects, fmt)
    ax.set_xticks(range(len(MODELS)), [name for _, name in MODELS])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def pct(x):
    return f"{x * 100:.0f}%"


def main():
    data = collect()
    (ROOT / "docs").mkdir(exist_ok=True)
    slim = {m: {v: {k: (dict(e, log_history=None) if k == "train" else e) for k, e in d.items()}
                for v, d in vs.items()} for m, vs in data.items()}
    (ROOT / "docs/results-2500-v2.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")

    # 1) Headline: numeric accuracy on calculation questions (test, verified annotations).
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    grouped(ax, data, lambda e: e.get("test", {}).get("numeric_accuracy"), pct,
            "Numeric accuracy", (0, 1.12))
    ax.yaxis.set_major_formatter(lambda y, _: f"{y * 100:.0f}%")
    ax.set_title("Calculation questions answered with the correct final number (test, 72 rows)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
    save(fig, "numeric-accuracy-test.png")

    # 2) Text quality: seen templates (test) vs unseen templates (held-out) - small multiples.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    grouped(axes[0], data, lambda e: e.get("test", {}).get("text_f1"), lambda x: f"{x:.2f}",
            "Character F1", (0, 1.12))
    axes[0].set_title("Explanation questions, seen templates (test)", loc="left", fontsize=10.5)
    grouped(axes[1], data, lambda e: e.get("heldout", {}).get("character_f1"), lambda x: f"{x:.2f}",
            "", (0, 1.12))
    axes[1].set_title("Unseen answer templates (held-out, 78 rows)", loc="left", fontsize=10.5)
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
    save(fig, "text-f1-test-vs-heldout.png")

    # 3) Training cost - two measures, two charts (never a dual axis).
    sft = VARIANTS[1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    grouped(axes[0], data, lambda e: e.get("train", {}).get("train_seconds", None) and
            e["train"]["train_seconds"] / 60, lambda x: f"{x:.1f}", "Minutes", None, sft)
    axes[0].set_title("Training time (incl. validation)", loc="left", fontsize=10.5)
    grouped(axes[1], data, lambda e: e.get("train", {}).get("peak_reserved_gib"),
            lambda x: f"{x:.1f}", "GiB", (0, 12), sft)
    axes[1].axhline(11.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[1].annotate("RTX 5070 usable 11.5 GiB", (2.45, 11.5), xytext=(0, 3), textcoords="offset points",
                     ha="right", fontsize=7.5, color=MUTED)
    axes[1].set_title("Peak reserved VRAM", loc="left", fontsize=10.5)
    axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
    save(fig, "training-cost.png")

    # 4) Loss curves per model (train loss; v2-steps trains on longer targets, so its scale differs).
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    for ax, (m, name) in zip(axes, MODELS):
        ax.grid(True, axis="both")
        for v, label in sft:
            hist = data.get(m, {}).get(v, {}).get("train", {}).get("log_history") or []
            pts = [(h["step"], h["loss"]) for h in hist if "loss" in h]
            if pts:
                ax.plot(*zip(*pts), color=COLORS[v], linewidth=2, label=label)
            ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
            if ev:
                ax.plot(*zip(*ev), linestyle="none", marker="o", markersize=6, color=COLORS[v],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False, zorder=5)
        ax.set_title(name, loc="left", fontsize=10.5)
        ax.set_xlabel("Optimizer step")
        ax.set_ylim(bottom=-0.06 * ax.get_ylim()[1])
    axes[0].set_ylabel("Loss (line = train, dots = validation)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    save(fig, "loss-curves.png")

    # 5) Per-topic numeric accuracy heatmap (sequential single hue; values printed in cells).
    cols = [(m, v) for m, _ in MODELS for v, _ in VARIANTS if data.get(m, {}).get(v, {}).get("test")]
    if cols:
        topics = sorted({t for m, v in cols for t in data[m][v]["test"]["by_topic_numeric"]})
        grid = [[data[m][v]["test"]["by_topic_numeric"].get(t, float("nan")) for m, v in cols] for t in topics]
        fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(cols), 0.32 * len(topics) + 1.6))
        cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val == val:
                    ax.text(j, i, f"{val * 100:.0f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if val >= 0.5 else INK)
        short = {"base": "base", "v2": "v2", "v2-steps": "steps", "v2-cap15": "cap15"}
        ax.set_xticks(range(len(cols)), [f"{dict(MODELS)[m].split()[0]}\n{short[v]}" for m, v in cols],
                      fontsize=7.5)
        ax.set_yticks(range(len(topics)), topics, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Numeric accuracy by calculation topic, % (test, 3 rows per topic)",
                     loc="left", fontsize=10.5)
        save(fig, "numeric-accuracy-by-topic.png")


if __name__ == "__main__":
    main()
