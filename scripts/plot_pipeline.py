"""Pipeline diagram for the README.  .venv/bin/python scripts/plot_pipeline.py -> docs/images/pipeline.png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "pipeline.png"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
TINT = {BLUE: "#e8f1fc", ORANGE: "#fdeee7", GREEN: "#e5f6ef", GREY: "#efeeea"}

fig, ax = plt.subplots(figsize=(15.7, 5.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 15.7)
ax.set_ylim(0, 5.6)
ax.axis("off")


def box(x, y, w, h, color, fill=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", linewidth=1.4,
                                edgecolor=color, facecolor=TINT[color] if fill else SURFACE, zorder=2))


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6, shrinkA=0, shrinkB=0))


def stage(cx, n, title, color):
    ax.add_patch(plt.Circle((cx, 5.15), 0.2, color=color, zorder=3))
    ax.text(cx, 5.15, str(n), ha="center", va="center", color="white", fontsize=11, weight="bold", zorder=4)
    ax.text(cx, 4.7, title, ha="center", va="center", color=INK, fontsize=11.5, weight="bold")


# 1 source
stage(1.1, 1, "Source data", GREY)
box(0.2, 2.2, 1.8, 2.0, GREY)
ax.text(1.1, 3.55, "2,500 Thai\nfinance Q&A", ha="center", va="center", fontsize=11, weight="bold")
ax.text(1.1, 2.75, "audited, never\nmodified", ha="center", va="center", fontsize=9, color=INK2)
arrow(2.05, 3.2, 2.55, 3.2)

# 2 build
stage(3.65, 2, "Build splits", BLUE)
box(2.6, 2.2, 2.1, 2.0, BLUE)
for i, t in enumerate(["80/10/10 split", "numbers re-checked", "6 templates held out"]):
    ax.text(2.75, 3.75 - i * 0.55, "•  " + t, ha="left", va="center", fontsize=9, color=INK)
arrow(4.75, 3.2, 5.25, 3.2)

# 3 three training sets
stage(6.35, 3, "3 training sets", ORANGE)
sets = [("v2", "plain answers", 1884, BLUE), ("v2-steps", "step-by-step working", 1884, ORANGE),
        ("v2-cap15", "≤15 rows per template", 1137, GREEN)]
for i, (name, desc, n, c) in enumerate(sets):
    y = 3.35 - i * 0.95
    box(5.3, y, 2.1, 0.85, c)
    ax.text(5.45, y + 0.58, name, fontsize=10.5, weight="bold", va="center")
    ax.text(5.45, y + 0.28, f"{desc}\n{n:,} train rows", fontsize=8.2, color=INK2, va="center")
arrow(7.45, 3.2, 7.95, 3.2)

# 4 optimize before training
stage(9.05, 4, "Optimize per model", BLUE)
box(8.0, 2.2, 2.1, 2.0, BLUE)
for i, t in enumerate(["best micro-batch", "attention backend", "VRAM / speed check"]):
    ax.text(8.15, 3.75 - i * 0.55, "•  " + t, ha="left", va="center", fontsize=9, color=INK)
arrow(10.15, 3.2, 10.65, 3.2)

# 5 train
D = 2.7
stage(9.05 + D, 5, "QLoRA fine-tune", ORANGE)
box(8.0 + D, 2.2, 2.1, 2.0, ORANGE)
ax.text(9.05 + D, 3.65, "3 models × 3 sets", ha="center", va="center", fontsize=10.5, weight="bold")
ax.text(9.05 + D, 3.15, "Qwen3 · Gemma 3 · Phi-4", ha="center", va="center", fontsize=8.8, color=INK2)
ax.text(9.05 + D, 2.6, "= 9 adapters", ha="center", va="center", fontsize=11, weight="bold", color=ORANGE)
arrow(10.15 + D, 3.2, 10.6 + D, 3.2)

# 6 evaluate
stage(11.85 + D, 6, "Evaluate & compare", GREEN)
box(10.65 + D, 2.2, 2.25, 2.0, GREEN)
for i, t in enumerate(["test: 211 questions", "held-out: 78 questions", "vs. untuned base"]):
    ax.text(10.8 + D, 3.75 - i * 0.55, "•  " + t, ha="left", va="center", fontsize=9, color=INK)

# base model feeding evaluation
box(8.0 + D, 0.55, 2.1, 0.8, GREY)
ax.text(9.05 + D, 0.95, "Base models", ha="center", va="center", fontsize=10, weight="bold")
arrow(10.15 + D, 0.95, 11.0 + D, 2.15)

ax.text(11.775 + D, 1.0, "→ metrics, charts,\nREADME, playground", ha="center", va="center", fontsize=9, color=INK2)
ax.text(0.2, 0.95, "Metrics: exact match · character F1 · numeric accuracy (checked by recomputation)",
        fontsize=9, color=INK2, va="center")
ax.text(0.2, 0.55, "Order matters: one GPU job at a time; adapters trained on a different dataset are rejected.",
        fontsize=9, color=INK2, va="center")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE)
print(OUT)
