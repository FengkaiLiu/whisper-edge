"""Charts for week 2 from results/<tag>.csv.

  <tag>_tradeoff.png   error rate vs RTF, one panel per language
  <tag>_memory.png     model file size vs peak process memory (and peak VRAM if recorded)
  <tag>_breakdown.png  where the time goes per utterance (en)

Usage: py scripts/plot_week2.py [--tag week2_cpu] [--label "Ryzen 5 4600H, 6 threads"]
"""
import argparse
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import QUANTS, RESULTS, SIZES

# Categorical palette, fixed order (validated: CVD / normal-vision separation pass on the light surface).
# Three slots are below 3:1 contrast, so every mark also carries a direct text label.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
QUANT_COLOR = dict(zip(QUANTS, SERIES))
SIZE_MARKER = {"base": "o", "small": "s"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LANG_TITLE = {"en": "English · LibriSpeech test-clean · WER", "zh": "Chinese · FLEURS · CER", "ja": "Japanese · FLEURS · CER"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def load(tag):
    with open(RESULTS / f"{tag}.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["repeat"] == "0"]
    for r in rows:
        r["size"], r["quant"] = r["model"].split("-", 1)
        for k in ["err", "rtf", "peak_mb", "file_mb", "encode", "decode", "batchd", "sample", "mel", "prompt", "vram_mb"]:
            if k in r:
                r[k] = float(r[k])
    return rows


def load_delta(tag):
    """Bootstrap CIs from bootstrap_ci.py, keyed by (lang, model); empty if not run yet."""
    path = RESULTS / f"{tag}_delta.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {(r["lang"], r["model"]): r for r in csv.DictReader(f)}


def spread_labels(ax, items, min_gap=11):
    """items: (x, y, text). Each label goes right of its point, or left when another
    point sits just to the right; labels on the same side are pushed apart vertically
    (in pixels) so none overlap."""
    to_px = ax.transData.transform
    pts = [(*to_px((x, y)), x, y, text) for x, y, text in items]
    placed = {1: [], -1: []}
    for px, py, x, y, text in sorted(pts, key=lambda t: -t[1]):
        crowded = any(0 < qx - px < 45 and abs(qy - py) < 14 for qx, qy, *_ in pts)
        side = -1 if crowded else 1
        ly = py
        for qy in placed[side]:
            if abs(ly - qy) < min_gap:
                ly = qy - min_gap
        placed[side].append(ly)
        ax.annotate(text, (x, y), xytext=(8 * side, ly - py), textcoords="offset points", va="center",
                    ha="left" if side > 0 else "right", fontsize=8, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(ly - py) > 4 else None)


def tradeoff(rows, tag, label):
    langs = [l for l in ["en", "zh", "ja"] if any(r["lang"] == l for r in rows)]
    delta = load_delta(tag)
    fig, axes = plt.subplots(len(SIZES), len(langs), figsize=(4.6 * len(langs), 3.6 * len(SIZES)), squeeze=False)
    for col, lang in enumerate(langs):
        for row, size in enumerate(SIZES):
            ax = axes[row][col]
            pts = [r for r in rows if r["lang"] == lang and r["size"] == size]
            f16 = next(r for r in pts if r["quant"] == "f16")
            ax.axhline(f16["err"] * 100, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
            for r in pts:
                d = delta.get((lang, r["model"]))
                if d:
                    lo, hi = (f16["err"] + float(d["ci_lo"])) * 100, (f16["err"] + float(d["ci_hi"])) * 100
                    ax.plot([r["rtf"]] * 2, [lo, hi], color=QUANT_COLOR[r["quant"]], lw=2, alpha=0.45,
                            solid_capstyle="round", zorder=2)
                ax.scatter(r["rtf"], r["err"] * 100, s=64, marker=SIZE_MARKER[size], color=QUANT_COLOR[r["quant"]],
                           edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.set_xlim(0, max(r["rtf"] for r in pts) * 1.35)
            ax.xaxis.set_major_locator(MaxNLocator(5))  # small GPU RTFs otherwise get 0.0025 steps whose labels collide
            ax.margins(y=0.25)
            fig.canvas.draw()  # fix the limits before converting label positions to pixels
            spread_labels(ax, [(r["rtf"], r["err"] * 100, r["quant"]) for r in pts])
            ax.set_title(f"{size} · {LANG_TITLE[lang]}", fontsize=9.5, loc="left", color=INK)
            if row == len(SIZES) - 1:
                ax.set_xlabel("RTF — lower is faster")
            ax.set_ylabel(("WER" if lang == "en" else "CER") + " (%)")
    fig.suptitle(f"Accuracy vs speed — {label}.  Dashed: f16.  "
                 "Bars: 95% paired-bootstrap CI of the error vs f16.", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{tag}_tradeoff.png", dpi=150)


def memory(rows, tag):
    # Peak memory is per model, so take the max over languages.
    models = [f"{s}-{q}" for s in SIZES for q in QUANTS]
    series = [("file_mb", "model file"), ("peak_mb", "peak process memory")]
    if "vram_mb" in rows[0]:
        series.append(("vram_mb", "peak GPU memory, rise above idle (median of languages)"))
    # VRAM is read device-wide, so other programs can only push it up: the median resists those spikes.
    agg = {"vram_mb": statistics.median}
    vals = {k: {m: agg.get(k, max)([r[k] for r in rows if r["model"] == m]) for m in models} for k, _ in series}
    fig, ax = plt.subplots(figsize=(8, 4.8 if len(series) == 2 else 6))
    h = 0.76 / len(series)
    for j, ((k, name), color) in enumerate(zip(series, SERIES)):
        off = (j - (len(series) - 1) / 2) * (h + 0.02)
        ax.barh([i + off for i in range(len(models))], [vals[k][m] for m in models], height=h, color=color, label=name)
        for i, m in enumerate(models):
            ax.text(vals[k][m] + 6, i + off, f"{vals[k][m]:.0f}", va="center", fontsize=7.5, color=INK_2)
    y = range(len(models))
    ax.set_yticks(list(y), models)
    ax.invert_yaxis()
    ax.set_xlabel("MB")
    ax.grid(axis="y", visible=False)
    if len(series) == 2:
        ax.legend(frameon=False, loc="lower right")
        ax.set_title("Model file size vs peak memory of whisper-cli", loc="left", fontsize=11)
    else:
        ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5)
        ax.set_title("Model file size vs peak memory of whisper-cli", loc="left", fontsize=11, pad=26)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{tag}_memory.png", dpi=150)


def breakdown(rows, tag, label, lang="en"):
    lr = {r["model"]: r for r in rows if r["lang"] == lang}
    models = [f"{s}-{q}" for s in SIZES for q in QUANTS if f"{s}-{q}" in lr]
    parts = [("encode", lambda r: r["encode"]), ("decode", lambda r: r["decode"] + r["batchd"] + r["prompt"]),
             ("sample", lambda r: r["sample"]), ("mel", lambda r: r["mel"])]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    left = [0.0] * len(models)
    for (name, fn), color in zip(parts, SERIES):
        vals = [fn(lr[m]) for m in models]
        ax.barh(models, vals, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.text(total + max(left) * 0.01, i, f"{total:.0f} ms", va="center", fontsize=7.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlabel("ms per utterance (LibriSpeech avg %.1f s audio)" % (float(lr[models[0]]["audio_s"]) / int(lr[models[0]]["n_utts"])))
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.set_title(f"Where the time goes — English, {label}", loc="left", fontsize=11, pad=26)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{tag}_breakdown.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="week2_cpu")
    ap.add_argument("--label", default="Ryzen 5 4600H, 6 threads", help="hardware description for the chart titles")
    args = ap.parse_args()
    tag = args.tag
    rows = load(tag)
    tradeoff(rows, tag, args.label)
    memory(rows, tag)
    breakdown(rows, tag, args.label)
    print(f"wrote results/{tag}_tradeoff.png, _memory.png, _breakdown.png")
