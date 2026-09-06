"""Generate every chart used in the ProofyX investor deck.

All figures are drawn on the deck's deep-indigo ground so they sit flush on the
slide with no visible plot box. Numbers come from docs/PROOFYX_COMPLETE_ANALYSIS.md
(sourced market research) and from the model evaluation runs recorded there;
nothing here is invented at render time.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "pitch" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0E042D"
CREAM = "#FFF6ED"
MUTED = "#A79EC4"
TEAL = "#4FD1C5"
VIOLET = "#8B6CFF"
CYAN = "#4CC9F0"
MAGENTA = "#FF4D9D"
GRID = "#2A2145"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": CREAM,
    "axes.labelcolor": CREAM,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
})


def _save(fig, name):
    """Transparent PNG so the figure sits on the slide's gradient rather than
    punching a flat rectangle of BG through it."""
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.12,
                transparent=True)
    plt.close(fig)
    print("wrote", name)


def market_growth():
    """Deepfake detection market: $170M (2025) to $5.6B (2034), 47.6% CAGR."""
    years = list(range(2025, 2035))
    # 47.6% CAGR compounded off the 2025 base, landing on the published 2034 figure.
    vals = [0.170 * (1.476 ** i) for i in range(len(years))]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.fill_between(years, vals, color=VIOLET, alpha=0.22)
    ax.plot(years, vals, color=CYAN, lw=3, zorder=3)
    ax.scatter([2025, 2034], [vals[0], vals[-1]], s=90, color=TEAL, zorder=4,
               edgecolor=BG, linewidth=2)

    ax.annotate("$170M\n2025", (2025, vals[0]), xytext=(6, 26),
                textcoords="offset points", fontsize=12, fontweight="bold",
                color=CREAM)
    ax.annotate("$5.6B\n2034", (2034, vals[-1]), xytext=(-54, -8),
                textcoords="offset points", fontsize=15, fontweight="bold",
                color=TEAL)
    ax.text(2026.3, vals[-1] * 0.64, "47.6% CAGR", fontsize=17,
            fontweight="bold", color=CREAM)
    ax.text(2026.3, vals[-1] * 0.52, "Deepfake detection market", fontsize=11,
            color=MUTED)

    ax.set_ylabel("Market size (US$ billions)", fontsize=10)
    ax.set_xticks([2025, 2027, 2029, 2031, 2034])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    _save(fig, "fig_market_growth.png")


def tam_sam_som():
    """TAM / SAM / SOM as nested horizontal bands."""
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    rows = [
        ("TAM", "$15-20B",
         "Every org handling digital media, identity\nor communications (by 2035)",
         1.00, VIOLET),
        ("SAM", "$5-8B",
         "Organisations actively buying deepfake\ndetection (by 2034)",
         0.62, CYAN),
        ("SOM", "$100-200M",
         "ProofyX ARR at year 5 = 2-4% of SAM",
         0.26, TEAL),
    ]
    for i, (label, value, desc, width, colour) in enumerate(rows):
        y = 2 - i
        ax.add_patch(FancyBboxPatch((0, y - 0.34), width, 0.68,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=colour, alpha=0.30,
                                    edgecolor=colour, lw=2))
        ax.text(0.022, y + 0.11, label, fontsize=15, fontweight="bold",
                color=CREAM, va="center")
        ax.text(0.022, y - 0.16, desc, fontsize=9, color=MUTED, va="center")
        # A narrow band has no room for the figure inside it - set it outside
        # so the value never lands on top of the label.
        if width < 0.42:
            # Clear of the longest description line, not just of the band edge.
            ax.text(0.46, y, value, fontsize=19, fontweight="bold",
                    color=colour, ha="left", va="center")
        else:
            ax.text(width - 0.022, y, value, fontsize=19, fontweight="bold",
                    color=colour, ha="right", va="center")

    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(-0.55, 2.55)
    ax.axis("off")
    _save(fig, "fig_tam_sam_som.png")


def revenue_projection():
    """Conservative / target ARR path. Explicitly labelled as a projection."""
    years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    low = [1, 5, 15, 45, 100]
    high = [5, 14, 50, 110, 200]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = list(range(len(years)))
    ax.fill_between(x, low, high, color=VIOLET, alpha=0.24)
    ax.plot(x, high, color=CYAN, lw=2.6, marker="o", ms=7, label="Target case")
    ax.plot(x, low, color=TEAL, lw=2.6, marker="o", ms=7, ls="--",
            label="Conservative case")

    for i, (lo, hi) in enumerate(zip(low, high)):
        ax.annotate(f"${hi}M", (i, hi), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color=CYAN)
        ax.annotate(f"${lo}M", (i, lo), xytext=(0, -20),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color=TEAL)

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11)
    ax.set_ylabel("ARR (US$ millions)", fontsize=10)
    # Headroom below zero for the conservative-case labels. The "projection"
    # caveat lives on the slide itself, so no footnote competes for this space.
    ax.set_ylim(-34, 238)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(CREAM)
    _save(fig, "fig_revenue_projection.png")


def detection_gap():
    """The confidence gap, from the iProov study (2,000 UK/US consumers, 2025).

    An earlier version of this chart put iProov's 0.1% next to ProofyX's 82.5%
    on a shared accuracy axis. That was a category error: the 0.1% is the share
    of *people* who classified every item correctly, not per-item accuracy, so
    the two numbers do not share a unit. This version compares only the two
    figures from the same study, which do.
    """
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    labels = ["Believe they could\nspot a deepfake",
              "Actually identified every\nreal and fake item"]
    vals = [60.0, 0.1]
    bars = ax.barh(labels, vals, height=0.46, color=[VIOLET, MAGENTA], alpha=0.92)
    for bar, val in zip(bars, vals):
        ax.text(val + 2.4, bar.get_y() + bar.get_height() / 2,
                f"{val:g}%", va="center", fontsize=19, fontweight="bold",
                color=CREAM)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of 2,000 UK and US consumers tested (%)", fontsize=10)
    ax.tick_params(labelsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    _save(fig, "fig_detection_gap.png")


def pipeline_diagram():
    """Three modalities to a model bank to learned fusion to a Trust Score."""
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x, y, w, h, text, colour, fs=9.5, bold=False, alpha=0.18):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=colour, alpha=alpha,
                                    edgecolor=colour, lw=1.8))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fs, color=CREAM,
                    fontweight="bold" if bold else "normal", linespacing=1.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6,
                                    shrinkA=0, shrinkB=0))

    for name, y in [("Image", 3.25), ("Video", 1.95), ("Audio", 0.65)]:
        box(0.12, y, 1.5, 0.85, name, CYAN, fs=11, bold=True)
        arrow(1.68, y + 0.42, 2.28, y + 0.42)

    box(2.32, 0.40, 2.5, 3.72, "", VIOLET, alpha=0.10)
    ax.text(3.57, 3.90, "MODEL BANK", ha="center", fontsize=9,
            fontweight="bold", color=VIOLET)
    for name, y in [("DINOv2  /  ViT", 3.28), ("EfficientNet-B4", 2.72),
                    ("Frequency CNN", 2.16), ("Face / ResNet50", 1.60),
                    ("CorefakeNet (fast)", 1.02), ("Wav2Vec2 audio", 0.48)]:
        box(2.46, y - 0.02, 2.22, 0.44, name, VIOLET, fs=9, alpha=0.22)

    arrow(4.9, 2.3, 5.5, 2.3)
    box(5.54, 1.55, 1.85, 1.5, "Learned\nFusion MLP\n+ calibration",
        TEAL, fs=10, bold=True, alpha=0.22)
    arrow(7.47, 2.3, 8.05, 2.3)
    box(8.08, 1.35, 1.82, 1.9,
        "TRUST SCORE\n0-100\n\n+ GradCAM heatmap\n+ per-model breakdown",
        TEAL, fs=9.5, alpha=0.30)
    ax.text(8.99, 3.44, "EXPLAINABLE OUTPUT", ha="center", fontsize=9,
            fontweight="bold", color=TEAL)

    ax.text(5.0, 0.16,
            "Cross-modal fusion: a video is scored on its frames AND its audio "
            "track, and disagreement between modalities is surfaced, not hidden.",
            ha="center", fontsize=8.5, color=MUTED, style="italic")
    _save(fig, "fig_pipeline.png")


if __name__ == "__main__":
    market_growth()
    tam_sam_som()
    revenue_projection()
    detection_gap()
    pipeline_diagram()
