"""Compact, paper-style figures for paper/report.pdf, drawn from the same data as reports/.

    python -m tools.paper_figures      # writes paper/figures/*.png
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from loop.metrics import distinct_mask  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(ROOT, "paper", "figures")

BLUE, ORANGE, GREEN, GREY, RED = "#1f5fa8", "#d9622b", "#2e8b57", "#8a8a8a", "#b23a3a"
METHODS = [("random", "Random", GREY, ":"), ("cmaes", "CMA-ES", "#555555", "--"),
           ("cvae+feedback", "cVAE + fb", GREEN, "-."), ("diffusion", "Diffusion", ORANGE, "-"),
           ("diffusion+feedback", "Diffusion + fb (ours)", BLUE, "-")]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.5, "legend.frameon": False,
        "lines.linewidth": 1.4, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def overview():
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    ax.set_xlim(-1, 121.5), ax.set_ylim(-1, 36), ax.axis("off")
    # (x, y, w, h, text, colour); columns: data | generator | simulator | causes | LLM
    boxes = [
        (0, 22, 19, 12, "Real data\nSDD obstacle speeds\nNOAA wind, gusts", "#f2f2f2"),
        (0, 4, 19, 12, "Procedural scenarios\n(10k, real statistics)\n+ validity rules", "#f2f2f2"),
        (26, 13, 21, 14, "Conditional diffusion\ngenerator\n(DDPM + CFG,\ncond. on difficulty)", "#dbe7f5"),
        (54, 13, 21, 14, "2D UAV simulator\nRRT / RRT* planner\n(unmodified), wind,\ndelayed noisy sensing", "#fbe3d6"),
        (54, 0, 21, 8, "Failure buffer\n(distinct, F ≥ 0.5)", "#e3f1e7"),
        (82, 13, 17, 14, "Counterfactual\nre-runs (6 single-\nfactor changes)\n= measured causes", "#efe6f5"),
        (103, 13, 17, 14, "Grounded LLM\nanalyst (Gemini)\n+ citation check", "#efe6f5"),
        (82, 0, 38, 8, "Targeted hardening of the planner stack,\nthen re-attack the hardened system", "#f7f0d4"),
    ]
    for x, y, w, h, txt, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=fc, ec="#333333", lw=0.6))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.3, linespacing=1.3)

    def arrow(a, b, txt=None, rad=0.0, tx=0.0, ty=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#333333",
                                     connectionstyle=f"arc3,rad={rad}"))
        if txt:
            ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, txt, ha="center", va="center",
                    fontsize=6, style="italic", color="#333333")
    arrow((9.5, 21.7), (9.5, 16.4))
    arrow((19.4, 10), (25.6, 16), "train", tx=-1.8, ty=5.0)
    arrow((47.4, 20), (53.6, 20))
    arrow((75.4, 20), (81.6, 20))
    arrow((99.4, 20), (102.6, 20))
    arrow((64.5, 12.7), (64.5, 8.4))
    arrow((53.6, 3), (38, 12.6), "fine-tune\n(5 rounds)", rad=-0.3, tx=-6.5, ty=-5.5)
    arrow((111.5, 12.7), (111.5, 8.4))
    fig.savefig(os.path.join(OUT, "fig1_overview.png"))
    plt.close(fig)


def example():
    """Re-use the stage-1 flight picture, without its slide headline."""
    im = Image.open(os.path.join(REP, "01_random_baseline", "1_runs", "03_crash_no_wind.png")).convert("RGB")
    w, h = im.size
    im.crop((0, int(h * 0.17), w, h)).save(os.path.join(OUT, "fig2_example.png"))


def discovery():
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for key, label, c, ls in METHODS:
        curves = []
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "comparison", f"{key}_s[0-9].parquet"))):
            curves.append(np.cumsum(distinct_mask(pd.read_parquet(f))))
        n = min(len(c_) for c_ in curves)
        m = np.mean([c_[:n] for c_ in curves], axis=0)
        ax.plot(np.arange(1, n + 1), m, color=c, ls=ls, lw=1.8 if "ours" in label else 1.2, label=label)
    ax.set_xlabel("simulated flights"), ax.set_ylabel("distinct failures found")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT, "fig3_discovery.png"))
    plt.close(fig)


def generators():
    st = pd.read_csv(os.path.join(REP, "02_generators", "2_results", "steering.csv"))
    pr = pd.read_csv(os.path.join(REP, "03_red_team_loop", "2_results", "per_round.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 1.75))
    for cfg, lab, c, ls in [("diffusion w=4", "Diffusion w=4", BLUE, "-"), ("diffusion w=2", "Diffusion w=2", ORANGE, "-"),
                            ("cvae", "cVAE", GREEN, "-.")]:
        d = st[st.config == cfg]
        a.plot(d.difficulty, 100 * d.failure_rate, color=c, ls=ls, marker="o", ms=2.2, label=lab)
    a.axhline(11.1, color=GREY, ls=":", lw=1)
    a.text(1.02, 8.0, "random", ha="right", fontsize=6, color=GREY)
    a.set_ylim(0, 50), a.set_xlabel("requested difficulty"), a.set_ylabel("failure rate (%)"), a.set_title("(a) steering", loc="left")
    a.legend(loc="upper left", fontsize=5.8, handlelength=1.6)
    for key, lab, c, ls in [("diffusion+feedback", "Diff. + fb", BLUE, "-"), ("diffusion", "Diffusion", ORANGE, "--"),
                            ("cvae+feedback", "cVAE + fb", GREEN, "-."), ("cvae", "cVAE", GREEN, ":")]:
        d = pr[pr.method == key]
        b.plot(d["round"] + 1, 100 * d.failure_rate, color=c, ls=ls, marker="o", ms=2.2, label=lab)
    b.set_ylim(8, 52), b.set_xlabel("round"), b.set_title("(b) feedback loop", loc="left"), b.set_xticks([1, 2, 3, 4, 5])
    b.legend(loc="upper left", fontsize=5.6, handlelength=1.6, ncol=2, columnspacing=0.8)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig4_generators.png"))
    plt.close(fig)


def llm():
    s = pd.read_csv(os.path.join(REP, "04_llm_analyst", "2_results", "summary.csv"))
    names = {"trace only": "trace\nonly", "+ similar cases": "+ similar\ncases (RAG)",
             "+ counterfactuals": "+ counter-\nfactuals", "full": "all\nevidence"}
    order = ["trace only", "+ similar cases", "+ counterfactuals", "full"]
    s = s.set_index("variant").loc[order].reset_index()
    labels = [names[v] for v in s.variant]
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    w = 0.26
    for i, (col, lab, c) in enumerate([("precision", "precision", BLUE), ("recall", "recall", ORANGE),
                                       ("contradicted", "contradicted", RED)]):
        ax.bar(x + (i - 1) * w, 100 * s[col], w, color=c, label=lab)
    ax.set_xticks(x, labels), ax.set_ylabel("% of explanations / causes"), ax.set_ylim(0, 112)
    ax.legend(ncol=3, loc="upper left", fontsize=6.5)
    fig.savefig(os.path.join(OUT, "fig5_llm.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    style()
    overview(), example(), discovery(), generators(), llm()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
