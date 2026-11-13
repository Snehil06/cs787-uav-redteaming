"""One plot style for every figure in reports/: light surface, thin marks, hairline grid.

Colors are the validated reference palette (categorical slots 1-3, status set, blue ramp).
"""
import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

OUTCOME_COLORS = {"success": GOOD, "near_failure": WARNING, "failure": CRITICAL}
OUTCOME_LABELS = {"success": "Success", "near_failure": "Near-failure", "failure": "Failure"}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "figure.titlesize": 13, "figure.titleweight": "semibold",
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 8.5,
        "savefig.dpi": 130, "savefig.bbox": "tight",
    })


def finding(fig, title: str, subtitle: str = ""):
    """Figure headline = the finding; subtitle = what is plotted."""
    h = fig.get_figheight()
    fig.text(0.01, 1 + 0.5 / h, title, ha="left", va="bottom", fontsize=13, weight="semibold", color=INK)
    if subtitle:
        fig.text(0.01, 1 + 0.42 / h, subtitle, ha="left", va="top", color=INK_2, fontsize=9)


apply()
