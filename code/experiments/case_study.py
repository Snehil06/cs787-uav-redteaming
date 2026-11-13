"""One real case followed through the whole pipeline, as a single PDF storyboard and a single video.

    python -m experiments.case_study

1. the diffusion red team generates a scenario in which RRT crashes;
2. counterfactual re-runs measure why;
3. the LLM analyst explains it from the evidence record, and a script checks every citation;
4. the hardening that targets the measured cause is switched on: the same scenario is now flown safely;
5. the red team attacks the hardened system again and finds a new crash, which is analysed the same way.

Everything is computed live from the stored runs (no hand-written numbers). Needs GEMINI_API_KEY, unless the
LLM answers are already in the cache.
"""
import os
import subprocess
import tempfile
import textwrap

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from analyst.analyst import explain
from analyst.counterfactuals import CAUSES, DESCRIPTIONS, analyse, intervene
from analyst.evidence import build, failure_type
from experiments.common import DATA_DIR
from scenarios.spec import VECTOR_NAMES, Scenario
from sim.simulator import HARDENED, SimConfig, simulate
from viz import style as st
from viz.charts import add_derived
from viz.runs import _draw_map, clearance_series, run_figure
from viz.video import render_video

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "08_case_study")
FF = imageio_ffmpeg.get_ffmpeg_exe()

SHORT = {"no_wind": "wind and gusts removed", "full_sensing": "sensing range raised to 25 m",
         "no_noise": "sensor noise removed", "no_popups": "pop-up obstacles present from the start",
         "slower": "cruise speed reduced by 30%", "reseed": "re-planned with 3 other RRT seeds"}

# which hardening option targets which measured cause (sim.simulator.HARDENED)
FIX_FOR = {
    "no_wind": ("wind feed-forward", "the controller estimates the wind force and cancels it"),
    "slower": ("speed governor", "speed is capped so the UAV can stop within its sensing range"),
    "full_sensing": ("speed governor", "speed is capped so the UAV can stop within its sensing range"),
    "reseed": ("3 planning attempts", "RRT is run 3 times per replan and the shortest path is kept"),
    "no_popups": ("longer obstacle sweep", "moving obstacles are swept 2 s ahead when planning (was 1 s)"),
}
HARDENING_ROWS = [
    ("wind feed-forward", "wind / gusts", "controller cancels the estimated wind force"),
    ("speed governor", "UAV too fast / late detection", "speed capped so the UAV can stop within sensing range"),
    ("3 planning attempts", "unlucky planner randomness", "RRT run 3x per replan, shortest path kept"),
    ("longer obstacle sweep", "moving / pop-up obstacles", "moving obstacles swept 2 s ahead (was 1 s)"),
]


# ------------------------------------------------------------------------------------------------ data
def _cf_table(path):
    cf = pd.read_csv(path, index_col=0)
    for k in CAUSES:
        cf[k] = cf[k].map({True: True, False: False, "True": True, "False": False})
    return cf


def _fly(row, cfg=None, vec=None):
    s = Scenario.from_vector(row[VECTOR_NAMES].to_numpy(float) if vec is None else vec)
    tr = simulate(s, planner=row["planner"], seed=int(row["seed"]), cfg=SimConfig(**(cfg or {})), record=True)
    return s, tr


def _row_from_trace(row, tr):
    """The stored row with its outcome/trace columns replaced by a fresh run (for evidence records)."""
    r = row.copy()
    for k, v in tr.metrics.items():
        if k in r.index:
            r[k] = v
    return r


def _llm(row, cf_row):
    ev = build(row, cf_row, None, include_counterfactuals=True)
    return ev, explain(ev)


def pick_first_case():
    """A late-round diffusion+feedback failure with a clear measured cause that hardening removes."""
    df = add_derived(pd.read_parquet(os.path.join(DATA_DIR, "comparison", "diffusion+feedback_s0.parquet")))
    cf = _cf_table(os.path.join(DATA_DIR, "comparison", "diffusion+feedback_s0_cf.csv"))
    fails = df[(df.outcome == "failure") & df.index.isin(cf.index)].copy()
    fails["n_fix"] = cf.loc[fails.index, "n_fixes"]
    fails["fixed_by"] = cf.loc[fails.index, "fixed_by"]
    fails = fails[(fails.n_fix >= 1) & (fails.n_fix <= 2)]
    fails = fails[fails.fixed_by.str.contains("no_wind|slower|full_sensing", na=False)]
    fails = fails.sort_values(["round", "F"], ascending=[False, False])
    for i, row in fails.head(40).iterrows():
        s, tr0 = _fly(row)
        if tr0.metrics["goal_reached"]:          # must reproduce the failure
            continue
        _, tr1 = _fly(row, HARDENED)
        if tr1.metrics["goal_reached"]:
            return row, cf.loc[i].to_dict(), (s, tr0, tr1)
    raise RuntimeError("no suitable first case")


def pick_second_case(exclude_cause):
    """A failure the red team found against the hardened system, with a measured single cause."""
    df = add_derived(pd.read_parquet(os.path.join(DATA_DIR, "hardening", "diffusion+feedback_s0.parquet")))
    fails = df[df.outcome == "failure"].sort_values(["round", "F"], ascending=[False, False])
    fallback = None
    for i, row in fails.head(40).iterrows():
        cfr = analyse(row[VECTOR_NAMES].to_numpy(float), row["planner"], int(row["seed"]), HARDENED)
        if not cfr["fixed_by"]:
            fallback = fallback or (row, cfr)
            continue
        if exclude_cause not in cfr["fixed_by"].split(","):
            return row, cfr
        fallback = fallback or (row, cfr)
    return fallback


# ------------------------------------------------------------------------------------------------ figures
def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def maps_figure(panels, path):
    """Side-by-side flight maps: [(scenario, trace, title, colour)]."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.0))
    for ax, (s, tr, title, col) in zip(np.atleast_1d(axes), panels):
        clear = clearance_series(s, tr.series)
        i = int(np.nanargmin(clear)) if np.isfinite(clear).any() else len(tr.series["t"]) - 1
        _draw_map(ax, s, tr, i)
        ax.set_title(title, fontsize=12, color=col, weight="bold")
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    return save(fig, path)


def cf_figure(cf_row, path, title):
    names = list(CAUSES)
    vals = [cf_row.get(k) for k in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for y, (k, v) in enumerate(zip(names, vals)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            col, lab = "#d0d0d0", "not applicable"
        elif v:
            col, lab = "#2e8b57", "crash prevented  \u2192  measured cause"
        else:
            col, lab = st.CRITICAL, "still crashes"
        ax.barh(y, 1, color=col, height=0.7)
        ax.text(0.02, y, SHORT[k], va="center", ha="left", fontsize=8.5, color="white" if col != "#d0d0d0" else "#333")
        ax.text(1.02, y, lab, va="center", ha="left", fontsize=9, color="#222", weight="bold" if v is True else None)
    ax.set_xlim(0, 1.75), ax.set_ylim(len(names) - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, loc="left")
    return save(fig, path)


# ------------------------------------------------------------------------------------------------ pdf
def _fonts():
    pdfmetrics.registerFont(TTFont("Ar", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Ar-B", "C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("Ar-I", "C:/Windows/Fonts/ariali.ttf"))
    pdfmetrics.registerFontFamily("Ar", normal="Ar", bold="Ar-B", italic="Ar-I", boldItalic="Ar-B")


S = {}


def _styles():
    S["h"] = ParagraphStyle("h", fontName="Ar-B", fontSize=20, leading=24, textColor=colors.HexColor("#1f2a44"))
    S["step"] = ParagraphStyle("step", fontName="Ar-B", fontSize=11, leading=14, textColor=colors.HexColor("#1f4e8c"))
    S["b"] = ParagraphStyle("b", fontName="Ar", fontSize=10, leading=13.5, alignment=TA_LEFT)
    S["s"] = ParagraphStyle("s", fontName="Ar", fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
    S["q"] = ParagraphStyle("q", fontName="Ar-I", fontSize=10.5, leading=14.5, leftIndent=8,
                            textColor=colors.HexColor("#1f2a44"))
    S["c"] = ParagraphStyle("c", fontName="Ar", fontSize=8.5, leading=10.5)
    S["cb"] = ParagraphStyle("cb", fontName="Ar-B", fontSize=8.5, leading=10.5)


def P(t, s="b"):
    return Paragraph(t, S[s])


def img(path, w):
    from PIL import Image as PI
    W, H = PI.open(path).size
    return Image(path, width=w, height=w * H / W)


def boxed(flowables, bg="#f4f7fb", edge="#1f4e8c", w=None):
    t = Table([[flowables]], colWidths=[w] if w else None)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def grid(rows, widths, header=True, hl=()):
    data = [[P(c, "cb" if (header and r == 0) else "c") for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths)
    style = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black))
    for r in hl:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e3f1e7")))
    t.setStyle(TableStyle(style))
    return t


def side(left, right, wl, wr):
    t = Table([[left, right]], colWidths=[wl, wr])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
    return t


def llm_block(res):
    a, chk = res["answer"], res["checks"]
    facs = "<br/>".join(f"\u2022 <b>{f['factor']}</b> &nbsp;cites&nbsp; <font face='Courier' size='8.5'>"
                        f"{f['evidence_field']}</font>" for f in a["primary_factors"])
    ok = chk["n_unsupported"] == 0 and chk["outcome_matches"]
    verdict = ("PASSED: every cited field exists in this case's evidence record; outcome copied correctly"
               if ok else f"FAILED: {chk['n_unsupported']} unsupported citation(s)")
    return [P(f"<b>{res['model'].split(':')[-1]}</b> &nbsp;\u00b7&nbsp; failure type: <b>{a['failure_type']}</b>"
              f" &nbsp;\u00b7&nbsp; confidence: {a['confidence']}", "s"), Spacer(1, 4),
            P(f"\u201c{a['explanation']}\u201d", "q"), Spacer(1, 6),
            P("<b>Factors and the evidence they cite</b>", "c"), P(facs, "c"), Spacer(1, 5),
            P(f"<b>Grounding check (script):</b> {verdict}", "c")]


def footer(c, doc):
    c.saveState()
    c.setFont("Ar", 8)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(0.5 * inch, 0.32 * inch, "Generative AI-Based Red-Teaming for Autonomous UAV Navigation  \u00b7  "
                 "end-to-end case study (all values computed by experiments/case_study.py)")
    c.drawRightString(11 * inch - 0.5 * inch, 0.32 * inch, str(doc.page))
    c.restoreState()


# ------------------------------------------------------------------------------------------------ video
def card(path, title, lines, image=None):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.text(0.06, 0.88, title, fontsize=28, weight="semibold", color=st.INK, va="top")
    y = 0.76
    text_w = 0.9 if image is None else 0.42
    for ln in lines:
        wrapped = textwrap.fill(ln, 95 if image is None else 46)
        fig.text(0.06, y, wrapped, fontsize=15, color=st.INK_2, va="top", linespacing=1.45)
        y -= 0.065 * (wrapped.count("\n") + 1) + 0.035
    if image is not None:
        ax = fig.add_axes([0.06 + text_w + 0.02, 0.08, 0.92 - text_w - 0.04, 0.72])
        ax.imshow(plt.imread(image)), ax.axis("off")
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(path, dpi=100, facecolor=st.SURFACE)
    plt.close(fig)
    return path


def stitch(parts, out):
    inputs, filters = [], []
    for kind, path, dur in parts:
        if kind == "card":
            inputs += ["-loop", "1", "-t", str(dur), "-i", path]
        else:
            inputs += ["-i", path]
        filters.append(f"[{len(filters)}:v]scale=1280:720:force_original_aspect_ratio=decrease,"
                       f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:white,setsar=1,fps=30,format=yuv420p[v{len(filters)}]")
    n = len(filters)
    graph = ";".join(filters) + ";" + "".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    subprocess.run([FF, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out], check=True)


# ------------------------------------------------------------------------------------------------ main
def main():
    os.makedirs(ROOT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="case_")
    print("picking the first case ...", flush=True)
    rowA, cfA, (sA, trA0, trA1) = pick_first_case()
    causesA = cfA["fixed_by"].split(",")
    main_cause = next(k for k in causesA if k in FIX_FOR)
    fix_name, fix_desc = FIX_FOR[main_cause]
    cs = intervene(sA, main_cause)
    _, trAcf = _fly(rowA, None, cs.to_vector()) if cs is not None else (None, None)

    print("LLM analysis of the first case ...", flush=True)
    evA, llmA = _llm(_row_from_trace(rowA, trA0), cfA)

    print("picking a failure of the hardened system ...", flush=True)
    rowB, cfB = pick_second_case(main_cause)
    sB, trB = _fly(rowB, HARDENED)
    evB, llmB = _llm(_row_from_trace(rowB, trB), cfB)
    causesB = [k for k in (cfB["fixed_by"] or "").split(",") if k]
    next_fix = ("no single change prevents it (a compound failure): a candidate for a new, combined fix"
                if not causesB else
                "even with all four fixes switched on, removing any one of these prevents this crash: " +
                ", ".join(CAUSES[k] for k in causesB) +
                ". The existing fixes do not cover this combination, so it becomes the target of the next "
                "round of hardening, and the loop continues.")

    typeA, typeB = failure_type(cfA), failure_type(cfB)
    rnd = int(rowA["round"]) if "round" in rowA and not pd.isna(rowA["round"]) else None

    # ---- figures
    fA0 = save(run_figure(sA, trA0, "Step 1: the generated scenario, original system", f"RRT, status: {trA0.metrics['status']}"),
               os.path.join(ROOT, "step1_crash.png"))
    fCF = cf_figure(cfA, os.path.join(ROOT, "step2_counterfactuals.png"),
                    "Same scenario re-flown six times, one factor changed each time")
    panels = [(sA, trA0, "original scenario: crash", st.CRITICAL)]
    if trAcf is not None:
        panels.append((cs, trAcf, f"{DESCRIPTIONS[main_cause]}: safe", "#2e7d4f"))
    fCFm = maps_figure(panels, os.path.join(ROOT, "step2_counterfactual_maps.png"))
    fFix = maps_figure([(sA, trA0, "original system: crash", st.CRITICAL),
                        (sA, trA1, "hardened system: reaches the goal", "#2e7d4f")],
                       os.path.join(ROOT, "step4_before_after.png"))
    fB = save(run_figure(sB, trB, "Step 5: new crash found against the hardened system", typeB),
              os.path.join(ROOT, "step5_new_crash.png"))
    fBm = maps_figure([(sB, trB, "hardened system: new crash", st.CRITICAL)],
                      os.path.join(ROOT, "step5_new_crash_map.png"))
    fCFB = cf_figure(cfB, os.path.join(ROOT, "step5_counterfactuals.png"),
                     "New crash, re-flown on the hardened system with one factor changed each time")

    # ---- PDF storyboard
    _fonts(), _styles()
    W = 10.0 * inch
    doc = SimpleDocTemplate(os.path.join(ROOT, "case_study.pdf"), pagesize=landscape(letter),
                            leftMargin=0.5 * inch, rightMargin=0.5 * inch, topMargin=0.45 * inch,
                            bottomMargin=0.55 * inch, title="End-to-end case study", author="Snehil Chatterjee")
    m0, m1 = trA0.metrics, trA1.metrics
    story = [P("One real case, end to end", "h"), Spacer(1, 4),
             P("A crash generated by the diffusion red team is measured, explained, fixed, re-flown, and then the "
               "red team attacks the fixed system again. Every picture and number on these pages is produced by "
               "<font face='Courier'>experiments/case_study.py</font> from the project's stored runs; the LLM text "
               "is Gemini's actual answer.", "b"), Spacer(1, 10)]
    flow = [["1  Crash created", "2  Measured causes", "3  LLM explanation", "4  Fix and re-fly", "5  Attacked again"],
            [f"Diffusion + feedback{f' (round {rnd})' if rnd is not None else ''} generates a scenario; RRT "
             f"{m0['status'].replace('_', ' ')}.",
             f"Six one-factor re-runs: <b>{typeA}</b>.",
             f"Gemini names {', '.join(f['factor'] for f in llmA['answer']['primary_factors'])}; every citation "
             f"checks out." if llmA["checks"]["n_unsupported"] == 0 else "Gemini's explanation, citation-checked.",
             f"<b>{fix_name}</b> switched on; the same scenario now reaches the goal.",
             f"Red team finds a new crash: <b>{typeB}</b>."]]
    t = Table([[P(c, "step") for c in flow[0]], [P(c, "c") for c in flow[1]]], colWidths=[W / 5] * 5)
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#1f4e8c")),
                           ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fb4d3")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [t, Spacer(1, 12), side(img(fFix, 6.0 * inch),
                                     [P("<b>The short version.</b>", "b"), Spacer(1, 4),
                                      P(f"Left: the generated scenario on the original planner stack (crash). Right: "
                                        f"the same scenario, same RRT random seed, after the hardening derived from "
                                        f"the measured causes. Minimum clearance {m0['min_clearance']:.2f} m "
                                        f"\u2192 {m1['min_clearance']:.2f} m.", "b"), Spacer(1, 8),
                                      P("The loop does not end there: page 6 shows the next crash the red team "
                                        "found against the hardened system.", "b")],
                                     6.2 * inch, W - 6.2 * inch), PageBreak()]

    # step 1
    story += [P("Step 1 \u00b7 The red team creates a crash", "h"), Spacer(1, 6),
              P(f"The conditional diffusion model, fine-tuned on failures it found in earlier rounds"
                f"{f' (this scenario is from round {rnd})' if rnd is not None else ''}, generates a scenario "
                f"(46 numbers: obstacles, wind, gusts, sensing). The unmodified PythonRobotics RRT flies it in the "
                f"simulator. Status: <b>{m0['status'].replace('_', ' ')}</b>, failure score F = {rowA['F']:.2f}.",
                "b"), Spacer(1, 6), img(fA0, 9.2 * inch), Spacer(1, 6),
              grid([["cruise speed", "obstacles (moving)", "wind", "gust", "sensing range", "sensor noise"],
                    [evA["scenario"]["uav_cruise_speed_m_s"].__str__() + " m/s",
                     f"{evA['scenario']['n_obstacles']} ({evA['scenario']['share_moving_obstacles']:.0%})",
                     evA["scenario"]["wind"],
                     "none" if not evA["scenario"]["gust"] or evA["scenario"]["gust"]["amplitude_m_s"] < 0.1 else
                     f"{evA['scenario']['gust']['amplitude_m_s']} m/s at t = {evA['scenario']['gust']['starts_at_s']} s",
                     f"{evA['scenario']['sensing_range_m']} m", f"{evA['scenario']['sensor_noise_std_m']} m"]],
                   [W / 6] * 6), PageBreak()]

    # step 2
    story += [P("Step 2 \u00b7 Measure why it crashed", "h"), Spacer(1, 6),
              P("The simulator re-flies the identical scenario six times, changing one factor each time (same "
                "planner seed). A change that turns the crash into a safe flight is a <b>measured cause</b>. This "
                "is evidence, not a guess, and it is what the LLM is given.", "b"), Spacer(1, 8),
              side(img(fCF, 5.6 * inch),
                   boxed([P(f"<b>Measured failure type:</b> {typeA}", "b"), Spacer(1, 4),
                          P(f"Only these changes turn the crash into a safe flight, so they are the measured causes. "
                            f"Below: the original flight next to the re-run with the first of them "
                            f"({SHORT[main_cause]}).", "c")], w=W - 5.9 * inch),
                   5.8 * inch, W - 5.8 * inch), Spacer(1, 8),
              img(fCFm, 6.6 * inch), PageBreak()]

    # step 3
    trace = evA["trace"]
    story += [P("Step 3 \u00b7 The LLM explains it, and a script checks the explanation", "h"), Spacer(1, 6),
              P("Gemini receives only this case's evidence record: the scenario, what the simulator measured, and the "
                "counterfactual results. It must name factors from a fixed vocabulary and cite, for each, the field of "
                "the record that supports it. A script then verifies every citation.", "b"), Spacer(1, 8),
              side([P("<b>Evidence record (excerpt)</b>", "b"), Spacer(1, 3),
                    grid([["field", "value"],
                          ["outcome / end status", f"{evA['outcome']} / {evA['end_status']}"],
                          ["min clearance", f"{trace['min_clearance_m']} m at t = {trace['time_of_min_clearance_s']} s"],
                          ["closest obstacle detected at", f"{trace['closest_obstacle_detected_at_s']} s "
                                                           f"(gap {trace['gap_when_detected_m']} m)"],
                          ["detection \u2192 closest approach", f"{trace['seconds_between_detection_and_closest_approach']} s"],
                          ["max wind felt", f"{trace['max_wind_felt_m_s']} m/s"],
                          ["replans / planner failures", f"{trace['replans']} / {trace['planner_failures']}"]] +
                         [[f"counterfactuals.{k}", v["result"]] for k, v in evA["counterfactuals"].items()],
                         [1.9 * inch, 2.1 * inch])],
                   boxed(llm_block(llmA), w=W - 4.3 * inch), 4.3 * inch, W - 4.3 * inch), Spacer(1, 14),
              boxed([P("<b>Why the counterfactuals matter.</b> Across 50 scored crashes, the same LLM given only the "
                       "flight log names 34% of the measured causes and contradicts them in 44% of answers; given the "
                       "counterfactual results as well, it names 96% with 2% contradictions (report, Table 3).", "b")],
                    bg="#fdf6e3", edge="#b8860b", w=W), PageBreak()]

    # step 4
    hl = [i + 1 for i, r in enumerate(HARDENING_ROWS) if r[0] == fix_name]
    story += [P("Step 4 \u00b7 Fix the cause, fly the same scenario again", "h"), Spacer(1, 6),
              P("Hardening changes only the code around RRT (controller, speed limit, number of planning attempts, "
                "obstacle sweep); the RRT algorithm itself is untouched. Each option targets a cause measured across "
                f"all failures. The one that targets this case's cause is highlighted: <b>{fix_name}</b>, "
                f"{fix_desc}.", "b"), Spacer(1, 6),
              grid([["hardening option", "targets measured cause", "what it does"]] +
                   [list(r) for r in HARDENING_ROWS], [1.8 * inch, 2.4 * inch, W - 4.2 * inch], hl=hl),
              Spacer(1, 8), img(fFix, 8.2 * inch), Spacer(1, 4),
              P(f"Same scenario, same RRT seed. Original: <b>{m0['status'].replace('_', ' ')}</b>, min clearance "
                f"{m0['min_clearance']:.2f} m. Hardened: <b>{m1['status'].replace('_', ' ')}</b>, min clearance "
                f"{m1['min_clearance']:.2f} m, flight time {m1['t_end']:.1f} s. Across all 8,650 failures found so "
                f"far, the hardened system survives 79%.", "b"), PageBreak()]

    # step 5
    story += [P("Step 5 \u00b7 The red team attacks the hardened system", "h"), Spacer(1, 6),
              P("Diffusion + feedback is run again, now against the hardened system. Failures are rarer (the failure "
                "rate on random scenarios fell from 12.0% to 4.7%), but the red team still finds them, 2.1\u00d7 "
                "faster than random sampling. Here is one, analysed exactly like the first.", "b"), Spacer(1, 6),
              side(img(fBm, 3.9 * inch), [img(fCFB, W - 4.1 * inch), Spacer(1, 6),
                                           boxed(llm_block(llmB), w=W - 4.1 * inch)], 4.1 * inch, W - 4.1 * inch),
              Spacer(1, 6), boxed([P(f"<b>What happens next:</b> {next_fix}", "b")], bg="#fdf6e3", edge="#b8860b",
                                  w=W)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    # ---- video
    print("rendering video ...", flush=True)
    vA0, vA1, vB = (os.path.join(tmp, n) for n in ("a0.mp4", "a1.mp4", "b.mp4"))
    render_video(sA, trA0, vA0, "Step 1: a crash created by the red team", "original system")
    render_video(sA, trA1, vA1, "Step 4: same scenario, hardened system", f"fix: {fix_name}")
    render_video(sB, trB, vB, "Step 5: the red team finds a new crash", "hardened system")
    a = llmA["answer"]
    parts = [
        ("card", card(os.path.join(tmp, "c0.png"), "One real case, end to end",
                      ["Crash created \u2192 causes measured \u2192 LLM explanation \u2192 fix \u2192 no crash "
                       "\u2192 the red team finds a new crash.",
                       "Everything shown is computed from the project's runs; the LLM text is Gemini's real answer."]), 5),
        ("card", card(os.path.join(tmp, "c1.png"), "1. The red team creates a crash",
                      [f"The diffusion model{f' (round {rnd} of fine-tuning)' if rnd is not None else ''} generates a "
                       f"scenario. The unmodified RRT planner flies it."]), 3),
        ("video", vA0, None),
        ("card", card(os.path.join(tmp, "c2.png"), "2. Measure why",
                      ["Re-fly the same scenario six times, one factor changed each time.",
                       f"Measured cause: {typeA}."], fCF), 7),
        ("card", card(os.path.join(tmp, "c3.png"), "3. The LLM explains it",
                      [f"\u201c{a['explanation']}\u201d",
                       "Cited evidence: " + ", ".join(f["evidence_field"] for f in a["primary_factors"]),
                       "Grounding check: " + ("every citation exists in the record" if llmA["checks"]["n_unsupported"] == 0
                                              else "unsupported citations found")]), 10),
        ("card", card(os.path.join(tmp, "c4.png"), "4. Fix the measured cause",
                      [f"Switch on {fix_name}: {fix_desc}.", "RRT itself is unchanged. Fly the same scenario again."]), 5),
        ("video", vA1, None),
        ("card", card(os.path.join(tmp, "c5.png"), "5. Attack the hardened system again",
                      ["The red team keeps searching, now against the fixed system.",
                       "Hardening survives 79% of past failures, yet new ones are still found 2.1x faster than random."]), 5),
        ("video", vB, None),
        ("card", card(os.path.join(tmp, "c6.png"), "The new crash, measured and explained",
                      [f"Measured cause: {typeB}.", f"\u201c{llmB['answer']['explanation']}\u201d",
                       "This becomes the input to the next round of fixes: red-teaming is a loop, not a one-off test."],
                      fCFB), 10),
    ]
    stitch(parts, os.path.join(ROOT, "case_study.mp4"))

    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# One real case, end to end

The whole pipeline on a single case, as one document and one video:

- **[case_study.pdf](case_study.pdf)**: 6-page visual storyboard
- **[case_study.mp4](case_study.mp4)**: the same story with the flights animated

| step | what happens in this case |
|---|---|
| 1. crash created | diffusion + feedback{f' (round {rnd})' if rnd is not None else ''} generates a scenario; RRT ends with `{m0['status']}` |
| 2. measured causes | one-factor re-runs: **{typeA}** |
| 3. LLM explanation | {res_line(llmA)} |
| 4. fix and re-fly | **{fix_name}** switched on; same scenario, same seed: `{m1['status']}` |
| 5. attacked again | the red team finds a new crash on the hardened system: **{typeB}** ({res_line(llmB)}) |

Regenerate with `python -m experiments.case_study` (needs `GEMINI_API_KEY` unless the answers are cached).
""")
    print("wrote", ROOT)


def res_line(res):
    a = res["answer"]
    ok = res["checks"]["n_unsupported"] == 0
    return (f"Gemini names {', '.join(f['factor'] for f in a['primary_factors'])}; "
            f"{'all citations verified' if ok else 'some citations unsupported'}")


if __name__ == "__main__":
    main()
