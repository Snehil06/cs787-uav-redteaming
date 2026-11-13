"""Build paper/report.pdf: the project write-up as a two-column research paper (AAAI-like layout).

    python -m tools.paper_figures     # figures first
    python -m tools.build_paper

Numbers in the text are taken from reports/ (v2 run); keep them in sync if the experiments are re-run.
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, FrameBreak, Image, KeepTogether,
                                NextPageTemplate, PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
OUT = os.path.join(ROOT, "paper", "report.pdf")

FONTS = "C:/Windows/Fonts"
for name, f in [("TNR", "times.ttf"), ("TNR-B", "timesbd.ttf"), ("TNR-I", "timesi.ttf"), ("TNR-BI", "timesbi.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, f)))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

PW, PH = letter
M = 0.75 * inch
GAP = 0.33 * inch
COLW = (PW - 2 * M - GAP) / 2
FULLW = PW - 2 * M

S = {
    "title": ParagraphStyle("title", fontName="TNR-B", fontSize=16.5, leading=20, alignment=TA_CENTER),
    "author": ParagraphStyle("author", fontName="TNR", fontSize=10.5, leading=13, alignment=TA_CENTER),
    "affil": ParagraphStyle("affil", fontName="TNR", fontSize=9, leading=11, alignment=TA_CENTER),
    "abs_h": ParagraphStyle("abs_h", fontName="TNR-B", fontSize=10, leading=12, alignment=TA_CENTER,
                            spaceAfter=3),
    "abs": ParagraphStyle("abs", fontName="TNR", fontSize=9, leading=10.8, alignment=TA_JUSTIFY,
                          leftIndent=10, rightIndent=10),
    "h1": ParagraphStyle("h1", fontName="TNR-B", fontSize=11.5, leading=14, alignment=TA_CENTER,
                         spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="TNR-B", fontSize=10, leading=12, spaceBefore=5, spaceAfter=2,
                         keepWithNext=1),
    "body": ParagraphStyle("body", fontName="TNR", fontSize=10, leading=12, alignment=TA_JUSTIFY,
                           firstLineIndent=10),
    "body0": ParagraphStyle("body0", fontName="TNR", fontSize=10, leading=12, alignment=TA_JUSTIFY),
    "bullet": ParagraphStyle("bullet", fontName="TNR", fontSize=10, leading=12, alignment=TA_JUSTIFY,
                             leftIndent=11, bulletIndent=2),
    "cap": ParagraphStyle("cap", fontName="TNR", fontSize=8.8, leading=10.4, alignment=TA_JUSTIFY,
                          spaceBefore=3, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="TNR", fontSize=8, leading=9.4, alignment=TA_CENTER),
    "cell_l": ParagraphStyle("cell_l", fontName="TNR", fontSize=8, leading=9.4, alignment=TA_LEFT),
    "algo": ParagraphStyle("algo", fontName="TNR", fontSize=8.6, leading=10.6, alignment=TA_LEFT),
    "eq": ParagraphStyle("eq", fontName="TNR", fontSize=9.5, leading=12, alignment=TA_CENTER),
    "ref": ParagraphStyle("ref", fontName="TNR", fontSize=8.5, leading=10, alignment=TA_JUSTIFY,
                          leftIndent=9, firstLineIndent=-9, spaceAfter=2.2),
}


S["bodyl"] = ParagraphStyle("bodyl", parent=S["body0"], alignment=TA_LEFT)


def P(text, style="body"):
    return Paragraph(text, S[style])


REPO = "https://github.com/Snehil06/cs787-uav-redteaming"


def link(url, text=None):
    """Clickable hyperlink in the paper's link colour."""
    return f'<a href="{url}" color="#1f4e8c">{text or url}</a>'


def repo(path, text=None):
    """Link to a folder or file in the GitHub repository."""
    kind = "blob" if "." in path.rsplit("/", 1)[-1] else "tree"
    return link(f"{REPO}/{kind}/main/{path}", text or path)


def repo_banner():
    """Boxed, centred repository link under the title so it is hard to miss."""
    st = ParagraphStyle("banner", fontName="TNR", fontSize=10, leading=13, alignment=TA_CENTER)
    p = Paragraph(f"<b>Code, results and demo video:</b> &nbsp;"
                  f'<a href="{REPO}" color="#1f4e8c"><b>{REPO}</b></a>', st)
    t = Table([[p]], colWidths=[4.9 * inch])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1f4e8c")),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3fa")),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def H1(text):
    return P(text, "h1")


def H2(text):
    return P(text, "h2")


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="\u2022") for t in items]


def eq(text, num):
    t = Table([[Paragraph(text, S["eq"]), Paragraph(f"({num})", S["eq"])]],
              colWidths=[COLW - 0.35 * inch, 0.35 * inch])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def figure(path, width, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(path, width=width, height=width * h / w)
    return KeepTogether([img, P(caption, "cap")])


def booktabs(rows, widths, caption, bold_rows=(), left_cols=(0,), font=8):
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, v in enumerate(row):
            st = S["cell_l"] if c in left_cols else S["cell"]
            txt = f"<b>{v}</b>" if r == 0 else v
            cells.append(Paragraph(txt, st))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
             ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
             ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]
    for r in bold_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef3fa")))
    t.setStyle(TableStyle(style))
    return KeepTogether([P(caption, "cap"), t, Spacer(1, 8)])


def algorithm():
    lines = [
        "<b>Algorithm 1</b> Closed-loop diffusion red-teaming",
        "<b>Input:</b> seed set D<sub>0</sub> = {(x<sub>i</sub>, F<sub>i</sub>)}, denoiser \u03b5<sub>\u03b8</sub>, "
        "budget B, rounds R, guidance w, threshold \u03c4<sub>F</sub>",
        "1: train \u03b5<sub>\u03b8</sub> on D<sub>0</sub> with condition c = (C(x), F)",
        "2: buffer <i>M</i> \u2190 empty set",
        "3: <b>for</b> r = 1 \u2026 R <b>do</b>",
        "4: &nbsp;&nbsp;&nbsp;sample x ~ p<sub>\u03b8</sub>(x | F = 1) with CFG weight w; keep valid x",
        "5: &nbsp;&nbsp;&nbsp;fly each x in the simulator (B/R flights); compute F(x)",
        "6: &nbsp;&nbsp;&nbsp;add x with F(x) \u2265 \u03c4<sub>F</sub> to <i>M</i> unless a near-duplicate is already in <i>M</i>",
        "7: &nbsp;&nbsp;&nbsp;fine-tune \u03b5<sub>\u03b8</sub> on D<sub>0</sub> plus 5 copies of <i>M</i> (800 steps)",
        "8: <b>end for</b>",
        "9: re-run each failure under 6 single-factor interventions (Eq. 5)",
        "<b>Output:</b> failures with measured causes, passed to the LLM analyst",
    ]
    data = [[Paragraph(lines[0], S["algo"])]] + [[Paragraph(t, S["algo"])] for t in lines[1:]]
    t = Table(data, colWidths=[COLW])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                           ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
                           ("TOPPADDING", (0, 0), (-1, -1), 0.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    return KeepTogether([t, Spacer(1, 8)])


# ------------------------------------------------------------------------------------------------ layout
TITLE_H = 4.3 * inch          # title block + teaser figure on page 1
def _wide_h():
    from PIL import Image as PILImage
    w, h = PILImage.open(os.path.join(FIG, "fig2_example.png")).size
    return FULLW * h / w + 0.75 * inch     # image + caption + gap


WIDE_FIG_H = _wide_h()          # full-width figure drawn at the top of one later page


def footer(canv, doc):
    canv.saveState()
    canv.setFont("TNR", 8.5)
    canv.drawCentredString(PW / 2, 0.45 * inch, str(doc.page))
    canv.restoreState()


def wide_figure_page(canv, doc):
    footer(canv, doc)
    path = os.path.join(FIG, "fig2_example.png")
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    iw = FULLW
    ih = iw * h / w
    top = PH - M
    canv.drawImage(path, M, top - ih, width=iw, height=ih)
    cap = Paragraph(
        "<b>Figure 2:</b> A failure from the random baseline, as recorded by the simulator. Left: flight path at "
        "the closest approach; the circle is the sensing range and obstacle #0 appears late (pop-up), detected at "
        "t = 6 s. Right: "
        "clearance and speed over time. A 4 m/s gust at t = 8 s pushes the UAV into the obstacle it is avoiding. "
        "Re-flying the same scenario with wind removed (and nothing else changed) gives a safe flight, so wind is a "
        "<i>measured</i> cause; no other single intervention prevents the crash.", S["cap"])
    _, ch = cap.wrap(FULLW, 100)
    cap.drawOn(canv, M, top - ih - ch - 2)


def build(story):
    doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                          title="Generative AI-Based Red-Teaming for Autonomous UAV Navigation",
                          author="Snehil Chatterjee", subject="CS787 course project report")
    body_h = PH - 2 * M
    first = [Frame(M, PH - M - TITLE_H, FULLW, TITLE_H, id="title", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0),
             Frame(M, M, COLW, body_h - TITLE_H, id="c1", leftPadding=0, rightPadding=0, topPadding=0,
                   bottomPadding=0),
             Frame(M + COLW + GAP, M, COLW, body_h - TITLE_H, id="c2", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0)]
    two = [Frame(M, M, COLW, body_h, id="l", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
           Frame(M + COLW + GAP, M, COLW, body_h, id="r", leftPadding=0, rightPadding=0, topPadding=0,
                 bottomPadding=0)]
    wide = [Frame(M, M, COLW, body_h - WIDE_FIG_H, id="wl", leftPadding=0, rightPadding=0, topPadding=0,
                  bottomPadding=0),
            Frame(M + COLW + GAP, M, COLW, body_h - WIDE_FIG_H, id="wr", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)]
    doc.addPageTemplates([PageTemplate("first", first, onPage=footer),
                          PageTemplate("two", two, onPage=footer),
                          PageTemplate("wide", wide, onPage=wide_figure_page)])
    doc.build(story)


# ------------------------------------------------------------------------------------------------ content
def content():
    st = []
    # ---- title block (full width) + teaser figure
    st += [Spacer(1, 2),
           P("Generative AI-Based Red-Teaming for<br/>Autonomous UAV Navigation", "title"),
           Spacer(1, 8),
           P("CS787 Course Project", "author"),
           Spacer(1, 2),
           P("Snehil Chatterjee (241026) &nbsp;\u00b7&nbsp; snehil24@iitk.ac.in", "affil"),
           Spacer(1, 7),
           repo_banner(),
           Spacer(1, 9)]
    p = os.path.join(FIG, "fig1_overview.png")
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    fw = FULLW * 0.96
    st += [Image(p, width=fw, height=fw * h / w),
           P("<b>Figure 1:</b> Overview. A conditional diffusion model, trained on scenarios whose obstacle speeds "
             "and winds follow real data, generates scenarios for an unmodified RRT/RRT* UAV planner. Simulated "
             "failures fill a de-duplicated buffer that fine-tunes the generator each round. Every failure is "
             "re-flown under single-factor interventions to measure its causes, which a grounded LLM turns into "
             "explanations and which guide targeted hardening; the hardened system is then attacked again. Shaded "
             "blocks form the closed red-teaming loop (Algorithm 1).", "cap"),
           NextPageTemplate("two"), FrameBreak()]

    # ---- abstract
    st += [P("Abstract", "abs_h"),
           P("Sampling-based planners such as RRT are widely used for UAV navigation, yet their failures under wind, "
             "delayed sensing and moving obstacles are rare and hard to find by random testing. Following DiffScene, "
             "which showed that diffusion models generate realistic safety-critical driving scenarios, we build a "
             "red-teaming framework for UAV path planning that <i>generates</i>, <i>explains</i> and <i>fixes</i> "
             "navigation failures. A conditional diffusion model over a 46-dimensional scenario space, trained on "
             "scenarios grounded in the Stanford Drone Dataset and NOAA wind records, is steered toward difficulty with "
             "classifier-free guidance and fine-tuned in closed loop on the failures it finds. Each failure is re-flown "
             "under single-factor interventions to measure its causes, and an LLM (Gemini) explains it under a "
             "citation check. With 2,000 simulated flights per run, our method finds 552 distinct failures versus 510 "
             "for CMA-ES and 214 for random sampling, spread over 3.5\u00d7 more of the scenario space than CMA-ES. "
             "Diffusion is far more valid and realistic than a tuned conditional VAE (87% vs. 50% valid). Measured "
             "counterfactuals raise the LLM's recall of true causes from 34% to 96%, whereas retrieving similar past "
             "cases does not help. Fixes derived from the measured causes survive 79% of past failures and cut the "
             "failure rate from 12.0% to 4.7%; re-attacking the hardened system, our red team still finds 2.1\u00d7 "
             "more failures than random sampling.", "abs"),
           Spacer(1, 4)]

    # ---- 1 Introduction
    st += [H1("1 &nbsp;Introduction"),
           P("Autonomous UAVs rely on path planners that must keep working when the world is not what the planner "
             "expects: wind pushes the vehicle off its path, obstacles appear late or move, and sensing is limited, "
             "noisy and delayed. Such failures are rare: on random scenarios our test planner fails about 11% of the "
             "time, and the most informative failures arise from <i>combinations</i> of factors that random sampling "
             "seldom produces. Black-box search finds failures faster but tends to collapse onto one failure mode "
             "(Corso et al. 2021). Testing therefore needs a generator that is both <i>targeted</i> and <i>diverse</i>, "
             "and that stays within physically plausible scenarios.", "body0"),
           P("DiffScene (Xu et al. 2023; 2025) addressed the same tension for autonomous driving by using a diffusion "
             "model as a realistic prior over scenarios and guiding it toward safety-critical events. It argued that "
             "diffusion represents the low-density tail of a scenario distribution better than VAE-based generators "
             "such as STRIVE (Rempe et al. 2022) and stays more realistic than adversarial optimisation such as AdvSim "
             "(Wang et al. 2021). We take this idea to UAV navigation and extend it in three directions."),
           P("<b>Contributions.</b>", "body0")]
    st += bullets([
        "<b>Closed-loop diffusion red-teaming</b> for an unmodified open-source RRT/RRT* stack: a conditional DDPM "
        "with classifier-free guidance, fine-tuned round by round on simulator-confirmed failures, instead of "
        "gradient guidance through a learned surrogate.",
        "<b>Real-data grounding and a fairness rule.</b> Obstacle speeds come from the Stanford Drone Dataset and "
        "wind from NOAA records; scenarios in which an obstacle appears inside the UAV's reaction distance are "
        "rejected, so reported failures are avoidable in principle.",
        "<b>Measured causes and a grounded LLM analyst.</b> Six single-factor counterfactual re-runs label every "
        "failure with its causes; an LLM explains failures under a script-checked citation rule, and an ablation "
        "measures which evidence makes its explanations faithful.",
        "<b>Closing the loop.</b> The measured causes drive four targeted fixes; we measure regression, side "
        "effects and a re-attack of the hardened system.",
    ])

    # ---- 2 Related work
    st += [H1("2 &nbsp;Related Work"),
           P("<b>Safety-critical scenario generation.</b> A survey of black-box safety validation (Corso et al. 2021) "
             "distinguishes sampling, optimisation and learned generators. AdvSim (Wang et al. 2021) perturbs actor trajectories adversarially; "
             "STRIVE (Rempe et al. 2022) optimises in the latent space of a learned VAE traffic model. DiffScene (Xu et al. 2023; 2025) replaces the VAE prior "
             "with a diffusion model and guides each denoising step with safety, functionality and constraint "
             "objectives, reporting more realistic and more transferable scenarios than six baselines. Our work "
             "follows DiffScene's prior-plus-guidance design but targets UAV planners, uses classifier-free guidance "
             "(Ho and Salimans 2021) with closed-loop fine-tuning, and adds causal analysis.", "body0"),
           P("<b>Generative models.</b> We use DDPMs (Ho, Jain, and Abbeel 2020) with a cosine schedule and FiLM "
             "conditioning, compared against a conditional VAE (Sohn, Lee, and Yan 2015). Our search baseline is CMA-ES (Hansen 2016)."),
           P("<b>Planners and explanation.</b> RRT (LaValle 1998) and RRT* (Karaman and Frazzoli 2011) are taken "
             "unmodified from PythonRobotics (Sakai et al. 2018). Causes are defined by intervention. "
             "Our LLM analyst uses Gemini with retrieval of similar past failures (retrieval-augmented generation); "
             "we test whether retrieval or measured "
             "interventions make explanations faithful."),
           ]

    # ---- 3 Problem formulation
    st += [H1("3 &nbsp;Problem Formulation"),
           P("A scenario is a vector x in <b>R</b><super>46</super>: start, goal and cruise speed of the UAV; "
             "five obstacle slots (position, velocity, radius, appearance time, active flag); mean wind, gust "
             "amplitude and onset; sensing range and position-noise level. A rule set V(x) rejects "
             "physically impossible scenarios (overlapping start and obstacles, out-of-range values, too-short "
             "missions) and, new in this work, unfair ones (Sec. 4.1). The simulator f flies the planner through x "
             "and returns a trace; the failure score is", "body0"),
           eq("F = 0.4\u00b71<sub>coll</sub> + 0.2\u00b71<sub>no goal</sub> + 0.2\u00b7\u03c3((2 \u2212 d<sub>min</sub>)/0.5)"
              "<br/>+ 0.1\u00b7min(n<sub>replan</sub>/20, 1) + 0.1\u00b7dev,", 1),
           P("where d<sub>min</sub> is the minimum clearance in metres and dev the normalised path deviation. A "
             "flight <i>fails</i> if it collides, leaves the geofence or times out. Analogous to DiffScene's safety, "
             "functionality and constraint objectives, F mixes safety (collision, clearance) and mission "
             "(goal, replanning) terms, while V plays the role of the constraint. Given a budget of B simulated "
             "flights, the red team seeks a set X of valid scenarios maximising the number of <i>distinct</i> "
             "failures,", "body0"),
           eq("max<sub>|X| = B</sub> |D(X)|, &nbsp; D(X) = { x in X : fail(x), and<br/>"
              "||z(x) \u2212 z(x')|| \u2265 1 for every earlier failure x' },", 2),
           P("where z(\u00b7) is the normalised scenario encoding, so that near-duplicates of an earlier failure do "
             "not count.", "body0")]

    # ---- 4 Method
    st += [H1("4 &nbsp;Method"),
           H2("4.1 &nbsp;Simulator, Planner and Real-Data Prior"),
           P("The 2D simulator (60\u00d760 m, 5 m geofence, dt = 0.05 s) models a UAV that tracks the planner's "
             "path with an acceleration limit of 5 m/s<super>2</super> and a turn-rate limit of 1.5 rad/s. Wind acts "
             "as a force with optional 3 s gusts. The planner perceives obstacles only within its sensing range, "
             "0.2 s late and with Gaussian position noise, and replans every 3 s, on a new detection or when its path "
             "is blocked. Figure 2 shows a recorded failure.", "body0"),
           P("<b>Real-data prior.</b> The procedural generator that produces the 10,000 training scenarios draws "
             "moving-obstacle speeds from 149,000 one-second track segments of the Stanford Drone Dataset "
             "(Robicquet et al. 2016), obtained via OpenTraj: pedestrians move at a median "
             "1.2 m/s and cyclists at 3.5 m/s. Wind and gusts come from 8,753 hourly NOAA ISD observations at San "
             "Francisco airport in 2023 (NOAA NCEI 2023): median wind 4.1 m/s, gusts in 13% of hours."),
           P("<b>Fairness rule.</b> In an early version, search methods learned to spawn obstacles directly in front "
             "of the UAV, producing crashes no planner could avoid. We reject any scenario in which an obstacle "
             "appears closer to the UAV's nominal route position than its reaction distance"),
           eq("r = v\u03c4 + v<super>2</super>/(2a) + |v<sub>o</sub>|(\u03c4 + v/a) + 1 m,", 3),
           P("with UAV speed v, sensing delay \u03c4 = 0.2 s, deceleration a = 5 m/s<super>2</super> and obstacle "
             "speed v<sub>o</sub>. The rule cut unavoidable crashes from 18% to 6.5% of our method's failures and "
             "from 41% to 18% of CMA-ES's; the remainder arise when the UAV drifts off its nominal route and are "
             "flagged in the analysis.", "body0"),
           H2("4.2 &nbsp;Conditional Diffusion Generator"),
           P("Scenario vectors are normalised to [\u22121, 1] and modelled with a DDPM (T = 200, cosine schedule). "
             "The denoiser is a residual MLP (width 512, 4 blocks, 3.4M parameters) with FiLM conditioning on the "
             "time step and a condition vector c = (C(x), F): seven coarse descriptors C (cruise speed, obstacle "
             "count, share of moving obstacles, wind magnitude and direction relative to travel, sensing range) and "
             "the difficulty F. Conditions are randomly masked during training so that any subset can be specified at "
             "sampling time. The model is trained with the standard objective", "body0"),
           eq("L = E<sub>x, \u03b5, t</sub> ||\u03b5 \u2212 \u03b5<sub>\u03b8</sub>(x<sub>t</sub>, t, c)||<super>2</super>", 4),
           P("and sampled with classifier-free guidance, \u03b5\u0302 = \u03b5<sub>u</sub> + w(\u03b5<sub>c</sub> "
             "\u2212 \u03b5<sub>u</sub>), where \u03b5<sub>u</sub> uses the null condition. Asking for F = 1 with "
             "w &gt; 1 steers samples toward difficulty. DiffScene instead adds gradients of a learned adversarial "
             "objective at each denoising step; our variant needs no differentiable surrogate of the planner, which "
             "is a non-differentiable sampling-based algorithm.", "body0"),
           H2("4.3 &nbsp;Closed-Loop Red-Teaming"),
           P("Algorithm 1 spends the budget over R = 5 rounds. Failures with F \u2265 0.5 enter a buffer unless they "
             "lie within 0.5 of a buffered scenario; after each round the generator is fine-tuned on the seed data "
             "plus five copies of the buffer, so that the model's notion of \u201chard\u201d moves toward the "
             "failures the planner actually exhibits.", "body0"),
           algorithm(),
           H2("4.4 &nbsp;Measured Causes"),
           P("Each failure x is re-flown with the same planner seed under six interventions T<sub>k</sub>: no wind, "
             "full sensing range (25 m), no sensor noise, late obstacles present from t = 0, 30% slower cruise speed, "
             "and three fresh planner seeds (fixed if at least two succeed). Factor k is a <i>measured cause</i> if", "body0"),
           eq("cause<sub>k</sub>(x) = fail(x) and not fail(T<sub>k</sub>(x)),", 5),
           P("and the set of causes defines the failure <i>type</i>; a failure that no single intervention prevents "
             "is <i>compound</i>.", "body0"),
           H2("4.5 &nbsp;Grounded LLM Analyst"),
           P("For each failure we assemble an evidence record: scenario parameters, trace measurements (clearance, "
             "detection time and gap, replans, wind felt), counterfactual outcomes, and the three most similar past "
             "failures retrieved by k-nearest neighbours from the failure library together with their measured types. "
             "Gemini (gemini-3.1-flash-lite, free tier) must return JSON naming factors from a fixed vocabulary, each "
             "citing the evidence field that supports it. A script rejects citations of missing fields and of the "
             "retrieved cases; the simulator, never the LLM, decides the outcome.", "body0")]

    # ---- 5 Experiments
    st += [NextPageTemplate(["wide", "*", "two"]),
           H1("5 &nbsp;Experiments"),
           P("<b>Setup.</b> Six methods each fly 2,000 scenarios per run, over three seeds: random sampling of the "
             "procedural generator; CMA-ES maximising F directly over the 46 numbers (population 16, 4 restarts); "
             "the cVAE and the diffusion model asked for maximum difficulty (w = 2), each with and without feedback. "
             "The cVAE (latent 32, \u03b2 = 0.3 with KL warm-up and free bits) was selected by a "
             "four-point \u03b2 sweep after a first version suffered posterior collapse.", "body0"),
           P("<b>Metrics.</b> Failures; <i>distinct</i> failures (Eq. 2); <i>coverage</i>, the number of occupied "
             "cells of a 384-cell grid over speed, obstacle count, moving share, wind and sensing range; failure "
             "<i>types</i> among each run's first 200 failures (fair across methods with different failure counts); "
             "and the number of flights needed to reach the 50th distinct failure.")]

    t1 = [["Method", "Failures", "Distinct", "Cover.", "Types", "To 50th"],
          ["Random", "215\u00b115", "214\u00b115", "112\u00b12", "28\u00b12", "461\u00b151"],
          ["CMA-ES", "<b>743\u00b1109</b>", "510\u00b171", "31\u00b12", "8\u00b13", "234\u00b151"],
          ["cVAE", "262\u00b14", "216\u00b12", "84\u00b15", "<b>36\u00b14</b>", "374\u00b148"],
          ["cVAE + fb", "295\u00b19", "209\u00b18", "80\u00b13", "35\u00b14", "375\u00b151"],
          ["Diffusion", "445\u00b119", "445\u00b120", "<b>116\u00b11</b>", "<b>36\u00b13</b>", "<b>230\u00b122</b>"],
          ["Diff. + fb (ours)", "553\u00b122", "<b>552\u00b121</b>", "109\u00b12", "35\u00b14", "<b>230\u00b122</b>"]]
    st += [booktabs(t1, [0.86 * inch, 0.56 * inch, 0.53 * inch, 0.43 * inch, 0.38 * inch, 0.52 * inch],
                    "<b>Table 1:</b> Method comparison, 2,000 flights per run, mean \u00b1 std over 3 seeds. "
                    "Cover. = occupied cells of 384; Types = failure types among the first 200 failures; To 50th = "
                    "flights until the 50th distinct failure.", bold_rows=(6,))]

    st += [H2("5.1 &nbsp;Main Comparison"),
           P("Table 1 and Figure 3 summarise the comparison. Diffusion with feedback finds the most distinct "
             "failures (552), 2.6\u00d7 random sampling. CMA-ES finds the most <i>raw</i> failures (743), but 31% of "
             "them are near-duplicates and 88% belong to a single type (wind only, slow UAV in strong wind): it "
             "covers 31 grid cells against our 109 and finds 8 failure types in its first 200 failures against our "
             "35. Its discovery curve also flattens late in the budget, while ours keeps rising (Fig. 3). This mirrors "
             "DiffScene's observation that search-based adversaries find many but narrow failures, while a learned "
             "prior keeps them diverse. Plain diffusion and random sampling cover slightly more cells than our "
             "method, which concentrates its budget where failures are; the cVAE matches us on failure types per "
             "200 failures but finds 2.6\u00d7 fewer distinct failures.", "body0"),
           figure(os.path.join(FIG, "fig3_discovery.png"), COLW,
                  "<b>Figure 3:</b> Cumulative distinct failures as the flight budget is spent (mean of 3 seeds). "
                  "CMA-ES leads early, then flattens as it exhausts one failure mode."),
           H2("5.2 &nbsp;Why Diffusion Rather than a VAE?"),
           P("DiffScene motivated diffusion over VAE priors by realism in the tail of the distribution. Table 2 "
             "reproduces this argument in the UAV domain. Without conditions, 87% of diffusion samples pass the "
             "validity rules against 50% for the best cVAE, and diffusion's condition features are 2.1\u00d7 closer "
             "(Wasserstein distance normalised by the data's standard deviation) to the real training distribution. "
             "Lowering the cVAE's KL weight sharpens its samples but makes validity worse (22\u201339%), so no "
             "setting achieves both. Diffusion is also steerable: asking for higher difficulty raises its failure "
             "rate monotonically to 35% at w = 4, while the cVAE never exceeds 13% (Fig. 4a). Generated scenarios are "
             "novel, not memorised: their median distance to the nearest training scenario (1.04) is close to that of "
             "fresh random scenarios (1.10).", "body0")]

    t2 = [["Generator", "Valid", "W-dist. \u2193", "Fail@F=1"],
          ["cVAE \u03b2 = 0.02", "23%", "0.31", "\u2013"],
          ["cVAE \u03b2 = 0.05", "32%", "0.31", "\u2013"],
          ["cVAE \u03b2 = 0.1", "39%", "0.36", "\u2013"],
          ["cVAE \u03b2 = 0.3 (used)", "50%", "0.36", "13%"],
          ["Diffusion (w = 2 / 4)", "<b>87%</b>", "<b>0.17</b>", "<b>23% / 35%</b>"]]
    st += [booktabs(t2, [1.25 * inch, 0.5 * inch, 0.65 * inch, 0.85 * inch],
                    "<b>Table 2:</b> Generator quality. Valid = share of unconditional samples passing the validity "
                    "rules; W-dist. = mean normalised Wasserstein distance of condition features to the training "
                    "data (lower is more realistic); Fail@F=1 = failure rate when asked for maximum difficulty.",
                    bold_rows=(5,))]

    st += [H2("5.3 &nbsp;Does Feedback Help?"),
           P("For diffusion, yes: over five rounds of fine-tuning on its own failures the failure rate rises from 21% "
             "to 36%, while without feedback it stays near 22% (Fig. 4b). Validity also rises (76% to 84%) because "
             "buffered failures are valid by construction. The cVAE barely benefits (13% to 15%): fine-tuning cannot "
             "fix a prior that produces mostly invalid samples.", "body0"),
           figure(os.path.join(FIG, "fig4_generators.png"), COLW,
                  "<b>Figure 4:</b> (a) Failure rate when the generator is asked for a given difficulty; the dotted "
                  "line is random sampling. (b) Failure rate per round of the red-team loop (3 seeds)."),
           H2("5.4 &nbsp;Are the LLM's Explanations Faithful?"),
           P("We explained 50 failures four times each with different evidence (200 explanations) and scored every "
             "named factor against the measured causes (Table 3, Fig. 5). From the trace alone the LLM names only 34% "
             "of the true causes and 44% of its explanations blame a factor the re-runs rule out. Adding retrieved "
             "similar cases, a form of retrieval-augmented generation, does not help: recall stays at 36% and "
             "contradictions rise to 60%, because the model tends to copy the failure type of its neighbours. With "
             "measured counterfactuals, recall reaches 92\u201396% with at most 2% contradictions, and no explanation "
             "cited missing evidence. Because the counterfactual variants are scored against evidence they were "
             "shown, their scores measure faithful <i>use</i> of evidence; the independent finding is the first two "
             "rows: causes should be measured by the simulator and only interpreted by the LLM.", "body0")]

    t3 = [["Evidence given", "Precision", "Recall", "Contradicted"],
          ["Trace only", "63%", "34%", "44%"],
          ["+ similar cases (retrieval)", "59%", "36%", "60%"],
          ["+ counterfactuals", "<b>100%</b>", "92%", "<b>0%</b>"],
          ["All evidence", "98%", "<b>96%</b>", "2%"]]
    st += [booktabs(t3, [1.4 * inch, 0.6 * inch, 0.55 * inch, 0.7 * inch],
                    "<b>Table 3:</b> LLM analyst ablation (50 failures \u00d7 4 evidence variants). Contradicted = "
                    "share of explanations naming a cause that the counterfactual re-runs rule out.")]
    st += [figure(os.path.join(FIG, "fig5_llm.png"), COLW,
                  "<b>Figure 5:</b> Faithfulness of the LLM's explanations by evidence given.")]

    st += [H2("5.5 &nbsp;Are the Failures Real?"),
           P("A 2D simulator invites the question whether failures are numerical artefacts. We re-flew 500 failures "
             "from random sampling and 500 from our method at the original step and at 2\u00d7 and 4\u00d7 finer "
             "steps, each with three fresh planner seeds. 78% of our failures still fail at the original step (the "
             "rest depend on RRT's random sampling) and 76% at the 4\u00d7 finer step, the same as random-sampling "
             "failures (78% / 76%). On 2,000 random scenarios RRT* fails 12.4% (RRT 12.2%), and 77% of the failures "
             "found against RRT also break RRT*, with the same measured causes. The red team thus finds planner-level "
             "weaknesses rather than simulator artefacts or quirks of one planner.", "body0"),
           H2("5.6 &nbsp;Closing the Loop: Harden and Re-attack"),
           P("The measured causes point to four fixes around the unmodified planner: wind feed-forward in the "
             "controller (wind is a cause in 79% of baseline failures), a speed governor tied to sensing range "
             "(\u201ctoo fast to react\u201d, 46%), best-of-three planning (planner randomness, 25%) and a two-second "
             "swept footprint for moving obstacles. Replaying all 8,650 failures found so far with the same planner "
             "seeds, the hardened system survives 79% (Table 4). Compound failures are hardest to fix (42% "
             "survived), and failures found by CMA-ES are the most persistent (65%). On 3,000 ordinary scenarios the "
             "failure rate falls from 12.0% to 4.7% at the cost of 6% longer flights and 61 new failures. Re-attacking "
             "the hardened system, our red team finds 176 distinct failures per 2,000 flights against 84 for random "
             "sampling, and late detection rises to the second most common cause, pointing to the next fix.", "body0")]

    t4 = [["Measure", "Original", "Hardened"],
          ["Past failures survived (of 8,650)", "0%", "<b>79%</b>"],
          ["Failure rate, ordinary scenarios", "12.0%", "<b>4.7%</b>"],
          ["Near-failure rate", "4.0%", "0.9%"],
          ["Mean flight time to goal", "12.0 s", "12.7 s"],
          ["Re-attack: random, distinct / 2k", "220", "84"],
          ["Re-attack: ours, distinct / 2k", "552", "176"]]
    st += [booktabs(t4, [1.75 * inch, 0.7 * inch, 0.7 * inch],
                    "<b>Table 4:</b> Hardening derived from measured causes: regression on past failures, side "
                    "effects on 3,000 ordinary scenarios, and a re-attack (2 seeds).")]

    # ---- 6 Limitations
    st += [H1("6 &nbsp;Limitations"),
           *bullets([
               "<b>Simplified world.</b> 2D kinematics with acceleration and turn limits, a wind force and delayed "
               "noisy sensing, but no aerodynamics or 3D geometry; results do not transfer directly to real "
               "quadrotors.",
               "<b>Partial grounding.</b> Obstacle speeds and wind follow real data; obstacle layouts, sensing "
               "parameters and the UAV model are synthetic, and the wind comes from one airport.",
               "<b>Residual unavoidable failures.</b> The fairness rule uses the nominal route; 6.5% of our failures "
               "(18% of CMA-ES's) still involve an obstacle appearing within braking time after the UAV drifted.",
               "<b>Metric choices.</b> The duplicate threshold, coverage grid and type definition are ours; with "
               "raw failure counts CMA-ES ranks first. Only two failure types were found exclusively by our method.",
               "<b>Evaluation scale.</b> Three seeds per method; the LLM was scored automatically on 50 cases with one "
               "model and no human rating.",
           ])]

    # ---- 7 Conclusion
    st += [H1("7 &nbsp;Conclusion"),
           P("We transferred DiffScene's central idea, a diffusion model as a realistic prior for safety-critical "
             "scenario generation, from driving to UAV path planning, and closed the loop from generation to "
             "explanation to repair. A closed-loop conditional diffusion model finds the most distinct failures of an "
             "unmodified RRT stack and spreads them over far more of the scenario space than direct search, while "
             "remaining valid and realistic where a tuned cVAE does not. Measured counterfactual causes, not "
             "retrieval, make LLM explanations faithful, and they translate into targeted fixes that remove most "
             "known failures. The hardened planner still fails, and the red team still finds those failures twice as "
             "fast as random testing, which is the point of red-teaming as a continuing process. Natural next steps "
             "are 3D dynamics, wind fields learned from more stations, and guidance with a learned failure critic as "
             "in DiffScene.", "body0")]

    # ---- availability
    st += [H1("Code and Data Availability"),
           P(f"All code, trained-model recipes, results and the demo video are at {link(REPO)}. "
             f"Appendix A lists where each part of this paper lives in the repository. The planner is "
             f"PythonRobotics ({link('https://github.com/AtsushiSakai/PythonRobotics')}); obstacle speeds come from "
             f"the Stanford Drone Dataset via OpenTraj ({link('https://github.com/crowdbotp/OpenTraj')}); wind comes "
             f"from NOAA ISD ({link('https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database')}"
             f"); the LLM is the Gemini API ({link('https://ai.google.dev')}).", "bodyl")]

    # ---- references
    st += [H1("References")]
    refs = [
        "Corso, A.; et al. 2021. A Survey of Algorithms for Black-Box Safety Validation of Cyber-Physical Systems. "
        "<i>Journal of Artificial Intelligence Research</i> 72: 377\u2013428.",
        "Hansen, N. 2016. The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772.",
        "Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Diffusion Probabilistic Models. In <i>NeurIPS</i>.",
        "Ho, J.; and Salimans, T. 2021. Classifier-Free Diffusion Guidance. In <i>NeurIPS Workshop on Deep "
        "Generative Models</i>.",
        "Karaman, S.; and Frazzoli, E. 2011. Sampling-Based Algorithms for Optimal Motion Planning. <i>IJRR</i> "
        "30(7): 846\u2013894.",
        "LaValle, S. M. 1998. Rapidly-Exploring Random Trees: A New Tool for Path Planning. Tech. Rep. 98-11, "
        "Iowa State University.",
        "NOAA NCEI. 2023. Integrated Surface Database, station 72494023234 (San Francisco Intl.).",
        "Rempe, D.; et al. 2022. Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior. "
        "In <i>CVPR</i>.",
        "Robicquet, A.; et al. 2016. Learning Social Etiquette: Human Trajectory Understanding in Crowded Scenes. "
        "In <i>ECCV</i>.",
        "Sakai, A.; et al. 2018. PythonRobotics: A Python Code Collection of Robotics Algorithms. arXiv:1808.10703.",
        "Sohn, K.; Lee, H.; and Yan, X. 2015. Learning Structured Output Representation Using Deep Conditional "
        "Generative Models. In <i>NeurIPS</i>.",
        "Wang, J.; et al. 2021. AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles. In <i>CVPR</i>.",
        "Xu, C.; Petiushko, A.; Zhao, D.; and Li, B. 2025. DiffScene: Diffusion-Based Safety-Critical Scenario "
        "Generation for Autonomous Vehicles. In <i>AAAI</i> 39(8): 8797\u20138805.",
        "Xu, C.; Zhao, D.; Sangiovanni-Vincentelli, A.; and Li, B. 2023. DiffScene: Diffusion-Based Safety-Critical "
        "Scenario Generation for Autonomous Vehicles. In <i>AdvML-Frontiers Workshop at ICML</i>.",
    ]
    st += [P(r, "ref") for r in refs]

    # ---- appendix: repository guide
    R = "results/"
    rows = [["Path (link)", "What it contains"],
            [repo("README.md"), "Start here: overview, headline numbers, map from paper sections to results"],
            [repo("report/report.pdf"), "This paper"],
            [repo("results/demo.mp4"), "2.5-minute video: example flights and the main results"],
            [repo("results/08_case_study/case_study.pdf"), "<b>One real case end to end</b>: crash created, causes measured, LLM explanation, fix, re-flown safely, attacked again (also as a video)"],
            [repo("results/README.md"), "Index of all results, stage by stage"],
            [repo(R + "01_random_baseline"), "Random baseline and the real-data scenario prior (Sec. 4.1, Fig. 2)"],
            [repo(R + "02_generators"), "Diffusion vs. cVAE: validity, realism, steering (Sec. 5.2, Table 2)"],
            [repo(R + "03_red_team_loop"), "Effect of feedback fine-tuning, per round (Sec. 5.3)"],
            [repo(R + "04_llm_analyst"), "LLM ablation and every explanation with its score (Sec. 5.4, Table 3)"],
            [repo(R + "05_comparison"), "Main comparison of all methods (Sec. 5.1, Table 1, Fig. 3)"],
            [repo(R + "06_fidelity_checks"), "Finer time steps and RRT* transfer (Sec. 5.5)"],
            [repo(R + "07_harden_and_reattack"), "Hardening, side effects, re-attack (Sec. 5.6, Table 4)"],
            [repo("code/README.md"), "Setup and the command for every stage"],
            [repo("code/scenarios"), "Scenario vector, validity and fairness rules, real-data statistics"],
            [repo("code/sim/simulator.py"), "2D UAV simulator (dynamics, wind, delayed sensing, replanning)"],
            [repo("code/planners"), "Unmodified PythonRobotics RRT / RRT* and its adapter"],
            [repo("code/generators/diffusion.py"), "Conditional DDPM with classifier-free guidance (Sec. 4.2)"],
            [repo("code/generators"), "Also the cVAE and CMA-ES baselines"],
            [repo("code/loop/scoring.py"), "Failure score F (Eq. 1)"],
            [repo("code/loop/redteam.py"), "Closed red-teaming loop (Sec. 4.3, Algorithm 1)"],
            [repo("code/loop/metrics.py"), "Distinct failures, coverage, failure types"],
            [repo("code/analyst"), "Counterfactual re-runs, evidence records, Gemini analyst, grounding check (Sec. 4.4-4.5)"],
            [repo("code/experiments/run_all.py"), "Runs the whole pipeline in order (about 2.5 h)"]]
    st += [CondPageBreak(2.5 * inch), H1("Appendix A &nbsp;Repository Guide"),
           P("The repository has three folders. <b>report/</b> holds this paper. <b>results/</b> holds one folder "
             "per experiment; inside each, <i>1_runs</i> has flight pictures and videos, <i>2_results</i> the "
             "charts and CSV tables, and <i>3_crash_reasons</i> the measured causes and explanations of individual "
             "failures. <b>code/</b> holds the full source, laid out as in Figure 1. A quick tour: read the "
             "top-level README, open the end-to-end case study (one real crash followed from generation to fix "
             "and re-attack), then the results folder of any section of interest. "
             "All paths below are links.", "body0"),
           Spacer(1, 4),
           booktabs(rows, [1.62 * inch, COLW - 1.62 * inch], "<b>Table A1:</b> Where to find each part of the "
                    "paper in the repository.", left_cols=(0, 1), font=7.5)]
    return st


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    build(content())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
