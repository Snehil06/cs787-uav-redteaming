"""Build the project proposal PDF (plain style, matching the original document).

    python tools/build_proposal_pdf.py
"""
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, KeepTogether, ListFlowable, ListItem, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Generative AI Red-Teaming for UAV Navigation (1).pdf")
FONTS = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Sans", os.path.join(FONTS, "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", os.path.join(FONTS, "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Italic", os.path.join(FONTS, "segoeuii.ttf")))
pdfmetrics.registerFont(TTFont("Sans-BoldItalic", os.path.join(FONTS, "segoeuiz.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONTS, "consola.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic",
                              boldItalic="Sans-BoldItalic")

INK, GREY, RULE = colors.HexColor("#111111"), colors.HexColor("#444444"), colors.HexColor("#cccccc")
S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=22, leading=28, textColor=INK, spaceAfter=14),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=17, leading=22, textColor=INK, spaceBefore=6,
                         spaceAfter=10, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=12, leading=16, textColor=INK, spaceBefore=6,
                         spaceAfter=6, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=10, leading=15, textColor=INK, spaceAfter=8),
    "lead": ParagraphStyle("lead", fontName="Sans", fontSize=10, leading=15, textColor=INK, spaceAfter=8,
                           keepWithNext=1),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=10, leading=14.5, textColor=INK),
    "formula": ParagraphStyle("formula", fontName="Sans-Italic", fontSize=10.5, leading=15, alignment=TA_CENTER),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=9, leading=12.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=9, leading=12.5, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="Mono", fontSize=8.6, leading=11.2, textColor=INK),
}
story = []


def H1(t):
    story.append(Paragraph(t, S["title"]))


def H2(t):
    story.append(Paragraph(t, S["h2"]))


def H3(t):
    story.append(Paragraph(t, S["h3"]))


def P(t):
    """A body paragraph; one ending in ':' introduces what follows and stays on its page."""
    story.append(Paragraph(t, S["lead"] if t.rstrip().endswith(":") else S["body"]))


def rule():
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=4, spaceAfter=12))


def bullets(items, numbered=False):
    lst = ListFlowable([ListItem(Paragraph(i, S["bullet"]), spaceAfter=2) for i in items],
                       bulletType="1" if numbered else "bullet", start="1" if numbered else "•",
                       bulletFormat="%s." if numbered else None, bulletFontName="Sans", bulletFontSize=9.5,
                       leftIndent=30, bulletDedent=14)
    story.append(lst)
    story.append(Spacer(1, 8))


def box(text):
    t = Table([[Preformatted(text.strip("\n"), S["mono"])]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(KeepTogether([t, Spacer(1, 10)]))


def formula(text, boxed=False):
    p = Paragraph(text, S["formula"])
    if boxed:
        plain = re.sub(r"<[^>]+>", "", text)
        bold = "<b>" in text
        w = pdfmetrics.stringWidth(plain, "Sans-BoldItalic" if bold else "Sans-Italic", 10.5) + 24
        t = Table([[p]], colWidths=[min(w, 170 * mm)])
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, INK), ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(t)
    else:
        story.append(p)
    story.append(Spacer(1, 10))


def table(rows, widths):
    data = [[Paragraph(c, S["cellb"] if r == 0 else S["cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(t)
    story.append(Spacer(1, 10))


# ---------------------------------------------------------------- content
H1("Generative AI-Based Red-Teaming for<br/>Autonomous UAV Navigation")

H2("1. Project Overview")
P("Autonomous UAV navigation systems such as RRT/RRT* are designed to find collision-free paths through an "
  "environment. However, their robustness depends heavily on the situations in which they are tested. "
  "Conventional testing generally relies on manually designed or randomly generated environments, which may fail "
  "to expose rare combinations of disturbances and dynamic events that can cause navigation failures.")
P("This project proposes a <b>Generative AI-based red-teaming framework for autonomous UAV navigation</b>. "
  "Instead of using Generative AI to directly control the UAV or replace the existing path planner, the system uses "
  "GenAI to <b>automatically generate novel and challenging UAV scenarios</b> and test an existing navigation "
  "algorithm against them. The planner under test is the open-source RRT/RRT* implementation from the "
  "PythonRobotics library, used unmodified.")
P("The framework additionally uses a <b>grounded Large Language Model (LLM)</b> to analyze simulation traces and "
  "provide human-readable explanations for why a UAV navigation attempt succeeded, failed, or encountered a "
  "near-failure. The causes it reports are measured by re-running each failure in the simulator, not guessed by "
  "the LLM.")
P("Finally, the measured causes are used to harden the system around the planner, and the hardened system is "
  "attacked again. The overall objective is to build an automated system that can:")
formula("<b>Generate → Test → Discover → Explain → Fix</b>")
P("challenging UAV navigation scenarios.")
P("<b>Revision note.</b> This version describes the system as it was built: every setting below is the one used. "
  "Two parts were added during the project (fidelity checks and a harden-and-re-attack stage), and items that "
  "were planned but not carried out are marked. Results are reported separately.")
rule()

H2("2. Problem Statement")
P("Testing autonomous UAV navigation systems is difficult because the space of possible environments is extremely "
  "large. A UAV may encounter different combinations of:")
bullets(["static and dynamic obstacles,", "obstacle velocities and trajectories,", "UAV velocities,",
         "wind disturbances and gusts,", "sensor noise and limited sensing range,", "obstacle appearance timing, and",
         "obstacle density."])
P("Most randomly generated scenarios will be relatively ordinary, while failures may occur only in a small region of "
  "this scenario space.")
P("The project investigates whether a generative model can learn the distribution of valid UAV scenarios and "
  "generate novel scenarios that are useful for <b>automated red-teaming</b>, i.e., deliberately searching for "
  "situations that expose weaknesses in an autonomous navigation system. It asks six questions:")
bullets([
    "<b>Learned vs. unlearned generation.</b> Does a generative model find more failures, and more varied ones, "
    "than random sampling for the same number of simulator runs?",
    "<b>Why diffusion.</b> Does a conditional diffusion model beat a conditional VAE on validity, diversity and "
    "failure discovery?",
    "<b>Generation vs. search.</b> Does failure-guided generation find more varied failures than directly "
    "optimising a failure score with a black-box optimiser (CMA-ES), at a similar cost?",
    "<b>Grounded explanation.</b> Do measured counterfactual evidence and similar past cases make the LLM's "
    "explanations more faithful to the simulator?",
    "<b>Real or artefact.</b> Do discovered failures persist at finer simulation steps, and do they also break a "
    "second planner (RRT*)?",
    "<b>Closing the loop.</b> Can the measured causes guide fixes around the planner, and does the red team still "
    "find failures once those fixes are in place?",
], numbered=True)
rule()

H2("3. Core Idea")
P("The project uses an existing navigation algorithm, RRT/RRT*, as the system under test.")
P("The GenAI system does not replace the planner.")
P("Instead:")
formula("GenAI generates scenario → RRT navigates → Simulator evaluates → GenAI explains outcome → fixes are tested",
        boxed=True)
P("The system progressively searches for challenging scenarios and builds a collection of automatically discovered "
  "UAV edge cases, grouped by their measured causes.")
rule()

H2("4. System Architecture")
box("""
                 Scenario conditions C
               (incl. target difficulty)
                          │
                          ▼
             ┌────────────────────────┐
             │  Conditional Diffusion │◄──────────────────┐
             │         Model          │    fine-tune on   │
             └───────────┬────────────┘    found failures │
                         ▼                                │
                Generated scenario                        │
                         │                                │
                         ▼                                │
                  Validity checks                         │
                         │                                │
                         ▼                                │
             ┌────────────────────────┐                   │
             │      UAV Simulator     │                   │
             │ UAV + Obstacles + Wind │                   │
             │   RRT / RRT* planner   │                   │
             └───────────┬────────────┘                   │
                         ▼                                │
           Execution trace + failure score F              │
                         │                                │
              ┌──────────┴───────────┐                    │
              ▼                      ▼                    │
      Counterfactual reruns    Edge-case library ─────────┘
      (measured causes)        (failure types)
              │                      │
              └──────────┬───────────┘
                         ▼
                 ┌───────────────┐
                 │ Grounded LLM  │
                 │    Analyst    │
                 └───────┬───────┘
                         ▼
                Explanation / Report
                         │
                         ▼
     Fixes around the planner → regression test → re-attack
""")
diagram, heading = story.pop(), story.pop()          # keep the heading on the same page as its diagram
story.append(KeepTogether([heading, diagram]))
H3("Simulator and planner")
P("The world is 60 × 60 m with a 5 m geofence. The UAV tracks a desired velocity with a 5 m/s² acceleration limit "
  "and a 1.5 rad/s turn-rate limit, and wind (with gusts) acts as a force on it. The planner sees obstacles only "
  "within its sensing range, 0.2 s late and with position noise. It replans when a new obstacle is sensed, when "
  "its current path becomes blocked, or every 3 s. RRT and RRT* come from PythonRobotics and are used unmodified; "
  "a thin adapter only seeds them, silences their logging and shortcut-smooths their paths. A flight fails if it "
  "collides, leaves the geofence, or does not reach the goal within 3× its nominal flight time.")
P("The generator, feedback loop and analyst reach the simulator only through <font name='Mono'>simulate(scenario) "
  "→ trace</font>, so a 3D simulator could replace it without changing the rest.")
rule()

H2("5. GenAI Component 1: Conditional Diffusion for Scenario Generation")
P("The primary generative model is a <b>conditional diffusion model operating on structured numerical UAV "
  "scenarios</b>. Unlike image diffusion, the model does not generate images. It generates the parameters of a "
  "scenario, and the simulator plays them out.")
H3("Scenario representation")
P("A scenario is a fixed-size vector of 46 numbers:")
bullets(["<b>UAV:</b> start (x, y), goal (x, y) and cruise speed.",
         "<b>Obstacles:</b> five slots, each with position and velocity at its appearance time, radius, appearance "
         "time and an active flag. Unused slots are switched off by the flag; active obstacles are stored first, "
         "sorted along the direction of travel.",
         "<b>Wind:</b> mean wind (w<sub>x</sub>, w<sub>y</sub>), gust strength and gust start time.",
         "<b>Sensing:</b> sensing range and position-noise standard deviation."])
P("A condition vector specifies coarse characteristics of the requested scenario:")
formula("C = [UAV speed, obstacle count, moving fraction, wind strength, cos φ, sin φ, sensing range, "
        "target difficulty]")
P("Here φ is the wind direction relative to the direction of travel, and target difficulty is the failure score F "
  "the scenario should reach (0 = easy, 1 = hardest). Conditions can be given partly: any condition left out is "
  "masked, so a request such as \"difficulty 1.0\" alone lets the model choose everything else.")
P("The diffusion model learns:")
formula("p(S | C)")
P("and can therefore generate multiple novel scenarios satisfying specified conditions. Every generated scenario "
  "passes rule-based validity checks before it is flown: speeds, wind, obstacle sizes and appearance times within "
  "limits; start and goal inside the map and at least 30 m apart; no obstacle on the start at t = 0; no static "
  "obstacle on the goal; and no overlapping static obstacles.")
H3("Training")
P("A seed dataset of 10,000 valid scenarios is generated with a procedural generator, tuned so that RRT fails in "
  "about 12% of them, and each is flown and labelled with its failure score. The model is a DDPM with a 200-step "
  "cosine noise schedule and a residual MLP denoiser (4 blocks of width 512, conditioned on the timestep and C), "
  "about 3.4M parameters, trained for 6,000 steps on the local GPU. During diffusion training:")
bullets(["A valid scenario S<sub>0</sub> is selected.", "Gaussian noise is added to produce S<sub>t</sub>.",
         "A neural network receives the noisy scenario, diffusion timestep, and condition vector (each condition "
         "hidden 30% of the time, all of them 15% of the time).",
         "The network learns to predict the noise that was added."], numbered=True)
P("The training objective is:")
formula("L = E [ || ε − ε<sub>θ</sub>(S<sub>t</sub>, t, C) ||<super>2</super> ]")
P("After training, new scenarios are generated from random noise with classifier-free guidance, "
  "ε<sub>u</sub> + w (ε<sub>c</sub> − ε<sub>u</sub>), at guidance scale w = 2.")
rule()

H2("6. GenAI Component 2: Failure and Success Reasoning Using an LLM")
P("The second GenAI component is a <b>grounded LLM-based test analyst</b>. The LLM is not responsible for deciding "
  "whether the UAV actually collided or succeeded. Those facts come directly from the simulator.")
P("For every test, an evidence record is built from three sources:")
bullets(["<b>Trace summary</b> from the simulator, for example:"])
box("""
UAV cruise speed:                          7.5 m/s
Wind:                                      crosswind from the left, 6.2 m/s
Closest obstacle detected at:              1.8 s   (gap 6.9 m)
Seconds between detection and closest:     0.7 s
Minimum clearance:                         -0.07 m
Replans:                                   3
Goal reached:                              No
End status:                                Collision
""")
bullets(["<b>Counterfactual reruns.</b> Each failure is re-flown with one factor changed at a time: no wind or "
         "gusts; sensing range raised to its 25 m maximum; no sensor noise; late-appearing obstacles present from "
         "the start on the same paths; UAV speed reduced 30%; and three fresh planner seeds (counted as fixed if at "
         "least two succeed). A change that prevents the failure is a <i>measured</i> cause, and the set of such "
         "changes defines the failure's type.",
         "<b>Similar past cases.</b> The 3 nearest failures from the edge-case library (scikit-learn nearest "
         "neighbours over 11 standardised features), with their failure types, given as background only."])
P("The LLM (gemini-3.1-flash-lite on the free Gemini API tier, temperature 0, JSON schema enforced; newer Gemini "
  "models and a local Qwen 2.5 7B via Ollama as fallbacks) then produces an interpretable explanation such as:")
story.append(Paragraph(
    "<b>Failure explanation:</b> The UAV collided with a static obstacle 2.5 s into the flight. A 6.2 m/s crosswind "
    "and a cruise speed of 7.5 m/s left too little margin; the counterfactual reruns confirm that removing the wind, "
    "reducing the speed, or removing the sensor noise would each have prevented the collision.",
    ParagraphStyle("quote", parent=S["body"], leftIndent=28, rightIndent=20)))
P("Each answer names its primary factors from a fixed vocabulary (wind, UAV speed, late detection, sensor noise, "
  "pop-up obstacle, planner randomness, moving obstacle, obstacle density, other), and every factor must cite the "
  "evidence field that supports it. A script rejects citations to fields that do not exist or to the similar-cases "
  "background. The outcome is copied from the simulator, never generated. Every response is cached on disk so "
  "reruns cost no quota. This creates a distinction between:")
formula("Simulator → objective facts and measured causes")
P("and")
formula("LLM → interpretation of those facts")
P("The explanation is therefore grounded in measurable simulation evidence rather than allowing the LLM to invent "
  "the cause of failure.")
rule()

H2("7. Red-Teaming and Failure-Guided Generation")
P("The generated scenarios are evaluated using the UAV simulator and RRT/RRT*. Each scenario receives a failure "
  "score F in [0, 1], with weights fixed before any experiment:")
formula("F = 0.4·collision + 0.2·(goal not reached) + 0.2·σ((2 m − d<sub>min</sub>) / 0.5 m) + 0.1·replans + "
        "0.1·deviation")
P("where d<sub>min</sub> is the minimum clearance, replans saturate at 20 and deviation at twice the straight-line "
  "distance. A run that reaches the goal with less than 0.5 m clearance is a <b>near-failure</b>.")
P("Each run of the red team spends its budget over 5 rounds:")
bullets(["Sample scenarios asking for difficulty 1.0 (guidance w = 2); invalid samples are discarded and replaced.",
         "Fly them in the simulator and compute F (one fifth of the budget per round).",
         "Add scenarios with F ≥ 0.5 to a buffer of discovered failures, skipping near-copies of ones already there.",
         "Fine-tune the model for 800 steps on the seed data plus the buffer (repeated 5 times), each labelled with "
         "the F it actually reached. Keeping the seed data prevents the model from forgetting ordinary scenarios."],
        numbered=True)
P("For example:")
box("""
High UAV velocity
        +
Late obstacle detection
        +
Strong lateral wind
        ↓
High failure rate
        ↓
Generate more variations
        ↓
Test again
""")
P("This creates a feedback loop:")
formula("Generate → Test → Score → Focus on difficult cases → Generate again", boxed=True)
P("The goal is to automatically discover edge cases rather than merely generate random synthetic data. "
  "(Classifier guidance from a learned F predictor was considered and not pursued.)")
rule()

H2("8. Example Use Case")
P("Consider a UAV travelling toward a target. A generated scenario contains:")
bullets(["high UAV velocity,", "a moving obstacle crossing the UAV's path,", "sudden lateral wind,",
         "delayed obstacle appearance."])
P("RRT attempts to navigate the environment. Suppose the UAV collides with the obstacle. The simulator records the "
  "complete execution trace, re-flies the scenario with one factor changed at a time, and produces an illustrative "
  "record such as:")
box("""
Test Case #427
────────────────────────
Outcome: FAILURE (collision)

Counterfactual reruns:
• No wind                    → failure prevented
• UAV 30% slower             → failure prevented
• Full sensing range         → still fails
• Obstacle present from t=0  → failure prevented
• No sensor noise            → still fails
• Other planner seeds        → still fails

Failure type: wind + speed + pop-up obstacle
Minimum clearance: -0.12 m      Replans: 3
""")
P("The LLM converts this into an interpretable explanation. The scenario is retained as a red-team test case, "
  "similar scenarios can be generated for further testing, and the case is later replayed to check whether a fix "
  "works.")
rule()

H2("9. Closing the Loop: Harden and Re-attack")
P("The red team's measured causes are used to improve the system around the planner, and the improved system is "
  "then attacked again. The RRT/RRT* code is never changed: each fix is a switchable setting of the control and "
  "planning layer, aimed at a top measured cause.")
table([["Fix", "What it does", "Targets"],
       ["Wind feed-forward", "the controller cancels the estimated wind force", "wind / gusts"],
       ["Speed governor", "never faster than the UAV can stop within its sensing range, and slower near sensed "
                          "obstacles", "UAV speed, late detection"],
       ["Best of 3 plans", "RRT runs three times per replan and the shortest path is kept", "planner randomness"],
       ["Wider moving-obstacle footprint", "2 s of obstacle motion swept into the planning footprint instead of 1 s",
        "moving obstacles"]], [45, 85, 40])
P("Three measurements follow:")
bullets(["<b>Regression:</b> replay every failure found so far, with the same scenario and planner seed, on the "
         "hardened system.",
         "<b>Side effects:</b> fly 3,000 ordinary scenarios on both systems and compare failure rate, near-misses, "
         "flight time and planning time, including scenarios that newly fail.",
         "<b>Re-attack:</b> run random sampling and the red team (2,000 flights, 2 seeds) against the hardened "
         "system and measure the causes of what still fails."], numbered=True)
rule()

H2("10. Experimental Evaluation")
P("The project compares the proposed GenAI-based red-teaming approach against conventional and alternative "
  "scenario generation. Every method gets the same budget of 2,000 simulator runs per run, repeated with 3 seeds, "
  "with the same planner, simulator settings and validity checks.")
table([["Method", "What it is"],
       ["Random", "sampling from the procedural generator (baseline)"],
       ["CMA-ES search", "black-box optimiser maximising F directly over the 46 scenario numbers (population 16, "
                         "4 restarts); invalid candidates are not flown"],
       ["cVAE", "conditional VAE (latent size 32) with the same data, conditioning and training steps; KL weight "
                "β = 0.3 chosen from a sweep of four values, with KL warm-up and free bits"],
       ["cVAE + feedback", "the cVAE inside the same feedback loop"],
       ["Diffusion", "diffusion sampled at difficulty 1.0 without fine-tuning"],
       ["Diffusion + feedback (proposed)", "the full method of section 7"],
       ["CTGAN", "planned as optional; not run"]], [50, 120])
P("The comparison examines:")
H3("Scenario quality")
P("Are generated scenarios physically and logically valid? (share passing the validity checks)")
H3("Realism, diversity and novelty")
P("Do generated scenarios match the real distribution, and are they new rather than copies? (distribution match, "
  "and distance to the nearest training scenario compared with fresh random scenarios)")
H3("Condition adherence")
P("If a specific speed, obstacle count or wind is requested, is it delivered?")
H3("Failure discovery")
P("How many failures and near-failures are found, and how many are distinct? A failure counts as distinct only if it "
  "is at least 1.0 away, in normalised scenario space, from every earlier failure. Regions covered: cells of a "
  "384-cell grid over speed, obstacle count, moving obstacles, wind and sensing range that contain a failure. "
  "Failure types: groups with the same measured causes, counted in total and among the first 200 failures.")
H3("Search efficiency")
P("How many test cases are required to find the 1st, 10th and 50th distinct failure?")
H3("Fidelity")
P("Do failures persist when re-flown at 2× and 4× finer simulation steps (three fresh planner seeds each, "
  "reproduced if at least two fail)? Do they also break RRT*, and for the same causes?")
H3("Explanation quality")
P("Are LLM-generated explanations consistent with the underlying simulation traces? 50 crashes, chosen to cover "
  "every measured failure type, are each explained four times: from the trace only, with counterfactual reruns, "
  "with similar past cases, and with all evidence. Measures: precision (named causes the reruns confirm), recall "
  "(confirmed causes that get named), explanations naming a cause the reruns rule out, and explanations citing "
  "evidence that does not exist.")
P("<b>Not carried out:</b> the held-out-region robustness test and a human rating of explanation usefulness.")
rule()

H2("11. Scope and Feasibility")
P("To ensure the project is achievable within 5 days (4 working days plus 1 buffer day), the system uses a "
  "lightweight <b>2D UAV simulator</b> rather than a full 3D flight simulator. The project focuses on:")
bullets(["2D UAV motion with acceleration and turn-rate limits,", "static, moving and late-appearing obstacles,",
         "wind as a force, with gusts,", "range-limited, delayed and noisy sensing,",
         "RRT/RRT* navigation (PythonRobotics) with replanning,", "structured numerical scenario generation,",
         "lightweight conditional diffusion,", "grounded LLM-based analysis."])
P("The generative model operates on scenario parameters rather than images or video, so it trains in about a minute "
  "on a laptop GPU (RTX 4050, 6 GB). To fit the time, the budget is 2,000 runs per method, the seed set has 10,000 "
  "scenarios, scenarios have at most 5 obstacles, and the LLM ablation uses 50 crashes.")
P("Because the simulator is 2D, three measures check that discovered failures are real planner weaknesses rather "
  "than simulator artefacts: physically limited dynamics, the time-step and second-planner checks of section 10, and "
  "a simulator-independent interface. More complex components such as 3D environments (for example "
  "gym-pybullet-drones), realistic aerodynamics, camera-based perception, and real UAV deployment are future "
  "extensions rather than requirements.")
rule()

H2("12. Software and Tools")
P("The project is implemented entirely using free/open-source tools:")
bullets(["Python", "PyTorch", "NumPy", "SciPy", "Pandas", "scikit-learn", "cma (CMA-ES)",
         "PythonRobotics (RRT/RRT*)", "Matplotlib, with imageio-ffmpeg for flight videos", "Git / GitHub",
         "Gemini API free tier (google-genai) for the explanation component",
         "Ollama with Qwen 2.5 7B as a local fallback"])
P("No paid cloud infrastructure is required; everything runs on one laptop.")
rule()

H2("13. Expected Outcome")
P("The final system provides an automated testing pipeline capable of:")
bullets(["Generating novel UAV scenarios using a conditional diffusion model.",
         "Testing an autonomous navigation algorithm (RRT/RRT*) against those scenarios.",
         "Identifying collisions, near-failures, and difficult navigation situations.",
         "Using failure information to guide further scenario generation.",
         "Measuring the causes of each failure with counterfactual reruns.",
         "Producing grounded natural-language explanations of why each test case succeeded or failed.",
         "Building a library of automatically discovered UAV edge cases, grouped by failure type.",
         "Checking that the failures are real (finer time step, second planner).",
         "Using the measured causes to harden the system, then testing the fixes and attacking again.",
         "Reporting every stage with flight videos, charts and measured crash causes."], numbered=True)
P("The final demonstration shows a sequence such as:")
box("""
Generate Scenario
        ↓
Run RRT
        ↓
Collision Detected
        ↓
Counterfactual Reruns
        ↓
LLM Explanation
        ↓
Generate Related Edge Cases
        ↓
Test Again
        ↓
Fix the System → Replay Past Failures → Attack Again
""")
rule()

H2("14. Final Objective")
story.append(Paragraph(
    "<b>To develop a Generative AI-based red-teaming framework for autonomous UAV navigation that learns to generate "
    "novel and challenging flight scenarios, automatically evaluates an existing navigation system such as RRT/RRT*, "
    "discovers failure-inducing edge cases through feedback, uses a grounded LLM to provide interpretable "
    "explanations of measured failure causes, and uses those causes to guide and verify fixes.</b>",
    ParagraphStyle("objective", parent=S["body"], leftIndent=28, rightIndent=20)))
P("The central philosophy of the project is:")
formula("<b>Do not use GenAI to replace the planner.</b>", boxed=True)
P("Instead:")
formula("<b>Use GenAI to test, challenge, and understand the planner.</b>", boxed=True)
P("This allows the project to exploit GenAI for two tasks where generative models are particularly useful:")
formula("Generating new possibilities + Reasoning over complex test outcomes", boxed=True)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8.5)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=22 * mm,
                        bottomMargin=22 * mm, title="Generative AI-Based Red-Teaming for Autonomous UAV Navigation",
                        author="Snehil Chatterjee")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
