"""Assemble reports/demo.mp4: title cards + the key flights, in the order of the story.

    python -m experiments.make_demo
"""
import os
import subprocess
import tempfile

import imageio_ffmpeg
import matplotlib.pyplot as plt

from viz import style as st

REPORTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
FF = imageio_ffmpeg.get_ffmpeg_exe()

STORY = [  # (card title, card text, video under reports/ or None)
    ("Red-teaming a UAV path planner with generative AI",
     "RRT / RRT* (open-source PythonRobotics) flies a 2D simulator with wind, moving and pop-up obstacles,\n"
     "and limited, noisy, delayed sensing. Generative models search for the scenarios that break it.", None),
    ("1. What a normal flight looks like", "A safe flight past moving obstacles.", "01_random_baseline/1_runs/01_success.mp4"),
    ("2. A random-scenario crash", "Re-running it without wind prevents the crash, so wind is a measured cause.",
     "01_random_baseline/1_runs/03_crash_no_wind.mp4"),
    ("3. Diffusion, asked for a hard scenario", "The model generates a scenario on request; RRT crashes in it.",
     "02_generators/1_runs/01_diffusion_crash_no_wind.mp4"),
    ("4. After learning from its own finds", "A crash the red team found in its final round of fine-tuning.",
     "03_red_team_loop/1_runs/01_last_round_crash.mp4"),
    ("5. A compound failure", "Found by diffusion + feedback; no single change prevents it.",
     "05_comparison/1_runs/01_ours_compound.mp4"),
    ("6. Same scenario, RRT*", "RRT* fails for the same reasons: here both planners crash.",
     "06_fidelity_checks/1_runs/02_both_crash_rrt_star.mp4"),
    ("7. Closing the loop: before hardening", "A past failure on the original system...",
     "07_harden_and_reattack/1_runs/01_fixed_before.mp4"),
    ("   ...and after hardening", "...survived by the hardened system (same scenario, same planner seed).",
     "07_harden_and_reattack/1_runs/01_fixed_after.mp4"),
    ("8. The red team attacks again", "A new weakness found against the hardened system.",
     "07_harden_and_reattack/1_runs/02_new_failure_hardened.mp4"),
    ("Results",
     "Diffusion + feedback: 552 distinct failures per 2,000 flights (CMA-ES 510, random 214),\n"
     "spread over 109 regions of scenario space (CMA-ES 31); 88% of CMA-ES failures are one type.\n"
     "Grounded LLM explanations name 96% of measured causes (34% from the trace alone).\n"
     "Hardening fixes 79% of past failures; the red team still finds 2x more than random afterwards.", None),
]


def card(path, title, text):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.text(0.07, 0.58, title, fontsize=30, weight="semibold", color=st.INK, va="bottom")
    fig.text(0.07, 0.52, text, fontsize=15, color=st.INK_2, va="top", linespacing=1.6)
    with plt.rc_context({"savefig.bbox": "standard"}):       # exact 1280x720, no tight cropping
        fig.savefig(path, dpi=100, facecolor=st.SURFACE)
    plt.close(fig)


def main():
    tmp = tempfile.mkdtemp(prefix="demo_")
    inputs, filters = [], []
    for i, (title, text, video) in enumerate(STORY):
        png = os.path.join(tmp, f"card{i}.png")
        card(png, title, text)
        inputs += ["-loop", "1", "-t", "4" if video else "6", "-i", png]
        filters.append(f"[{len(filters)}:v]scale=1280:720,setsar=1,fps=30,format=yuv420p[v{len(filters)}]")
        if video:
            inputs += ["-i", os.path.join(REPORTS, video)]
            filters.append(f"[{len(filters)}:v]scale=1280:720,setsar=1,fps=30,format=yuv420p[v{len(filters)}]")
    n = len(filters)
    graph = ";".join(filters) + ";" + "".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    out = os.path.join(REPORTS, "demo.mp4")
    subprocess.run([FF, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
