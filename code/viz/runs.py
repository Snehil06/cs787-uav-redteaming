"""Static picture of a single simulated flight (the video lives in viz/video.py)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from loop.scoring import NEAR_MISS
from scenarios.spec import MAP_SIZE, UAV_RADIUS, Scenario
from viz import style as st

STATUS_TEXT = {"goal": "reached goal", "collision": "COLLISION", "out_of_bounds": "left the geofence",
               "timeout": "timed out"}


def clearance_series(s: Scenario, ser) -> np.ndarray:
    """Distance from the UAV's edge to the nearest obstacle's edge at every step."""
    gaps = np.hypot(*(ser["obs"] - ser["pos"][:, None, :]).transpose(2, 0, 1)) - s.obstacles[:, 4] - UAV_RADIUS
    gaps = np.where(np.isnan(gaps), np.inf, gaps)
    return gaps.min(axis=1)


def _draw_map(ax, s: Scenario, tr, i_focus: int, show_plans=True):
    ser, m = tr.series, tr.metrics
    ax.set_xlim(-6, MAP_SIZE + 6)
    ax.set_ylim(-6, MAP_SIZE + 6)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.add_patch(Rectangle((0, 0), MAP_SIZE, MAP_SIZE, fill=False, lw=0.8, ec=st.AXIS))
    if show_plans:
        for _, path in ser["plans"]:
            ax.plot(path[:, 0], path[:, 1], color=st.BLUE, alpha=0.12, lw=1)
    for k in np.flatnonzero(s.active):
        track = ser["obs"][:, k]
        ok = ~np.isnan(track[:, 0])
        if not ok.any():
            continue
        if np.hypot(*s.obstacles[k, 2:4]) > 0.1:
            ax.plot(track[ok, 0], track[ok, 1], color=st.ORANGE, lw=1, alpha=0.6)
        p = track[i_focus] if ok[i_focus] else track[ok][-1 if i_focus > np.flatnonzero(ok)[-1] else 0]
        key = k == m["closest_obstacle"]
        ax.add_patch(Circle(p, s.obstacles[k, 4], color=st.CRITICAL if key else st.ORANGE,
                            alpha=0.55 if key else 0.35, lw=0))
        tag = f"#{k}" + (" pop-up" if s.obstacles[k, 5] > 0 else "")
        ax.text(p[0], p[1] + s.obstacles[k, 4] + 1.0, tag, ha="center", fontsize=7.5, color=st.INK_2)
    ax.plot(ser["pos"][: i_focus + 1, 0], ser["pos"][: i_focus + 1, 1], color=st.BLUE, lw=2)
    ax.plot(ser["pos"][i_focus:, 0], ser["pos"][i_focus:, 1], color=st.BLUE, lw=1, alpha=0.35)
    ax.add_patch(Circle(ser["pos"][i_focus], s.sense_range, fill=False, lw=0.8, ec=st.MUTED))
    ax.plot(*ser["pos"][i_focus], "o", color=st.BLUE, ms=6, mec=st.SURFACE, mew=1.5)
    ax.plot(*s.start, "o", color=st.GOOD, ms=8, mec=st.SURFACE, mew=1.5)
    ax.text(s.start[0], s.start[1] - 3.2, "start", ha="center", fontsize=8, color=st.INK_2)
    ax.plot(*s.goal, "*", color=st.INK, ms=13, mec=st.SURFACE)
    ax.text(s.goal[0], s.goal[1] - 3.4, "goal", ha="center", fontsize=8, color=st.INK_2)
    w = s.wind
    wm = np.hypot(*w)
    if wm > 0.2:
        d = w / wm * 6
        ax.annotate("", xy=(4 + d[0], MAP_SIZE + 2 + d[1] * 0.3), xytext=(4, MAP_SIZE + 2),
                    arrowprops=dict(arrowstyle="-|>", color=st.INK_2, lw=1.2))
        ax.text(12, MAP_SIZE + 2, f"wind {wm:.1f} m/s", va="center", fontsize=8, color=st.INK_2)


def thumbnail(ax, s: Scenario, label: str = "", color: str = st.INK_2):
    """Small static picture of a scenario at t = 0: route, obstacles (velocity arrows), wind."""
    ax.set_xlim(-4, MAP_SIZE + 4)
    ax.set_ylim(-4, MAP_SIZE + 4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(Rectangle((0, 0), MAP_SIZE, MAP_SIZE, fill=False, lw=0.6, ec=st.AXIS))
    ax.plot([s.start[0], s.goal[0]], [s.start[1], s.goal[1]], color=st.BLUE, lw=0.8, alpha=0.5)
    ax.plot(*s.start, "o", color=st.GOOD, ms=4)
    ax.plot(*s.goal, "*", color=st.INK, ms=7)
    for k in np.flatnonzero(s.active):
        x, y, vx, vy, r, ta, _ = s.obstacles[k]
        ax.add_patch(Circle((x, y), r, fc=st.ORANGE, alpha=0.6 if ta == 0 else 0.25, ec="none"))
        if np.hypot(vx, vy) > 0.1:
            ax.annotate("", xy=(x + 2 * vx, y + 2 * vy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=st.INK_2, lw=0.7))
    w = s.wind
    if np.hypot(*w) > 0.3:
        ax.annotate("", xy=(MAP_SIZE - 6 + w[0], MAP_SIZE - 4 + w[1]), xytext=(MAP_SIZE - 6, MAP_SIZE - 4),
                    arrowprops=dict(arrowstyle="-|>", color=st.MUTED, lw=0.9))
    if label:
        ax.set_title(label, fontsize=8, color=color, loc="center")


def run_figure(s: Scenario, tr, title: str, note: str = ""):
    """Map at the moment of closest approach + clearance and speed over time."""
    ser, m = tr.series, tr.metrics
    t = ser["t"]
    clear = clearance_series(s, ser)
    i_focus = int(np.argmin(clear)) if np.isfinite(clear).any() else len(t) - 1

    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.45, wspace=0.18)
    ax_map = fig.add_subplot(gs[:, 0])
    _draw_map(ax_map, s, tr, i_focus)
    ax_map.set_title(f"Flight path (snapshot at closest approach, t = {t[i_focus]:.1f} s)", fontsize=9.5)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")

    ax_c = fig.add_subplot(gs[0, 1])
    shown = np.where(np.isfinite(clear), clear, np.nan)
    ax_c.axhspan(min(-0.5, np.nanmin(shown) if np.isfinite(clear).any() else 0), 0, color=st.CRITICAL, alpha=0.08, lw=0)
    ax_c.axhspan(0, NEAR_MISS, color=st.WARNING, alpha=0.12, lw=0)
    ax_c.plot(t, shown, color=st.BLUE)
    ax_c.axhline(0, color=st.CRITICAL, lw=0.8)
    top = np.nanmax(shown) if np.isfinite(clear).any() else 10
    ax_c.set_ylim(-1, min(15, max(3, top * 1.05)))
    ax_c.text(t[0], NEAR_MISS + 0.15, " near-miss zone", va="bottom", fontsize=7.5, color=st.INK_2)
    xt = ax_c.get_xaxis_transform()
    for tp, _ in ser["plans"]:
        ax_c.plot([tp, tp], [0.94, 1.0], color=st.MUTED, lw=0.9, transform=xt)
    if np.isfinite(m["closest_detect_time"]):
        td = m["closest_detect_time"]
        ax_c.axvline(td, color=st.INK_2, lw=0.8)
        ax_c.text(td, 0.84, f" obstacle #{m['closest_obstacle']} detected", fontsize=7.5, color=st.INK_2,
                  transform=xt)
    ax_c.set_title("Clearance to nearest obstacle (ticks on top = replans)", fontsize=9.5)
    ax_c.set_ylabel("clearance (m)")

    ax_v = fig.add_subplot(gs[1, 1], sharex=ax_c)
    speed = np.hypot(*ser["vel"].T)
    wind = np.hypot(*ser["wind"].T)
    ax_v.plot(t, speed, color=st.BLUE)
    ax_v.plot(t, wind, color=st.ORANGE, lw=1.5)
    ax_v.axhline(s.speed, color=st.MUTED, lw=0.8)
    ax_v.text(t[-1], speed[-1], "  UAV speed", va="center", fontsize=8, color=st.BLUE)
    ax_v.text(t[-1], wind[-1], "  wind", va="center", fontsize=8, color=st.ORANGE)
    ax_v.text(t[0], s.speed, "cruise speed", va="bottom", fontsize=7.5, color=st.MUTED)
    ax_v.set_title("UAV speed and wind speed", fontsize=9.5)
    ax_v.set_ylabel("m/s")
    ax_v.set_xlabel("time (s)")

    st.finding(fig, title, note or (
        f"Outcome: {STATUS_TEXT[m['status']]} · min clearance {m['min_clearance']:.2f} m · "
        f"{m['n_replans']} replans · failure score F = {tr.metrics.get('F', float('nan')):.2f}"))
    return fig
