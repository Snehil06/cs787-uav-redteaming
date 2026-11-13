"""Render a simulated flight as an MP4 'mission view' video.

Left: top-down map with a quadrotor icon, wind streaks, obstacles (faint until the UAV
has sensed them), the sensing range, the current plan and a fading trail.
Right: live dashboard and a clearance-over-time strip. A crash gets an impact burst,
a COLLISION banner and a slow-motion replay of its last moments.
"""
import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Polygon, Rectangle

from loop.scoring import NEAR_MISS
from scenarios.spec import GUST_DURATION, MAP_SIZE, Scenario
from viz import style as st
from viz.runs import clearance_series

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

BADGE = {"collision": ("COLLISION", st.CRITICAL), "goal": ("GOAL REACHED", st.GOOD),
         "out_of_bounds": ("LEFT GEOFENCE", st.CRITICAL), "timeout": ("TIMED OUT", st.SERIOUS)}


def _drone(ax, pos, heading, spin, scale=1.0, color=st.INK):
    arm = 1.3 * scale
    for k, a in enumerate(heading + np.pi / 4 + np.arange(4) * np.pi / 2):
        end = pos + arm * np.array([np.cos(a), np.sin(a)])
        ax.plot([pos[0], end[0]], [pos[1], end[1]], color=color, lw=2.2, solid_capstyle="round", zorder=6)
        ax.add_patch(Circle(end, 0.6 * scale, fc=st.BLUE, ec=color, lw=1,
                            alpha=0.45 + 0.35 * ((k + spin) % 2), zorder=7))
    nose = pos + 0.9 * scale * np.array([np.cos(heading), np.sin(heading)])
    ax.add_patch(Circle(pos, 0.55 * scale, fc=color, ec=st.SURFACE, lw=1, zorder=8))
    ax.plot([pos[0], nose[0]], [pos[1], nose[1]], color=st.WARNING, lw=2, zorder=9)


def _burst(ax, pos, size, rng):
    for r, col, alpha in ((size, st.WARNING, 0.9), (size * 0.6, st.CRITICAL, 0.95)):
        ang = np.linspace(0, 2 * np.pi, 19)[:-1]
        rad = np.where(np.arange(18) % 2 == 0, r, r * 0.45) * rng.uniform(0.85, 1.15, 18)
        pts = pos + np.column_stack([np.cos(ang) * rad, np.sin(ang) * rad])
        ax.add_patch(Polygon(pts, closed=True, fc=col, ec="none", alpha=alpha, zorder=10))


def render_video(s: Scenario, tr, path: str, title: str, subtitle: str = "", speedup: float = 1.5, fps: int = 30):
    """speedup = simulated seconds shown per second of video during live flight."""
    ser, m = tr.series, tr.metrics
    t, pos, vel, wind = ser["t"], ser["pos"], ser["vel"], ser["wind"]
    n, dt = len(t), (t[1] - t[0]) if len(t) > 1 else 0.05
    clear = clearance_series(s, ser)
    clear_plot = np.where(np.isfinite(clear), clear, np.nan)
    plan_t = np.array([p[0] for p in ser["plans"]]) if ser["plans"] else np.array([])
    act = np.flatnonzero(s.active)

    # which obstacles the UAV has sensed by each step (cumulative)
    gaps = np.hypot(*(ser["obs"] - pos[:, None, :]).transpose(2, 0, 1)) - s.obstacles[:, 4]
    seen = np.maximum.accumulate(np.nan_to_num(gaps, nan=np.inf) <= s.sense_range, axis=0)

    # heading follows velocity, held when nearly stopped
    heading = np.zeros(n)
    h = np.arctan2(*(s.goal - s.start)[::-1])
    for i in range(n):
        if np.hypot(*vel[i]) > 0.3:
            h = np.arctan2(vel[i, 1], vel[i, 0])
        heading[i] = h

    # wind streaks: particles advected by the wind field
    rng = np.random.default_rng(0)
    p0 = rng.uniform(-6, MAP_SIZE + 6, (70, 2))
    drift = np.cumsum(wind, axis=0) * dt * 1.5

    # frame plan: 1 s on the start, live flight paced to 4-14 s of video, then
    # (crash) impact + half-speed replay of the last 1.5 s, or a held final frame
    live_secs = float(np.clip(n * dt / speedup, 4.0, 14.0))
    live = np.linspace(0, n - 1, int(live_secs * fps)).round().astype(int)
    frames = [("hold", 0)] * fps + [("live", int(i)) for i in live]
    crashed = m["status"] == "collision"
    if crashed:
        frames += [("impact", k) for k in range(18)]
        start = max(0, n - int(1.5 / dt))
        frames += [("replay", int(i)) for i in np.linspace(start, n - 1, int(3.0 * fps)).round()]
        frames += [("impact", 17)] * int(1.5 * fps)
    else:
        frames += [("hold", n - 1)] * int(1.5 * fps)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.01, 0.02, 0.53, 0.96])
    ax_c = fig.add_axes([0.6, 0.09, 0.37, 0.27])
    hud = fig.add_axes([0.58, 0.42, 0.4, 0.55])
    hud.axis("off")
    burst_rng = np.random.default_rng(1)

    def draw(frame):
        mode, i = frame
        if mode == "impact":
            i = n - 1
        for a in (ax, ax_c, hud):
            a.clear()
        hud.axis("off")

        # ---- map
        ax.set_xlim(-6, MAP_SIZE + 6)
        ax.set_ylim(-6, MAP_SIZE + 6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_patch(Rectangle((-6, -6), MAP_SIZE + 12, MAP_SIZE + 12, fc="#f3f2ee", ec="none", zorder=0))
        ax.add_patch(Rectangle((0, 0), MAP_SIZE, MAP_SIZE, fc=st.SURFACE, ec=st.AXIS, lw=1, zorder=0))
        wv = wind[i]
        wm = np.hypot(*wv)
        if wm > 0.3:
            pp = (p0 + drift[i] + 6) % (MAP_SIZE + 12) - 6
            seg = wv / wm * min(3.5, 0.6 * wm)
            for q in pp:
                ax.plot([q[0] - seg[0], q[0]], [q[1] - seg[1], q[1]], color=st.MUTED, lw=0.8, alpha=0.5, zorder=1)
        ax.plot(*s.start, "o", color=st.GOOD, ms=9, mec=st.SURFACE, mew=1.5, zorder=3)
        ax.plot(*s.goal, "*", color=st.INK, ms=16, mec=st.SURFACE, zorder=3)
        ax.text(s.goal[0], s.goal[1] - 3.6, "goal", ha="center", fontsize=9, color=st.INK_2)

        for k in act:
            p = ser["obs"][i, k]
            if np.isnan(p[0]):
                continue
            r = s.obstacles[k, 4]
            hit = crashed and k == m["closest_obstacle"]
            if seen[i, k]:
                ax.add_patch(Circle(p, r, fc=st.CRITICAL if hit else st.ORANGE, alpha=0.75, ec="none", zorder=4))
            else:
                ax.add_patch(Circle(p, r, fc="none", ec=st.ORANGE, lw=1.2, ls=(0, (2, 2)), alpha=0.8, zorder=4))
                ax.text(p[0], p[1], "unseen", ha="center", va="center", fontsize=7, color=st.ORANGE, zorder=4)
            v = s.obstacles[k, 2:4]
            if np.hypot(*v) > 0.1:
                ax.annotate("", xy=p + v * 0.8, xytext=p, arrowprops=dict(arrowstyle="-|>", color=st.INK_2, lw=1),
                            zorder=5)

        if len(plan_t):
            j = np.searchsorted(plan_t, t[i], side="right") - 1
            if j >= 0:
                pl = ser["plans"][j][1]
                ax.plot(pl[:, 0], pl[:, 1], color=st.BLUE, lw=1.2, alpha=0.55, zorder=2)
        a = max(0, i - 60)
        ax.plot(pos[: a + 1, 0], pos[: a + 1, 1], color=st.BLUE, lw=1.2, alpha=0.35, zorder=3)
        ax.plot(pos[a: i + 1, 0], pos[a: i + 1, 1], color=st.BLUE, lw=2.5, zorder=3)
        ax.add_patch(Circle(pos[i], s.sense_range, fc=st.BLUE, alpha=0.05, ec=st.BLUE, lw=0.6, zorder=1))
        _drone(ax, pos[i], heading[i], spin=i)

        if mode == "impact":
            _burst(ax, pos[i], 1.5 + 0.35 * frame[1] if frame[1] < 12 else 5.5, burst_rng)
        if mode == "replay":
            ax.text(0.5, 0.97, "SLOW-MOTION REPLAY", transform=ax.transAxes, ha="center", va="top",
                    fontsize=12, weight="bold", color=st.INK_2)
        done = mode == "impact" or (mode in ("hold", "live") and i == n - 1)
        if done:
            label, col = BADGE[m["status"]]
            ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center", fontsize=30,
                    weight="bold", color=st.SURFACE,
                    bbox=dict(boxstyle="round,pad=0.4", fc=col, ec="none", alpha=0.92), zorder=20)

        # ---- dashboard
        hud.text(0, 1.0, title, fontsize=15, weight="semibold", color=st.INK, va="top", transform=hud.transAxes)
        if subtitle:
            hud.text(0, 0.9, subtitle, fontsize=9.5, color=st.INK_2, va="top", transform=hud.transAxes, wrap=True)
        gust = s.gust_amp > 0 and s.gust_t <= t[i] < s.gust_t + GUST_DURATION
        c_now = clear[i] if np.isfinite(clear[i]) else None
        rows = [
            ("time", f"{t[i]:5.1f} s"),
            ("UAV speed", f"{np.hypot(*vel[i]):4.1f} m/s   (cruise {s.speed:.1f})"),
            ("wind", f"{wm:4.1f} m/s" + ("   GUST" if gust else "")),
            ("clearance", "no obstacle present" if c_now is None else f"{c_now:5.2f} m"),
            ("obstacles sensed", f"{int(seen[i, act].sum()) if len(act) else 0} of {len(act)}"),
            ("replans so far", f"{int((plan_t <= t[i]).sum()) - 1 if len(plan_t) else 0}"),
        ]
        for r_, (k_, v_) in enumerate(rows):
            y = 0.72 - r_ * 0.1
            hud.text(0.0, y, k_, fontsize=10.5, color=st.INK_2, transform=hud.transAxes)
            warn = k_ == "clearance" and c_now is not None and c_now < NEAR_MISS
            hud.text(0.42, y, v_, fontsize=10.5, color=st.CRITICAL if warn else st.INK,
                     weight="bold" if warn else "normal", family="monospace", transform=hud.transAxes)

        # ---- clearance strip
        ax_c.axhspan(-1, 0, color=st.CRITICAL, alpha=0.08, lw=0)
        ax_c.axhspan(0, NEAR_MISS, color=st.WARNING, alpha=0.15, lw=0)
        ax_c.axhline(0, color=st.CRITICAL, lw=0.8)
        ax_c.plot(t[: i + 1], clear_plot[: i + 1], color=st.BLUE, lw=2)
        ax_c.set_xlim(0, t[-1])
        top = np.nanmax(clear_plot) if np.isfinite(clear).any() else 10
        ax_c.set_ylim(-1, min(15, max(3, top * 1.05)))
        ax_c.set_title("clearance to nearest obstacle (m) - yellow band = near-miss zone", fontsize=9)
        ax_c.set_xlabel("time (s)")
        return []

    writer = FFMpegWriter(fps=fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, path, dpi=100):
        for f in frames:
            draw(f)
            writer.grab_frame()
    plt.close(fig)
