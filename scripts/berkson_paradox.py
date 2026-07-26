"""Berkson's paradox animation, in Python.

Regenerates the animated GIF embedded in
content/posts/2021-02-14-berkson-s-paradox.md. The post's R/ggplot2 code
block is a *reference*, not something this script ports line-for-line: the
statistical setup (independent Niceness/Attractiveness, mu=50, var=200; two
parallel cutoff lines at x+y=85 and x+y=115) is the same, but here it's a
single continuous animation instead of three PNGs stitched together with an
external GIF tool.

Styled with the site's Flexoki palette (styles/main.css) so it fits the
blog's look.

Usage:
    python3 scripts/berkson_paradox.py

Requires: numpy, matplotlib, pillow
Writes:   content/static/img/berksonParadox.gif
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ── Flexoki palette (light theme, styles/main.css) ──────────────────────────
BG = "#FFFCF0"       # --bg
UI = "#E6E4D9"       # --ui (borders / grid)
TX = "#100F0F"       # --tx (primary text)
TX_2 = "#6F6E69"      # --tx-2 (secondary text / axis labels)
TX_3 = "#B7B5AC"      # --tx-3 (excluded points)
ACCENT = "#205EA6"    # --syn-blue / --accent (kept points)
RED = "#AF3029"       # --syn-red (cutoff lines)

OUT_PATH = Path(__file__).resolve().parent.parent / "content/static/img/berksonParadox.gif"

N = 5000
MU = 50.0
SD = np.sqrt(200.0)
LOWER_B, UPPER_B = 85.0, 115.0  # x + y thresholds; same band as the R code


def lower_line(x, b):
    return -x + b


def upper_line(x, b):
    return -x + b


def corr(mask, x, y):
    if mask.sum() < 2:
        return float("nan")
    return np.corrcoef(x[mask], y[mask])[0, 1]


def _ramp(local_i, sweep_len, fade_frames=10):
    """0 -> 1 over the last `fade_frames` of a `sweep_len`-frame sweep."""
    start = sweep_len - fade_frames
    return max(0.0, min(1.0, (local_i - start) / fade_frames))


def build_schedule():
    """Frame-by-frame (b_lower, b_upper, phase, label1, label2) schedule.

    Phases: 0 = full sample, 1 = lower cutoff sweeping in, 2 = lower cutoff
    settled, 3 = upper cutoff sweeping in, 4 = both cutoffs settled (hold).
    label1/label2 are the alpha of the "YOU WOULD NOT DATE" / "WOULD NOT
    DATE YOU" annotations, computed directly per-frame so they can't drift
    out of sync with the sweep that triggers them.
    """
    frames = []
    frames += [(None, None, 0, 0.0, 0.0)] * 18              # hold: full sample

    sweep1 = np.linspace(190, LOWER_B, 28)
    for i, b in enumerate(sweep1):                          # sweep lower line in
        frames.append((b, None, 1, _ramp(i, len(sweep1)), 0.0))

    frames += [(LOWER_B, None, 2, 1.0, 0.0)] * 18            # hold: one cutoff

    sweep2 = np.linspace(230, UPPER_B, 28)
    for i, b in enumerate(sweep2):                           # sweep upper line in
        frames.append((LOWER_B, b, 3, 1.0, _ramp(i, len(sweep2))))

    frames += [(LOWER_B, UPPER_B, 4, 1.0, 1.0)] * 34         # hold: both cutoffs
    return frames


def main():
    rng = np.random.default_rng(323)
    x = rng.normal(MU, SD, N)
    y = rng.normal(MU, SD, N)

    cor_full = corr(np.ones(N, dtype=bool), x, y)

    plt.rcParams.update(
        {
            "font.size": 15,
            "text.color": TX,
            "axes.edgecolor": UI,
            "axes.labelcolor": TX_2,
            "xtick.color": TX_2,
            "ytick.color": TX_2,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    scatter = ax.scatter(x, y, s=2, alpha=0.35, linewidths=0, color=ACCENT)
    (line_lo,) = ax.plot([], [], color=RED, linewidth=1.6, zorder=3)
    (line_hi,) = ax.plot([], [], color=RED, linewidth=1.6, zorder=3)
    title = ax.set_title("", fontsize=17, pad=14)

    label_not_date = ax.text(
        20, 20, "YOU WOULD\nNOT DATE", ha="center", va="center",
        fontsize=11, color=TX, alpha=0, zorder=4,
    )
    label_not_date_you = ax.text(
        80, 80, "WOULD NOT\nDATE YOU", ha="center", va="center",
        fontsize=11, color=TX, alpha=0, zorder=4,
    )

    ax.set_xlim(-25, 125)
    ax.set_ylim(-25, 125)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    for spine in ax.spines.values():
        spine.set_color(UI)
    ax.tick_params(length=0)
    ax.set_xlabel("Niceness")
    ax.set_ylabel("Attractiveness")

    corner_kwargs = dict(fontsize=12, color=TX_2, ha="center", va="center")
    ax.text(10, -15, "JERK", **corner_kwargs)
    ax.text(90, -15, "NICE", **corner_kwargs)
    ax.text(-14, 10, "NOT", **corner_kwargs)
    ax.text(-14, 90, "HOT", **corner_kwargs)

    fig.subplots_adjust(left=0.16, right=0.95, top=0.9, bottom=0.14)

    xs = np.linspace(-25, 125, 2)
    schedule = build_schedule()

    def frame(i):
        b_lo, b_hi, phase, alpha1, alpha2 = schedule[i]
        colors = np.full(N, ACCENT, dtype=object)

        if phase == 0:
            line_lo.set_data([], [])
            line_hi.set_data([], [])
            rho = cor_full
        elif phase in (1, 2):
            line_lo.set_data(xs, lower_line(xs, b_lo))
            line_hi.set_data([], [])
            excluded = (x + y) <= b_lo
            colors[excluded] = TX_3
            rho = corr(excluded, x, y)
        else:
            line_lo.set_data(xs, lower_line(xs, LOWER_B))
            line_hi.set_data(xs, upper_line(xs, b_hi))
            band = ((x + y) > LOWER_B) & ((x + y) <= b_hi)
            colors[~band] = TX_3
            rho = corr(band, x, y)

        label_not_date.set_alpha(alpha1)
        label_not_date_you.set_alpha(alpha2)
        scatter.set_color(colors)
        title.set_text(rf"$\rho$ = {rho:.2f}")
        return scatter, line_lo, line_hi, title, label_not_date, label_not_date_you

    anim = FuncAnimation(fig, frame, frames=len(schedule), blit=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_PATH, writer=PillowWriter(fps=22))
    plt.close(fig)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
