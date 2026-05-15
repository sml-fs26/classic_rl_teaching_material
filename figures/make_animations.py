"""Animated GIFs for the SARSA lecture deck.

Outputs land beside the static PNGs in ``figures/``. The animation pipeline
deliberately uses Pillow (no ffmpeg) so the deck builds on any machine with
matplotlib + PIL.

Run:
    .venv/bin/python figures/make_animations.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    COL_ACTION,
    COL_MUTED,
    COL_REWARD,
    COL_STATE,
    COL_TEXT,
    draw_warehouse,
    setup_style,
)
from make_figures import (  # noqa: E402
    _make_q_overlay_episode_0,
    _make_q_overlay_episode_50,
    _make_q_overlay_episode_500,
)

setup_style()

OUT_DIR = HERE
SLIP_CELLS = [(1, 1), (1, 2), (2, 1), (2, 2)]


def _interp(d0, d1, t):
    """Linearly interpolate two q-overlay dicts at t in [0, 1]."""
    out = {}
    keys = set(d0) | set(d1)
    for cell in keys:
        if cell not in d0:
            out[cell] = d1[cell]
            continue
        if cell not in d1:
            out[cell] = d0[cell]
            continue
        a = d0[cell]
        b = d1[cell]
        out[cell] = {k: round((1 - t) * a[k] + t * b[k], 1) for k in a}
    return out


def _ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)


def animate_q_evolution(out_path: Path):
    snap0 = _make_q_overlay_episode_0()
    snap50 = _make_q_overlay_episode_50()
    snap500 = _make_q_overlay_episode_500()

    n_a = 4
    n_b = 5
    hold = 2
    frames = []
    for i in range(n_a):
        t = _ease(i / (n_a - 1))
        frames.append((int(round(t * 50)), _interp(snap0, snap50, t)))
    for _ in range(hold):
        frames.append((50, snap50))
    for i in range(n_b):
        t = _ease(i / (n_b - 1))
        ep = int(round(50 + t * 450))
        frames.append((ep, _interp(snap50, snap500, t)))
    for _ in range(hold + 1):
        frames.append((500, snap500))

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=85)

    def render(frame_idx):
        ax.clear()
        ep, overlay = frames[frame_idx]
        draw_warehouse(
            ax,
            robot=None,
            package=None,
            depot=(3, 0),
            slip_cells=SLIP_CELLS,
            q_overlay=overlay,
            title=f"Episode {ep}",
        )

    a = anim.FuncAnimation(fig, render, frames=len(frames), interval=120)
    a.save(out_path, writer=anim.PillowWriter(fps=8))
    plt.close(fig)
    print(f"saved {out_path.name} ({out_path.stat().st_size // 1024} KB)")


def animate_cliff_walking(out_path: Path):
    W, H = 12, 4
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-0.05, W + 0.05)
    ax.set_ylim(-0.4, H + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    sarsa_path = [(0, 0), (0, 1), (0, 2)] + [(x, 2) for x in range(1, 12)] + [(11, 1), (11, 0)]
    q_path = [(0, 0), (0, 1)] + [(x, 1) for x in range(1, 12)] + [(11, 0)]
    n_steps = max(len(sarsa_path), len(q_path))

    def draw_grid():
        for x in range(W):
            for y in range(H):
                face = "white"
                if y == 0 and 0 < x < W - 1:
                    face = COL_TEXT
                if (x, y) == (0, 0):
                    face = "#dbeafe"
                if (x, y) == (W - 1, 0):
                    face = "#fef3c7"
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face,
                                        edgecolor=COL_MUTED, lw=0.4))
        ax.text(W / 2, 0.5, "T H E   C L I F F", ha="center", va="center",
                color="white", fontweight="bold", fontsize=12, zorder=8)
        ax.text(0.5, 0.5, "S", ha="center", va="center",
                color=COL_TEXT, fontweight="bold", fontsize=12, zorder=10)
        ax.text(W - 0.5, 0.5, "G", ha="center", va="center",
                color=COL_TEXT, fontweight="bold", fontsize=12, zorder=10)

    pad_after = 6

    def render(t):
        ax.clear()
        ax.set_xlim(-0.05, W + 0.05)
        ax.set_ylim(-0.4, H + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        draw_grid()
        for path, color in [(sarsa_path, COL_STATE), (q_path, COL_REWARD)]:
            i = min(t + 1, len(path))
            xs = [p[0] + 0.5 for p in path[:i]]
            ys = [p[1] + 0.5 for p in path[:i]]
            ax.plot(xs, ys, "-", color=color, lw=2.5, alpha=0.85, zorder=8)
            x, y = path[min(t, len(path) - 1)]
            ax.add_patch(plt.Circle((x + 0.5, y + 0.5), 0.22,
                                     facecolor=color, edgecolor="white",
                                     lw=1.5, zorder=12))
        ax.plot([], [], "-", color=COL_STATE, lw=2.5, label="SARSA — safe")
        ax.plot([], [], "-", color=COL_REWARD, lw=2.5, label="Q-learning — optimal/risky")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
                  frameon=False, fontsize=10)

    total = n_steps + pad_after
    a = anim.FuncAnimation(fig, render, frames=total, interval=180)
    a.save(out_path, writer=anim.PillowWriter(fps=6))
    plt.close(fig)
    print(f"saved {out_path.name} ({out_path.stat().st_size // 1024} KB)")


def animate_driving_home(out_path: Path):
    """Animate the driving-home target estimates updating across stops."""
    stops = ["Leave\noffice", "Reach\ncar", "Exit\nhighway", "Behind\ntruck",
             "Home\nstreet", "Arrive\nhome"]
    initial = np.array([30, 35, 15, 10, 3, 0])
    actual_remaining = np.array([43, 38, 23, 13, 3, 0])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(stops))

    def render(t):
        ax.clear()
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(xs)
        ax.set_xticklabels(stops, fontsize=10)
        ax.set_ylabel("predicted minutes to home", color=COL_TEXT)
        ax.set_ylim(-3, 50)
        ax.grid(axis="y", color="#e5e7eb", lw=0.5)
        # Initial predictions: faded line
        ax.plot(xs, initial, "o-", color=COL_MUTED, lw=1.5, markersize=6,
                alpha=0.6, label="initial guess")
        # Updated predictions: gradually replace from left to right
        n_revealed = min(t + 1, len(stops))
        revealed = initial.copy()
        revealed[:n_revealed] = actual_remaining[:n_revealed]
        ax.plot(xs[:n_revealed], revealed[:n_revealed], "o-",
                color=COL_STATE, lw=2.5, markersize=8,
                label="after observing stop")
        # Highlight current update
        if 0 < t < len(stops):
            ax.annotate("", xy=(t, actual_remaining[t]),
                        xytext=(t, initial[t]),
                        arrowprops=dict(arrowstyle="->", color=COL_REWARD,
                                        lw=2.0))
        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Driving-home: targets land one step at a time",
                     color=COL_TEXT, fontsize=12, pad=10)
        ax.tick_params(colors=COL_TEXT)

    pad_after = 5
    a = anim.FuncAnimation(fig, render,
                           frames=len(stops) + pad_after, interval=600)
    a.save(out_path, writer=anim.PillowWriter(fps=2))
    plt.close(fig)
    print(f"saved {out_path.name} ({out_path.stat().st_size // 1024} KB)")


def animate_bellman_value_flow(out_path: Path):
    """Animate value backups propagating along a 5-state chain.

    States S1..S5 with reward +10 only when entering S5 from S4.
    γ = 0.9. Values converge: V[i] = 10 * 0.9**(4-i).
    """
    n_states = 5
    gamma = 0.9
    V_target = np.array([10 * gamma ** (n_states - 1 - i) for i in range(n_states)])
    V_target[-1] = 0.0  # terminal
    sweeps = 6
    V = np.zeros(n_states)
    history = [V.copy()]
    for _ in range(sweeps):
        for i in range(n_states - 1):
            r = 10.0 if i == n_states - 2 else 0.0
            V[i] = r + gamma * V[i + 1]
            history.append(V.copy())

    fig, ax = plt.subplots(figsize=(9, 3.5))

    def render(t):
        ax.clear()
        ax.set_xlim(-0.5, n_states - 0.5)
        ax.set_ylim(-3.5, 12)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        Vt = history[min(t, len(history) - 1)]
        # Cells
        for i in range(n_states):
            color = "#fef3c7" if i == n_states - 1 else "white"
            ax.add_patch(Rectangle((i - 0.4, -0.5), 0.8, 1,
                                    facecolor=color, edgecolor=COL_MUTED, lw=1.2))
            label = f"$S_{{{i+1}}}$" if i < n_states - 1 else "T"
            ax.text(i, 0, label, ha="center", va="center",
                    color=COL_TEXT, fontsize=14, fontweight="bold")
            ax.text(i, -1.5, f"V = {Vt[i]:.2f}", ha="center", va="top",
                    color=COL_STATE, fontsize=12, fontweight="bold")
        # Arrows
        for i in range(n_states - 1):
            ax.annotate("", xy=(i + 0.55, 0.6), xytext=(i + 0.45, 0.6),
                        arrowprops=dict(arrowstyle="->", color=COL_ACTION,
                                        lw=1.5))
            r = 10.0 if i == n_states - 2 else 0.0
            ax.text(i + 0.5, 1.1, f"r={int(r)}", ha="center", va="bottom",
                    color=COL_REWARD, fontsize=10)
        # Active update indicator
        if t > 0 and t <= sweeps * (n_states - 1):
            sweep_pos = (t - 1) % (n_states - 1)
            ax.add_patch(Rectangle((sweep_pos - 0.45, -0.55), 0.9, 1.1,
                                    facecolor="none", edgecolor=COL_REWARD,
                                    lw=3, zorder=15))
        sweep = (t - 1) // (n_states - 1) + 1 if t > 0 else 0
        ax.set_title(f"Bellman backup — sweep {min(sweep, sweeps)} / {sweeps}",
                     color=COL_TEXT, fontsize=13, pad=10)

    a = anim.FuncAnimation(fig, render,
                           frames=len(history) + 4, interval=450)
    a.save(out_path, writer=anim.PillowWriter(fps=3))
    plt.close(fig)
    print(f"saved {out_path.name} ({out_path.stat().st_size // 1024} KB)")


def main():
    print("Generating animations…")
    animate_q_evolution(OUT_DIR / "q_evolution.gif")
    animate_cliff_walking(OUT_DIR / "cliff_walking.gif")
    animate_driving_home(OUT_DIR / "driving_home.gif")
    animate_bellman_value_flow(OUT_DIR / "bellman_flow.gif")
    print("Done.")


if __name__ == "__main__":
    main()
