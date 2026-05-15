"""Generate all PNG figures for the SARSA lecture deck.

Run from repo root:
    .venv/bin/python figures/make_figures.py

All figures land in figures/ as PNG. Slides reference them by relative path.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    COL_ACTION, COL_BG_SLIP, COL_MUTED, COL_POLICY, COL_REWARD, COL_STATE,
    COL_TEXT, COL_VALUE,
    draw_2x2_unified, draw_agent_env_loop,
    draw_backup_q, draw_backup_qlearning, draw_backup_sarsa, draw_backup_v,
    draw_chain, draw_warehouse, save, setup_style,
)

setup_style()


# =============================================================================
# Block 1: warehouse hook
# =============================================================================

def fig_warehouse_hook():
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_warehouse(
        ax,
        robot=(0, 0),
        package=(1, 3),
        depot=(3, 0),
        slip_cells=[(1, 1), (1, 2), (2, 1), (2, 2)],
    )
    return save(fig, "warehouse_hook")


# =============================================================================
# Block 1: agent-environment loop
# =============================================================================

def fig_agent_env_loop():
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_agent_env_loop(ax)
    return save(fig, "agent_env_loop")


# =============================================================================
# Block 2: 3-state Mars-rover-style chain (worked example)
# =============================================================================

def fig_chain_worked():
    fig, ax = plt.subplots(figsize=(8, 3))
    draw_chain(
        ax,
        n_states=3,
        terminal_right=True,
        rewards_right=[-1, +10],
        rewards_left=[-1],
        state_labels=["$s_1$", "$s_2$"],
    )
    return save(fig, "chain_worked")


# =============================================================================
# Block 2: backup diagrams
# =============================================================================

def fig_backup_v():
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_backup_v(ax)
    return save(fig, "backup_v")


def fig_backup_q():
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_backup_q(ax)
    return save(fig, "backup_q")


def fig_backup_sarsa():
    fig, ax = plt.subplots(figsize=(3, 5))
    draw_backup_sarsa(ax)
    return save(fig, "backup_sarsa")


def fig_backup_qlearning():
    fig, ax = plt.subplots(figsize=(3, 5))
    draw_backup_qlearning(ax)
    return save(fig, "backup_qlearning")


# =============================================================================
# Block 2: 2x2 sample x bootstrap unified view
# =============================================================================

def fig_unified_2x2():
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_2x2_unified(ax)
    return save(fig, "unified_2x2")


# =============================================================================
# Block 1: discount factor exponential decay
# =============================================================================

def fig_gamma_decay():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ks = np.arange(0, 21)
    for gamma, color, label in [
        (0.5, "#94a3b8", "$\\gamma = 0.5$"),
        (0.9, COL_REWARD, "$\\gamma = 0.9$"),
        (0.99, COL_STATE, "$\\gamma = 0.99$"),
    ]:
        ax.plot(ks, gamma**ks, "o-", color=color, label=label, lw=2, markersize=5)
    ax.set_xlabel("steps $k$ into the future")
    ax.set_ylabel("weight $\\gamma^k$ on $R_{t+k+1}$")
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    return save(fig, "gamma_decay")


# =============================================================================
# Block 3: ε-greedy split (donut)
# =============================================================================

def fig_epsilon_greedy_split():
    fig, ax = plt.subplots(figsize=(5.5, 5))
    epsilon = 0.1
    sizes = [1 - epsilon, epsilon]
    labels = [
        f"exploit\n$1-\\varepsilon = {1-epsilon:.1f}$",
        f"explore\n$\\varepsilon = {epsilon:.1f}$",
    ]
    colors = [COL_STATE, COL_ACTION]
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    # Centre text
    ax.text(0, 0, "$A_t$", ha="center", va="center",
            fontsize=22, color=COL_TEXT, fontweight="bold")
    # Labels outside
    for w, lab, col in zip(wedges, labels, colors):
        ang = (w.theta2 + w.theta1) / 2
        x = 1.25 * np.cos(np.deg2rad(ang))
        y = 1.25 * np.sin(np.deg2rad(ang))
        ax.text(x, y, lab, ha="center", va="center", color=col,
                fontsize=12, fontweight="bold")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    return save(fig, "epsilon_greedy_split")


# =============================================================================
# Block 3: ε schedules
# =============================================================================

def fig_epsilon_schedules():
    fig, ax = plt.subplots(figsize=(7, 4))
    ts = np.arange(1, 201)
    ax.plot(ts, 0.1 * np.ones_like(ts), color=COL_MUTED, lw=2,
            label="constant $\\varepsilon = 0.1$")
    ax.plot(ts, 1 / ts, color=COL_REWARD, lw=2, label="$\\varepsilon_t = 1/t$")
    ax.plot(ts, np.maximum(0.05, 0.5 * 0.97**ts), color=COL_STATE, lw=2,
            label="exponential decay")
    ax.set_xlabel("step $t$")
    ax.set_ylabel("$\\varepsilon$")
    ax.set_ylim(-0.02, 0.55)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    return save(fig, "epsilon_schedules")


# =============================================================================
# Block 3: driving home (S&B Fig 6.1) — MC vs TD update timing
# =============================================================================

def fig_driving_home():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    landmarks = ["leave\noffice", "reach\ncar", "exit\nhighway",
                 "behind\ntruck", "home\nstreet", "arrive\nhome"]
    elapsed = np.array([0, 5, 20, 30, 40, 43])
    predicted_total = np.array([30, 35, 35, 50, 43, 43])
    predicted_remaining = predicted_total - elapsed
    actual_total = 43

    xs = np.arange(len(landmarks))

    for ax in axes:
        ax.plot(xs, predicted_total, "o-", color=COL_STATE, lw=2,
                markersize=8, zorder=5)
        ax.axhline(actual_total, color=COL_MUTED, ls="--", lw=1.5,
                   label="actual outcome (43 min)")
        ax.set_xticks(xs)
        ax.set_xticklabels(landmarks, fontsize=9)
        ax.set_ylim(20, 60)
        ax.set_ylabel("predicted total time")
        ax.legend(loc="lower right", frameon=False, fontsize=10)

    # Left: Monte Carlo — every prediction's arrow goes to actual outcome
    axes[0].set_title("Monte Carlo update", color=COL_TEXT, fontsize=13)
    for x, y in zip(xs[:-1], predicted_total[:-1]):
        axes[0].annotate(
            "", xy=(x, actual_total), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=COL_REWARD, lw=1.5,
                            mutation_scale=12, alpha=0.9),
        )

    # Right: TD — each prediction's arrow goes to the next prediction
    axes[1].set_title("TD update", color=COL_TEXT, fontsize=13)
    for x, y, y_next in zip(xs[:-1], predicted_total[:-1], predicted_total[1:]):
        axes[1].annotate(
            "", xy=(x + 0.85, y_next), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=COL_REWARD, lw=1.5,
                            mutation_scale=12, alpha=0.9),
        )

    fig.tight_layout()
    return save(fig, "driving_home")


# =============================================================================
# Block 1: stochastic transition (P slide) — single-cell zoom
# =============================================================================

def fig_stochastic_step():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Centre cell with robot
    from matplotlib.patches import Circle, Rectangle
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, facecolor="white",
                            edgecolor=COL_TEXT, linewidth=1.2))
    ax.add_patch(Circle((0, 0), 0.18, facecolor=COL_STATE, edgecolor=COL_TEXT, lw=1.2))
    ax.text(0, 0, "R", ha="center", va="center", color="white",
            fontweight="bold", fontsize=11)
    # Surrounding faded cells
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ax.add_patch(Rectangle((dx - 0.5, dy - 0.5), 1, 1,
                               facecolor=COL_BG_SLIP, edgecolor=COL_MUTED, lw=0.8))
    # Three outcome arrows from intended N action
    outcomes = [
        ((0, 0.95), 0.8, "intended N", COL_REWARD),
        ((-0.95, 0), 0.1, "slip W", COL_ACTION),
        ((0.95, 0), 0.1, "slip E", COL_ACTION),
    ]
    for (xt, yt), p, lab, col in outcomes:
        ax.annotate("", xy=(xt, yt), xytext=(0, 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5,
                                    mutation_scale=18))
        ax.text(xt, yt + (0.25 if yt > 0 else 0.0) + (0.0 if xt == 0 else 0.0),
                f"$p={p}$", ha="center", va="bottom" if yt > 0 else "center",
                color=col, fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=col, boxstyle="round,pad=0.2"))
    # Action label at center
    ax.text(0, -1.3, "intended action: N", ha="center", color=COL_ACTION,
            fontsize=12, fontweight="bold")
    return save(fig, "stochastic_step")


# =============================================================================
# Block 1: policies — deterministic vs stochastic gridworld
# =============================================================================

def fig_policies():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    # Deterministic: single arrow per cell
    det_arrows = {
        (0, 0): "N", (0, 1): "N", (0, 2): "N", (0, 3): "E",
        (1, 0): "N", (1, 1): "N", (1, 2): "N", (1, 3): "E",
        (2, 0): "N", (2, 1): "N", (2, 2): "N", (2, 3): "E",
        (3, 0): "N", (3, 1): "N", (3, 2): "N",
    }
    draw_warehouse(
        axes[0], robot=None, package=None, depot=(3, 3),
        policy_arrows=det_arrows, title="Deterministic policy",
    )
    # Stochastic: show probability bars in each cell instead of single arrows
    axes[1].set_xlim(-0.05, 4.05); axes[1].set_ylim(-0.05, 4.05)
    axes[1].set_aspect("equal")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    for sp in axes[1].spines.values():
        sp.set_visible(False)
    from matplotlib.patches import Rectangle
    for x in range(4):
        for y in range(4):
            if (x, y) == (3, 3):
                axes[1].add_patch(Rectangle((x, y), 1, 1, facecolor="#fef3c7",
                                            edgecolor=COL_MUTED, lw=0.8))
                axes[1].text(x + 0.5, y + 0.5, "D", ha="center", va="center",
                             color=COL_TEXT, fontweight="bold", fontsize=22)
                continue
            axes[1].add_patch(Rectangle((x, y), 1, 1, facecolor="white",
                                        edgecolor=COL_MUTED, lw=0.8))
            # 4-bar mini chart inside cell
            probs = {"N": 0.5, "E": 0.4, "S": 0.05, "W": 0.05}
            bar_w = 0.13
            bar_y = y + 0.15
            cx = x + 0.5
            for i, (a, p) in enumerate(probs.items()):
                bx = cx - 2 * bar_w + i * bar_w
                axes[1].add_patch(Rectangle((bx, bar_y), bar_w * 0.85,
                                            0.55 * p, facecolor=COL_POLICY,
                                            edgecolor="none"))
                axes[1].text(bx + bar_w * 0.42, bar_y - 0.06, a,
                             ha="center", va="top", color=COL_TEXT, fontsize=7)
    axes[1].set_title("Stochastic policy (probabilities per action)",
                       fontsize=13, color=COL_TEXT, pad=10)
    fig.tight_layout()
    return save(fig, "policies")


# =============================================================================
# Block 2: warehouse with optimal policy arrows (Bellman optimality)
# =============================================================================

def fig_optimal_policy():
    fig, ax = plt.subplots(figsize=(6, 6))
    arrows = {
        (0, 0): "E", (1, 0): "E", (2, 0): "E",
        (0, 1): "E", (1, 1): "E", (2, 1): "E", (3, 1): "S",
        (0, 2): "E", (1, 2): "E", (2, 2): "E", (3, 2): "S",
        (0, 3): "E", (1, 3): "S", (2, 3): "S",
    }
    draw_warehouse(
        ax,
        robot=None, package=(1, 3), depot=(3, 0),
        slip_cells=[(1, 1), (1, 2), (2, 1), (2, 2)],
        policy_arrows=arrows,
        title="$\\pi_*$: optimal action per cell",
    )
    return save(fig, "optimal_policy")


# =============================================================================
# Block 3: quartered-Q gridworld snapshots (3 frames over training)
# =============================================================================

def _make_q_overlay_episode_0():
    return {(x, y): {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
            for x in range(4) for y in range(4) if (x, y) != (3, 0)}


def _make_q_overlay_episode_50():
    overlay = {}
    for x in range(4):
        for y in range(4):
            if (x, y) == (3, 0):
                continue
            # Optimal direction: prefer E + S to reach depot bottom-right
            base = -2 - abs(x - 3) - abs(y - 0)
            qs = {"N": base - 1, "S": base + 0.5, "E": base + 0.8, "W": base - 1}
            overlay[(x, y)] = {k: round(v, 1) for k, v in qs.items()}
    return overlay


def _make_q_overlay_episode_500():
    overlay = {}
    for x in range(4):
        for y in range(4):
            if (x, y) == (3, 0):
                continue
            dist = abs(x - 3) + abs(y - 0)
            best = 8.0 - dist * 1.0
            qs = {"N": best - 2, "S": best - 0.5 if y > 0 else best - 2.5,
                  "E": best - 0.5 if x < 3 else best - 2.5, "W": best - 2}
            overlay[(x, y)] = {k: round(v, 1) for k, v in qs.items()}
    return overlay


def fig_q_evolution():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    overlays = [
        ("episode 0", _make_q_overlay_episode_0()),
        ("episode 50", _make_q_overlay_episode_50()),
        ("episode 500", _make_q_overlay_episode_500()),
    ]
    for ax, (title, q) in zip(axes, overlays):
        draw_warehouse(
            ax, robot=None, package=None, depot=(3, 0),
            slip_cells=[(1, 1), (1, 2), (2, 1), (2, 2)],
            q_overlay=q, title=title,
        )
    fig.tight_layout()
    return save(fig, "q_evolution")


# =============================================================================
# Block 3: Windy Gridworld (S&B Fig 6.2 reproduction)
# =============================================================================

def fig_windy_gridworld():
    from matplotlib.patches import FancyArrow, Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw=dict(width_ratios=[1.3, 1]))
    # --- Gridworld ---
    ax = axes[0]
    W, H = 10, 7
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    start = (0, 3); goal = (7, 3)
    ax.set_xlim(-0.05, W + 0.05); ax.set_ylim(-0.65, H + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for x in range(W):
        for y in range(H):
            face = "white"
            if (x, y) == start: face = "#dbeafe"
            if (x, y) == goal:  face = "#fef3c7"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face,
                                    edgecolor=COL_MUTED, lw=0.5))
    # Wind strength labels
    for x, w in enumerate(wind):
        if w > 0:
            ax.annotate("", xy=(x + 0.5, 0.0), xytext=(x + 0.5, -0.4),
                        arrowprops=dict(arrowstyle="-|>", color=COL_ACTION,
                                        lw=1.0 + 0.5 * w, mutation_scale=10))
        ax.text(x + 0.5, -0.55, str(w), ha="center", va="center",
                color=COL_ACTION if w > 0 else COL_MUTED, fontsize=10)
    ax.text(start[0] + 0.5, start[1] + 0.5, "S", ha="center", va="center",
            color=COL_TEXT, fontweight="bold", fontsize=15, zorder=10)
    ax.text(goal[0] + 0.5, goal[1] + 0.5, "G", ha="center", va="center",
            color=COL_TEXT, fontweight="bold", fontsize=15, zorder=10)
    # Trained path (approximate, manually crafted)
    path = [(0, 3), (1, 3), (2, 3), (3, 3), (4, 4), (5, 5), (6, 6), (7, 6),
            (8, 6), (9, 5), (9, 4), (9, 3), (8, 3), (7, 3)]
    xs = [p[0] + 0.5 for p in path]; ys = [p[1] + 0.5 for p in path]
    ax.plot(xs, ys, "-", color=COL_REWARD, lw=2.5, zorder=8)
    ax.plot(xs, ys, "o", color=COL_REWARD, markersize=4, zorder=9)
    ax.set_title("Windy Gridworld — SARSA learned path", color=COL_TEXT,
                 fontsize=13, pad=10)

    # --- Learning curve ---
    ax = axes[1]
    np.random.seed(0)
    eps = np.arange(1, 171)
    # Synthetic SARSA-on-windy curve (cumulative episodes vs time-steps)
    raw = 600 * np.exp(-eps / 35) + 17 + np.random.normal(0, 1.2, len(eps))
    cum_steps = np.cumsum(raw)
    ax.plot(cum_steps, eps, color=COL_STATE, lw=2)
    ax.set_xlabel("time steps")
    ax.set_ylabel("episodes completed")
    ax.set_title("Learning curve  ($\\alpha=0.5,\\,\\varepsilon=0.1$)",
                 color=COL_TEXT, fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return save(fig, "windy_gridworld")


# =============================================================================
# Block 3: Cliff Walking (S&B Fig 6.3 + 6.4 combined)
# =============================================================================

def fig_cliff_walking():
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                             gridspec_kw=dict(width_ratios=[1.4, 1]))
    # --- Gridworld with two trajectories ---
    ax = axes[0]
    W, H = 12, 4
    ax.set_xlim(-0.05, W + 0.05); ax.set_ylim(-0.05, H + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for x in range(W):
        for y in range(H):
            face = "white"
            if y == 0 and 0 < x < W - 1:
                face = COL_TEXT  # cliff
            if (x, y) == (0, 0): face = "#dbeafe"
            if (x, y) == (W - 1, 0): face = "#fef3c7"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face,
                                    edgecolor=COL_MUTED, lw=0.4))
    ax.text(W / 2, 0.5, "T H E   C L I F F", ha="center", va="center",
            color="white", fontweight="bold", fontsize=14, zorder=8)
    ax.text(0.5, 0.5, "S", ha="center", va="center", color=COL_TEXT,
            fontweight="bold", fontsize=14, zorder=10)
    ax.text(W - 0.5, 0.5, "G", ha="center", va="center", color=COL_TEXT,
            fontweight="bold", fontsize=14, zorder=10)

    # SARSA path: safe — climbs to row 2 and back
    sarsa_path = [(0, 0), (0, 1), (0, 2)] + [(x, 2) for x in range(1, 12)] + [(11, 1), (11, 0)]
    # Q-learning path: optimal — hugs cliff edge at row 1
    q_path = [(0, 0), (0, 1)] + [(x, 1) for x in range(1, 12)] + [(11, 0)]
    for path, color, label, dy in [
        (sarsa_path, COL_STATE, "SARSA (safe)", 0.0),
        (q_path,    COL_REWARD, "Q-learning (optimal/risky)", 0.0),
    ]:
        xs = [p[0] + 0.5 for p in path]; ys = [p[1] + 0.5 + dy for p in path]
        ax.plot(xs, ys, "-", color=color, lw=2.5, label=label, zorder=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2,
              frameon=False, fontsize=11)
    ax.set_title("Cliff Walking — learned trajectories", color=COL_TEXT,
                 fontsize=13, pad=10)

    # --- Learning curves ---
    ax = axes[1]
    np.random.seed(1)
    eps = np.arange(1, 501)
    sarsa = -25 - 30 * np.exp(-eps / 80) + np.random.normal(0, 4, len(eps))
    qlearn = -50 - 40 * np.exp(-eps / 80) + np.random.normal(0, 8, len(eps))
    # Smooth
    def smooth(x, w=20):
        return np.convolve(x, np.ones(w) / w, mode="same")
    ax.plot(eps, smooth(sarsa), color=COL_STATE, lw=2, label="SARSA")
    ax.plot(eps, smooth(qlearn), color=COL_REWARD, lw=2, label="Q-learning")
    ax.set_xlabel("episodes")
    ax.set_ylabel("sum of rewards per episode")
    ax.set_ylim(-110, 0)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.set_title("Learning curves", color=COL_TEXT, fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return save(fig, "cliff_walking")


# =============================================================================
# Block 1: trajectory sequence (S, A, R, S', A', R', ...)
# =============================================================================

def fig_trajectory():
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.set_xlim(-0.5, 11.5); ax.set_ylim(-0.6, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    items = [
        ("$S_0$", COL_STATE), ("$A_0$", COL_ACTION), ("$R_1$", COL_REWARD),
        ("$S_1$", COL_STATE), ("$A_1$", COL_ACTION), ("$R_2$", COL_REWARD),
        ("$S_2$", COL_STATE), ("$A_2$", COL_ACTION), ("$R_3$", COL_REWARD),
        ("$S_3$", COL_STATE), ("$\\dots$", COL_TEXT),
    ]
    for i, (lab, col) in enumerate(items):
        ax.add_patch(Circle((i, 0), 0.32, facecolor="white", edgecolor=col, lw=2))
        ax.text(i, 0, lab, ha="center", va="center", color=col, fontsize=12,
                fontweight="bold")
        if i > 0:
            ax.annotate("", xy=(i - 0.34, 0), xytext=(i - 0.66, 0),
                        arrowprops=dict(arrowstyle="-|>", color=COL_MUTED, lw=1.5,
                                        mutation_scale=12))
    return save(fig, "trajectory")


def fig_state_factorization():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    boxes = [
        (1.2, "position\n$(x, y)$", COL_STATE),
        (4.2, "package?\n$\\{0, 1\\}$", COL_ACTION),
        (7.2, "battery\n$\\{0,\\dots,B_{\\max}\\}$", COL_REWARD),
    ]
    for cx, lab, col in boxes:
        ax.add_patch(FancyBboxPatch((cx - 1.0, 1.0), 2.0, 1.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor="white", edgecolor=col, lw=2))
        ax.text(cx, 1.7, lab, ha="center", va="center", color=col,
                fontsize=12, fontweight="bold")
    for x in (3.2, 6.2):
        ax.text(x, 1.7, "×", ha="center", va="center",
                color=COL_TEXT, fontsize=18, fontweight="bold")
    ax.text(5.0, 0.3, "$|\\mathcal{S}| = 16 \\times 2 \\times (B_{\\max}+1)$",
            ha="center", va="center", color=COL_TEXT, fontsize=13)
    return save(fig, "state_factorization")


def fig_actions_arrows():
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    centers_dirs = [
        (1.2, "N", (0, 0.55)),
        (3.2, "S", (0, -0.55)),
        (5.2, "E", (0.55, 0)),
        (7.2, "W", (-0.55, 0)),
    ]
    for cx, lab, (dx, dy) in centers_dirs:
        ax.add_patch(Circle((cx, 1.3), 0.55, facecolor="white",
                             edgecolor=COL_ACTION, lw=2))
        ax.annotate("", xy=(cx + dx, 1.3 + dy), xytext=(cx, 1.3),
                    arrowprops=dict(arrowstyle="-|>", color=COL_ACTION,
                                    lw=2.5, mutation_scale=15))
        ax.text(cx, 0.4, lab, ha="center", va="center", color=COL_ACTION,
                fontsize=14, fontweight="bold")
    # pick-up + drop as iconic boxes
    for cx, lab, sub in [(9.2, "pick-up", "(state-only)"), (11.0, "drop", "(state-only)")]:
        ax.add_patch(plt.Rectangle((cx - 0.7, 0.85), 1.4, 0.9,
                                    facecolor="#fef3c7",
                                    edgecolor=COL_ACTION, lw=2))
        ax.text(cx, 1.3, lab, ha="center", va="center",
                color=COL_TEXT, fontsize=11, fontweight="bold")
        ax.text(cx, 0.4, sub, ha="center", va="center", color=COL_MUTED,
                fontsize=9)
    return save(fig, "actions_arrows")


def fig_bus_route():
    from matplotlib.patches import Circle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8, 2.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    stops = [(1.5, "stop 0"), (4.5, "stop 1"), (7.5, "stop 2"),
             (9.5, "depot")]
    for x, lab in stops[:-1]:
        ax.add_patch(Circle((x, 1.0), 0.45, facecolor="white",
                             edgecolor=COL_STATE, lw=2))
        ax.text(x, 1.0, lab.split()[1], ha="center", va="center",
                color=COL_STATE, fontsize=12, fontweight="bold")
        ax.text(x, 0.25, lab, ha="center", va="center",
                color=COL_TEXT, fontsize=10)
    # depot terminal
    ax.add_patch(plt.Rectangle((9.0, 0.55), 0.9, 0.9,
                                facecolor="#fef3c7", edgecolor=COL_TEXT, lw=2))
    ax.text(9.45, 1.0, "T", ha="center", va="center",
            color=COL_TEXT, fontsize=14, fontweight="bold")
    # arrows between stops
    for (x0, _), (x1, _) in zip(stops[:-1], stops[1:]):
        ax.annotate("", xy=(x1 - 0.5, 1.0), xytext=(x0 + 0.5, 1.0),
                    arrowprops=dict(arrowstyle="-|>", color=COL_ACTION,
                                    lw=2, mutation_scale=15))
    ax.text(3.0, 1.45, "depart", ha="center", color=COL_ACTION, fontsize=10)
    ax.text(6.0, 1.45, "depart", ha="center", color=COL_ACTION, fontsize=10)
    ax.text(8.5, 1.45, "depart", ha="center", color=COL_ACTION, fontsize=10)
    return save(fig, "bus_route")


if __name__ == "__main__":
    figs = [
        fig_warehouse_hook,
        fig_agent_env_loop,
        fig_chain_worked,
        fig_backup_v,
        fig_backup_q,
        fig_backup_sarsa,
        fig_backup_qlearning,
        fig_unified_2x2,
        fig_gamma_decay,
        fig_epsilon_greedy_split,
        fig_epsilon_schedules,
        fig_driving_home,
        fig_stochastic_step,
        fig_policies,
        fig_optimal_policy,
        fig_q_evolution,
        fig_windy_gridworld,
        fig_cliff_walking,
        fig_trajectory,
        fig_state_factorization,
        fig_actions_arrows,
        fig_bus_route,
    ]
    for f in figs:
        path = f()
        print(f"  wrote {path.name}")
    print(f"\n{len(figs)} figure(s) generated in figures/")
