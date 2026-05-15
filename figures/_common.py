"""Shared helpers for RL teaching figures.

Locked palette (also used in slide equations via \\textcolor):
    blue   = state / value
    orange = action / policy
    red    = reward
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

# ---- locked palette ---------------------------------------------------------
COL_STATE = "#2563eb"
COL_VALUE = "#2563eb"
COL_ACTION = "#f97316"
COL_POLICY = "#f97316"
COL_REWARD = "#dc2626"
COL_TEXT = "#1f2937"
COL_MUTED = "#94a3b8"
COL_BG_SLIP = "#dbeafe"
COL_BG_DEPOT = "#fef3c7"
COL_NEG = "#fecaca"   # light red (negative q-value background)
COL_POS = "#bbf7d0"   # light green (positive q-value background)
COL_ZERO = "#f3f4f6"  # light grey (zero / unknown)

STYLE_PATH = Path(__file__).parent / "style.mplstyle"
FIG_DIR = Path(__file__).parent


def setup_style():
    plt.style.use(STYLE_PATH)


# ---- warehouse gridworld ---------------------------------------------------
# Convention: cell (x, y) with x = column 0..N-1 (left→right),
#                            y = row    0..N-1 (bottom→top).
# Action symbols: "N" (+y), "S" (-y), "E" (+x), "W" (-x).

ACTION_DXY = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}


def _q_color(q, vmin=-10, vmax=10):
    """Diverging colour: red for negative, green for positive, white at zero."""
    if q is None:
        return COL_ZERO
    t = max(min(q / max(abs(vmin), abs(vmax)), 1.0), -1.0)
    if t >= 0:
        # interpolate white -> COL_POS
        r, g, b = (1, 1, 1)
        r2, g2, b2 = (187 / 255, 247 / 255, 208 / 255)
        return (r + (r2 - r) * t, g + (g2 - g) * t, b + (b2 - b) * t)
    else:
        r, g, b = (1, 1, 1)
        r2, g2, b2 = (254 / 255, 202 / 255, 202 / 255)
        return (r + (r2 - r) * (-t), g + (g2 - g) * (-t), b + (b2 - b) * (-t))


def draw_warehouse(
    ax,
    grid_size=4,
    robot=(0, 0),
    package=(1, 3),
    depot=(3, 0),
    slip_cells=(),
    walls=(),
    policy_arrows=None,
    q_overlay=None,
    v_overlay=None,
    extra_paths=None,
    title=None,
    cell_label_fn=None,
):
    """Render a gridworld warehouse with optional overlays.

    robot/package/depot: (x, y) cell, or None to omit.
    slip_cells: iterable of (x, y) cells with slippery floor.
    walls: iterable of (x, y) cells that are walls (impassable).
    policy_arrows: dict cell -> action ("N"/"S"/"E"/"W") for greedy policy arrows.
    q_overlay: dict cell -> dict {"N": q, ...} for quartered Q visualization.
    v_overlay: dict cell -> float for state-value heatmap (mutually exclusive with q_overlay).
    extra_paths: list of dicts {"cells": [(x,y),...], "color": ..., "label": ..., "linewidth": ...}.
    cell_label_fn: callable (x, y) -> str | None to add text in each cell.
    """
    n = grid_size
    ax.set_xlim(-0.05, n + 0.05)
    ax.set_ylim(-0.05, n + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Backgrounds (v_overlay heatmap takes precedence)
    for x in range(n):
        for y in range(n):
            if (x, y) in walls:
                color = COL_TEXT
            elif v_overlay and (x, y) in v_overlay:
                color = _q_color(v_overlay[(x, y)])
            elif (x, y) == depot:
                color = COL_BG_DEPOT
            elif (x, y) in slip_cells:
                color = COL_BG_SLIP
            else:
                color = "white"
            ax.add_patch(
                Rectangle(
                    (x, y), 1, 1,
                    facecolor=color, edgecolor=COL_MUTED, linewidth=0.8,
                )
            )

    # Quartered Q-values overlay
    if q_overlay:
        for (x, y), qs in q_overlay.items():
            cx, cy = x + 0.5, y + 0.5
            corners = {
                "N": [(x, y + 1), (x + 1, y + 1), (cx, cy)],
                "S": [(x, y), (x + 1, y), (cx, cy)],
                "E": [(x + 1, y), (x + 1, y + 1), (cx, cy)],
                "W": [(x, y), (x, y + 1), (cx, cy)],
            }
            label_pos = {
                "N": (cx, y + 0.85), "S": (cx, y + 0.15),
                "E": (x + 0.85, cy), "W": (x + 0.15, cy),
            }
            for direction, q in qs.items():
                ax.add_patch(
                    Polygon(corners[direction], facecolor=_q_color(q),
                            edgecolor=COL_MUTED, linewidth=0.4)
                )
                ax.text(*label_pos[direction], f"{q:+.1f}",
                        ha="center", va="center", color=COL_TEXT, fontsize=8)

    # Policy arrows
    if policy_arrows:
        for (x, y), action in policy_arrows.items():
            cx, cy = x + 0.5, y + 0.5
            dx, dy = ACTION_DXY[action]
            ax.annotate(
                "", xy=(cx + 0.32 * dx, cy + 0.32 * dy),
                xytext=(cx - 0.18 * dx, cy - 0.18 * dy),
                arrowprops=dict(arrowstyle="-|>", color=COL_POLICY, lw=2.5,
                                mutation_scale=18),
                zorder=8,
            )

    # Custom cell labels
    if cell_label_fn:
        for x in range(n):
            for y in range(n):
                lab = cell_label_fn(x, y)
                if lab:
                    ax.text(x + 0.5, y + 0.5, lab, ha="center", va="center",
                            color=COL_TEXT, fontsize=11)

    # Extra paths (e.g. trajectory overlays)
    if extra_paths:
        for p in extra_paths:
            xs = [c[0] + 0.5 for c in p["cells"]]
            ys = [c[1] + 0.5 for c in p["cells"]]
            ax.plot(xs, ys, color=p.get("color", COL_TEXT),
                    linewidth=p.get("linewidth", 2.5),
                    label=p.get("label"), zorder=7)

    # Iconography for robot / package / depot (drawn last, on top)
    if depot is not None:
        dx, dy = depot
        ax.text(dx + 0.5, dy + 0.5, "D", ha="center", va="center",
                color=COL_TEXT, fontweight="bold", fontsize=22, zorder=9)
    if package is not None:
        px, py = package
        ax.add_patch(Rectangle(
            (px + 0.28, py + 0.28), 0.44, 0.44,
            facecolor=COL_REWARD, edgecolor=COL_TEXT, linewidth=1.2, zorder=10,
        ))
        ax.text(px + 0.5, py + 0.5, "P", ha="center", va="center",
                color="white", fontweight="bold", fontsize=14, zorder=11)
    if robot is not None:
        rx, ry = robot
        ax.add_patch(Circle(
            (rx + 0.5, ry + 0.5), 0.27,
            facecolor=COL_STATE, edgecolor=COL_TEXT, linewidth=1.2, zorder=10,
        ))
        ax.text(rx + 0.5, ry + 0.5, "R", ha="center", va="center",
                color="white", fontweight="bold", fontsize=14, zorder=11)

    if title:
        ax.set_title(title, fontsize=14, color=COL_TEXT, pad=10)


# ---- 3-state Mars-rover-style chain ----------------------------------------

def draw_chain(
    ax,
    n_states=3,
    terminal_right=True,
    rewards_right=None,
    rewards_left=None,
    state_labels=None,
    value_labels=None,
    highlight=None,
    title=None,
):
    """Render a horizontal n-state chain MDP.

    rewards_right/left: list of length n_states-1 (or n_states if terminal),
                       reward for taking right/left from each state.
    state_labels: list of length n_states; defaults to s_1..s_n.
    value_labels: dict state_index -> str for labels inside circles.
    highlight: state index to highlight in COL_REWARD.
    """
    state_labels = state_labels or [f"$s_{i+1}$" for i in range(n_states - (1 if terminal_right else 0))]
    if terminal_right:
        state_labels = state_labels + ["T"]

    radius = 0.32
    spacing = 1.5
    y0 = 0.0
    xs = [i * spacing for i in range(n_states)]

    ax.set_xlim(-0.7, xs[-1] + 0.7)
    ax.set_ylim(-1.0, 1.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Nodes
    for i, x in enumerate(xs):
        is_terminal = (i == n_states - 1) and terminal_right
        face = COL_REWARD if highlight == i else ("white" if not is_terminal else COL_BG_DEPOT)
        if is_terminal:
            ax.add_patch(Rectangle(
                (x - radius, y0 - radius), 2 * radius, 2 * radius,
                facecolor=face, edgecolor=COL_TEXT, linewidth=1.5, zorder=5,
            ))
        else:
            ax.add_patch(Circle((x, y0), radius, facecolor=face,
                                edgecolor=COL_TEXT, linewidth=1.5, zorder=5))
        ax.text(x, y0, state_labels[i], ha="center", va="center",
                color=COL_TEXT, fontsize=14, zorder=6)
        if value_labels and i in value_labels:
            ax.text(x, y0 - 0.7, value_labels[i], ha="center", va="center",
                    color=COL_VALUE, fontsize=12)

    # Right arrows + reward labels
    for i in range(n_states - 1):
        x1, x2 = xs[i] + radius + 0.05, xs[i + 1] - radius - 0.05
        ax.annotate("", xy=(x2, y0 + 0.18), xytext=(x1, y0 + 0.18),
                    arrowprops=dict(arrowstyle="-|>", color=COL_ACTION, lw=2,
                                    mutation_scale=14))
        if rewards_right and i < len(rewards_right):
            mid = (xs[i] + xs[i + 1]) / 2
            ax.text(mid, y0 + 0.45, f"$r{{=}}{rewards_right[i]:+}$",
                    ha="center", va="center", color=COL_REWARD, fontsize=11)

    # Left arrows + reward labels
    if rewards_left is not None:
        for i in range(1, n_states):
            x1, x2 = xs[i] - radius - 0.05, xs[i - 1] + radius + 0.05
            ax.annotate("", xy=(x2, y0 - 0.18), xytext=(x1, y0 - 0.18),
                        arrowprops=dict(arrowstyle="-|>", color=COL_ACTION,
                                        lw=2, mutation_scale=14))
            if i - 1 < len(rewards_left):
                mid = (xs[i] + xs[i - 1]) / 2
                ax.text(mid, y0 - 0.5, f"$r{{=}}{rewards_left[i - 1]:+}$",
                        ha="center", va="center", color=COL_REWARD, fontsize=11)

    if title:
        ax.set_title(title, fontsize=14, color=COL_TEXT, pad=10)


# ---- agent-environment loop ------------------------------------------------

def draw_agent_env_loop(ax, title=None):
    """Sutton & Barto Fig 3.1 style: two boxes, two arrows."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Agent box (top)
    ax.add_patch(Rectangle((1.5, 3.2), 3, 1.2,
                           facecolor=COL_STATE, edgecolor=COL_TEXT, linewidth=1.5))
    ax.text(3, 3.8, "Agent", ha="center", va="center", color="white",
            fontweight="bold", fontsize=15)

    # Environment box (bottom)
    ax.add_patch(Rectangle((1.5, 0.6), 3, 1.2,
                           facecolor=COL_ACTION, edgecolor=COL_TEXT, linewidth=1.5))
    ax.text(3, 1.2, "Environment", ha="center", va="center", color="white",
            fontweight="bold", fontsize=15)

    # Arrow: Agent -> Env (action)
    ax.annotate("", xy=(5.2, 1.2), xytext=(4.6, 3.8),
                arrowprops=dict(arrowstyle="-|>", color=COL_TEXT, lw=2,
                                connectionstyle="arc3,rad=-0.4",
                                mutation_scale=20))
    ax.text(7.3, 2.85, "action $A_t$", ha="center", va="center",
            color=COL_ACTION, fontsize=13, fontweight="bold")

    # Arrow: Env -> Agent (state, reward)
    ax.annotate("", xy=(1.5, 3.8), xytext=(0.9, 1.2),
                arrowprops=dict(arrowstyle="-|>", color=COL_TEXT, lw=2,
                                connectionstyle="arc3,rad=-0.4",
                                mutation_scale=20))
    ax.text(-0.4, 2.85, "state $S_{t+1}$\nreward $R_{t+1}$", ha="center",
            va="center", color=COL_TEXT, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, color=COL_TEXT, pad=10)


# ---- backup diagram primitives ---------------------------------------------
# Silver / S&B Fig 3.4 style: open circle = state, filled dot = action, square = terminal.

def _state_node(ax, x, y, r=0.18, color=COL_STATE, label=None, label_offset=(0, 0)):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, linewidth=2.0, zorder=5))
    if label:
        ax.text(x + label_offset[0], y + label_offset[1], label,
                ha="center", va="center", color=COL_TEXT, fontsize=11)


def _action_node(ax, x, y, r=0.10, color=COL_ACTION, label=None, label_offset=(0.0, -0.18)):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=color, linewidth=1.5, zorder=5))
    if label:
        ax.text(x + label_offset[0], y + label_offset[1], label,
                ha="center", va="center", color=COL_TEXT, fontsize=10)


def _edge(ax, p1, p2, label=None, dashed=False, color=None, lw=1.2, label_offset=(0.05, 0)):
    color = color or COL_TEXT
    ls = "--" if dashed else "-"
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], ls, color=color, linewidth=lw, zorder=3)
    if label:
        mid = ((p1[0] + p2[0]) / 2 + label_offset[0],
               (p1[1] + p2[1]) / 2 + label_offset[1])
        ax.text(*mid, label, ha="left", va="center", color=color, fontsize=9)


def draw_backup_v(ax, title=None):
    """Backup diagram for v_pi: state -> two actions -> two next states each."""
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Root state
    _state_node(ax, 0, 0.6, label="$s$", label_offset=(0, 0.45))
    # Two action nodes
    a_xs = [-0.9, 0.9]
    for ax_x in a_xs:
        _edge(ax, (0, 0.45), (ax_x, -0.4), color=COL_POLICY, lw=1.2,
              label="$\\pi(a|s)$" if ax_x == a_xs[0] else None,
              label_offset=(-0.15, 0.05))
        _action_node(ax, ax_x, -0.5)
        # From each action, two next states
        for ns_dx in [-0.55, 0.55]:
            ns_x = ax_x + ns_dx
            _edge(ax, (ax_x, -0.6), (ns_x, -1.5),
                  color=COL_TEXT, lw=1.0,
                  label=("$P(s'|s,a)$\n$R(s,a,s')$" if (ax_x == a_xs[0] and ns_dx < 0) else None),
                  label_offset=(-0.55, -0.0))
            _state_node(ax, ns_x, -1.55, r=0.13, color=COL_STATE)
    if title:
        ax.set_title(title, fontsize=12, color=COL_TEXT, pad=4)


def draw_backup_q(ax, title=None):
    """Backup diagram for q_pi: action -> two next states -> two next actions each."""
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Root action (with implicit state s above)
    _state_node(ax, 0, 0.7, label="$s$", label_offset=(-0.30, 0.0))
    _edge(ax, (0, 0.55), (0, 0.05), color=COL_POLICY, lw=1.2)
    _action_node(ax, 0, 0, label="$a$", label_offset=(0.25, 0.0))
    # Two next states
    ns_xs = [-1.0, 1.0]
    for ns_x in ns_xs:
        _edge(ax, (0, -0.1), (ns_x, -0.95), color=COL_TEXT, lw=1.0,
              label=("$P(s'|s,a)$\n$R(s,a,s')$" if ns_x == ns_xs[0] else None),
              label_offset=(-0.7, -0.05))
        _state_node(ax, ns_x, -1.0, label="$s'$", label_offset=(-0.25, 0.05))
        # From each next state, two next actions
        for na_dx in [-0.45, 0.45]:
            na_x = ns_x + na_dx
            _edge(ax, (ns_x, -1.18), (na_x, -1.6),
                  color=COL_POLICY, lw=1.0,
                  label=("$\\pi(a'|s')$" if (ns_x == ns_xs[0] and na_dx < 0) else None),
                  label_offset=(-0.45, -0.0))
            _action_node(ax, na_x, -1.65, r=0.08)
    if title:
        ax.set_title(title, fontsize=12, color=COL_TEXT, pad=4)


def draw_backup_sarsa(ax, title=None):
    """Vertical SARSA backup: S -> A -> R -> S' -> A'."""
    ax.set_xlim(-1.0, 2.4)
    ax.set_ylim(-2.4, 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    nodes = [
        (0, 0.0, "state"), (0, -0.6, "action"), (0, -1.2, "state"), (0, -1.8, "action"),
    ]
    labels = ["$S$", "$A$", "$S'$", "$A'$"]
    for i, ((x, y, kind), lab) in enumerate(zip(nodes, labels)):
        if kind == "state":
            _state_node(ax, x, y, label=lab, label_offset=(0.45, 0.0))
        else:
            _action_node(ax, x, y, label=lab, label_offset=(0.45, 0.0))
        if i > 0:
            ax.plot([0, 0], [y + 0.1, y + 0.5], "-", color=COL_TEXT, lw=1.2, zorder=3)
    # Reward label between A and S'
    ax.text(0.55, -0.9, "$R$", color=COL_REWARD, fontsize=12, va="center")
    if title:
        ax.set_title(title, fontsize=12, color=COL_TEXT, pad=4)


def draw_backup_qlearning(ax, title=None):
    """Vertical Q-learning backup: S -> A -> R -> S' -> max over actions (fan)."""
    ax.set_xlim(-1.6, 2.0)
    ax.set_ylim(-2.5, 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # S, A, S' chain
    _state_node(ax, 0, 0.0, label="$S$", label_offset=(0.45, 0.0))
    ax.plot([0, 0], [-0.1, -0.5], "-", color=COL_TEXT, lw=1.2, zorder=3)
    _action_node(ax, 0, -0.6, label="$A$", label_offset=(0.45, 0.0))
    ax.plot([0, 0], [-0.7, -1.1], "-", color=COL_TEXT, lw=1.2, zorder=3)
    _state_node(ax, 0, -1.2, label="$S'$", label_offset=(0.45, 0.0))
    ax.text(0.55, -0.9, "$R$", color=COL_REWARD, fontsize=12, va="center")
    # Fan of action nodes (max)
    for x_off in [-0.7, 0.0, 0.7]:
        ax.plot([0, x_off], [-1.3, -1.85], "-", color=COL_POLICY, lw=1.0, zorder=3)
        _action_node(ax, x_off, -1.95, r=0.08)
    ax.text(0, -2.3, "$\\max_{a'}$", ha="center", va="center",
            color=COL_POLICY, fontsize=12, fontstyle="italic")
    if title:
        ax.set_title(title, fontsize=12, color=COL_TEXT, pad=4)


def draw_2x2_unified(ax, title=None):
    """Sample x bootstrap 2x2 with mini backup diagrams in each corner."""
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Frame
    ax.add_patch(Rectangle((-2.2, -2.2), 4.4, 4.4,
                            facecolor="none", edgecolor=COL_MUTED, linewidth=0.8))
    ax.plot([0, 0], [-2.2, 2.2], color=COL_MUTED, lw=0.8)
    ax.plot([-2.2, 2.2], [0, 0], color=COL_MUTED, lw=0.8)

    # Axis labels (sampling = right, bootstrap = bottom)
    ax.text(0, 2.55, r"Sampling $\rightarrow$", ha="center", va="bottom",
            fontsize=12, color=COL_TEXT, fontweight="bold")
    ax.text(-2.55, 0, r"Bootstrapping $\rightarrow$", ha="center", va="center",
            fontsize=12, color=COL_TEXT, fontweight="bold", rotation=90)

    # Cell labels
    centers = {"NW": (-1.1, 1.1), "NE": (1.1, 1.1), "SW": (-1.1, -1.1), "SE": (1.1, -1.1)}
    methods = {
        "NW": "Exhaustive",
        "NE": "Monte Carlo",
        "SW": "Dynamic\nProgramming",
        "SE": r"Temporal" + "\n" + r"Difference  $\bigstar$",
    }
    colors = {"NW": COL_MUTED, "NE": COL_TEXT, "SW": COL_TEXT, "SE": COL_REWARD}
    for k, (cx, cy) in centers.items():
        ax.text(cx, cy + 0.85, methods[k], ha="center", va="center",
                color=colors[k], fontsize=12, fontweight="bold")

    # Mini backup pictograms at each corner (very tiny)
    def mini_state(x, y):
        ax.add_patch(Circle((x, y), 0.07, facecolor="white",
                             edgecolor=COL_STATE, linewidth=1.5, zorder=5))

    def mini_action(x, y, color=COL_ACTION):
        ax.add_patch(Circle((x, y), 0.05, facecolor=color,
                             edgecolor=color, linewidth=1.2, zorder=5))

    # SE (TD): state -> action -> reward -> next state -> next action
    cx, cy = centers["SE"]
    mini_state(cx, cy + 0.3)
    ax.plot([cx, cx], [cy + 0.23, cy], color=COL_TEXT, lw=0.8)
    mini_action(cx, cy - 0.05)
    ax.plot([cx, cx], [cy - 0.1, cy - 0.35], color=COL_TEXT, lw=0.8)
    mini_state(cx, cy - 0.4)

    # SW (DP): state -> all actions -> all next states
    cx, cy = centers["SW"]
    mini_state(cx, cy + 0.3)
    for dx in [-0.18, 0, 0.18]:
        ax.plot([cx, cx + dx], [cy + 0.23, cy], color=COL_TEXT, lw=0.6)
        mini_action(cx + dx, cy)
        for dx2 in [-0.08, 0.08]:
            ax.plot([cx + dx, cx + dx + dx2], [cy - 0.05, cy - 0.3],
                    color=COL_TEXT, lw=0.5)
            mini_state(cx + dx + dx2, cy - 0.35)

    # NE (MC): state -> action -> long chain to terminal
    cx, cy = centers["NE"]
    mini_state(cx, cy + 0.4)
    for i, sign in enumerate(["a", "s", "a", "s", "T"]):
        y0 = cy + 0.4 - 0.18 * (i + 1)
        ax.plot([cx, cx], [y0 + 0.15, y0 + 0.05], color=COL_TEXT, lw=0.7)
        if sign == "a":
            mini_action(cx, y0)
        elif sign == "s":
            mini_state(cx, y0)
        else:
            ax.add_patch(Rectangle((cx - 0.07, y0 - 0.07), 0.14, 0.14,
                                    facecolor=COL_BG_DEPOT, edgecolor=COL_TEXT,
                                    linewidth=1.0))

    if title:
        ax.set_title(title, fontsize=14, color=COL_TEXT, pad=10)


# ---- save helper -----------------------------------------------------------

def save(fig, name):
    """Save figure to figures/<name>.png with consistent settings."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return path
