"""Draws every figure in docs/ from the machine-written files in results/.

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path("results")
DOCS = Path("docs")

PANEL = "#0c0e0d"
HOUSING = "#161a18"
RAIL = "#2a302d"
TEXT = "#d9dfda"
MUTED = "#8a968f"
LAMP = "#ffb000"
AMBER = "#c98200"  # searched network; palette checked with the dark-mode validator
BLUE = "#2a8fd0"  # raw network
RED = "#e0503e"  # third categorical slot
STRATA = ["opening", "early", "middle", "late", "endgame", "final"]
STRATA_LABEL = {
    "opening": "OPENING  ply 4-7",
    "early": "EARLY  ply 8-13",
    "middle": "MIDDLE  ply 14-19",
    "late": "LATE  ply 20-25",
    "endgame": "ENDGAME  ply 26-31",
    "final": "FINAL  ply 32-39",
}

plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.size": 10,
        "figure.facecolor": HOUSING,
        "savefig.facecolor": HOUSING,
        "axes.facecolor": PANEL,
        "axes.edgecolor": RAIL,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": RAIL,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "lines.linewidth": 2,
    }
)


def read_metrics() -> list[dict]:
    lines = (RESULTS / "training_metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def optimal_rate_figure(metrics: list[dict], random_floor: dict) -> None:
    games = np.array([m["total_games"] for m in metrics]) / 1000
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, STRATA, strict=True):
        search = [m["grade"]["search"]["strata"][name]["value_rate"] for m in metrics]
        raw = [m["grade"]["raw"]["strata"][name]["value_rate"] for m in metrics]
        floor = random_floor["strata"][name]["value_rate"]
        ax.plot(games, search, color=AMBER, label="network + 200-visit search")
        ax.plot(games, raw, color=BLUE, label="network policy alone")
        ax.axhline(floor, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, label="random mover")
        ax.annotate(
            f"{search[-1] * 100:.1f}",
            (games[-1], search[-1]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            color=TEXT,
        )
        ax.set_title(STRATA_LABEL[name])
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
        ax.grid(axis="y")
    for ax in axes[1]:
        ax.set_xlabel("self-play games (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("value kept, % of positions")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 0.945))
    fig.suptitle(
        "Moves that keep the game-theoretic result, scored by the solver",
        x=0.012,
        y=0.985,
        ha="left",
        color=LAMP,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(DOCS / "optimal_rate.png", dpi=150)
    plt.close(fig)


def training_figure(metrics: list[dict]) -> None:
    games = np.array([m["total_games"] for m in metrics]) / 1000
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    axes[0].plot(games, [m["train"]["policy_excess"] for m in metrics], color=AMBER)
    axes[0].set_title("Policy loss above target entropy")
    axes[1].plot(games, [m["train"]["value_loss"] for m in metrics], color=AMBER)
    axes[1].set_title("Value loss (MSE)")

    scores = np.array([m["arena"]["score"] for m in metrics])
    promoted = np.array([m["arena"]["promoted"] for m in metrics])
    axes[2].axhline(0.55, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, label="55% gate")
    axes[2].scatter(games[promoted], scores[promoted], s=26, color=AMBER, label="promoted")
    axes[2].scatter(games[~promoted], scores[~promoted], s=26, color=BLUE, label="rejected")
    axes[2].set_title("Candidate score vs gated net")
    axes[2].legend(loc="upper right")
    for ax in axes:
        ax.set_xlabel("self-play games (thousands)")
        ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(DOCS / "training_curves.png", dpi=150)
    plt.close(fig)


def throughput_figure(report: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    series = [
        ("mps", "evaluations_per_second", AMBER, "solid", "Metal GPU, warm"),
        ("mps", "cold_evaluations_per_second", AMBER, (0, (4, 3)), "Metal GPU, first pass"),
        ("cpu", "evaluations_per_second", BLUE, "solid", "CPU, 8 torch threads"),
    ]
    for device, field, color, style, label in series:
        rows = report["devices"][device]
        batches = [r["mean_batch"] for r in rows]
        rates = [r[field] for r in rows]
        ax.plot(batches, rates, color=color, linestyle=style, marker="o", markersize=7, label=label)
        ax.annotate(
            f"{rates[-1]:,.0f}/s",
            (batches[-1], rates[-1]),
            xytext=(-2, -16),
            textcoords="offset points",
            ha="right",
            color=TEXT,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean leaves per network batch")
    ax.set_ylabel("network evaluations per second")
    ax.grid(True, which="major")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "throughput.png", dpi=150)
    plt.close(fig)


def record_cell(entry: dict) -> str:
    return f"{entry['wins']}-{entry['draws']}-{entry['losses']}"


def scoreboard_figure(tournament: dict, grades: dict, ceiling: dict, simulations: int) -> None:
    opponents = [
        ("random", "RANDOM"),
        ("tactical", "TACTICAL"),
        ("alphabeta-4", "ALPHA-BETA D4"),
        ("alphabeta-8", "ALPHA-BETA D8"),
        ("rollout-1k", "ROLLOUT MCTS 1K"),
        ("rollout-10k", "ROLLOUT MCTS 10K"),
        ("perfect", "PERFECT SOLVER"),
    ]
    fig = plt.figure(figsize=(9, 5.1))
    ax = fig.add_axes((0.02, 0.03, 0.96, 0.94))
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RAIL)

    ax.text(3, 91, "CONNECTZERO", color=LAMP, fontsize=22, fontweight="bold")
    ax.text(
        97,
        92,
        f"FINAL STANDINGS  {simulations} VISITS PER MOVE",
        color=MUTED,
        fontsize=9,
        ha="right",
    )
    header_y = 81
    ax.text(3, header_y, "OPPONENT", color=MUTED, fontsize=9)
    ax.text(50, header_y, "W-D-L", color=MUTED, fontsize=9)
    ax.text(72, header_y, "SCORE", color=MUTED, fontsize=9, ha="right")
    ax.text(97, header_y, "95% WILSON", color=MUTED, fontsize=9, ha="right")
    ax.plot([3, 97], [78, 78], color=RAIL, linewidth=1)

    y = 72.5
    for key, label in opponents:
        entry = tournament[f"network_{simulations}_vs_{key}"]
        ax.text(3, y, label, color=TEXT, fontsize=12)
        ax.text(50, y, record_cell(entry), color=TEXT, fontsize=12)
        ax.text(72, y, f"{entry['score'] * 100:.1f}", color=LAMP, fontsize=14, ha="right")
        low, high = entry["wilson_low"] * 100, entry["wilson_high"] * 100
        ax.text(97, y, f"{low:.1f} - {high:.1f}", color=MUTED, fontsize=10, ha="right")
        y -= 7.4

    ax.plot([3, 97], [18, 18], color=RAIL, linewidth=1)
    graded = grades[f"network_{simulations}"]
    empty = tournament[f"network_{simulations}_vs_perfect_from_empty"]
    ax.text(
        3,
        12.5,
        f"FROM THE EMPTY BOARD, MOVING FIRST vs PERFECT: {empty['first_wins']} WINS "
        f"{empty['draws']} DRAW OF {empty['first_games']}",
        color=TEXT,
        fontsize=10,
    )
    ax.text(
        3,
        7,
        "PERFECT SOLVER ROW STARTS FROM ALL 49 TWO-PLY OPENINGS. FLAWLESS PLAY THERE: "
        f"{ceiling['ceiling_wins']}-{ceiling['ceiling_draws']}-{ceiling['ceiling_losses']}",
        color=MUTED,
        fontsize=9,
    )
    ax.text(
        3,
        2,
        f"SOLVER AUDIT  value kept {graded['value_rate'] * 100:.1f}%"
        f"   top score {graded['strict_rate'] * 100:.1f}%"
        f"   {graded['positions']} positions",
        color=MUTED,
        fontsize=9,
    )
    fig.savefig(DOCS / "scoreboard.png", dpi=150)
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    metrics = read_metrics()
    grades = json.loads((RESULTS / "final_grades.json").read_text())
    optimal_rate_figure(metrics, grades["random_expected"])
    training_figure(metrics)
    throughput_figure(json.loads((RESULTS / "throughput.json").read_text()))
    scoreboard_figure(
        json.loads((RESULTS / "tournament.json").read_text()),
        grades,
        json.loads((RESULTS / "perfect_ceiling.json").read_text()),
        simulations=800,
    )


if __name__ == "__main__":
    main()
