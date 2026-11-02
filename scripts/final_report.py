"""Final grading and tournament for a trained network.

Writes results/final_grades.json (benchmark rates for the network at several simulation
counts and for every scripted baseline) and results/tournament.json (match scores with
Wilson intervals, plus games against the perfect solver from the empty board).

Usage: python scripts/final_report.py --weights weights/connectzero.pt --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from connectzero._engine import WIDTH, Board, Seat, seat_moves

from connectzero.arena import play_match
from connectzero.benchmark import STRATA, expected_random_grade, grade, load_benchmark
from connectzero.checkpoint import load_weights
from connectzero.net import Evaluator, NetConfig, PolicyValueNet
from connectzero.search import search_positions

OPPONENTS = {
    "random": Seat.random(),
    "tactical": Seat.tactical(),
    "alphabeta-4": Seat.alphabeta(4),
    "alphabeta-8": Seat.alphabeta(8),
    "rollout-1k": Seat.rollout(1_000),
    "rollout-10k": Seat.rollout(10_000),
    "perfect": Seat.perfect(),
}


def load_evaluator(path: Path, device: str) -> tuple[Evaluator, dict]:
    state = load_weights(path)
    config = state["config"]
    model = PolicyValueNet(NetConfig(blocks=config["blocks"], filters=config["filters"]))
    model.load_state_dict(state["model"])
    return Evaluator(model, device), config


def two_ply_openings() -> list[str]:
    return [f"{a}{b}" for a in range(1, WIDTH + 1) for b in range(1, WIDTH + 1)]


def grade_outcomes(benchmark, chosen: np.ndarray) -> dict:
    """Rates split by the true result, which separates converting from holding."""
    report = {}
    for label, value in (("won", 1), ("drawn", 0), ("lost", -1)):
        mask = benchmark.outcome == value
        rows = np.flatnonzero(mask)
        strict = benchmark.strictly_optimal()[rows, chosen[rows]]
        preserve = benchmark.value_preserving()[rows, chosen[rows]]
        report[label] = {
            "positions": int(mask.sum()),
            "strict_rate": float(strict.mean()),
            "value_rate": float(preserve.mean()),
        }
    return report


def grades(evaluator: Evaluator, benchmark, args, c_puct: float) -> dict:
    boards = benchmark.boards()
    report: dict = {"random_expected": expected_random_grade(benchmark)}

    planes = np.stack([board.planes() for board in boards])
    policy, _ = evaluator(planes)
    raw = np.where(benchmark.legal, policy, -1.0).argmax(axis=1)
    report["network_raw"] = grade(benchmark, raw) | {"by_result": grade_outcomes(benchmark, raw)}

    for simulations in args.simulations:
        visits, _ = search_positions(boards, evaluator, simulations, c_puct, threads=args.threads)
        chosen = visits.argmax(axis=1)
        report[f"network_{simulations}"] = grade(benchmark, chosen) | {
            "by_result": grade_outcomes(benchmark, chosen)
        }
        print("graded network", simulations, report[f"network_{simulations}"]["value_rate"])

    for name in ("tactical", "alphabeta-4", "alphabeta-8", "rollout-1k", "rollout-10k"):
        chosen = np.array(
            seat_moves(boards, OPPONENTS[name], c_puct=c_puct, threads=args.threads, seed=args.seed)
        )
        report[name] = grade(benchmark, chosen) | {"by_result": grade_outcomes(benchmark, chosen)}
        print("graded", name, report[name]["value_rate"], flush=True)
    return report


def tournament(evaluator: Evaluator, args, c_puct: float) -> dict:
    report: dict = {}
    openings = two_ply_openings()
    games = 2 * len(openings)
    for simulations in args.tournament_simulations:
        for name, seat in OPPONENTS.items():
            started = time.perf_counter()
            score, _ = play_match(
                Seat.network(0, simulations),
                seat,
                [evaluator],
                games=games,
                c_puct=c_puct,
                seed=args.seed,
                threads=args.threads,
                concurrency=32,
                openings=openings,
                solver_log2_entries=26,
            )
            key = f"network_{simulations}_vs_{name}"
            report[key] = score.as_dict() | {"seconds": time.perf_counter() - started}
            print(key, f"{score.wins}-{score.draws}-{score.losses}", flush=True)

    # From the empty board the solver's verdict is known: first player wins, so the network
    # must play a perfect game as first player to score at all against this opponent.
    games = args.empty_board_games
    for simulations in args.tournament_simulations:
        score, results = play_match(
            Seat.network(0, simulations),
            Seat.perfect(),
            [evaluator],
            games=games,
            c_puct=c_puct,
            seed=args.seed + 1,
            threads=args.threads,
            concurrency=8,
            solver_log2_entries=26,
        )
        key = f"network_{simulations}_vs_perfect_from_empty"
        report[key] = score.as_dict() | {
            "first_player_games": [r.moves for r in results if r.first_seat == 0][:4],
            "second_player_games": [r.moves for r in results if r.first_seat == 1][:4],
        }
        print(key, f"{score.wins}-{score.draws}-{score.losses}", flush=True)
    return report


def opening_choices(evaluator: Evaluator, simulations: int, c_puct: float) -> dict:
    board = Board()
    visits, values = search_positions([board], evaluator, simulations, c_puct, threads=1)
    return {
        "simulations": simulations,
        "first_move_visits": visits[0].tolist(),
        "first_move_value": float(values[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/connectzero.pt"))
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--simulations", type=int, nargs="+", default=[50, 200, 800, 3200])
    parser.add_argument("--tournament-simulations", type=int, nargs="+", default=[200, 800])
    parser.add_argument("--empty-board-games", type=int, default=20)
    parser.add_argument("--only", choices=["grades", "tournament"])
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    evaluator, config = load_evaluator(args.weights, args.device)
    c_puct = config["c_puct"]
    benchmark = load_benchmark(args.benchmark)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only != "tournament":
        report = grades(evaluator, benchmark, args, c_puct)
        report["strata"] = [list(stratum) for stratum in STRATA]
        report["opening"] = opening_choices(evaluator, 800, c_puct)
        (args.out / "final_grades.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.only != "grades":
        report = tournament(evaluator, args, c_puct)
        (args.out / "tournament.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
