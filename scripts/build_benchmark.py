"""Builds data/benchmark.csv, the stratified solver-scored position set.

Candidates come from games where each side plays a tactical move half the time and a
random move otherwise. Pure random play produces positions decided by blunders, and
tactical play alone is too narrow, so the mix reaches a wide range of sensible positions.

Usage: python scripts/build_benchmark.py --per-stratum 300 --opening-quota 200 --threads 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from connectzero._engine import Board, Rng, random_move, tactical_move

from connectzero.benchmark import STRATA, Benchmark, build_stratum, save_benchmark, stratum_of


def candidate_lines(games: int, seed: int) -> dict[str, set[str]]:
    """One position per game for each stratum, deduplicated up to mirror image."""
    engine_rng = Rng(seed)
    rng = np.random.default_rng(seed)
    pools: dict[str, set[str]] = {name: set() for name, _, _ in STRATA}
    seen: dict[str, set[int]] = {name: set() for name, _, _ in STRATA}
    for _ in range(games):
        board = Board()
        line: list[int] = []
        snapshots: dict[str, list[tuple[str, int]]] = {name: [] for name, _, _ in STRATA}
        while not board.is_over():
            ply = board.moves
            if STRATA[0][1] <= ply <= STRATA[-1][2]:
                text = "".join(str(col + 1) for col in line)
                snapshots[stratum_of(ply)].append((text, board.canonical_key()))
            move = (
                tactical_move(board, engine_rng)
                if rng.random() < 0.5
                else random_move(board, engine_rng)
            )
            line.append(move)
            board = board.play(move)
        for name, entries in snapshots.items():
            if not entries:
                continue
            text, key = entries[int(rng.integers(len(entries)))]
            if key not in seen[name]:
                seen[name].add(key)
                pools[name].add(text)
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-stratum", type=int, default=300)
    parser.add_argument("--opening-quota", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=4, help="multiple of the quota to solve")
    parser.add_argument("--games", type=int, default=60_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log2-entries", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("data/benchmark.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/benchmark_build.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pools = candidate_lines(games=args.games, seed=args.seed)

    moves: list[str] = []
    strata: list[str] = []
    scores: list[np.ndarray] = []
    timings = {}
    for name, _, _ in STRATA:
        started = time.perf_counter()
        pool = sorted(pools[name])
        rng.shuffle(pool)
        is_opening = name == "opening"
        quota = args.opening_quota if is_opening else args.per_stratum
        target = quota * 3 // 2 if is_opening else quota * args.candidates
        chosen, exact = build_stratum(pool[:target], quota, args.threads, rng, args.log2_entries)
        moves.extend(chosen)
        strata.extend([name] * len(chosen))
        scores.append(exact)
        timings[name] = {
            "candidates_available": len(pool),
            "candidates_solved": min(target, len(pool)),
            "kept": len(chosen),
            "seconds": time.perf_counter() - started,
        }
        print(name, timings[name], flush=True)

    benchmark = Benchmark(moves, np.array(strata), np.concatenate(scores))
    save_benchmark(benchmark, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "per_stratum": args.per_stratum,
                "opening_quota": args.opening_quota,
                "threads": args.threads,
                "strata": timings,
                "outcomes": {
                    str(value): int((benchmark.outcome == value).sum()) for value in (-1, 0, 1)
                },
                "value_contested": int(benchmark.value_contested().sum()),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
