"""Solves the empty board move by move and records the result and solver throughput.

Usage: python scripts/solve_opening.py --log2-entries 26
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from connectzero._engine import Board, Solver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log2-entries", type=int, default=26)
    parser.add_argument("--out", type=Path, default=Path("results/solver_opening.json"))
    args = parser.parse_args()

    solver = Solver(args.log2_entries)
    started = time.perf_counter()
    root_score = solver.solve(Board())
    root_seconds = time.perf_counter() - started
    root_nodes = solver.nodes

    solver.reset_nodes()
    started = time.perf_counter()
    scores = solver.analyze(Board())
    move_seconds = time.perf_counter() - started

    report = {
        "root_score": root_score,
        "root_seconds": root_seconds,
        "root_nodes": root_nodes,
        "root_nodes_per_second": root_nodes / root_seconds,
        "first_move_scores": scores,
        "first_move_seconds_with_warm_table": move_seconds,
        "first_move_nodes_with_warm_table": solver.nodes,
        "table_entries": 2**args.log2_entries,
        "threads": 1,
        "platform": f"{platform.system()} {platform.machine()}",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
