"""Measures self-play throughput: batching, device, search threads, tree reuse.

Each row plays real self-play games with a freshly initialised network, so the figures
describe the engine and not a particular trained net. Results depend on machine load, which
is recorded alongside them.

Usage: python scripts/throughput.py --devices cpu mps
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from connectzero._engine import GamePool, SearchParams, Seat

from connectzero.net import Evaluator, NetConfig, PolicyValueNet, UniformEvaluator
from connectzero.search import drive


def self_play_rate(
    evaluate,
    simulations: int,
    games: int,
    concurrency: int,
    leaves: int,
    threads: int,
) -> dict:
    pool = GamePool(
        Seat.network(0, simulations),
        Seat.network(0, simulations),
        num_games=games,
        concurrency=concurrency,
        leaves_per_tree=leaves,
        threads=threads,
        seed=3,
        search=SearchParams(dirichlet_fraction=0.25),
        temperature_plies=12,
        share_tree=True,
        record_samples=True,
    )
    started = time.perf_counter()
    drive(pool, [evaluate])
    seconds = time.perf_counter() - started
    moves = sum(len(r.moves) for r in pool.results)
    return {
        "simulations": simulations,
        "games": games,
        "concurrency": concurrency,
        "leaves_per_tree": leaves,
        "threads": threads,
        "seconds": seconds,
        "evaluations": pool.evaluations,
        "evaluations_per_second": pool.evaluations / seconds,
        "games_per_second": games / seconds,
        "moves": moves,
        "mean_batch": pool.evaluations / max(pool.batches, 1),
        "evaluations_per_move": pool.evaluations / moves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/throughput.json"))
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    report: dict = {
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "network": {"blocks": args.blocks, "filters": args.filters},
        "torch_threads": args.torch_threads,
        "load_average_start": list(os.getloadavg()),
    }

    # The engine alone, with a trivial evaluator: what the C++ side can sustain.
    report["engine_only"] = [
        self_play_rate(UniformEvaluator(), 200, 256, 256, 4, threads) for threads in (1, 2, 4, 8)
    ]
    print(json.dumps(report["engine_only"], indent=1), flush=True)

    grid = [(1, 1, 8), (8, 4, 16), (32, 4, 48), (128, 4, 192), (256, 4, 384), (512, 4, 768)]
    report["devices"] = {}
    for device in args.devices:
        model = PolicyValueNet(NetConfig(blocks=args.blocks, filters=args.filters))
        evaluator = Evaluator(model, device)
        rows = []
        for concurrency, leaves, games in grid:
            count = max(concurrency, games // 4) if device == "cpu" else games
            row = self_play_rate(evaluator, 100, count, concurrency, leaves, 4)
            rows.append(row)
            print(device, concurrency, leaves, round(row["evaluations_per_second"]), flush=True)
        report["devices"][device] = rows

    best = max(args.devices, key=lambda d: report["devices"][d][-1]["evaluations_per_second"])
    rows = report["devices"][best]
    report["batching_speedup"] = {
        "device": best,
        "unbatched": rows[0]["evaluations_per_second"],
        "best_batched": max(r["evaluations_per_second"] for r in rows),
    }
    report["batching_speedup"]["ratio"] = (
        report["batching_speedup"]["best_batched"] / report["batching_speedup"]["unbatched"]
    )
    report["tree_reuse"] = {
        "evaluations_per_move_at_100": float(np.mean([r["evaluations_per_move"] for r in rows])),
        "note": "A move at 100 visits would cost 100 evaluations with no reuse.",
    }
    report["load_average_end"] = list(os.getloadavg())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
