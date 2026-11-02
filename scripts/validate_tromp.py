"""Cross-checks the solver against John Tromp's 8-ply database.

The dataset (https://doi.org/10.24432/C59P43, CC BY 4.0) lists every legal 8-ply position in
which neither side has won and the next move is not forced, with the game-theoretic result for
the first player. It predates and is independent of this code. The first player is to move
after 8 plies, so its labels can be compared directly with the solver's weak result.

Usage: python scripts/validate_tromp.py path/to/connect-4.data --threads 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from connectzero._engine import Board, solve_many

LABELS = {"win": 1, "draw": 0, "loss": -1}


def parse_row(row: str) -> tuple[str, str]:
    fields = row.strip().split(",")
    return "".join(fields[:42]), fields[42]


def board_from_cells(cells: str) -> Board:
    current = occupied = 0
    for index, cell in enumerate(cells):
        col, row = divmod(index, 6)
        mask = 1 << (col * 7 + row)
        if cell != "b":
            occupied |= mask
        if cell == "x":
            current |= mask
    return Board(current, occupied)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0, help="check a seeded subset of this size")
    parser.add_argument("--log2-entries", type=int, default=23)
    parser.add_argument("--out", type=Path, default=Path("results/solver_tromp.json"))
    parser.add_argument("--sample-out", type=Path, help="write a small sample CSV for the tests")
    parser.add_argument("--sample-size", type=int, default=60)
    args = parser.parse_args()

    raw = args.data.read_bytes()
    rows = [parse_row(line) for line in raw.decode().splitlines() if line.strip()]
    rng = np.random.default_rng(1995)

    if args.sample_out:
        picks = rng.choice(len(rows), args.sample_size, replace=False)
        lines = ["cells,outcome"] + [f"{rows[i][0]},{rows[i][1]}" for i in sorted(picks)]
        args.sample_out.parent.mkdir(parents=True, exist_ok=True)
        args.sample_out.write_text("\n".join(lines) + "\n")

    selected = rows
    if args.limit:
        selected = [rows[i] for i in sorted(rng.choice(len(rows), args.limit, replace=False))]

    digest = hashlib.sha256(raw).hexdigest()
    mismatches: list[str] = []
    checked = 0
    earlier_seconds = 0.0
    if args.out.exists():
        previous = json.loads(args.out.read_text())
        if previous.get("dataset_sha256") == digest and previous.get("limit") == args.limit:
            checked = previous["rows_checked"]
            mismatches = previous["mismatch_examples"]
            earlier_seconds = previous["seconds"]
    started = time.perf_counter()
    for begin in range(checked, len(selected), args.chunk):
        chunk = selected[begin : begin + args.chunk]
        boards = [board_from_cells(cells) for cells, _ in chunk]
        results = solve_many(
            boards, threads=args.threads, weak=True, log2_entries=args.log2_entries
        )
        for (cells, label), result in zip(chunk, results, strict=True):
            if LABELS[label] != result:
                mismatches.append(cells)
        checked += len(chunk)
        elapsed = earlier_seconds + time.perf_counter() - started
        print(
            f"{checked}/{len(selected)} mismatches={len(mismatches)} t={elapsed:.0f}s", flush=True
        )
        report = {
            "dataset": "Tromp, J. (1995). Connect-4. UCI ML Repository, doi:10.24432/C59P43",
            "licence": "CC BY 4.0",
            "dataset_sha256": digest,
            "limit": args.limit,
            "rows_in_dataset": len(rows),
            "rows_checked": checked,
            "mismatches": len(mismatches),
            "seconds": elapsed,
            "threads": args.threads,
            "mismatch_examples": mismatches,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
