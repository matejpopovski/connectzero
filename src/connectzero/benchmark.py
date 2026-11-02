"""Stratified position set scored by the solver, and grading of move choices against it.

Random games give positions that are mostly already decided, and asking a player to find
the only move in a position with no wrong move measures nothing. So positions are sampled
per game phase, balanced across win, draw and loss for the side to move, and every kept
position has at least one column that scores worse than the best.

Two grades are reported because they answer different questions:

* value rate: the chosen move keeps the game-theoretic result. Counted only on positions
  the side to move does not lose, since every move in a lost position keeps the result.
* strict rate: the chosen move has the solver's top score, which also means the fastest win
  or the longest resistance. AlphaZero's value target has no notion of game length, so this
  grade is harder for it by construction.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from connectzero._engine import ILLEGAL, Board, analyze_many

STRATA: tuple[tuple[str, int, int], ...] = (
    ("opening", 4, 7),
    ("early", 8, 13),
    ("middle", 14, 19),
    ("late", 20, 25),
    ("endgame", 26, 31),
    ("final", 32, 39),
)


def stratum_of(ply: int) -> str:
    for name, low, high in STRATA:
        if low <= ply <= high:
            return name
    raise ValueError(f"ply {ply} is outside every stratum")


@dataclass(frozen=True)
class Benchmark:
    moves: list[str]
    strata: np.ndarray  # stratum name per position
    scores: np.ndarray  # (n, 7) solver score per column, ILLEGAL where full

    def __len__(self) -> int:
        return len(self.moves)

    def boards(self) -> list[Board]:
        return [Board.from_moves(text) for text in self.moves]

    @property
    def legal(self) -> np.ndarray:
        return self.scores != ILLEGAL

    @property
    def best(self) -> np.ndarray:
        return np.where(self.legal, self.scores, -100).max(axis=1)

    @property
    def outcome(self) -> np.ndarray:
        """+1, 0 or -1 for the side to move under perfect play."""
        return np.sign(self.best)

    def value_preserving(self) -> np.ndarray:
        """Boolean (n, 7): the column keeps the position's game-theoretic result."""
        results = np.where(self.legal, np.sign(self.scores), -2)
        return results == self.outcome[:, None]

    def strictly_optimal(self) -> np.ndarray:
        return self.scores == self.best[:, None]

    def value_contested(self) -> np.ndarray:
        """Positions where a legal move would throw away the result."""
        preserving = self.value_preserving()
        return (self.outcome >= 0) & (preserving.sum(axis=1) < self.legal.sum(axis=1))


def save_benchmark(benchmark: Benchmark, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["moves", "stratum", *[f"col{c}" for c in range(1, 8)]])
        for text, stratum, row in zip(
            benchmark.moves, benchmark.strata, benchmark.scores, strict=True
        ):
            cells = ["" if value == ILLEGAL else int(value) for value in row]
            writer.writerow([text, stratum, *cells])


def load_benchmark(path: Path) -> Benchmark:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = np.array(
        [[ILLEGAL if r[f"col{c}"] == "" else int(r[f"col{c}"]) for c in range(1, 8)] for r in rows],
        dtype=np.int32,
    )
    return Benchmark([r["moves"] for r in rows], np.array([r["stratum"] for r in rows]), scores)


def grade(benchmark: Benchmark, chosen: np.ndarray) -> dict:
    """Value and strict rates for one column per position, overall and per stratum."""
    rows = np.arange(len(benchmark))
    value_hit = benchmark.value_preserving()[rows, chosen]
    strict_hit = benchmark.strictly_optimal()[rows, chosen]
    contested = benchmark.value_contested()

    def rates(mask: np.ndarray) -> dict:
        value_mask = mask & contested
        return {
            "positions": int(mask.sum()),
            "value_positions": int(value_mask.sum()),
            "value_rate": float(value_hit[value_mask].mean()) if value_mask.any() else None,
            "strict_rate": float(strict_hit[mask].mean()) if mask.any() else None,
        }

    report = rates(np.ones(len(benchmark), dtype=bool))
    report["strata"] = {
        name: rates(benchmark.strata == name) for name, _, _ in STRATA if name in benchmark.strata
    }
    return report


def expected_random_grade(benchmark: Benchmark) -> dict:
    """Exact expected rates of a uniform random mover, the floor for every chart."""
    legal = benchmark.legal.sum(axis=1)
    value_share = benchmark.value_preserving().sum(axis=1) / legal
    strict_share = benchmark.strictly_optimal().sum(axis=1) / legal
    contested = benchmark.value_contested()

    def rates(mask: np.ndarray) -> dict:
        value_mask = mask & contested
        return {
            "positions": int(mask.sum()),
            "value_positions": int(value_mask.sum()),
            "value_rate": float(value_share[value_mask].mean()) if value_mask.any() else None,
            "strict_rate": float(strict_share[mask].mean()) if mask.any() else None,
        }

    report = rates(np.ones(len(benchmark), dtype=bool))
    report["strata"] = {
        name: rates(benchmark.strata == name) for name, _, _ in STRATA if name in benchmark.strata
    }
    return report


def _balanced_pick(outcome: np.ndarray, quota: int, rng: np.random.Generator) -> list[int]:
    """Up to quota indices, split evenly across outcomes, topped up from the fullest."""
    groups = {
        value: list(rng.permutation(np.flatnonzero(outcome == value))) for value in (-1, 0, 1)
    }
    picked: list[int] = []
    share = quota // 3
    for group in groups.values():
        picked.extend(int(i) for i in group[:share])
        del group[:share]
    leftovers = [int(i) for group in groups.values() for i in group]
    rng.shuffle(leftovers)
    picked.extend(leftovers[: quota - len(picked)])
    return picked


def filter_contested(scores: np.ndarray) -> np.ndarray:
    """Keeps positions where the solver separates the legal moves at all."""
    legal = scores != ILLEGAL
    high = np.where(legal, scores, -100).max(axis=1)
    low = np.where(legal, scores, 100).min(axis=1)
    return high > low


def build_stratum(
    candidates: Iterable[str],
    quota: int,
    threads: int,
    rng: np.random.Generator,
    log2_entries: int = 23,
) -> tuple[list[str], np.ndarray]:
    """Solves candidates exactly, drops those with no wrong move, balances across outcomes.

    A weak (result-only) screen would be cheaper, but it sees every move in a lost position
    as equal and so would discard all lost positions before the balancing step.
    """
    moves = list(candidates)
    boards = [Board.from_moves(text) for text in moves]
    exact = analyze_many(boards, threads=threads, weak=False, log2_entries=log2_entries)
    keep = np.flatnonzero(filter_contested(exact))
    outcome = np.sign(np.where(exact[keep] != ILLEGAL, exact[keep], -100).max(axis=1))
    picked = keep[_balanced_pick(outcome, quota, rng)]
    return [moves[i] for i in picked], np.asarray(exact[picked], dtype=np.int32)
