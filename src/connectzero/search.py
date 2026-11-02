"""Drives the C++ batched search with Python-side networks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from connectzero._engine import Board, GamePool, PositionSearch, SearchParams

Evaluate = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def drive(engine: GamePool | PositionSearch, evaluators: Sequence[Evaluate]) -> int:
    """Runs the collect/apply loop until the engine asks for nothing. Returns leaves evaluated."""
    evaluated = 0
    while True:
        planes, owners = engine.collect()
        rows = len(owners)
        if rows == 0:
            return evaluated
        evaluated += rows
        if len(evaluators) == 1:
            policy, values = evaluators[0](planes)
        else:
            policy, values = _split_by_owner(planes, owners, evaluators)
        engine.apply(policy, values, rows)


def _split_by_owner(
    planes: np.ndarray, owners: np.ndarray, evaluators: Sequence[Evaluate]
) -> tuple[np.ndarray, np.ndarray]:
    policy = np.empty((len(owners), 7), dtype=np.float32)
    values = np.empty(len(owners), dtype=np.float32)
    for owner, evaluate in enumerate(evaluators):
        rows = np.flatnonzero(owners == owner)
        if rows.size:
            policy[rows], values[rows] = evaluate(planes[rows])
    return policy, values


def search_positions(
    boards: Sequence[Board],
    evaluate: Evaluate,
    simulations: int,
    c_puct: float,
    threads: int = 4,
    leaves_per_tree: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Visit counts (n, 7) and root values (n,) after a noise-free search of each position."""
    engine = PositionSearch(
        list(boards), SearchParams(c_puct=c_puct), simulations, leaves_per_tree, threads
    )
    drive(engine, [evaluate])
    visits = np.array([engine.visits(i) for i in range(len(boards))], dtype=np.int32)
    values = np.array([engine.value(i) for i in range(len(boards))], dtype=np.float32)
    return visits, values
