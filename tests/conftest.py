import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from connectzero._engine import Board  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260211)


def random_line(rng: np.random.Generator, plies: int) -> str | None:
    """A random move string reaching `plies` with the game still open, or None if it ended."""
    board = Board()
    text = ""
    for _ in range(plies):
        legal = [c for c in board.legal_moves() if not board.is_winning_move(c)]
        if not legal:
            return None
        col = int(rng.choice(legal))
        board = board.play(col)
        text += str(col + 1)
    return None if board.is_over() else text


def random_lines(rng: np.random.Generator, plies: int, count: int) -> list[str]:
    lines: list[str] = []
    while len(lines) < count:
        line = random_line(rng, plies)
        if line is not None:
            lines.append(line)
    return lines
