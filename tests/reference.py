"""A deliberately naive Connect 4 written against a plain grid.

It shares no code or technique with the C++ engine, so agreement between the two is evidence
about the rules rather than about one implementation agreeing with itself.
"""

from __future__ import annotations

from functools import cache

WIDTH = 7
HEIGHT = 6
CELLS = WIDTH * HEIGHT


class Grid:
    def __init__(self) -> None:
        self.columns: list[list[int]] = [[] for _ in range(WIDTH)]
        self.turn = 0
        self.winner: int | None = None

    @property
    def moves(self) -> int:
        return sum(len(column) for column in self.columns)

    def legal(self) -> list[int]:
        if self.winner is not None:
            return []
        return [c for c in range(WIDTH) if len(self.columns[c]) < HEIGHT]

    def cell(self, col: int, row: int) -> int | None:
        if 0 <= col < WIDTH and 0 <= row < len(self.columns[col]):
            return self.columns[col][row]
        return None

    def wins_at(self, col: int, row: int, player: int) -> bool:
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            run = 1
            for sign in (1, -1):
                x, y = col + sign * dx, row + sign * dy
                while self.cell(x, y) == player:
                    run += 1
                    x += sign * dx
                    y += sign * dy
            if run >= 4:
                return True
        return False

    def would_win(self, col: int) -> bool:
        row = len(self.columns[col])
        self.columns[col].append(self.turn)
        result = self.wins_at(col, row, self.turn)
        self.columns[col].pop()
        return result

    def play(self, col: int) -> None:
        row = len(self.columns[col])
        self.columns[col].append(self.turn)
        if self.wins_at(col, row, self.turn):
            self.winner = self.turn
        self.turn = 1 - self.turn

    def is_over(self) -> bool:
        return self.winner is not None or self.moves == CELLS

    def stones(self, player: int) -> set[tuple[int, int]]:
        return {
            (col, row)
            for col in range(WIDTH)
            for row, owner in enumerate(self.columns[col])
            if owner == player
        }


def brute_force_score(line: tuple[int, ...]) -> int:
    """Exact solver-convention score by full minimax over a move list. Use near game end."""

    @cache
    def score_of(state: tuple[tuple[int, ...], ...], turn: int) -> int:
        grid = Grid()
        grid.columns = [list(column) for column in state]
        grid.turn = turn
        moves = grid.moves
        if moves == CELLS:
            return 0
        best = -100
        for col in range(WIDTH):
            if len(grid.columns[col]) == HEIGHT:
                continue
            if grid.would_win(col):
                return (CELLS + 1 - moves) // 2
        for col in range(WIDTH):
            if len(grid.columns[col]) == HEIGHT:
                continue
            grid.columns[col].append(turn)
            child = tuple(tuple(column) for column in grid.columns)
            grid.columns[col].pop()
            best = max(best, -score_of(child, 1 - turn))
        return best

    grid = Grid()
    for col in line:
        grid.play(col)
    return score_of(tuple(tuple(column) for column in grid.columns), grid.turn)
