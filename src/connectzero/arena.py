"""Head-to-head matches: gating, baselines, and the final tournament."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from connectzero._engine import GamePool, GameResult, SearchParams, Seat, TranspositionTable
from connectzero.search import Evaluate, drive


@dataclass(frozen=True)
class MatchScore:
    """Result from the point of view of seat A. Colours alternate game by game."""

    games: int
    wins: int
    draws: int
    losses: int
    first_wins: int
    first_games: int
    second_wins: int
    second_games: int

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games

    def wilson_interval(self, z: float = 1.96) -> tuple[float, float]:
        """95% Wilson interval on the score, counting a draw as half a win."""
        n, p = self.games, self.score
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, centre - half), min(1.0, centre + half)

    def as_dict(self) -> dict:
        low, high = self.wilson_interval()
        return asdict(self) | {"score": self.score, "wilson_low": low, "wilson_high": high}


def summarize(results: Sequence[GameResult]) -> MatchScore:
    wins = sum(r.winner_seat == 0 for r in results)
    losses = sum(r.winner_seat == 1 for r in results)
    first = [r for r in results if r.first_seat == 0]
    second = [r for r in results if r.first_seat == 1]
    return MatchScore(
        games=len(results),
        wins=wins,
        draws=len(results) - wins - losses,
        losses=losses,
        first_wins=sum(r.winner_seat == 0 for r in first),
        first_games=len(first),
        second_wins=sum(r.winner_seat == 0 for r in second),
        second_games=len(second),
    )


def play_match(
    seat_a: Seat,
    seat_b: Seat,
    evaluators: Sequence[Evaluate],
    games: int,
    c_puct: float,
    seed: int,
    threads: int = 4,
    concurrency: int = 64,
    leaves_per_tree: int = 8,
    temperature_plies: int = 0,
    openings: Sequence[str] = (),
    solver_table: TranspositionTable | None = None,
) -> tuple[MatchScore, list[GameResult]]:
    pool = GamePool(
        seat_a,
        seat_b,
        num_games=games,
        concurrency=concurrency,
        leaves_per_tree=leaves_per_tree,
        threads=threads,
        seed=seed,
        search=SearchParams(c_puct=c_puct),
        temperature_plies=temperature_plies,
        openings=list(openings),
        solver_table=solver_table,
    )
    drive(pool, evaluators)
    results = list(pool.results)
    return summarize(results), results
