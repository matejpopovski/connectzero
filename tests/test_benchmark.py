"""The grading file and the grading arithmetic."""

from pathlib import Path

import numpy as np
import pytest
from connectzero._engine import ILLEGAL, Board, Solver

from connectzero.benchmark import (
    STRATA,
    Benchmark,
    expected_random_grade,
    grade,
    load_benchmark,
    save_benchmark,
    stratum_of,
)

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def benchmark() -> Benchmark:
    return load_benchmark(REPO / "data" / "benchmark.csv")


def toy() -> Benchmark:
    scores = np.array(
        [
            [2, 1, -1, ILLEGAL, -3, -3, -3],  # won: two moves keep the win, one is fastest
            [0, -1, -2, -2, -2, -2, -2],  # drawn: one move keeps the draw
            [-1, -1, -3, -3, -3, -3, -3],  # lost: no value to keep
        ],
        dtype=np.int32,
    )
    return Benchmark(["1", "2", "3"], np.array(["early", "early", "late"]), scores)


def test_value_and_strict_grades_on_a_hand_worked_table() -> None:
    report = grade(toy(), np.array([1, 0, 0]))
    assert report["strict_rate"] == pytest.approx(2 / 3)
    assert report["value_positions"] == 2
    assert report["value_rate"] == pytest.approx(1.0)
    report = grade(toy(), np.array([2, 1, 2]))
    assert report["strict_rate"] == 0.0
    assert report["value_rate"] == 0.0


def test_random_floor_on_a_hand_worked_table() -> None:
    report = expected_random_grade(toy())
    assert report["value_rate"] == pytest.approx((2 / 6 + 1 / 7) / 2)
    assert report["strict_rate"] == pytest.approx((1 / 6 + 1 / 7 + 2 / 7) / 3)


def test_csv_round_trip(tmp_path: Path) -> None:
    original = toy()
    save_benchmark(original, tmp_path / "b.csv")
    copy = load_benchmark(tmp_path / "b.csv")
    assert copy.moves == original.moves
    np.testing.assert_array_equal(copy.scores, original.scores)
    np.testing.assert_array_equal(copy.strata, original.strata)


def test_strata_boundaries() -> None:
    assert stratum_of(4) == "opening"
    assert stratum_of(13) == "early"
    assert stratum_of(39) == "final"
    with pytest.raises(ValueError):
        stratum_of(41)


def test_file_is_stratified_and_every_position_is_open(benchmark: Benchmark) -> None:
    for name, low, high in STRATA:
        members = [m for m, s in zip(benchmark.moves, benchmark.strata, strict=True) if s == name]
        assert len(members) >= 100
        assert all(low <= len(m) <= high for m in members)
    keys = set()
    for board in benchmark.boards():
        assert not board.is_over()
        keys.add(board.canonical_key())
    assert len(keys) == len(benchmark)


def test_every_position_has_a_wrong_move(benchmark: Benchmark) -> None:
    legal_scores = np.where(benchmark.legal, benchmark.scores, 99)
    assert (legal_scores.min(axis=1) < benchmark.best).all()


def test_each_outcome_is_represented_in_each_stratum(benchmark: Benchmark) -> None:
    for name, _, _ in STRATA:
        outcomes = set(benchmark.outcome[benchmark.strata == name].tolist())
        assert {-1, 1} <= outcomes


def test_recorded_scores_agree_with_a_fresh_solve(benchmark: Benchmark) -> None:
    solver = Solver(21)
    late = [i for i, s in enumerate(benchmark.strata) if s in ("endgame", "final")]
    for row in late[:: max(1, len(late) // 40)]:
        board = Board.from_moves(benchmark.moves[row])
        assert solver.analyze(board) == benchmark.scores[row].tolist()


def test_random_grade_is_a_floor_for_the_perfect_chooser(benchmark: Benchmark) -> None:
    chosen = np.where(benchmark.legal, benchmark.scores, -100).argmax(axis=1)
    perfect = grade(benchmark, chosen)
    floor = expected_random_grade(benchmark)
    assert perfect["strict_rate"] == 1.0
    assert perfect["value_rate"] == 1.0
    assert floor["strict_rate"] < 0.6
