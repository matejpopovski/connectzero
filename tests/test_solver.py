"""The solver is the ground truth of this project, so it gets the hardest checks."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import random_lines
from connectzero._engine import ILLEGAL, Board, Solver, analyze_many, solve_many
from reference import brute_force_score

REPO = Path(__file__).parent.parent
TROMP_SAMPLE = Path(__file__).parent / "data" / "tromp_8ply_sample.csv"


@pytest.fixture(scope="module")
def solver() -> Solver:
    return Solver(22)


def line_to_columns(line: str) -> tuple[int, ...]:
    return tuple(int(c) - 1 for c in line)


@pytest.mark.parametrize("plies", [38, 36, 34, 32])
def test_exact_score_matches_brute_force_minimax(plies: int, solver: Solver, rng) -> None:
    for line in random_lines(rng, plies, 12):
        assert solver.solve(Board.from_moves(line)) == brute_force_score(line_to_columns(line))


def test_analysis_matches_brute_force_per_move(solver: Solver, rng) -> None:
    for line in random_lines(rng, 33, 12):
        board = Board.from_moves(line)
        scores = solver.analyze(board)
        for col in range(7):
            if not board.can_play(col):
                assert scores[col] == ILLEGAL
            elif board.is_winning_move(col):
                assert scores[col] == (43 - board.moves) // 2
            else:
                child = line_to_columns(line) + (col,)
                assert scores[col] == -brute_force_score(child)


@pytest.mark.parametrize("plies", [14, 18, 22, 26])
def test_position_score_is_the_best_move_score(plies: int, solver: Solver, rng) -> None:
    for line in random_lines(rng, plies, 8):
        board = Board.from_moves(line)
        scores = [s for s in solver.analyze(board) if s != ILLEGAL]
        assert max(scores) == solver.solve(board)


def test_mirrored_positions_score_the_same(solver: Solver, rng) -> None:
    for line in random_lines(rng, 16, 10):
        board = Board.from_moves(line)
        assert solver.solve(board) == solver.solve(board.mirrored())
        assert solver.analyze(board) == solver.analyze(board.mirrored())[::-1]


def test_weak_solve_gives_the_sign_of_the_exact_score(solver: Solver, rng) -> None:
    for line in random_lines(rng, 12, 8):
        board = Board.from_moves(line)
        assert solver.solve(board, weak=True) == np.sign(solver.solve(board))


def test_weak_result_stays_in_range_for_large_wins() -> None:
    # Regression, found by the Tromp cross-check: this 8-ply win scores +16, and a fail-soft
    # weak search used to return that bound instead of +1.
    board = board_from_cells("bbbbbbbbbbbbxxbbbbxoxobbbbbbbboobbbbbbbbbb")
    assert Solver(22).solve(board) == 16
    assert Solver(22).solve(board, weak=True) == 1


def test_a_tiny_table_changes_speed_but_not_answers(rng) -> None:
    tiny = Solver(10)
    large = Solver(22)
    for line in random_lines(rng, 22, 6):
        board = Board.from_moves(line)
        assert tiny.solve(board) == large.solve(board)
    assert tiny.nodes > large.nodes


def test_immediate_win_is_scored_by_stones_left() -> None:
    board = Board.from_moves("121212")
    scores = Solver(16).analyze(board)
    assert board.is_winning_move(0)
    assert scores[0] == (43 - 6) // 2


def test_node_budget_aborts_and_leaves_the_solver_usable() -> None:
    solver = Solver(20)
    board = Board.from_moves("4444")
    assert solver.solve(board, budget=1000) is None
    assert solver.analyze(board, budget=1000) is None
    late = Board.from_moves(random_lines(np.random.default_rng(3), 26, 1)[0])
    assert solver.solve(late) == Solver(20).solve(late)


def test_threaded_solving_agrees_with_serial(solver: Solver, rng) -> None:
    boards = [Board.from_moves(line) for line in random_lines(rng, 20, 16)]
    scores = solve_many(boards, threads=4)
    analyses = analyze_many(boards, threads=4)
    for board, score, analysis in zip(boards, scores, analyses, strict=True):
        assert score == solver.solve(board)
        assert list(analysis) == solver.analyze(board)


def test_opening_record_matches_the_published_solution() -> None:
    """Allis (1988): the centre opening wins, its neighbours draw, the rest lose."""
    recorded = json.loads((REPO / "results" / "solver_opening.json").read_text())
    scores = recorded["first_move_scores"]
    assert [int(np.sign(s)) for s in scores] == [-1, -1, 0, 1, 0, -1, -1]
    assert scores == scores[::-1]


@pytest.mark.slow
def test_recompute_opening_from_scratch() -> None:
    recorded = json.loads((REPO / "results" / "solver_opening.json").read_text())
    assert Solver(25).analyze(Board()) == recorded["first_move_scores"]


def tromp_rows() -> list[tuple[str, str]]:
    with TROMP_SAMPLE.open() as handle:
        return [(row["cells"], row["outcome"]) for row in csv.DictReader(handle)]


def board_from_cells(cells: str) -> Board:
    """Tromp's 42 cells run a1..a6, b1..b6 and so on, with x moving first."""
    current = occupied = 0
    for index, cell in enumerate(cells):
        col, row = divmod(index, 6)
        mask = 1 << (col * 7 + row)
        if cell != "b":
            occupied |= mask
        if cell == "x":
            current |= mask
    return Board(current, occupied)


def test_results_match_tromps_8_ply_database() -> None:
    """Third-party ground truth: x is to move after 8 plies, so labels are for the mover."""
    rows = tromp_rows()
    assert len(rows) >= 50
    boards = [board_from_cells(cells) for cells, _ in rows]
    assert all(board.moves == 8 for board in boards)
    results = solve_many(boards, threads=2, weak=True, log2_entries=21)
    expected = [{"win": 1, "draw": 0, "loss": -1}[outcome] for _, outcome in rows]
    assert results == expected
