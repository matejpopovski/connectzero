"""Baselines must be what the report says they are, or the win rates mean nothing."""

import numpy as np
from conftest import random_lines
from connectzero._engine import (
    Board,
    Rng,
    Seat,
    Solver,
    TranspositionTable,
    alphabeta_move,
    analyze_many,
    perfect_move,
    random_move,
    rollout_mcts_move,
    tactical_move,
)

from connectzero.arena import play_match


def test_tactical_takes_a_win() -> None:
    assert tactical_move(Board.from_moves("121212"), Rng(1)) == 0


def test_tactical_blocks_a_loss() -> None:
    board = Board.from_moves("12131")
    for seed in range(20):
        assert tactical_move(board, Rng(seed)) == 0


def test_tactical_never_plays_under_an_opposing_win(rng) -> None:
    gen = Rng(9)
    for line in random_lines(rng, 21, 60):
        board = Board.from_moves(line)
        safe = board.non_losing_columns()
        if safe and not board.can_win_next():
            assert tactical_move(board, gen) in safe


def test_random_is_uniform_over_legal_columns() -> None:
    board = Board.from_moves("444444")
    gen = Rng(3)
    counts = np.bincount([random_move(board, gen) for _ in range(6000)], minlength=7)
    assert counts[3] == 0
    assert counts[[0, 1, 2, 4, 5, 6]].min() > 800


def test_alphabeta_finds_a_forced_win_inside_its_horizon(rng) -> None:
    solver = Solver(20)
    tested = 0
    for line in random_lines(rng, 28, 300):
        board = Board.from_moves(line)
        if board.can_win_next() or solver.solve(board) < (42 - board.moves) // 2 - 1:
            continue
        move = alphabeta_move(board, 7, Rng(1))
        assert solver.analyze(board)[move] > 0
        tested += 1
    assert tested >= 3


def test_perfect_move_only_chooses_top_scores() -> None:
    gen = Rng(2)
    scores = [-1, 3, 127, 3, 0, -2, 1]
    assert {perfect_move(scores, gen) for _ in range(50)} == {1, 3}


def test_rollout_search_finds_a_win() -> None:
    assert rollout_mcts_move(Board.from_moves("121212"), 200, 1.5, Rng(1)) == 0


def test_strength_order_tactical_beats_random() -> None:
    score, _ = play_match(Seat.tactical(), Seat.random(), [], 200, 1.5, seed=1, threads=2)
    assert score.score > 0.8


def test_strength_order_rollout_beats_tactical() -> None:
    score, _ = play_match(Seat.rollout(500), Seat.tactical(), [], 60, 1.5, seed=1, threads=2)
    assert score.score > 0.7


def test_perfect_player_is_never_beaten_from_a_drawn_or_won_seat() -> None:
    """Starting late keeps this fast: the perfect seat must not lose a theoretical non-loss."""
    rng = np.random.default_rng(5)
    openings = []
    solver = Solver(20)
    for line in random_lines(rng, 18, 60):
        if solver.solve(Board.from_moves(line)) == 0:
            openings.append(line)
    assert openings
    score, _ = play_match(
        Seat.perfect(), Seat.tactical(), [], len(openings) * 2, 1.5, seed=1, openings=openings[:30]
    )
    assert score.losses == 0


def test_a_warmed_table_is_reused_by_later_solvers() -> None:
    table = TranspositionTable(22)
    boards = [Board.from_moves(line) for line in random_lines(np.random.default_rng(4), 12, 4)]
    first = analyze_many(boards, threads=2, table=table)
    warm = Solver(table)
    assert [warm.analyze(b) for b in boards] == first.tolist()
    cold = Solver(22)
    for board in boards:
        cold.analyze(board)
    assert warm.nodes < cold.nodes / 2
