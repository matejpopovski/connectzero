"""Bitboard rules against an independent grid implementation and published counts."""

import numpy as np
import pytest
from conftest import random_lines
from connectzero._engine import Board, encode_batch, has_four
from reference import CELLS, HEIGHT, WIDTH, Grid

# OEIS A212693 (John Tromp): legal 7x6 positions after n plies.
TROMP_POSITION_COUNTS = [1, 7, 49, 238, 1120, 4263, 16422, 54859, 184275]


def bit(col: int, row: int) -> int:
    return 1 << (col * (HEIGHT + 1) + row)


def stones_to_bits(cells: set[tuple[int, int]]) -> int:
    return sum(bit(col, row) for col, row in cells)


def line_cells(col: int, row: int, dx: int, dy: int) -> list[tuple[int, int]]:
    return [(col + i * dx, row + i * dy) for i in range(4)]


def all_lines() -> list[list[tuple[int, int]]]:
    lines = []
    for col in range(WIDTH):
        for row in range(HEIGHT):
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = line_cells(col, row, dx, dy)
                if all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in cells):
                    lines.append(cells)
    return lines


def test_board_has_the_69_known_lines() -> None:
    assert len(all_lines()) == 69


@pytest.mark.parametrize("cells", all_lines())
def test_every_line_is_a_win(cells: list[tuple[int, int]]) -> None:
    assert has_four(stones_to_bits(set(cells)))


@pytest.mark.parametrize("cells", all_lines())
def test_a_line_missing_any_stone_is_not_a_win(cells: list[tuple[int, int]]) -> None:
    for missing in cells:
        assert not has_four(stones_to_bits(set(cells) - {missing}))


def test_runs_do_not_wrap_between_columns() -> None:
    # Top of one column and bottom of the next are adjacent in bit index only.
    wrap_vertical = {(0, 4), (0, 5), (1, 0), (1, 1)}
    wrap_diagonal = {(0, 2), (1, 3), (2, 4), (3, 5), (4, 0)}
    assert not has_four(stones_to_bits(wrap_vertical))
    assert not has_four(stones_to_bits(wrap_diagonal - {(3, 5)}))


@pytest.mark.parametrize("seed", range(40))
def test_random_games_match_the_grid_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    board = Board()
    grid = Grid()
    for _ in range(CELLS):
        assert board.moves == grid.moves
        assert board.is_over() == grid.is_over()
        assert board.legal_moves() == grid.legal()
        if grid.is_over():
            break
        for col in grid.legal():
            assert board.is_winning_move(col) == grid.would_win(col)
        assert board.can_win_next() == any(grid.would_win(c) for c in grid.legal())
        assert board.current == stones_to_bits(grid.stones(grid.turn))
        assert board.previous == stones_to_bits(grid.stones(1 - grid.turn))
        col = int(rng.choice(grid.legal()))
        board = board.play(col)
        grid.play(col)
    assert board.last_move_won() == (grid.winner is not None)


def test_position_counts_match_tromp() -> None:
    frontier = {Board()}
    for ply, expected in enumerate(TROMP_POSITION_COUNTS):
        assert len(frontier) == expected, f"ply {ply}"
        if ply + 1 == len(TROMP_POSITION_COUNTS):
            break
        frontier = {b.play(col) for b in frontier if not b.is_over() for col in b.legal_moves()}


def test_key_is_unique_up_to_ply_six() -> None:
    frontier = {Board()}
    for _ in range(6):
        frontier = {b.play(col) for b in frontier for col in b.legal_moves()}
        assert len({b.key() for b in frontier}) == len(frontier)


def test_transposed_move_orders_reach_the_same_key() -> None:
    assert Board.from_moves("1234").key() == Board.from_moves("3214").key()
    assert Board.from_moves("1234").key() != Board.from_moves("1243").key()


@pytest.mark.parametrize("plies", [3, 11, 19, 27])
def test_non_losing_columns_are_exactly_the_safe_ones(plies: int, rng) -> None:
    for line in random_lines(rng, plies, 40):
        board = Board.from_moves(line)
        if board.can_win_next():
            continue
        safe = []
        for col in board.legal_moves():
            child = board.play(col)
            if not child.can_win_next():
                safe.append(col)
        assert board.non_losing_columns() == safe


def test_mirror_is_an_involution_and_preserves_rules(rng) -> None:
    for line in random_lines(rng, 15, 60):
        board = Board.from_moves(line)
        flipped = board.mirrored()
        assert flipped.mirrored() == board
        assert flipped == Board.from_moves("".join(str(8 - int(c)) for c in line))
        for col in board.legal_moves():
            assert board.is_winning_move(col) == flipped.is_winning_move(6 - col)
        assert board.canonical_key() == flipped.canonical_key()


def test_bad_move_strings_are_rejected() -> None:
    with pytest.raises(ValueError):
        Board.from_moves("8")
    with pytest.raises(ValueError):
        Board.from_moves("1111111")
    with pytest.raises(ValueError):
        Board.from_moves("12121212")


def decode_planes(planes: np.ndarray) -> Board:
    """Inverse of the network encoding, written from the plane layout alone."""
    current = occupied = 0
    for row in range(HEIGHT):
        for col in range(WIDTH):
            mine = planes[0, row, col] == 1.0
            other = planes[1, row, col] == 1.0
            assert not (mine and other)
            if mine:
                current |= bit(col, row)
            if mine or other:
                occupied |= bit(col, row)
    return Board(current, occupied)


def test_planes_round_trip(rng) -> None:
    for plies in (0, 1, 2, 9, 20, 33):
        for line in random_lines(rng, plies, 12):
            board = Board.from_moves(line)
            assert decode_planes(board.planes()) == board


def test_planes_put_row_zero_at_the_bottom_for_the_side_to_move() -> None:
    planes = Board.from_moves("4").planes()
    assert planes.shape == (3, 6, 7)
    assert planes[0].sum() == 0
    assert planes[1, 0, 3] == 1.0
    assert planes[1].sum() == 1


def test_parity_plane_tracks_who_moved_first() -> None:
    assert (Board().planes()[2] == 1.0).all()
    assert (Board.from_moves("4").planes()[2] == 0.0).all()
    assert (Board.from_moves("44").planes()[2] == 1.0).all()


def test_batch_encoding_matches_single_encoding(rng) -> None:
    boards = [Board.from_moves(line) for line in random_lines(rng, 13, 16)]
    current = np.array([b.current for b in boards], dtype=np.uint64)
    occupied = np.array([b.occupied for b in boards], dtype=np.uint64)
    batch = encode_batch(current, occupied)
    assert batch.shape == (16, 3, 6, 7)
    for row, board in enumerate(boards):
        np.testing.assert_array_equal(batch[row], board.planes())


def test_mirrored_planes_are_flipped_planes(rng) -> None:
    for line in random_lines(rng, 12, 12):
        board = Board.from_moves(line)
        np.testing.assert_array_equal(board.mirrored().planes(), board.planes()[:, :, ::-1])
