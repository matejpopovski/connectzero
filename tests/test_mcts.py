"""Search invariants: statistics must add up however the batches fall."""

import numpy as np
import pytest
from conftest import random_lines
from connectzero._engine import (
    Board,
    GamePool,
    Leaf,
    PositionSearch,
    Rng,
    SearchParams,
    Seat,
    Selection,
    Solver,
    Tree,
)

from connectzero.net import UniformEvaluator
from connectzero.search import drive, search_positions

UNIFORM = [1 / 7] * 7


def run_sequential(tree: Tree, simulations: int, policy=UNIFORM, value: float = 0.0) -> None:
    leaf = Leaf()
    while tree.root_visits < simulations:
        if tree.select(leaf) == Selection.NEEDS_EVALUATION:
            tree.expand(leaf, policy, value)


def run_batched(tree: Tree, simulations: int, batch: int) -> int:
    """Selects up to `batch` leaves before evaluating any of them. Returns collisions seen."""
    collisions = 0
    while tree.root_visits < simulations:
        pending = []
        while len(pending) < batch and tree.root_visits + len(pending) < simulations:
            leaf = Leaf()
            outcome = tree.select(leaf)
            if outcome == Selection.NEEDS_EVALUATION:
                pending.append(leaf)
            elif outcome == Selection.COLLISION:
                collisions += 1
                break
        for leaf in pending:
            tree.expand(leaf, UNIFORM, 0.0)
    return collisions


def assert_consistent(tree: Tree) -> None:
    """Every node's visits equal one for itself plus its children's visits."""
    nodes = tree.nodes
    for node in nodes:
        assert node.inflight == 0
        if node.first_child >= 0:
            children = nodes[node.first_child : node.first_child + node.num_children]
            assert node.visits == sum(c.visits for c in children) + 1
            assert sum(c.prior for c in children) == pytest.approx(1.0, abs=1e-4)
            assert abs(node.value_sum) <= node.visits + 1e-3


@pytest.mark.parametrize("line", ["", "4", "4433", "1111122222"])
@pytest.mark.parametrize("simulations", [1, 2, 7, 64, 400])
def test_visits_are_conserved(line: str, simulations: int) -> None:
    tree = Tree()
    tree.reset(Board.from_moves(line))
    run_sequential(tree, simulations)
    assert tree.root_visits == simulations
    if simulations > 1:
        assert sum(tree.child_visits) == simulations - 1
    assert_consistent(tree)


@pytest.mark.parametrize("batch", [2, 8, 32])
def test_batched_selection_conserves_visits(batch: int) -> None:
    tree = Tree()
    tree.reset(Board.from_moves("4433"))
    run_batched(tree, 500, batch)
    assert tree.root_visits == 500
    assert_consistent(tree)


def test_leaves_in_one_batch_are_distinct() -> None:
    tree = Tree()
    tree.reset(Board())
    run_sequential(tree, 30)
    keys = []
    pending = []
    for _ in range(12):
        leaf = Leaf()
        if tree.select(leaf) != Selection.NEEDS_EVALUATION:
            break
        pending.append(leaf)
        keys.append(leaf.board.key())
    assert len(pending) >= 8
    assert len(set(keys)) == len(keys)
    for item in pending:
        tree.expand(item, UNIFORM, 0.0)
    assert_consistent(tree)


def test_collision_leaves_no_inflight_residue() -> None:
    tree = Tree()
    tree.reset(Board())
    first = Leaf()
    assert tree.select(first) == Selection.NEEDS_EVALUATION
    second = Leaf()
    assert tree.select(second) == Selection.COLLISION
    tree.expand(first, UNIFORM, 0.0)
    assert_consistent(tree)


def test_full_columns_receive_no_visits_or_prior() -> None:
    tree = Tree()
    tree.reset(Board.from_moves("444444"))
    run_sequential(tree, 200)
    assert tree.child_visits[3] == 0
    assert tree.child_priors[3] == 0.0
    assert sum(tree.child_priors) == pytest.approx(1.0)
    assert_consistent(tree)


def test_value_signs_alternate_down_the_tree() -> None:
    tree = Tree()
    tree.reset(Board())
    leaf = Leaf()
    tree.select(leaf)
    tree.expand(leaf, UNIFORM, 0.5)  # root is worth +0.5 to the side to move
    assert tree.root_value == pytest.approx(0.5)
    tree.select(leaf)
    tree.expand(leaf, UNIFORM, 0.5)  # the child is worth +0.5 to the reply, so -0.5 to the root
    assert tree.root_value == pytest.approx(0.0)


def test_search_takes_an_immediate_win_without_a_network() -> None:
    board = Board.from_moves("121212")
    assert board.is_winning_move(0)
    tree = Tree()
    tree.reset(board)
    run_sequential(tree, 300)
    assert int(np.argmax(tree.child_visits)) == 0
    assert tree.root_value > 0.5


def test_search_blocks_an_immediate_loss_without_a_network() -> None:
    board = Board.from_moves("12131")
    assert not board.can_win_next()
    assert board.non_losing_columns() == [0]
    tree = Tree()
    tree.reset(board)
    run_sequential(tree, 2000)
    assert int(np.argmax(tree.child_visits)) == 0


def test_search_converges_to_the_solver_on_a_short_endgame() -> None:
    line = random_lines(np.random.default_rng(11), 34, 1)[0]
    board = Board.from_moves(line)
    scores = Solver(18).analyze(board)
    tree = Tree(SearchParams(c_puct=1.0))
    tree.reset(board)
    run_sequential(tree, 30000)
    best = max(s for s in scores if s != 127)
    assert np.sign(scores[int(np.argmax(tree.child_visits))]) == np.sign(best)


def test_terminal_visits_back_up_without_evaluation() -> None:
    tree = Tree()
    tree.reset(Board.from_moves("121212"))
    leaf = Leaf()
    while tree.select(leaf) != Selection.FINISHED:
        tree.expand(leaf, UNIFORM, 0.0)
    visits = tree.root_visits
    wins = tree.child_visits[0]
    assert tree.select(leaf) == Selection.FINISHED
    assert tree.root_visits == visits + 1
    assert tree.child_visits[0] == wins + 1
    assert_consistent(tree)


def test_advance_keeps_the_chosen_subtree_exactly() -> None:
    tree = Tree()
    tree.reset(Board())
    run_sequential(tree, 500)
    visits = tree.child_visits
    column = int(np.argmax(visits))
    before = len(tree.nodes)
    tree.advance(column)
    assert tree.root_visits == visits[column]
    assert tree.root_board == Board().play(column)
    assert len(tree.nodes) < before
    assert_consistent(tree)
    run_sequential(tree, 800)
    assert_consistent(tree)


def test_advance_into_an_unexplored_column_starts_clean() -> None:
    tree = Tree()
    tree.reset(Board())
    tree.advance(2)
    assert tree.root_visits == 0
    assert tree.root_board == Board().play(2)


def test_root_noise_changes_priors_but_keeps_them_a_distribution() -> None:
    tree = Tree(SearchParams(dirichlet_alpha=1.0, dirichlet_fraction=0.25))
    tree.reset(Board.from_moves("444444"))
    run_sequential(tree, 1)
    before = list(tree.child_priors)
    tree.add_root_noise(Rng(5))
    after = list(tree.child_priors)
    assert after != before
    assert after[3] == 0.0
    assert sum(after) == pytest.approx(1.0, abs=1e-5)
    assert min(a for i, a in enumerate(after) if i != 3) >= 0.75 * min(
        b for i, b in enumerate(before) if i != 3
    )


def test_position_search_with_one_leaf_equals_a_sequential_tree() -> None:
    board = Board.from_moves("4436")
    tree = Tree()
    tree.reset(board)
    run_sequential(tree, 300)
    visits, values = search_positions(
        [board], UniformEvaluator(), 300, 1.5, threads=1, leaves_per_tree=1
    )
    assert list(visits[0]) == list(tree.child_visits)
    assert values[0] == pytest.approx(tree.root_value)


def make_pool(**overrides) -> GamePool:
    settings = {
        "seat_a": Seat.network(0, 40),
        "seat_b": Seat.network(0, 40),
        "num_games": 12,
        "concurrency": 5,
        "leaves_per_tree": 4,
        "threads": 3,
        "seed": 7,
        "search": SearchParams(dirichlet_fraction=0.25),
        "temperature_plies": 8,
        "share_tree": True,
        "record_samples": True,
    }
    return GamePool(**(settings | overrides))


def test_every_game_replays_legally_to_its_recorded_result() -> None:
    pool = make_pool()
    drive(pool, [UniformEvaluator()])
    assert pool.finished
    for result in pool.results:
        board = Board()
        for digit in result.moves:
            assert not board.is_over()
            assert board.can_play(int(digit) - 1)
            board = board.play(int(digit) - 1)
        assert board.is_over()
        if result.winner_seat < 0:
            assert board.is_full()
            assert not board.last_move_won()
        else:
            mover_is_first = len(result.moves) % 2 == 1
            winner = result.first_seat if mover_is_first else 1 - result.first_seat
            assert result.winner_seat == winner


def test_samples_cover_every_position_with_correct_targets() -> None:
    pool = make_pool()
    drive(pool, [UniformEvaluator()])
    current, occupied, policy, outcome = pool.drain_samples()
    assert len(current) == sum(len(r.moves) for r in pool.results)
    np.testing.assert_allclose(policy.sum(axis=1), 1.0, atol=1e-5)
    row = 0
    for result in pool.results:
        board = Board()
        for ply, digit in enumerate(result.moves):
            assert current[row] == board.current
            assert occupied[row] == board.occupied
            for col in range(7):
                if not board.can_play(col):
                    assert policy[row, col] == 0.0
            plies_left = len(result.moves) - ply
            if result.winner_seat < 0:
                assert outcome[row] == 0.0
            else:
                assert outcome[row] == (1.0 if plies_left % 2 == 1 else -1.0)
            board = board.play(int(digit) - 1)
            row += 1


def test_games_are_reproducible_regardless_of_thread_count() -> None:
    games = []
    for threads in (1, 4):
        pool = make_pool(threads=threads, concurrency=12)
        drive(pool, [UniformEvaluator()])
        games.append([r.moves for r in pool.results])
    assert games[0] == games[1]


def test_batches_are_larger_when_more_trees_run_at_once() -> None:
    wide = make_pool(concurrency=12)
    narrow = make_pool(concurrency=1, leaves_per_tree=1)
    drive(wide, [UniformEvaluator()])
    drive(narrow, [UniformEvaluator()])
    assert wide.evaluations / wide.batches > 8 * narrow.evaluations / narrow.batches


def test_colours_alternate_by_game_index() -> None:
    pool = make_pool(seat_a=Seat.random(), seat_b=Seat.random(), share_tree=False)
    drive(pool, [])
    assert [r.first_seat for r in pool.results] == [i % 2 for i in range(12)]


def test_two_evaluators_serve_their_own_seats() -> None:
    calls = {"a": 0, "b": 0}

    def counted(name):
        evaluator = UniformEvaluator()

        def run(planes):
            calls[name] += len(planes)
            return evaluator(planes)

        return run

    pool = make_pool(
        seat_a=Seat.network(0, 20),
        seat_b=Seat.network(1, 20),
        share_tree=False,
        record_samples=False,
        search=SearchParams(),
    )
    drive(pool, [counted("a"), counted("b")])
    assert calls["a"] > 0
    assert calls["b"] > 0
    assert calls["a"] + calls["b"] == pool.evaluations


def test_openings_are_respected_and_seat_turns_follow_them() -> None:
    pool = make_pool(
        seat_a=Seat.tactical(),
        seat_b=Seat.random(),
        share_tree=False,
        record_samples=False,
        openings=["445", "1267"],
    )
    drive(pool, [])
    assert all(
        r.moves.startswith("445" if i % 2 == 0 else "1267") for i, r in enumerate(pool.results)
    )


def test_an_opening_that_ends_the_game_is_refused() -> None:
    with pytest.raises(ValueError):
        make_pool(openings=["1212121"])


def test_position_search_finishes_on_terminal_heavy_trees() -> None:
    board = Board.from_moves("44444455555")
    engine = PositionSearch([board], SearchParams(), 300, 8, 1)
    assert drive(engine, [UniformEvaluator()]) > 0
    assert sum(engine.visits(0)) == 299
