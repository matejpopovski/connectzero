"""Learning machinery: augmentation, persistence, and the loop on a tiny budget."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import random_lines
from connectzero._engine import Board, Seat

from connectzero import supervise as supervise_module
from connectzero.arena import play_match, summarize
from connectzero.benchmark import load_benchmark
from connectzero.checkpoint import atomic_save, load, load_weights
from connectzero.net import Evaluator, NetConfig, PolicyValueNet
from connectzero.replay import ReplayBuffer
from connectzero.train import TrainConfig, Trainer, learning_rate, optimise

REPO = Path(__file__).parent.parent
TINY = TrainConfig(
    blocks=1,
    filters=8,
    games_per_generation=6,
    simulations=12,
    concurrency=6,
    leaves_per_tree=2,
    mcts_threads=2,
    replay_capacity=2000,
    batch_size=16,
    arena_games=4,
    arena_simulations=8,
    grade_simulations=8,
)


def test_network_shapes_and_value_range() -> None:
    net = PolicyValueNet(NetConfig(blocks=1, filters=8)).eval()
    logits, value = net(torch.rand(5, 3, 6, 7))
    assert logits.shape == (5, 7)
    assert value.shape == (5,)
    assert ((value >= -1) & (value <= 1)).all()


def test_evaluator_chunks_match_a_single_batch() -> None:
    net = PolicyValueNet(NetConfig(blocks=1, filters=8)).eval()
    planes = np.random.default_rng(0).random((10, 3, 6, 7)).astype(np.float32)
    whole_policy, whole_value = Evaluator(net, max_batch=10)(planes)
    split_policy, split_value = Evaluator(net, max_batch=3)(planes)
    np.testing.assert_allclose(whole_policy, split_policy, atol=1e-5)
    np.testing.assert_allclose(whole_value, split_value, atol=1e-5)


def test_replay_wraps_and_keeps_only_the_newest() -> None:
    buffer = ReplayBuffer(4)
    for start in (0, 3):
        ids = np.arange(start, start + 3, dtype=np.uint64)
        buffer.add(ids, ids, np.zeros((3, 7), np.float32), ids.astype(np.float32))
    assert buffer.size == 4
    assert sorted(buffer.current.tolist()) == [2, 3, 4, 5]


def test_replay_mirror_flips_board_and_policy_together(rng) -> None:
    board = Board.from_moves("1123")
    buffer = ReplayBuffer(8)
    policy = np.array([[0.7, 0.3, 0, 0, 0, 0, 0]], dtype=np.float32)
    buffer.add(
        np.array([board.current], np.uint64),
        np.array([board.occupied], np.uint64),
        policy,
        np.array([1.0], np.float32),
    )
    planes, targets, _ = buffer.sample(64, rng)
    original = board.planes()
    flipped = board.mirrored().planes()
    seen_flip = False
    for row in range(64):
        if targets[row, 0] > 0:
            np.testing.assert_array_equal(planes[row], original)
        else:
            seen_flip = True
            np.testing.assert_array_equal(planes[row], flipped)
            assert targets[row, 6] == pytest.approx(0.7)
    assert seen_flip


def test_replay_state_round_trip() -> None:
    buffer = ReplayBuffer(6)
    ids = np.arange(9, dtype=np.uint64)
    buffer.add(ids, ids, np.ones((9, 7), np.float32) / 7, np.ones(9, np.float32))
    copy = ReplayBuffer.from_state_dict(buffer.state_dict())
    assert copy.size == buffer.size
    assert copy.cursor == buffer.cursor
    np.testing.assert_array_equal(copy.current, buffer.current)


def test_atomic_save_keeps_the_old_file_when_writing_fails(tmp_path: Path) -> None:
    path = tmp_path / "state.pt"
    atomic_save({"step": 1}, path)

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("disk gone")

    with pytest.raises(RuntimeError):
        atomic_save({"step": 2, "bad": Unpicklable()}, path)
    assert load(path)["step"] == 1
    assert [p.name for p in tmp_path.iterdir()] == ["state.pt"]


def test_weights_file_loads_with_restricted_unpickling(tmp_path: Path) -> None:
    net = PolicyValueNet(NetConfig(blocks=1, filters=8))
    atomic_save({"config": {"blocks": 1, "c_puct": 1.5}, "model": net.state_dict()}, tmp_path / "w")
    assert load_weights(tmp_path / "w")["config"]["blocks"] == 1


def test_checkpoints_hold_cpu_tensors_only(tmp_path: Path) -> None:
    net = PolicyValueNet(NetConfig(blocks=1, filters=8))
    atomic_save({"model": net.state_dict(), "nested": [torch.zeros(2)]}, tmp_path / "a.pt")
    state = load(tmp_path / "a.pt")
    assert all(t.device.type == "cpu" for t in state["model"].values())


def test_training_reduces_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    config = replace(TINY, batch_size=32)
    model = PolicyValueNet(NetConfig(blocks=1, filters=16))
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lines = random_lines(np.random.default_rng(0), 8, 32)
    boards = [Board.from_moves(line) for line in lines]
    buffer = ReplayBuffer(32)
    targets = np.zeros((32, 7), np.float32)
    targets[np.arange(32), np.arange(32) % 7] = 1.0
    buffer.add(
        np.array([b.current for b in boards], np.uint64),
        np.array([b.occupied for b in boards], np.uint64),
        targets,
        np.where(np.arange(32) % 2 == 0, 1.0, -1.0).astype(np.float32),
    )
    rng = np.random.default_rng(1)
    early = optimise(model, optimiser, buffer, 5, config, "cpu", rng)
    optimise(model, optimiser, buffer, 150, config, "cpu", rng)
    late = optimise(model, optimiser, buffer, 5, config, "cpu", rng)
    assert late["policy_loss"] < early["policy_loss"] * 0.8
    assert late["value_loss"] < early["value_loss"] * 0.8


def test_learning_rate_schedule_endpoints() -> None:
    config = TrainConfig()
    assert learning_rate(config, 0.0) == pytest.approx(config.learning_rate)
    assert learning_rate(config, 1.0) == pytest.approx(config.final_learning_rate)
    assert learning_rate(config, 2.0) == pytest.approx(config.final_learning_rate)


def test_identical_networks_split_an_arena_evenly_by_colour() -> None:
    net = PolicyValueNet(NetConfig(blocks=1, filters=8)).eval()
    evaluator = Evaluator(net)
    score, results = play_match(
        Seat.network(0, 8),
        Seat.network(1, 8),
        [evaluator, evaluator],
        games=8,
        c_puct=1.5,
        seed=2,
        threads=2,
        concurrency=8,
        temperature_plies=4,
    )
    assert score.games == 8
    assert score.first_games == score.second_games == 4
    assert summarize(results) == score
    low, high = score.wilson_interval()
    assert 0.0 <= low <= score.score <= high <= 1.0


def test_loop_runs_logs_and_resumes(tmp_path: Path) -> None:
    benchmark = load_benchmark(REPO / "data" / "benchmark.csv")
    subset = type(benchmark)(benchmark.moves[::40], benchmark.strata[::40], benchmark.scores[::40])
    run = tmp_path / "run"
    first = Trainer(run, replace(TINY, max_generations=1), subset, "cpu", 10.0)
    assert first.resume() is False
    first.run()
    assert first.generation == 1
    assert (run / "state.pt").exists()
    assert (run / "best.pt").exists()

    second = Trainer(run, replace(TINY, max_generations=2), subset, "cpu", 10.0)
    assert second.resume() is True
    assert second.generation == 1
    assert second.replay.size == first.replay.size
    for name, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, second.model.state_dict()[name])
    second.run()
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["generation"] for r in records] == [1, 2]
    assert records[1]["total_games"] == 2 * TINY.games_per_generation
    assert 0.0 <= records[1]["grade"]["search"]["strict_rate"] <= 1.0


def test_supervisor_restarts_a_hung_trainer(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    attempts = tmp_path / "attempts"
    script = (
        "import sys, pathlib, time\n"
        f"marker = pathlib.Path({str(attempts)!r})\n"
        "count = int(marker.read_text()) + 1 if marker.exists() else 1\n"
        "marker.write_text(str(count))\n"
        "time.sleep(30) if count == 1 else sys.exit(0)\n"
    )
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        supervise_module.subprocess,
        "Popen",
        lambda command: real_popen([sys.executable, "-c", script]),
    )
    code = supervise_module.supervise(["--run", str(run)], stall=0.5, poll=0.05, max_restarts=2)
    assert code == 0
    assert attempts.read_text() == "2"
