"""The demo's request logic, without opening a socket."""

from pathlib import Path

import numpy as np
import pytest
from connectzero._engine import ILLEGAL

from connectzero.benchmark import load_benchmark
from connectzero.checkpoint import atomic_save
from connectzero.demo import PAGE, Match
from connectzero.net import NetConfig, PolicyValueNet

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def match(tmp_path_factory) -> Match:
    weights: Path = tmp_path_factory.mktemp("weights") / "net.pt"
    state = PolicyValueNet(NetConfig(blocks=1, filters=8)).state_dict()
    atomic_save({"config": {"blocks": 1, "filters": 8, "c_puct": 1.5}, "model": state}, weights)
    return Match(weights, solver_budget=2_000_000)


def test_page_is_self_contained() -> None:
    page = PAGE.read_text()
    assert "/api/agent" in page
    assert "http://" not in page
    assert "https://" not in page


def test_position_reports_stones_and_turn(match: Match) -> None:
    position = match.position("4433")
    assert position["ply"] == 4
    assert position["to_move"] == 1
    assert position["columns"][3][:2] == [1, 2]
    assert position["winner"] is None


def test_position_reports_a_finished_game(match: Match) -> None:
    position = match.position("1212121")
    assert position["winner"] == 1
    assert position["legal"] == []


def test_referee_proves_a_late_position(match: Match) -> None:
    benchmark = load_benchmark(REPO / "data" / "benchmark.csv")
    row = int(np.flatnonzero(benchmark.strata == "final")[0])
    answer = match.referee(benchmark.moves[row])
    assert answer["proved"]
    expected = [None if s == ILLEGAL else int(s) for s in benchmark.scores[row]]
    assert answer["scores"] == expected


def test_referee_gives_up_honestly_when_the_budget_runs_out(match: Match) -> None:
    answer = match.referee("4")
    assert answer["proved"] is False
    assert answer["scores"] is None


def test_referee_marks_finished_games(match: Match) -> None:
    assert match.referee("1212121")["over"] is True


def test_agent_plays_a_legal_move_and_clamps_the_budget(match: Match) -> None:
    answer = match.agent("444444", 10)
    assert answer["column"] != 3
    assert sum(answer["visits"]) == 9
    assert sum(match.agent("", 0)["visits"]) == 0
