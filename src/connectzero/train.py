"""AlphaZero training loop: self-play, optimise, gate, grade, checkpoint, repeat.

The solver appears here only to grade. It never feeds a training target or a search, so the
curves show what self-play alone learned.

Usage: python -m connectzero.train --run runs/main --minutes 42 --device mps --torch-threads 2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from connectzero._engine import GamePool, SearchParams, Seat
from connectzero.arena import play_match
from connectzero.benchmark import Benchmark, grade, load_benchmark
from connectzero.checkpoint import atomic_save, load
from connectzero.net import Evaluator, NetConfig, PolicyValueNet
from connectzero.replay import ReplayBuffer
from connectzero.search import drive, search_positions


@dataclass(frozen=True)
class TrainConfig:
    blocks: int = 6
    filters: int = 64
    games_per_generation: int = 500
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_fraction: float = 0.25
    temperature_plies: int = 12
    concurrency: int = 512
    leaves_per_tree: int = 4
    mcts_threads: int = 4
    replay_capacity: int = 400_000
    batch_size: int = 512
    sample_reuse: float = 8.0
    learning_rate: float = 2e-3
    final_learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    arena_games: int = 100
    arena_simulations: int = 100
    arena_temperature_plies: int = 6
    gate_threshold: float = 0.55
    grade_simulations: int = 200
    max_generations: int = 1000
    seed: int = 0


def self_play(
    evaluator: Evaluator, config: TrainConfig, seed: int
) -> tuple[tuple[np.ndarray, ...], dict]:
    pool = GamePool(
        Seat.network(0, config.simulations),
        Seat.network(0, config.simulations),
        num_games=config.games_per_generation,
        concurrency=config.concurrency,
        leaves_per_tree=config.leaves_per_tree,
        threads=config.mcts_threads,
        seed=seed,
        search=SearchParams(
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_fraction=config.dirichlet_fraction,
        ),
        temperature_plies=config.temperature_plies,
        share_tree=True,
        record_samples=True,
    )
    start = time.perf_counter()
    drive(pool, [evaluator])
    elapsed = time.perf_counter() - start
    samples = pool.drain_samples()
    results = pool.results
    first_wins = sum(r.winner_seat == r.first_seat for r in results)
    draws = sum(r.winner_seat < 0 for r in results)
    stats = {
        "games": len(results),
        "positions": len(samples[0]),
        "seconds": elapsed,
        "evaluations": pool.evaluations,
        "evaluations_per_second": pool.evaluations / elapsed,
        "mean_batch": pool.evaluations / max(pool.batches, 1),
        "first_player_score": (first_wins + 0.5 * draws) / len(results),
        "draw_rate": draws / len(results),
        "mean_length": float(np.mean([len(r.moves) for r in results])),
    }
    return samples, stats


def optimise(
    model: PolicyValueNet,
    optimiser: torch.optim.Optimizer,
    replay: ReplayBuffer,
    steps: int,
    config: TrainConfig,
    device: str,
    rng: np.random.Generator,
) -> dict:
    model.train()
    totals = np.zeros(3)
    for _ in range(steps):
        planes, policy, outcome = replay.sample(config.batch_size, rng)
        planes = torch.from_numpy(planes).to(device)
        target_policy = torch.from_numpy(policy).to(device)
        target_value = torch.from_numpy(outcome).to(device)
        logits, value = model(planes)
        policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(value, target_value)
        loss = policy_loss + value_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        entropy = -(target_policy * torch.log(target_policy.clamp_min(1e-8))).sum(dim=1).mean()
        totals += [policy_loss.item(), value_loss.item(), entropy.item()]
    model.eval()
    policy_loss, value_loss, target_entropy = totals / max(steps, 1)
    return {
        "steps": steps,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "target_entropy": target_entropy,
        "policy_excess": policy_loss - target_entropy,
    }


def grade_network(
    evaluator: Evaluator, benchmark: Benchmark, simulations: int, c_puct: float, threads: int
) -> dict:
    """Grades the raw policy and the searched policy on every benchmark position."""
    boards = benchmark.boards()
    planes = np.stack([board.planes() for board in boards])
    policy, _ = evaluator(planes)
    raw = np.where(benchmark.legal, policy, -1.0).argmax(axis=1)
    visits, _ = search_positions(boards, evaluator, simulations, c_puct, threads=threads)
    searched = visits.argmax(axis=1)
    return {"raw": grade(benchmark, raw), "search": grade(benchmark, searched)}


def learning_rate(config: TrainConfig, progress: float) -> float:
    """Cosine decay on the share of the time budget used, so a resumed run stays on schedule."""
    fraction = min(max(progress, 0.0), 1.0)
    span = config.learning_rate - config.final_learning_rate
    return config.final_learning_rate + span * 0.5 * (1.0 + np.cos(np.pi * fraction))


def build_model(config: TrainConfig) -> PolicyValueNet:
    return PolicyValueNet(NetConfig(blocks=config.blocks, filters=config.filters))


class Trainer:
    def __init__(
        self,
        run_dir: Path,
        config: TrainConfig,
        benchmark: Benchmark,
        device: str,
        minutes: float,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.benchmark = benchmark
        self.device = device
        self.budget_seconds = minutes * 60.0
        self.model = build_model(config).to(device)
        self.best = build_model(config).to(device)
        self.best.load_state_dict(self.model.state_dict())
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.replay = ReplayBuffer(config.replay_capacity)
        self.generation = 0
        self.best_generation = 0
        self.elapsed = 0.0
        self.total_games = 0
        self.rng = np.random.default_rng(config.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "metrics.jsonl"
        self.state_path = run_dir / "state.pt"
        self.heartbeat_path = run_dir / "heartbeat"

    def beat(self) -> None:
        self.heartbeat_path.touch()

    def save(self) -> None:
        atomic_save(
            {
                "config": asdict(self.config),
                "model": self.model.state_dict(),
                "best": self.best.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "replay": self.replay.state_dict(),
                "generation": self.generation,
                "best_generation": self.best_generation,
                "elapsed": self.elapsed,
                "total_games": self.total_games,
                "rng": self.rng.bit_generator.state,
            },
            self.state_path,
        )

    def resume(self) -> bool:
        if not self.state_path.exists():
            return False
        state = load(self.state_path)
        self.model.load_state_dict(state["model"])
        self.best.load_state_dict(state["best"])
        self.optimiser.load_state_dict(state["optimiser"])
        self.replay = ReplayBuffer.from_state_dict(state["replay"])
        self.generation = state["generation"]
        self.best_generation = state["best_generation"]
        self.elapsed = state["elapsed"]
        self.total_games = state["total_games"]
        self.rng.bit_generator.state = state["rng"]
        return True

    def generation_seed(self, salt: int) -> int:
        return self.config.seed * 1_000_003 + self.generation * 1009 + salt

    def step(self) -> dict:
        config = self.config
        started = time.perf_counter()
        self.beat()
        self.best.eval()
        best_eval = Evaluator(self.best, self.device)
        samples, play_stats = self_play(best_eval, config, self.generation_seed(1))
        self.replay.add(*samples)
        self.total_games += play_stats["games"]
        self.beat()

        progress = self.elapsed / self.budget_seconds
        for group in self.optimiser.param_groups:
            group["lr"] = learning_rate(config, progress)
        steps = max(1, int(config.sample_reuse * play_stats["positions"] / config.batch_size))
        train_started = time.perf_counter()
        train_stats = optimise(
            self.model, self.optimiser, self.replay, steps, config, self.device, self.rng
        )
        train_stats["seconds"] = time.perf_counter() - train_started
        train_stats["learning_rate"] = self.optimiser.param_groups[0]["lr"]
        self.beat()

        candidate_eval = Evaluator(self.model, self.device)
        arena_started = time.perf_counter()
        score, _ = play_match(
            Seat.network(0, config.arena_simulations),
            Seat.network(1, config.arena_simulations),
            [candidate_eval, best_eval],
            games=config.arena_games,
            c_puct=config.c_puct,
            seed=self.generation_seed(2),
            threads=config.mcts_threads,
            temperature_plies=config.arena_temperature_plies,
        )
        promoted = score.score >= config.gate_threshold
        if promoted:
            self.best.load_state_dict(self.model.state_dict())
            self.best_generation = self.generation + 1
            best_eval = Evaluator(self.best, self.device)
        arena_stats = score.as_dict() | {
            "promoted": promoted,
            "seconds": time.perf_counter() - arena_started,
        }
        self.beat()

        grade_started = time.perf_counter()
        graded = grade_network(
            best_eval,
            self.benchmark,
            config.grade_simulations,
            config.c_puct,
            config.mcts_threads,
        )
        # A rejected candidate is graded too: comparing it with the gated net shows whether
        # the arena is keeping stronger nets or only delaying them.
        candidate_graded = (
            graded
            if promoted
            else grade_network(
                candidate_eval,
                self.benchmark,
                config.grade_simulations,
                config.c_puct,
                config.mcts_threads,
            )
        )
        grade_seconds = time.perf_counter() - grade_started

        self.generation += 1
        self.elapsed += time.perf_counter() - started
        record = {
            "generation": self.generation,
            "elapsed_seconds": self.elapsed,
            "total_games": self.total_games,
            "replay_size": self.replay.size,
            "best_generation": self.best_generation,
            "self_play": play_stats,
            "train": train_stats,
            "arena": arena_stats,
            "grade": graded,
            "candidate_grade": candidate_graded,
            "grade_seconds": grade_seconds,
        }
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if promoted or self.generation == 1:
            atomic_save(
                {
                    "config": asdict(config),
                    "generation": self.best_generation,
                    "total_games": self.total_games,
                    "model": self.best.state_dict(),
                },
                self.run_dir / "generations" / f"gen{self.best_generation:04d}.pt",
            )
        atomic_save(
            {"config": asdict(config), "model": self.best.state_dict()},
            self.run_dir / "best.pt",
        )
        self.save()
        self.beat()
        return record

    def run(self) -> None:
        while self.elapsed < self.budget_seconds and self.generation < self.config.max_generations:
            record = self.step()
            print(summary_line(record), flush=True)


def summary_line(record: dict) -> str:
    def rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    search = record["grade"]["search"]
    raw = record["grade"]["raw"]
    return (
        f"gen {record['generation']:3d} t={record['elapsed_seconds'] / 60:5.1f}m "
        f"games={record['total_games']} "
        f"evals/s={record['self_play']['evaluations_per_second']:.0f} "
        f"ploss={record['train']['policy_loss']:.3f} "
        f"vloss={record['train']['value_loss']:.3f} "
        f"arena={record['arena']['score']:.2f}{'*' if record['arena']['promoted'] else ' '} "
        f"raw v={rate(raw['value_rate'])} s={rate(raw['strict_rate'])} "
        f"mcts v={rate(search['value_rate'])} s={rate(search['strict_rate'])}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/main"))
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.csv"))
    parser.add_argument("--minutes", type=float, default=40.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    fields = {name: getattr(args, name) for name in asdict(TrainConfig())}
    config = replace(TrainConfig(), **fields)
    trainer = Trainer(args.run, config, load_benchmark(args.benchmark), args.device, args.minutes)
    if trainer.resume():
        print(f"resumed at generation {trainer.generation}", flush=True)
    trainer.run()


if __name__ == "__main__":
    main()
