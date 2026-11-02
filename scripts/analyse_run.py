"""Summarises results/training_metrics.jsonl into results/run_analysis.json.

Answers two questions the raw log leaves open: where the learning curve flattened, and
whether the arena kept better networks than the ones it rejected.

Usage: python scripts/analyse_run.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def gating(rows: list[dict]) -> dict:
    rejected = [row for row in rows if not row["arena"]["promoted"]]
    search = [
        row["candidate_grade"]["search"]["value_rate"] - row["grade"]["search"]["value_rate"]
        for row in rejected
    ]
    raw = [
        row["candidate_grade"]["raw"]["value_rate"] - row["grade"]["raw"]["value_rate"]
        for row in rejected
    ]
    return {
        "rejected_generations": len(rejected),
        "promoted_generations": len(rows) - len(rejected),
        "candidate_minus_gated_search_value_points": {
            "mean": 100 * statistics.mean(search),
            "stdev": 100 * statistics.pstdev(search),
            "best": 100 * max(search),
            "worst": 100 * min(search),
            "candidate_better_share": sum(delta > 0 for delta in search) / len(search),
        },
        "candidate_minus_gated_raw_value_points": {
            "mean": 100 * statistics.mean(raw),
            "stdev": 100 * statistics.pstdev(raw),
        },
        "last_promotion_generation": max(
            row["generation"] for row in rows if row["arena"]["promoted"]
        ),
    }


def curve(rows: list[dict], every: int = 10) -> list[dict]:
    picks = [row for row in rows if row["generation"] % every == 0 or row["generation"] == 1]
    if picks[-1] is not rows[-1]:
        picks.append(rows[-1])
    return [
        {
            "generation": row["generation"],
            "games": row["total_games"],
            "minutes": row["elapsed_seconds"] / 60,
            "search_value": row["grade"]["search"]["value_rate"],
            "search_strict": row["grade"]["search"]["strict_rate"],
            "raw_value": row["grade"]["raw"]["value_rate"],
            "raw_strict": row["grade"]["raw"]["strict_rate"],
            "first_player_score": row["self_play"]["first_player_score"],
            "draw_rate": row["self_play"]["draw_rate"],
            "mean_game_length": row["self_play"]["mean_length"],
        }
        for row in picks
    ]


def phases(rows: list[dict]) -> dict:
    return {
        "self_play_minutes": sum(r["self_play"]["seconds"] for r in rows) / 60,
        "optimiser_minutes": sum(r["train"]["seconds"] for r in rows) / 60,
        "arena_minutes": sum(r["arena"]["seconds"] for r in rows) / 60,
        "grading_minutes": sum(r["grade_seconds"] for r in rows) / 60,
    }


def main() -> None:
    log = Path("results/training_metrics.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in log]
    report = {"gating": gating(rows), "curve": curve(rows), "phases": phases(rows)}
    Path("results/run_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
