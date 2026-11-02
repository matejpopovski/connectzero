"""Play the trained network in a browser, with the solver refereeing every column.

Usage: connectzero-demo --weights weights/connectzero.pt --port 45754
Then open http://127.0.0.1:45754
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from connectzero._engine import ILLEGAL, Board, Solver
from connectzero.checkpoint import load_weights
from connectzero.net import Evaluator, NetConfig, PolicyValueNet
from connectzero.search import search_positions

PAGE = Path(__file__).parent / "web" / "index.html"
MAX_SIMULATIONS = 5000


class Match:
    """Stateless per request: the client sends the move list, the server answers for it."""

    def __init__(self, weights: Path, solver_budget: int) -> None:
        state = load_weights(weights)
        config = state["config"]
        model = PolicyValueNet(NetConfig(blocks=config["blocks"], filters=config["filters"]))
        model.load_state_dict(state["model"])
        self.evaluator = Evaluator(model, "cpu")
        self.c_puct = config["c_puct"]
        self.solver = Solver(24)
        self.solver_budget = solver_budget
        # A proof can take seconds, and the agent's reply must not wait for it, so requests
        # run on their own threads and only the solver itself is serialised.
        self.solver_lock = threading.Lock()

    def position(self, moves: str) -> dict:
        board = Board.from_moves(moves)
        columns = [[0] * 6 for _ in range(7)]
        replay = Board()
        for ply, digit in enumerate(moves):
            col = int(digit) - 1
            columns[col][replay.height(col)] = 1 if ply % 2 == 0 else 2
            replay = replay.play(col)
        winner = None
        if board.last_move_won():
            winner = 1 if board.moves % 2 == 1 else 2
        return {
            "moves": moves,
            "ply": board.moves,
            "columns": columns,
            "legal": board.legal_moves(),
            "winner": winner,
            "draw": board.is_full() and winner is None,
            "to_move": 1 if board.moves % 2 == 0 else 2,
        }

    def referee(self, moves: str) -> dict:
        board = Board.from_moves(moves)
        if board.is_over():
            return {"scores": None, "proved": False, "over": True}
        with self.solver_lock:
            scores = self.solver.analyze(board, budget=self.solver_budget)
        if scores is None:
            return {"scores": None, "proved": False, "over": False}
        return {
            "scores": [None if s == ILLEGAL else s for s in scores],
            "proved": True,
            "over": False,
        }

    def agent(self, moves: str, simulations: int) -> dict:
        board = Board.from_moves(moves)
        simulations = max(1, min(simulations, MAX_SIMULATIONS))
        visits, values = search_positions(
            [board], self.evaluator, simulations, self.c_puct, threads=2
        )
        return {
            "column": int(np.argmax(visits[0])),
            "visits": visits[0].tolist(),
            "value": float(values[0]),
        }


def make_handler(match: Match) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def send_body(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_body(PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send_body(b"not found", "text/plain", 404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            moves = str(request.get("moves", ""))
            try:
                if self.path == "/api/position":
                    answer = match.position(moves)
                elif self.path == "/api/referee":
                    answer = match.referee(moves)
                elif self.path == "/api/agent":
                    answer = match.agent(moves, int(request.get("simulations", 400)))
                else:
                    self.send_body(b"not found", "text/plain", 404)
                    return
            except ValueError as error:
                body = json.dumps({"error": str(error)}).encode()
                self.send_body(body, "application/json", 400)
                return
            self.send_body(json.dumps(answer).encode(), "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/connectzero.pt"))
    parser.add_argument("--port", type=int, default=45754)
    parser.add_argument("--solver-budget", type=int, default=30_000_000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(Match(args.weights, args.solver_budget))
    )
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
