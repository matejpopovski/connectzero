"""Fixed-size window of self-play positions, stored as bitboards rather than tensors."""

from __future__ import annotations

import numpy as np

from connectzero._engine import WIDTH, encode_batch


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.current = np.zeros(capacity, dtype=np.uint64)
        self.occupied = np.zeros(capacity, dtype=np.uint64)
        self.policy = np.zeros((capacity, WIDTH), dtype=np.float32)
        self.outcome = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def add(
        self, current: np.ndarray, occupied: np.ndarray, policy: np.ndarray, outcome: np.ndarray
    ) -> None:
        rows = (self.cursor + np.arange(len(current))) % self.capacity
        self.current[rows] = current
        self.occupied[rows] = occupied
        self.policy[rows] = policy
        self.outcome[rows] = outcome
        self.cursor = int((self.cursor + len(current)) % self.capacity)
        self.size = min(self.capacity, self.size + len(current))

    def sample(
        self, batch: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random batch with a random left-right mirror per position.

        The board and game rules are symmetric, so a mirror is a free extra sample, and
        it is drawn at sample time to avoid storing two copies of everything.
        """
        rows = rng.integers(0, self.size, size=batch)
        planes = encode_batch(self.current[rows], self.occupied[rows])
        policy = self.policy[rows].copy()
        outcome = self.outcome[rows].copy()
        flip = rng.random(batch) < 0.5
        planes[flip] = planes[flip][:, :, :, ::-1]
        policy[flip] = policy[flip][:, ::-1]
        return planes, policy, outcome

    def state_dict(self) -> dict:
        count = self.size
        return {
            "capacity": self.capacity,
            "cursor": self.cursor,
            "size": count,
            "current": self.current[:count].copy(),
            "occupied": self.occupied[:count].copy(),
            "policy": self.policy[:count].copy(),
            "outcome": self.outcome[:count].copy(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> ReplayBuffer:
        buffer = cls(state["capacity"])
        count = state["size"]
        buffer.current[:count] = state["current"]
        buffer.occupied[:count] = state["occupied"]
        buffer.policy[:count] = state["policy"]
        buffer.outcome[:count] = state["outcome"]
        buffer.size = count
        buffer.cursor = state["cursor"]
        return buffer
