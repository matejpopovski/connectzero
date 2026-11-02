"""Residual policy-value network and its batched inference wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from connectzero._engine import HEIGHT, PLANES, WIDTH


@dataclass(frozen=True)
class NetConfig:
    blocks: int = 6
    filters: int = 64


class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.norm2(self.conv2(y))
        return F.relu(x + y, inplace=True)


class PolicyValueNet(nn.Module):
    """Maps board planes to column logits and a value in [-1, 1] for the side to move."""

    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        filters = self.config.filters
        cells = HEIGHT * WIDTH
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*(ResidualBlock(filters) for _ in range(self.config.blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * cells, WIDTH),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * cells, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(self.stem(planes))
        return self.policy_head(features), self.value_head(features).squeeze(1)


class Evaluator:
    """Answers leaf batches from the C++ side with numpy arrays."""

    def __init__(self, model: PolicyValueNet, device: str = "cpu", max_batch: int = 4096) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch = max_batch

    @torch.inference_mode()
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Whole-benchmark searches can queue far more leaves than self-play does, so a cap
        # keeps GPU memory bounded whatever the caller's tree count.
        policies, values = [], []
        for start in range(0, len(planes), self.max_batch):
            chunk = torch.from_numpy(planes[start : start + self.max_batch]).to(self.device)
            logits, value = self.model(chunk)
            policies.append(F.softmax(logits, dim=1).cpu())
            values.append(value.cpu())
        return torch.cat(policies).numpy(), torch.cat(values).numpy()


class UniformEvaluator:
    """Uniform priors and a neutral value: search with no network, used by the tests."""

    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(planes)
        return np.full((count, WIDTH), 1.0 / WIDTH, dtype=np.float32), np.zeros(
            count, dtype=np.float32
        )


def config_dict(config: NetConfig) -> dict[str, int]:
    return asdict(config)
