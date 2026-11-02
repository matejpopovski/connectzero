"""Crash-safe persistence. A checkpoint is either complete or absent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def to_cpu(value: Any) -> Any:
    """Recursively detaches tensors to CPU copies so a checkpoint never holds device memory."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(to_cpu(item) for item in value)
    return value


def atomic_save(state: dict, path: Path) -> None:
    """Writes to a sibling temp file and renames, so a crash cannot damage the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(to_cpu(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path: Path) -> dict:
    """Loads a full training state, which holds numpy arrays. Only use on files you wrote."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_weights(path: Path) -> dict:
    """Loads shipped network weights with pickle restricted to plain data and tensors."""
    return torch.load(path, map_location="cpu", weights_only=True)
