"""Torch utilities for device, dtype, and seeding."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def resolve_device(name: str) -> str:
    """Resolve device name to available hardware."""
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return name


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """Resolve torch dtype with basic device guardrails."""
    if dtype == "float16":
        if device in {"cpu", "mps"}:
            logger.warning("float16 requested on %s; using float32 for stability.", device)
            return torch.float32
        return torch.float16
    return torch.float32


def set_seed(seed: int) -> None:
    """Set deterministic seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    worker_seed: Optional[int] = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader from numpy arrays."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    worker_init_fn: Optional[Callable[[int], None]] = None
    if worker_seed is not None and num_workers > 0:
        def _seed_worker(worker_id: int) -> None:
            seed = worker_seed + worker_id
            np.random.seed(seed)
            torch.manual_seed(seed)
        worker_init_fn = _seed_worker
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
    )
