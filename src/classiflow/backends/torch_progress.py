"""Lightweight fit-progress context for torch estimator logging."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class TorchFitProgressState:
    """Mutable progress state shared across sequential torch estimator fits."""

    label: str
    total: int
    current: int = 0


_TORCH_FIT_PROGRESS: ContextVar[Optional[TorchFitProgressState]] = ContextVar(
    "torch_fit_progress",
    default=None,
)


@contextmanager
def torch_fit_progress(label: str, total: int) -> Iterator[TorchFitProgressState]:
    """Install a fit-progress context for sequential torch estimator fits."""
    state = TorchFitProgressState(label=label, total=max(int(total), 1))
    token: Token[Optional[TorchFitProgressState]] = _TORCH_FIT_PROGRESS.set(state)
    try:
        yield state
    finally:
        _TORCH_FIT_PROGRESS.reset(token)


def next_torch_fit_progress() -> Optional[TorchFitProgressState]:
    """Advance and return the active fit-progress state, if any."""
    state = _TORCH_FIT_PROGRESS.get()
    if state is None:
        return None
    state.current += 1
    return state
