from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, Optional


class PreviousModelSelector:
    """Maintain a limited history of previous model checkpoints for evaluation."""

    def __init__(self, pool_size: int = 5) -> None:
        self.pool_size = pool_size
        self.checkpoints: Deque[Path] = deque(maxlen=pool_size)

    def add_checkpoint(self, path: Path | str) -> None:
        """Add a checkpoint path to the history."""
        self.checkpoints.append(Path(path))

    def get_random_checkpoint(self) -> Optional[Path]:
        """Return a random checkpoint from the history, or None if empty."""
        if not self.checkpoints:
            return None
        return random.choice(list(self.checkpoints))

    def get_all(self) -> Iterable[Path]:
        """Return all stored checkpoints in order of addition."""
        return list(self.checkpoints)
