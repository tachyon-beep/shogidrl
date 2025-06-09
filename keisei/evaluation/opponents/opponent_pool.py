from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional, Sequence
import random

from ..legacy.elo_registry import EloRegistry


@dataclass
class OpponentEntry:
    """Represents a stored opponent checkpoint."""

    path: Path


class OpponentPool:
    """Manage a pool of opponent checkpoints and their Elo ratings."""

    def __init__(self, pool_size: int = 5, elo_registry_path: Optional[str] = None) -> None:
        self.pool_size = pool_size
        self._entries: Deque[Path] = deque(maxlen=pool_size)
        self.elo_registry: Optional[EloRegistry] = None
        if elo_registry_path:
            self.elo_registry = EloRegistry(Path(elo_registry_path))

    # Management -----------------------------------------------------
    def add_checkpoint(self, path: Path | str) -> None:
        """Add a checkpoint to the pool, evicting the oldest if full."""
        p = Path(path)
        self._entries.append(p)
        if self.elo_registry:
            # Ensure rating entry exists
            self.elo_registry.get_rating(p.name)
            self.elo_registry.save()

    def get_all(self) -> Iterable[Path]:
        """Return all checkpoints in the pool."""
        return list(self._entries)

    # Selection ------------------------------------------------------
    def sample(self) -> Optional[Path]:
        """Return a random opponent checkpoint from the pool."""
        if not self._entries:
            return None
        return random.choice(list(self._entries))

    def champion(self) -> Optional[Path]:
        """Return the checkpoint with the highest Elo rating."""
        if not self._entries:
            return None
        if not self.elo_registry:
            return self.sample()
        return max(self._entries, key=lambda p: self.elo_registry.get_rating(p.name))

    # Elo updates ----------------------------------------------------
    def update_ratings(
        self,
        agent_id: str,
        opponent_id: str,
        results: Sequence[str],
    ) -> None:
        """Update Elo ratings for a match and persist."""
        if not self.elo_registry:
            return
        self.elo_registry.update_ratings(agent_id, opponent_id, list(results))
        self.elo_registry.save()
