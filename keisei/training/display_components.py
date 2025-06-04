from __future__ import annotations

"""Reusable Rich display components for the training TUI."""

from typing import Protocol, Optional, List

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text


class DisplayComponent(Protocol):
    """Protocol for display components used by :class:`TrainingDisplay`."""

    def render(self) -> RenderableType:
        """Return a Rich renderable representing this component."""
        raise NotImplementedError


class ShogiBoard:
    """Placeholder component for displaying an ASCII Shogi board."""

    def render(self) -> RenderableType:  # type: ignore[override]
        return Panel(Text("Board display not implemented"), title="Shogi Board")


class Sparkline:
    """Placeholder sparkline component for metric trends."""

    def __init__(self, width: int = 20) -> None:
        self.width = width
        self.history: List[float] = []

    def render(self) -> RenderableType:  # type: ignore[override]
        return Panel(Text("Trend data not available"), title="Trends")
