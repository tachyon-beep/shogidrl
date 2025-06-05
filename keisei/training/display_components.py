"""Reusable Rich display components for the training TUI."""

from __future__ import annotations

from typing import Protocol, Optional, List, Sequence
from wcwidth import wcswidth

from keisei.utils.unified_logger import log_error_to_stderr

from rich.console import RenderableType, Group
from rich.panel import Panel
from rich.text import Text


class DisplayComponent(Protocol):
    """Protocol for display components used by :class:`TrainingDisplay`."""

    def render(self) -> RenderableType:
        """Return a Rich renderable representing this component."""
        raise NotImplementedError


class ShogiBoard:
    """ASCII representation of the current Shogi board state."""

    def __init__(self, use_unicode: bool = True, show_moves: bool = False, max_moves: int = 10) -> None:
        self.use_unicode = use_unicode
        self.show_moves = show_moves
        self.max_moves = max_moves

    def _piece_to_symbol(self, piece) -> str:
        if not piece:
            return "・" if self.use_unicode else "."

        if self.use_unicode:
            symbols = {
                "PAWN": "歩",
                "LANCE": "香",
                "KNIGHT": "桂",
                "SILVER": "銀",
                "GOLD": "金",
                "BISHOP": "角",
                "ROOK": "飛",
                "KING": "王",
                "PROMOTED_PAWN": "と",
                "PROMOTED_LANCE": "成香",
                "PROMOTED_KNIGHT": "成桂",
                "PROMOTED_SILVER": "成銀",
                "PROMOTED_BISHOP": "馬",
                "PROMOTED_ROOK": "竜",
            }
            base = symbols.get(piece.type.name, "?")
            return base
        return piece.symbol()

    def _generate_ascii_board(self, board_state) -> str:
        lines: List[str] = ["  9 8 7 6 5 4 3 2 1"]
        cell_width = 2 if self.use_unicode else 1
        for r_idx, row in enumerate(board_state.board):
            line_parts: List[str] = [f"{9 - r_idx} "]
            for piece in reversed(row):
                symbol = self._piece_to_symbol(piece)
                padding = max(0, cell_width - wcswidth(symbol))
                line_parts.append(symbol + " " * (padding + 1))
            lines.append("".join(line_parts))
        return "\n".join(lines)

    def _move_to_usi(self, move_tuple, policy_mapper) -> str:
        try:
            return policy_mapper.shogi_move_to_usi(move_tuple)
        except (ValueError, KeyError):
            return str(move_tuple)
        except Exception as e:  # noqa: BLE001
            log_error_to_stderr("ShogiBoard", f"Unexpected error in _move_to_usi: {e}")
            raise

    def render(self, board_state=None, move_history=None, policy_mapper=None) -> RenderableType:  # type: ignore[override]
        if not board_state:
            return Panel(Text("No active game"), title="Shogi Board")

        ascii_board = self._generate_ascii_board(board_state)
        board_panel = Panel(Text(ascii_board), title="Current Position", border_style="blue")

        if not self.show_moves or not move_history or policy_mapper is None:
            return board_panel

        last_moves = move_history[-self.max_moves :]
        lines = [self._move_to_usi(mv, policy_mapper) for mv in last_moves]
        start_idx = len(move_history) - len(last_moves) + 1
        formatted = [f"{start_idx + i:3d}: {mv}" for i, mv in enumerate(lines)]
        moves_panel = Panel(Text("\n".join(formatted)), border_style="yellow", title="Recent Moves")
        return Group(board_panel, moves_panel)


class Sparkline:
    """Simple Unicode sparkline generator for metric trends."""

    def __init__(self, width: int = 20) -> None:
        self.width = width
        self.chars = "▁▂▃▄▅▆▇█"

    def generate(self, values: Sequence[float]) -> str:
        if len(values) < 2:
            return "".join(["─" for _ in range(self.width)])

        min_v = min(values)
        max_v = max(values)
        if max_v == min_v:
            normalized = [4] * len(values)
        else:
            rng = max_v - min_v
            normalized = [int((v - min_v) / rng * 7) for v in values]

        recent = normalized[-self.width :]
        spark = "".join(self.chars[n] for n in recent)
        if len(spark) < self.width:
            spark = "▁" * (self.width - len(spark)) + spark
        return spark

    def render(self, values: Optional[Sequence[float]] = None, title: str = "Trends") -> RenderableType:  # type: ignore[override]
        values = values or []
        spark = self.generate(list(values)) if values else "".join(["─" for _ in range(self.width)])
        return Panel(Text(f"{title}: {spark}", style="cyan"), border_style="cyan")
