"""Reusable Rich display components for the training TUI."""

from __future__ import annotations

from typing import Protocol, Optional, List, Sequence, Dict
from collections import deque
from wcwidth import wcswidth

from keisei.utils.unified_logger import log_error_to_stderr
from keisei.shogi.shogi_core_definitions import Color

from rich.console import RenderableType, Group
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout


class DisplayComponent(Protocol):
    """Protocol for display components used by :class:`TrainingDisplay`."""

    def render(self) -> RenderableType:
        """Return a Rich renderable representing this component."""
        raise NotImplementedError


class ShogiBoard:
    """ASCII representation of the current Shogi board state."""

    def __init__(
        self,
        use_unicode: bool = True,
        show_moves: bool = False,
        max_moves: int = 10,
    ) -> None:
        self.use_unicode = use_unicode
        self.show_moves = show_moves
        self.max_moves = max_moves

        if self.use_unicode:
            reference_symbols = [
                "歩",
                "香",
                "桂",
                "銀",
                "金",
                "角",
                "飛",
                "王",
                "と",
                "成香",
                "成桂",
                "成銀",
                "馬",
                "竜",
                "・",
            ]
        else:
            reference_symbols = [
                "P",
                "L",
                "N",
                "S",
                "G",
                "B",
                "R",
                "K",
                "+P",
                "+L",
                "+N",
                "+S",
                "+B",
                "+R",
                ".",
            ]
        widths = [wcswidth(sym) or len(sym) for sym in reference_symbols]
        self.cell_width = max(widths)

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

    def _colorize(self, symbol: str, piece) -> str:
        if piece.color == Color.BLACK:
            return f"[bright_red]{symbol}[/bright_red]"
        return f"[bright_blue]{symbol}[/bright_blue]"

    def _generate_ascii_board(self, board_state) -> str:
        cell_width = self.cell_width
        indent = ""
        start_padding = cell_width - 1 if self.use_unicode else cell_width
        header = (
            indent
            + " " * start_padding
            + " ".join(str(n).rjust(cell_width) for n in range(9, 0, -1))
        )
        lines: List[str] = [header]

        def pad(sym: str) -> str:
            width = wcswidth(sym) or len(sym)
            padding = cell_width - width
            return sym + (" " * padding)

        for r_idx, row in enumerate(board_state.board):
            row_header = str(9 - r_idx).rjust(cell_width) + " "
            line_parts: List[str] = [indent + row_header]
            for piece in reversed(row):
                symbol = self._piece_to_symbol(piece)
                colored = self._colorize(symbol, piece) if piece else symbol
                padded = pad(colored)
                line_parts.append(padded + " ")
        lines.append("".join(line_parts).rstrip())
        return "\n".join(lines)

    def _move_to_usi(self, move_tuple, policy_mapper) -> str:
        try:
            return policy_mapper.shogi_move_to_usi(move_tuple)
        except (ValueError, KeyError):
            return str(move_tuple)
        except Exception as e:  # noqa: BLE001
            log_error_to_stderr("ShogiBoard", f"Unexpected error in _move_to_usi: {e}")
            raise

    def render(self, board_state=None, **_kwargs) -> RenderableType:  # type: ignore[override]
        if not board_state:
            return Panel(Text("No active game"), title="Main Board")

        ascii_board = self._generate_ascii_board(board_state)
        board_panel = Panel(
            Text.from_markup(ascii_board),
            title="Main Board",
            border_style="blue",
        )
        return board_panel


class RecentMovesPanel:
    """Renders the list of recent moves."""

    def __init__(self, max_moves: int = 20):
        self.max_moves = max_moves

    def render(self, move_strings: Optional[List[str]] = None) -> RenderableType:
        if not move_strings:
            return Panel(
                Text("No moves yet."), title="Recent Moves", border_style="yellow"
            )
        indent = " "
        last_msgs = move_strings[-self.max_moves :]
        formatted = [f"{indent}{msg}" for msg in last_msgs]
        return Panel(
            Text("\n".join(formatted)), title="Recent Moves", border_style="yellow"
        )


class Sparkline:
    """Simple Unicode sparkline generator for metric trends."""

    def __init__(self, width: int = 20) -> None:
        self.width = width
        self.chars = "▁▂▃▄▅▆▇█"

    def generate(
        self,
        values: Sequence[float],
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> str:
        if not values:
            return " " * self.width
        if len(values) < 2:
            return "─" * self.width

        min_v = range_min if range_min is not None else min(values)
        max_v = range_max if range_max is not None else max(values)
        clipped = [min(max(v, min_v), max_v) for v in values]
        if max_v == min_v:
            normalized = [4] * len(clipped)
        else:
            rng = max_v - min_v
            normalized = [int((v - min_v) / rng * 7) for v in clipped]

        recent = normalized[-self.width :]
        spark = "".join(self.chars[n] for n in recent)
        if len(spark) < self.width:
            spark = "▁" * (self.width - len(spark)) + spark
        return spark


class RollingAverageCalculator:
    """Compute a rolling average and simple trend direction."""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)

    def add_value(self, value: float) -> float:
        self.values.append(value)
        return sum(self.values) / len(self.values)

    def get_trend_direction(self) -> str:
        if len(self.values) < 2:
            return "→"
        if self.values[-1] > self.values[0]:
            return "↑"
        if self.values[-1] < self.values[0]:
            return "↓"
        return "→"


class MultiMetricSparkline:
    """Render multiple metric sparklines in one panel."""

    def __init__(self, width: int, metrics: List[str]) -> None:
        self.width = width
        self.metrics = metrics
        self.data: Dict[str, List[float]] = {m: [] for m in metrics}
        self.spark = Sparkline(width=width)

    def add_data_point(self, metric_name: str, value: float) -> None:
        if metric_name in self.data:
            self.data[metric_name].append(value)

    def render_with_trendlines(self) -> RenderableType:
        lines: List[str] = []
        for name in self.metrics:
            values = self.data.get(name, [])
            spark = self.spark.generate(values[-self.width :])
            lines.append(f"{name}: {spark}")
        return Text("\n".join(lines), style="cyan")
