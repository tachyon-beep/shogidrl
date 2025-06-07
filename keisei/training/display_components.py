"""Reusable Rich display components for the training TUI."""

from __future__ import annotations

from typing import Protocol, Optional, List, Sequence, Dict
from collections import deque, Counter
from wcwidth import wcswidth

from keisei.utils.unified_logger import log_error_to_stderr
from keisei.shogi.shogi_core_definitions import Color

from rich.console import RenderableType, Group
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.table import Table
from rich.style import Style
from rich.align import Align


class DisplayComponent(Protocol):
    """Protocol for display components used by :class:`TrainingDisplay`."""

    def render(self) -> RenderableType:
        """Return a Rich renderable representing this component."""
        raise NotImplementedError


class ShogiBoard:
    """Rich-ified representation of the current Shogi board state using a Table."""

    def __init__(
        self,
        use_unicode: bool = True,
        show_moves: bool = False,  # This parameter is kept for compatibility but not used in this version
        max_moves: int = 10,  # This parameter is kept for compatibility but not used in this version
    ) -> None:
        self.use_unicode = use_unicode
        # Define reference symbols for width calculation
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
                "杏",
                "圭",
                "全",
                "馬",
                "龍",
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
        """Turn your internal piece-object into a single-string symbol."""
        if not piece:
            return "・"
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
                "PROMOTED_LANCE": "杏",
                "PROMOTED_KNIGHT": "圭",
                "PROMOTED_SILVER": "全",
                "PROMOTED_BISHOP": "馬",
                "PROMOTED_ROOK": "龍",
            }
            return symbols.get(piece.type.name, "?")
        return piece.symbol()

    def _colorize(self, symbol: str, piece) -> Text:
        """Apply per-piece colouring."""
        if not piece:
            return Text(symbol)
        if piece.color == Color.BLACK:
            return Text(symbol, style="bright_red")
        return Text(symbol, style="bright_blue")

    def _pad_symbol(self, symbol: str) -> str:
        """Pad a raw symbol to exactly self.cell_width characters."""
        width = wcswidth(symbol) or len(symbol)
        padding = self.cell_width - width
        return symbol + (" " * padding)

    def _get_shogi_notation(self, row: int, col: int) -> str:
        """Convert board indices to shogi square notation like '7f'."""
        file_num = 9 - col
        rank_letter = chr(ord("a") + row)
        return f"{file_num}{rank_letter}"

    def _generate_rich_table(self, board_state, hot_squares: Optional[List[str]] = None) -> Table:
        """Create a 10×10 Table for the board."""
        light_bg_color = "#EEC28A"
        dark_bg_color = "#C19A55"
        light_bg = Style(bgcolor=light_bg_color)
        dark_bg = Style(bgcolor=dark_bg_color)

        table = Table(
            show_header=False,
            box=None,
            pad_edge=False,
            padding=(0, 0),
            expand=False,
        )
        table.add_column("", width=self.cell_width, no_wrap=True, justify="center")
        for file_num in range(9, 0, -1):
            table.add_column(
                str(file_num),
                width=self.cell_width,
                justify="center",
                no_wrap=True,
            )

        file_labels = [""] + [str(n) for n in range(9, 0, -1)]
        table.add_row(*[Text(lbl, style="bold") for lbl in file_labels])

        hot_squares = hot_squares or []
        for r_idx, row in enumerate(board_state.board):
            rank_label = str(9 - r_idx)
            row_cells: List[Text] = [Text(rank_label, style="bold")]

            for c_idx, piece in enumerate(reversed(row)):
                is_light = (r_idx + c_idx) % 2 == 0
                bg_style = light_bg if is_light else dark_bg

                if piece:
                    raw_symbol = self._piece_to_symbol(piece)
                    padded = self._pad_symbol(raw_symbol)
                    cell_renderable = self._colorize(padded, piece)
                else:
                    raw_symbol = self._piece_to_symbol(piece)
                    padded = self._pad_symbol(raw_symbol)
                    dot_color = dark_bg_color if is_light else light_bg_color
                    cell_renderable = Text(padded, style=dot_color)

                cell_renderable.stylize(bg_style)
                board_col = len(row) - 1 - c_idx
                notation = self._get_shogi_notation(r_idx, board_col)
                if notation in hot_squares:
                    cell_renderable.stylize(Style(frame=True, color="red"))
                row_cells.append(cell_renderable)

            table.add_row(*row_cells)
        return table

    def render(self, board_state=None, hot_squares: Optional[List[str]] = None, **_kwargs) -> RenderableType:
        """Returns a Panel containing a Rich Table of the current board."""
        if not board_state:
            return Panel(Text("No active game"), title="Main Board", border_style="blue")

        board_table = self._generate_rich_table(board_state, hot_squares)
        return Panel(Align.center(board_table), title="Main Board", border_style="blue")


class RecentMovesPanel:
    """Renders the list of recent moves."""

    def __init__(self, max_moves: int = 20):
        self.max_moves = max_moves

    def render(self, move_strings: Optional[List[str]] = None) -> RenderableType:
        if not move_strings:
            return Panel(
                Text("No moves yet."),
                title="Recent Moves",
                border_style="yellow",
                expand=True,
            )
        indent = " "
        last_msgs = move_strings[-self.max_moves :]
        formatted = [f"{indent}{msg}" for msg in last_msgs]
        return Panel(
            Text("\n".join(formatted)),
            title="Recent Moves",
            border_style="yellow",
            expand=True,
        )


class PieceStandPanel:
    """Renders the captured pieces (komadai) for each player."""

    def _format_hand(self, hand: Dict[str, int]) -> str:
        symbols = {
            "PAWN": "歩",
            "LANCE": "香",
            "KNIGHT": "桂",
            "SILVER": "銀",
            "GOLD": "金",
            "BISHOP": "角",
            "ROOK": "飛",
        }
        parts = [f"{symbols.get(getattr(k, 'name', k), '?')}x{v}" for k, v in hand.items() if v > 0]
        return " ".join(parts) or ""

    def render(self, game) -> RenderableType:
        if not game:
            return Panel("...", title="Captured Pieces")

        sente_hand = self._format_hand(getattr(game, "hands", {}).get(Color.BLACK.value, {}))
        gote_hand = self._format_hand(getattr(game, "hands", {}).get(Color.WHITE.value, {}))

        return Panel(
            Group(
                Text.from_markup(f"[bold]Sente:[/b] {sente_hand}"),
                Text.from_markup(f"[bold]Gote: [/b] {gote_hand}"),
            ),
            title="Captured Pieces",
            border_style="yellow",
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


class GameStatisticsPanel:
    """Renders detailed statistics about the current game and session."""

    def _calculate_material(self, board, color):
        piece_values = {
            "PAWN": 1,
            "LANCE": 3,
            "KNIGHT": 3,
            "SILVER": 5,
            "GOLD": 6,
            "BISHOP": 8,
            "ROOK": 10,
        }
        total_value = 0
        for row in board.board:
            for piece in row:
                if piece and piece.color == color:
                    key = piece.type.name.replace("PROMOTED_", "")
                    total_value += piece_values.get(key, 0)
        return total_value

    def _format_hand(self, hand: Dict[str, int]) -> str:
        symbols = {
            "PAWN": "歩",
            "LANCE": "香",
            "KNIGHT": "桂",
            "SILVER": "銀",
            "GOLD": "金",
            "BISHOP": "角",
            "ROOK": "飛",
        }
        parts = [f"{symbols.get(getattr(k, 'name', k), '?')}x{v}" for k, v in hand.items() if v > 0]
        return " ".join(parts) or ""

    def render(
        self,
        game,
        move_history: Optional[List[str]] = None,
        metrics_manager=None,
        policy_mapper=None,
    ) -> RenderableType:
        if not game or not move_history or not metrics_manager:
            return Panel("Waiting for game to start...", title="Game Statistics", border_style="green")

        sente_material = self._calculate_material(game.board, Color.BLACK)
        gote_material = self._calculate_material(game.board, Color.WHITE)
        material_adv = sente_material - gote_material
        is_in_check = getattr(game, 'is_in_check', lambda: False)()
        hot_squares = metrics_manager.get_hot_squares(top_n=3)

        sente_openings = metrics_manager.sente_opening_history
        gote_openings = metrics_manager.gote_opening_history
        fav_sente_opening = Counter(sente_openings).most_common(1)[0][0] if sente_openings else "N/A"
        fav_gote_opening = Counter(gote_openings).most_common(1)[0][0] if gote_openings else "N/A"

        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()

        table.add_row("Check Status:", "[red]CHECK[/]" if is_in_check else "Clear")
        table.add_row("Material Adv:", f"{material_adv:+.1f} (Sente)")
        table.add_row("Hot Squares:", ", ".join(hot_squares) or "N/A")
        table.add_row("Fav. Sente Opening:", fav_sente_opening)
        table.add_row("Fav. Gote Opening:", fav_gote_opening)

        sente_hand_str = self._format_hand(getattr(game, 'hands', {}).get(Color.BLACK, {}))
        gote_hand_str = self._format_hand(getattr(game, 'hands', {}).get(Color.WHITE, {}))
        hand_info = Group(
             Text.from_markup("\n[bold]Sente's Hand:[/bold]"), Text(sente_hand_str or "None"),
             Text.from_markup("\n[bold]Gote's Hand:[/bold]"), Text(gote_hand_str or "None")
        )

        return Panel(Group(table, hand_info), title="Game Statistics", border_style="green")
