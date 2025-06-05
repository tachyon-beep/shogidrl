"""Helpers for choosing training display layouts based on terminal size."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from rich.console import Console
from rich.layout import Layout

from keisei.config_schema import DisplayConfig


@dataclass
class TerminalInfo:
    width: int
    height: int
    unicode_ok: bool


class AdaptiveDisplayManager:
    """Determine which layout variant should be used for the training TUI."""

    MIN_WIDTH_ENHANCED = 120
    MIN_HEIGHT_ENHANCED = 25

    def __init__(self, config: DisplayConfig) -> None:
        self.config = config

    def _get_terminal_size(self, console: Console) -> Tuple[int, int]:
        try:
            return console.size.width, console.size.height
        except Exception:
            try:
                size = os.get_terminal_size()
                return size.columns, size.lines
            except OSError:
                return 80, 24

    def get_terminal_info(self, console: Console) -> TerminalInfo:
        width, height = self._get_terminal_size(console)
        try:
            "▁".encode(console.encoding or "utf-8")
            unicode_ok = True
        except UnicodeEncodeError:
            unicode_ok = False
        return TerminalInfo(width=width, height=height, unicode_ok=unicode_ok)

    def choose_layout(self, console: Console) -> str:
        info = self.get_terminal_info(console)
        if (
            info.width >= self.MIN_WIDTH_ENHANCED
            and info.height >= self.MIN_HEIGHT_ENHANCED
            and self.config.enable_enhanced_layout
        ):
            return "enhanced"
        return "compact"
