"""
training/display.py: Rich UI management for the Shogi RL trainer.
"""

from typing import Any, Dict, List, Union

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class TrainingDisplay:
    def __init__(self, config, trainer, rich_console: Console):
        self.config = config
        self.trainer = trainer
        self.rich_console = rich_console
        self.rich_log_messages = trainer.rich_log_messages
        self.progress_bar, self.training_task, self.layout, self.log_panel = (
            self._setup_rich_progress_display()
        )

    def _setup_rich_progress_display(self):
        progress_columns: List[Union[str, ProgressColumn]]
        base_columns: List[Union[str, ProgressColumn]] = [
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn(
                "• Steps: {task.completed}/{task.total} ({task.fields[speed]:.1f} it/s)"
            ),
            TextColumn("• {task.fields[ep_metrics]}", style="bright_cyan"),
            TextColumn("• {task.fields[ppo_metrics]}", style="bright_yellow"),
            TextColumn(
                "• Wins B:{task.fields[black_wins_cum]} W:{task.fields[white_wins_cum]} D:{task.fields[draws_cum]}",
                style="bright_green",
            ),
            TextColumn(
                "• Rates B:{task.fields[black_win_rate]:.1%} W:{task.fields[white_win_rate]:.1%} D:{task.fields[draw_rate]:.1%}",
                style="bright_blue",
            ),
        ]
        enable_spinner = getattr(self.config.training, "enable_spinner", True)
        if enable_spinner:
            progress_columns = [SpinnerColumn()] + base_columns
        else:
            progress_columns = base_columns
        progress_bar = Progress(
            *progress_columns,
            console=self.rich_console,
            transient=False,
        )
        initial_black_win_rate = (
            self.trainer.black_wins / self.trainer.total_episodes_completed
            if self.trainer.total_episodes_completed > 0
            else 0.0
        )
        initial_white_win_rate = (
            self.trainer.white_wins / self.trainer.total_episodes_completed
            if self.trainer.total_episodes_completed > 0
            else 0.0
        )
        initial_draw_rate = (
            self.trainer.draws / self.trainer.total_episodes_completed
            if self.trainer.total_episodes_completed > 0
            else 0.0
        )
        training_task = progress_bar.add_task(
            "Training",
            total=self.config.training.total_timesteps,
            completed=self.trainer.global_timestep,
            ep_metrics="Ep L:0 R:0.0",
            ppo_metrics="",
            black_wins_cum=self.trainer.black_wins,
            white_wins_cum=self.trainer.white_wins,
            draws_cum=self.trainer.draws,
            black_win_rate=initial_black_win_rate,
            white_win_rate=initial_white_win_rate,
            draw_rate=initial_draw_rate,
            speed=0.0,
            start=(self.trainer.global_timestep < self.config.training.total_timesteps),
        )
        log_panel = Panel(
            Text(""),
            title="[b]Live Training Log[/b]",
            border_style="bright_green",
            expand=True,
        )
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="main_log", ratio=1),
            Layout(name="progress_display", size=4),
        )
        layout["main_log"].update(log_panel)
        layout["progress_display"].update(progress_bar)
        return progress_bar, training_task, layout, log_panel

    def update_progress(self, trainer, speed, pending_updates):
        update_data = {"completed": trainer.global_timestep, "speed": speed}
        update_data.update(pending_updates)
        self.progress_bar.update(self.training_task, **update_data)

    def update_log_panel(self, trainer):
        visible_rows = max(0, self.rich_console.size.height - 6)
        if trainer.rich_log_messages:
            display_messages = trainer.rich_log_messages[-visible_rows:]
            updated_panel_content = Group(*display_messages)
            self.log_panel.renderable = updated_panel_content
        else:
            self.log_panel.renderable = Text("")

    def start(self):
        return Live(
            self.layout,
            console=self.rich_console,
            refresh_per_second=self.config.training.refresh_per_second,
            transient=False,
        )
