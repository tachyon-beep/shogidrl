"""
training/display.py: Rich UI management for the Shogi RL trainer.
"""

from typing import List, Union, Optional

from rich.console import Console, Group
from rich.table import Table
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

from keisei.config_schema import DisplayConfig
from .display_components import ShogiBoard, Sparkline
from .adaptive_display import AdaptiveDisplayManager


class TrainingDisplay:
    def __init__(self, config, trainer, rich_console: Console):
        self.config = config
        self.display_config: DisplayConfig = getattr(config, "display", DisplayConfig())
        self.trainer = trainer
        self.rich_console = rich_console
        self.rich_log_messages = trainer.rich_log_messages

        self.board_component: Optional[ShogiBoard] = None
        self.trend_component: Optional[Sparkline] = None
        self.elo_component_enabled: bool = False
        self.using_enhanced_layout: bool = False

        if self.display_config.enable_board_display:
            self.board_component = ShogiBoard(
                use_unicode=self.display_config.board_unicode_pieces,
                show_moves=True,
                max_moves=self.display_config.move_list_length,
                indent_spaces=20,
                vertical_offset=2,
            )
        if self.display_config.enable_trend_visualization:
            self.trend_component = Sparkline(width=self.display_config.sparkline_width)
        if self.display_config.enable_elo_ratings:
            self.elo_component_enabled = True

        (
            self.progress_bar,
            self.training_task,
            self.layout,
            self.log_panel,
        ) = self._setup_rich_progress_display()

    def _create_compact_layout(
        self, log_panel: Panel, progress_bar: Progress
    ) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="main_log", ratio=1),
            Layout(name="progress_display", size=2),
        )
        layout["main_log"].update(log_panel)
        layout["progress_display"].update(progress_bar)
        return layout

    def _create_enhanced_layout(
        self, log_panel: Panel, progress_bar: Progress
    ) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="main_log", ratio=1),
            Layout(name="dashboard", ratio=self.display_config.dashboard_height_ratio),
            Layout(
                name="progress_display", size=self.display_config.progress_bar_height
            ),
        )
        layout["dashboard"].split_row(
            Layout(name="board_panel"),
            Layout(name="trends_panel"),
            Layout(name="elo_panel"),
        )
        layout["board_panel"].update(Panel(Text("..."), title="Shogi Board"))
        layout["trends_panel"].update(Panel(Text("Collecting data..."), title="Trends"))
        layout["elo_panel"].update(Panel(Text("Elo pending"), title="Elo Ratings"))
        layout["main_log"].update(log_panel)
        layout["progress_display"].update(progress_bar)
        return layout

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
        adaptive = AdaptiveDisplayManager(self.display_config)
        layout_type = adaptive.choose_layout(self.rich_console)
        if layout_type == "enhanced":
            layout = self._create_enhanced_layout(log_panel, progress_bar)
            self.using_enhanced_layout = True
        else:
            layout = self._create_compact_layout(log_panel, progress_bar)
            self.using_enhanced_layout = False
        return progress_bar, training_task, layout, log_panel

    def _build_metrics_table(self, history) -> Table:
        table = Table(box=None, show_edge=False, expand=False)
        table.add_column("Metric")
        table.add_column("Last", justify="right")
        table.add_column("Prev", justify="right")
        table.add_column("Prev2", justify="right")

        def vals(lst):
            v1 = f"{lst[-1]:.4f}" if len(lst) >= 1 else "-"
            v2 = f"{lst[-2]:.4f}" if len(lst) >= 2 else "-"
            v3 = f"{lst[-3]:.4f}" if len(lst) >= 3 else "-"
            return v1, v2, v3

        for name, lst in [
            ("LR", history.learning_rates),
            ("KL", history.kl_divergences),
            ("PolL", history.policy_losses),
            ("ValL", history.value_losses),
        ]:
            v1, v2, v3 = vals(lst)
            table.add_row(name, v1, v2, v3)

        table.caption = "LR=Learning rate, KL=approx KL divergence"
        return table

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
        if self.using_enhanced_layout:
            if self.board_component:
                try:
                    self.layout["board_panel"].update(
                        self.board_component.render(
                            trainer.game,
                            (
                                trainer.step_manager.move_history
                                if trainer.step_manager
                                else None
                            ),
                            trainer.policy_output_mapper,
                        )
                    )
                except Exception as e:
                    self.rich_console.log(
                        f"Error rendering board: {e}", style="bold red"
                    )
                    self.layout["board_panel"].update(Panel(Text("No board")))
            if self.trend_component:
                hist = trainer.metrics_manager.history
                w = self.display_config.sparkline_width
                trends = []
                if hist.learning_rates:
                    trends.append(
                        "LR: " + self.trend_component.generate(hist.learning_rates[-w:])
                    )
                if hist.policy_losses:
                    trends.append(
                        "PL: " + self.trend_component.generate(hist.policy_losses[-w:])
                    )
                if hist.value_losses:
                    trends.append(
                        "VL: " + self.trend_component.generate(hist.value_losses[-w:])
                    )
                if hist.kl_divergences:
                    trends.append(
                        "KL: " + self.trend_component.generate(hist.kl_divergences[-w:])
                    )
                if hist.win_rates_history:
                    wr_values = [
                        d.get("win_rate_black", 0.0) for d in hist.win_rates_history
                    ]
                    trends.append(
                        "Win%: " + self.trend_component.generate(wr_values[-w:])
                    )
                trend_text = "\n".join(trends) if trends else "Collecting data..."

                metrics_table = self._build_metrics_table(hist)
                model_graphic = Text(
                    "Model evolution: "
                    + "=" * len(trainer.previous_model_selector.get_all()),
                    style="magenta",
                )
                self.layout["trends_panel"].update(
                    Panel(
                        Group(
                            metrics_table, Text(trend_text, style="cyan"), model_graphic
                        ),
                        border_style="cyan",
                        title="Metric Trends",
                    )
                )
            if self.elo_component_enabled:
                elo = trainer.metrics_manager.elo_system
                lines = [
                    f"Black: {elo.black_rating:.0f}",
                    f"White: {elo.white_rating:.0f}",
                    f"Diff: {elo.black_rating - elo.white_rating:+.0f}",
                    "",
                    f"Assessment: {elo.get_strength_assessment()}",
                ]
                snap = getattr(trainer, "evaluation_elo_snapshot", None)
                if snap:
                    lines.extend(
                        [
                            "",
                            f"Current {snap['current_id']}: {snap['current_rating']:.0f}",
                            f"Opp {snap['opponent_id']}: {snap['opponent_rating']:.0f}",
                            f"Last result: {snap['last_outcome']}",
                        ]
                    )
                    if snap.get("top_ratings"):
                        lines.append("")
                        for mid, rating in snap["top_ratings"]:
                            lines.append(f"{mid}: {rating:.0f}")
                else:
                    lines.append("")
                    lines.append("No evaluation data yet")
                self.layout["elo_panel"].update(
                    Panel(
                        Text("\n".join(lines), style="yellow"),
                        border_style="yellow",
                        title="Elo Ratings",
                    )
                )

    def start(self):
        return Live(
            self.layout,
            console=self.rich_console,
            refresh_per_second=self.config.training.refresh_per_second,
            transient=False,
        )
