"""
training/display.py: Rich UI management for the Shogi RL trainer.
"""

from typing import List, Union, Optional

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

from keisei.config_schema import DisplayConfig
from .display_components import (
    ShogiBoard,
    RecentMovesPanel,
    Sparkline,
    MultiMetricSparkline,
    RollingAverageCalculator,
)
from .adaptive_display import AdaptiveDisplayManager


class TrainingDisplay:
    def __init__(self, config, trainer, rich_console: Console):
        self.config = config
        self.display_config: DisplayConfig = config.display
        self.trainer = trainer
        self.rich_console = rich_console
        self.rich_log_messages = trainer.rich_log_messages

        self.board_component: Optional[ShogiBoard] = None
        self.moves_component: Optional[RecentMovesPanel] = None
        self.trend_component: Optional[Sparkline] = None
        self.multi_trend_component: Optional[MultiMetricSparkline] = None
        self.completion_rate_calc: Optional[RollingAverageCalculator] = None
        self.elo_component_enabled: bool = False
        self.using_enhanced_layout: bool = False

        if self.display_config.enable_board_display:
            self.board_component = ShogiBoard(
                use_unicode=self.display_config.board_unicode_pieces,
                show_moves=True,
                max_moves=self.display_config.move_list_length,
            )
            self.moves_component = RecentMovesPanel(
                max_moves=self.display_config.move_list_length
            )
        if self.display_config.enable_trend_visualization:
            self.trend_component = Sparkline(width=self.display_config.sparkline_width)
            self.multi_trend_component = MultiMetricSparkline(
                width=self.display_config.sparkline_width,
                metrics=["Moves", "Turns"],
            )
            self.completion_rate_calc = RollingAverageCalculator(
                window_size=self.display_config.metrics_window_size
            )
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

        # --- Correct three-column layout with fixed board height ---
        layout["dashboard"].split_row(
            Layout(name="left_column", ratio=1),
            Layout(name="middle_column", ratio=1),
            Layout(name="right_column", ratio=1),
        )

        layout["left_column"].split_column(
            Layout(name="board_panel", size=12),  # Set size to 12 for the new Table-based board
            Layout(name="moves_panel"),
        )

        layout["middle_column"].split_column(
            Layout(name="trends_panel", ratio=1),
            Layout(name="stats_panel", ratio=1),
        )

        layout["right_column"].split_column(
            Layout(name="evolution_panel", ratio=1),
            Layout(name="elo_panel", ratio=1),
        )

        layout["board_panel"].update(Panel("...", title="Main Board"))
        layout["moves_panel"].update(Panel("...", title="Recent Moves"))
        layout["trends_panel"].update(Panel("...", title="Metric Trends"))
        layout["stats_panel"].update(Panel("...", title="Game Statistics"))
        layout["evolution_panel"].update(Panel("...", title="Model Evolution"))
        layout["elo_panel"].update(Panel("...", title="Elo Ratings"))

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

    def _build_metric_lines(self, history) -> List[str]:
        metrics = [
            ("Learning Rate", history.learning_rates),
            ("Policy Loss", history.policy_losses),
            ("Value Loss", history.value_losses),
            ("Entropy", history.entropies),
            ("KL Divergence", history.kl_divergences),
            ("Episode Length", history.episode_lengths),
            ("Episode Reward", history.episode_rewards),
        ]
        lines = []
        width = self.display_config.sparkline_width
        LABEL_WIDTH = 16

        def vals(lst):
            last = f"{lst[-1]:.4f}" if len(lst) >= 1 else "-"
            prev = f"{lst[-2]:.4f}" if len(lst) >= 2 else "-"
            return last, prev

        for name, lst in metrics:
            last, prev = vals(lst)
            spark = self.trend_component.generate(lst[-width:]) if lst else "-" * width
            lines.append(f"{name:<{LABEL_WIDTH}} [{last}] [{prev}] {spark}")

        moves = history.win_rates_history
        if moves:
            recent_b = [d.get("win_rate_black", 0.0) for d in moves]
            recent_w = [d.get("win_rate_white", 0.0) for d in moves]
            recent_d = [d.get("win_rate_draw", 0.0) for d in moves]
            spark_b = self.trend_component.generate(recent_b[-width:])
            spark_w = self.trend_component.generate(recent_w[-width:])
            spark_d = self.trend_component.generate(recent_d[-width:])
            last_b = f"{recent_b[-1]:.2f}"
            last_w = f"{recent_w[-1]:.2f}"
            last_d = f"{recent_d[-1]:.2f}"
            prev_b = f"{recent_b[-2]:.2f}" if len(recent_b) > 1 else "-"
            prev_w = f"{recent_w[-2]:.2f}" if len(recent_w) > 1 else "-"
            prev_d = f"{recent_d[-2]:.2f}" if len(recent_d) > 1 else "-"
            lines.append(
                f"{'Win Rate B':<{LABEL_WIDTH}} [{last_b}] [{prev_b}] {spark_b}"
            )
            lines.append(
                f"{'Win Rate W':<{LABEL_WIDTH}} [{last_w}] [{prev_w}] {spark_w}"
            )
            lines.append(
                f"{'Draw Rate':<{LABEL_WIDTH}} [{last_d}] [{prev_d}] {spark_d}"
            )

        return lines

    def update_progress(self, trainer, speed, pending_updates):
        update_data = {"completed": trainer.global_timestep, "speed": speed}
        update_data.update(pending_updates)
        self.progress_bar.update(self.training_task, **update_data)

    def refresh_dashboard_panels(self, trainer):
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
                        self.board_component.render(trainer.game)
                    )
                except Exception as e:
                    self.rich_console.log(
                        f"Error rendering board: {e}", style="bold red"
                    )
                    self.layout["board_panel"].update(Panel(Text("No board")))

            if self.moves_component:
                move_strings = (
                    trainer.step_manager.move_log if trainer.step_manager else None
                )
                self.layout["moves_panel"].update(
                    self.moves_component.render(move_strings)
                )

            if self.trend_component:
                hist = trainer.metrics_manager.history
                metric_lines = self._build_metric_lines(hist)
                group_items = [Text(line) for line in metric_lines]
                self.layout["trends_panel"].update(
                    Panel(
                        Group(*group_items), border_style="cyan", title="Metric Trends"
                    )
                )

                stats_lines: List[str] = []
                mm = trainer.metrics_manager
                width = self.display_config.sparkline_width
                stats_lines.append(f"Turns: {mm.global_timestep}")
                stats_lines.append(f"Games: {mm.total_episodes_completed}")
                if mm.turns_per_game:
                    avg_len = sum(mm.turns_per_game) / len(mm.turns_per_game)
                    len_spark = self.trend_component.generate(
                        list(mm.turns_per_game)[-width:], range_min=1, range_max=500
                    )
                    stats_lines.append(f"Avg Length: {avg_len:.1f} {len_spark}")
                if mm.history.win_rates_history:
                    recent_b = [
                        d.get("win_rate_black", 0.0)
                        for d in mm.history.win_rates_history
                    ]
                    recent_w = [
                        d.get("win_rate_white", 0.0)
                        for d in mm.history.win_rates_history
                    ]
                    recent_d = [
                        d.get("win_rate_draw", 0.0)
                        for d in mm.history.win_rates_history
                    ]
                    spark_b = self.trend_component.generate(
                        recent_b[-width:], range_min=0, range_max=100
                    )
                    spark_w = self.trend_component.generate(
                        recent_w[-width:], range_min=0, range_max=100
                    )
                    spark_d = self.trend_component.generate(
                        recent_d[-width:], range_min=0, range_max=100
                    )
                    stats_lines.append(f"B Win%: {recent_b[-1]:.1f} {spark_b}")
                    stats_lines.append(f"W Win%: {recent_w[-1]:.1f} {spark_w}")
                    stats_lines.append(f"Draw%: {recent_d[-1]:.1f} {spark_d}")
                self.layout["stats_panel"].update(
                    Panel(
                        Group(*[Text(l) for l in stats_lines]),
                        border_style="green",
                        title="Game Statistics",
                    )
                )

            model = getattr(trainer.agent, "model", None)
            if model is not None:
                stats_lines = []
                named_params = dict(model.named_parameters())
                for name, p in named_params.items():
                    if ".weight" in name and (
                        "policy_head" in name or "value_head" in name or "stem" in name
                    ):
                        data = p.data.float().cpu().numpy()
                        stats_lines.append(
                            f"Layer: {name}\n  Mean: {data.mean():.4f} | Std Dev: {data.std():.4f} | Min: {data.min():.2f} | Max: {data.max():.2f}"
                        )
                arch = (
                    "[Input: 9x9xN] -> [ResNet Core] -> [Policy Head]\n"
                    "                       -> [Value Head]"
                )
                evo_text = arch + "\n\n" + "\n\n".join(stats_lines)
                self.layout["evolution_panel"].update(
                    Panel(
                        Text(evo_text), border_style="magenta", title="Model Evolution"
                    )
                )
            if self.elo_component_enabled:
                snap = getattr(trainer, "evaluation_elo_snapshot", None)
                if snap and snap.get("top_ratings") and len(snap["top_ratings"]) >= 2:
                    lines = [
                        f"{mid}: {rating:.0f}" for mid, rating in snap["top_ratings"]
                    ]
                    content = Text("\n".join(lines), style="yellow")
                else:
                    content = Text(
                        "Waiting for initial model evaluations...",
                        style="yellow",
                    )
                self.layout["elo_panel"].update(
                    Panel(content, border_style="yellow", title="Elo Ratings")
                )

    def start(self):
        return Live(
            self.layout,
            console=self.rich_console,
            refresh_per_second=self.config.training.refresh_per_second,
            transient=False,
        )
