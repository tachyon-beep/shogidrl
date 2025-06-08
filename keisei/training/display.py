"""
training/display.py: Rich UI management for the Shogi RL trainer.
"""

import copy
from typing import Dict, List, Optional, Union, cast

from rich.console import Console, Group, RenderableType
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
from rich.table import Table
from rich.text import Text

from keisei.config_schema import DisplayConfig

from .adaptive_display import AdaptiveDisplayManager
from .display_components import (
    GameStatisticsPanel,
    MultiMetricSparkline,
    PieceStandPanel,
    RecentMovesPanel,
    RollingAverageCalculator,
    ShogiBoard,
    Sparkline,
)


class TrainingDisplay:
    def __init__(self, config, trainer, rich_console: Console):
        self.config = config
        self.display_config: DisplayConfig = config.display
        self.trainer = trainer
        self.rich_console = rich_console
        self.rich_log_messages = trainer.rich_log_messages

        self.board_component: Optional[ShogiBoard] = None
        self.moves_component: Optional[RecentMovesPanel] = None
        self.piece_stand_component: Optional[PieceStandPanel] = None
        self.trend_component: Optional[Sparkline] = None
        self.multi_trend_component: Optional[MultiMetricSparkline] = None
        self.completion_rate_calc: Optional[RollingAverageCalculator] = None
        self.game_stats_component: Optional[GameStatisticsPanel] = GameStatisticsPanel()
        self.elo_component_enabled: bool = False
        self.using_enhanced_layout: bool = False

        if self.display_config.enable_board_display:
            self.board_component = ShogiBoard(
                use_unicode=self.display_config.board_unicode_pieces,
                cell_width=self.display_config.board_cell_width,
                cell_height=self.display_config.board_cell_height,
            )
            self.moves_component = RecentMovesPanel(
                max_moves=self.display_config.move_list_length,
                newest_on_top=self.display_config.moves_latest_top,
                flash_ms=self.display_config.moves_flash_ms,
            )
            self.piece_stand_component = PieceStandPanel()
        if self.display_config.enable_trend_visualization:
            self.trend_component = Sparkline(width=self.display_config.sparkline_width)
            self.multi_trend_component = MultiMetricSparkline(
                width=self.display_config.sparkline_width,
                metrics=["Moves", "Turns"],
            )
            self.completion_rate_calc = RollingAverageCalculator(window_size=self.display_config.metrics_window_size)
        if self.display_config.enable_elo_ratings:
            self.elo_component_enabled = True

        (
            self.progress_bar,
            self.training_task,
            self.layout,
            self.log_panel,
        ) = self._setup_rich_progress_display()

        # Store previous stats for trend arrows in evolution panel
        self.previous_model_stats: Optional[Dict[str, Dict[str, float]]] = None
        self.config_panel_rendered: bool = False

    def _create_compact_layout(self, log_panel: Panel, progress_bar: Progress) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="main_log", ratio=1),
            Layout(name="progress_display", size=2),
        )
        layout["main_log"].update(log_panel)
        layout["progress_display"].update(progress_bar)
        return layout

    def _create_enhanced_layout(self, log_panel: Panel, progress_bar: Progress) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="main_log", ratio=1),
            Layout(name="dashboard", ratio=self.display_config.dashboard_height_ratio),
            Layout(name="progress_display", size=self.display_config.progress_bar_height),
        )

        # --- Correct three-column layout with fixed board height ---
        layout["dashboard"].split_row(
            Layout(name="left_column", ratio=1),
            Layout(name="middle_column", ratio=1),
            Layout(name="right_column", ratio=1),
        )

        layout["left_column"].split_column(
            Layout(name="board_panel", size=12),
            Layout(name="komadai_panel", size=5),
            Layout(
                name="moves_panel",
                ratio=1,
                minimum_size=self.display_config.move_list_length,
            ),
        )

        layout["middle_column"].split_column(
            Layout(name="trends_panel", ratio=2),
            Layout(name="stats_panel", ratio=2),
            Layout(name="config_panel", ratio=1),
        )

        layout["right_column"].split_column(
            Layout(name="evolution_panel", ratio=1),
            Layout(name="elo_panel", ratio=1),
        )

        layout["board_panel"].update(Panel("...", title="Main Board"))
        layout["komadai_panel"].update(Panel("...", title="Captured Pieces"))
        layout["moves_panel"].update(Panel("...", title="Recent Moves"))
        layout["trends_panel"].update(Panel("...", title="Metric Trends"))
        layout["stats_panel"].update(Panel("...", title="Game Statistics"))
        layout["config_panel"].update(Panel("...", title="Configuration"))
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
            TextColumn("• Steps: {task.completed}/{task.total} ({task.fields[speed]:.1f} it/s)"),
            TextColumn("• {task.fields[ep_metrics]}", style="bright_cyan"),
            TextColumn("• {task.fields[ppo_metrics]}", style="bright_yellow"),
            TextColumn(
                "• Wins S:{task.fields[black_wins_cum]} G:{task.fields[white_wins_cum]} D:{task.fields[draws_cum]}",
                style="bright_green",
            ),
            TextColumn(
                "• Rates S:{task.fields[black_win_rate]:.1%} "
                "G:{task.fields[white_win_rate]:.1%} "
                "D:{task.fields[draw_rate]:.1%}",
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

    def _build_metric_lines(self, history) -> List[RenderableType]:

        metrics_to_display = [
            ("Episode Length", "episode_lengths"),
            ("Episode Reward", "episode_rewards"),
            ("Policy Loss", "policy_losses"),
            ("Value Loss", "value_losses"),
            ("Entropy", "entropies"),
            ("KL Divergence", "kl_divergences"),
            ("PPO Clip Frac", "clip_fractions"),
            ("Win Rate", "win_rates_black"),
            ("Draw Rate", "draw_rates"),
        ]

        def fmt(val: Optional[float]) -> str:
            return "-" if val is None else f"{val:.4f}"

        table = Table(box=None, expand=True, show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan", no_wrap=True, ratio=3)
        table.add_column("Last", justify="right", ratio=2)
        table.add_column("Previous", justify="right", ratio=2)
        table.add_column("Average (5)", justify="right", ratio=2)
        table.add_column("Trend", no_wrap=True, ratio=4)

        SPARKLINE_WIDTH = self.display_config.sparkline_width
        assert self.trend_component is not None
        trend = self.trend_component

        for name, history_key in metrics_to_display:
            if history_key == "win_rates_black":
                data_list = [d.get("win_rate_black", 0.0) for d in history.win_rates_history]
            elif history_key == "draw_rates":
                data_list = [d.get("win_rate_draw", 0.0) for d in history.win_rates_history]
            else:
                data_list = getattr(history, history_key, [])

            last_val = data_list[-1] if len(data_list) >= 1 else None
            prev_val = data_list[-2] if len(data_list) >= 2 else None
            avg_slice = data_list[-5:]
            avg_val = sum(avg_slice) / len(avg_slice) if avg_slice else None

            spark = trend.generate(data_list[-SPARKLINE_WIDTH:]) if data_list else " " * SPARKLINE_WIDTH

            table.add_row(
                name,
                fmt(last_val),
                fmt(prev_val),
                fmt(avg_val),
                spark,
            )

        return [table]

    def update_progress(self, trainer, speed, pending_updates):
        update_data = {"completed": trainer.global_timestep, "speed": speed}
        update_data.update(pending_updates)
        self.progress_bar.update(self.training_task, **update_data)

    def refresh_dashboard_panels(self, trainer):
        # Update the main log panel at the top
        visible_rows = max(0, self.rich_console.size.height - 6)
        if trainer.rich_log_messages:
            display_messages = trainer.rich_log_messages[-visible_rows:]
            updated_panel_content = Group(*display_messages)
            self.log_panel.renderable = updated_panel_content
        else:
            self.log_panel.renderable = Text("")

        # Only update the dashboard if we are using the enhanced layout
        if not self.using_enhanced_layout:
            return

        # --- Panel Update Logic with Debug Borders ---

        # 1. Board Panel (Red Border)
        if self.board_component:
            try:
                hot_sq = trainer.metrics_manager.get_hot_squares(top_n=3)
                board_panel = self.board_component.render(trainer.game, highlight_squares=hot_sq)
                if isinstance(board_panel, Panel):
                    board_panel.border_style = "red"  # Override border style for debugging
                self.layout["board_panel"].update(board_panel)
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                self.rich_console.log(f"[bold red]Error rendering board: {e}[/]")
                self.layout["board_panel"].update(Panel("Error", border_style="red"))

        # 2. Komadai (Piece Stand) Panel (Green Border)
        if self.piece_stand_component:
            try:
                komadai_panel = self.piece_stand_component.render(trainer.game)
                if isinstance(komadai_panel, Panel):
                    komadai_panel.border_style = "green"  # Override border style
                self.layout["komadai_panel"].update(komadai_panel)
            except (AttributeError, KeyError, TypeError) as e:
                self.rich_console.log(f"[bold red]Error rendering piece stand: {e}[/]")
                self.layout["komadai_panel"].update(Panel("Error", border_style="green"))

        # 3. Moves Panel (Blue Border)
        # In refresh_dashboard_panels()
        if self.moves_component:
            try:
                move_strings = trainer.step_manager.move_log if trainer.step_manager else None
                pps = getattr(trainer, "last_ply_per_sec", 0.0)
                # The render method no longer needs available_height
                moves_panel = self.moves_component.render(move_strings, ply_per_sec=pps)
                self.layout["moves_panel"].update(moves_panel)
            except (AttributeError, TypeError) as e:
                self.rich_console.log(f"[bold red]Error rendering moves: {e}[/]")

        if self.trend_component:
            hist = trainer.metrics_manager.history
            renderables = self._build_metric_lines(hist)
            group_items = list(renderables)

            grad_norm = getattr(trainer, "last_gradient_norm", 0.0)
            grad_norm_scaled = min(grad_norm, 50.0)
            grad_bar = Progress(
                TextColumn("Gradient Norm  "),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
            )
            grad_bar.add_task("", total=50.0, completed=int(grad_norm_scaled))
            group_items.append(grad_bar)

            try:
                buffer_bar = Progress(
                    TextColumn("Replay Buffer"),
                    BarColumn(bar_width=None),
                    TextColumn("{task.percentage:>3.0f}%"),
                )
                buf = trainer.experience_buffer
                buffer_bar.add_task("", total=buf.capacity(), completed=buf.size())
                group_items.append(buffer_bar)
            except (AttributeError, TypeError):
                pass

            if getattr(trainer.metrics_manager, "processing", False):
                group_items.append(Text("PROCESSING", style="bold red"))

            self.layout["trends_panel"].update(Panel(Group(*group_items), border_style="cyan", title="Metric Trends"))

            # TODO: the panel should exist; asserting inside try for safety
            try:
                assert self.game_stats_component is not None

                panel = cast(
                    Panel,
                    self.game_stats_component.render(
                        trainer.game,
                        (trainer.step_manager.move_log if trainer.step_manager else None),
                        trainer.metrics_manager,
                        getattr(trainer.step_manager, "sente_best_capture", None),
                        getattr(trainer.step_manager, "gote_best_capture", None),
                        getattr(trainer.step_manager, "sente_capture_count", 0),
                        getattr(trainer.step_manager, "gote_capture_count", 0),
                        getattr(trainer.step_manager, "sente_drop_count", 0),
                        getattr(trainer.step_manager, "gote_drop_count", 0),
                        getattr(trainer.step_manager, "sente_promo_count", 0),
                        getattr(trainer.step_manager, "gote_promo_count", 0),
                    ),
                )
                group_stats: List[RenderableType] = [panel.renderable]
                self.layout["stats_panel"].update(
                    Panel(
                        Group(*group_stats),
                        title="Game Statistics",
                        border_style="green",
                    )
                )
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                AssertionError,
            ) as e:
                self.layout["stats_panel"].update(Panel(f"Error: {e}", title="Game Statistics"))

        if not self.config_panel_rendered:
            try:
                cfg = self.config
                batch_size = getattr(cfg.training, "minibatch_size", None)
                if batch_size is None:
                    batch_size = getattr(cfg.parallel, "batch_size", "?")

                config_table = Table.grid(padding=(0, 2))
                config_table.add_column(style="bold")
                config_table.add_column()
                config_table.add_row("Learning Rate:", str(cfg.training.learning_rate))
                config_table.add_row("Batch Size:", str(batch_size))
                config_table.add_row("Tower Depth:", str(cfg.training.tower_depth))
                config_table.add_row("SE Ratio:", str(cfg.training.se_ratio))

                self.layout["config_panel"].update(
                    Panel(
                        config_table,
                        title="Configuration",
                        border_style="green",
                    )
                )
                self.config_panel_rendered = True
            except (AttributeError, KeyError, TypeError) as e:
                self.layout["config_panel"].update(Panel(f"Error loading config:\n{e}", title="Configuration"))

        model = getattr(trainer.agent, "model", None)
        if model is not None:
            current_stats: Dict[str, Dict[str, float]] = {}
            named_params = dict(model.named_parameters())

            # --- 1. Calculate current statistics ---
            for name, p in named_params.items():
                if ".weight" in name and any(
                    keyword in name for keyword in self.display_config.log_layer_keyword_filters
                ):
                    data = p.data.float().cpu().numpy()
                    current_stats[name] = {
                        "mean": float(data.mean()),
                        "std": float(data.std()),
                        "min": float(data.min()),
                        "max": float(data.max()),
                    }

            # --- 2. Create and populate a Rich Table ---
            stats_table = Table(title="Weight Statistics", expand=True)
            stats_table.add_column("Layer", style="cyan", no_wrap=True, ratio=2)
            stats_table.add_column("Mean", justify="right", ratio=1)
            stats_table.add_column("Std Dev", justify="right", ratio=1)
            stats_table.add_column("Min", justify="right", ratio=1)
            stats_table.add_column("Max", justify="right", ratio=1)

            for name, stats in current_stats.items():
                trend_chars = {"mean": "→", "std": "→", "min": "→", "max": "→"}
                if self.previous_model_stats and name in self.previous_model_stats:
                    for key in trend_chars:
                        prev_val = self.previous_model_stats[name][key]
                        curr_val = stats[key]
                        if curr_val > prev_val:
                            trend_chars[key] = "↑"
                        elif curr_val < prev_val:
                            trend_chars[key] = "↓"

                # Add a row to the table with formatted stats and trend arrows
                stats_table.add_row(
                    name,
                    f"{stats['mean']:.4f} {trend_chars['mean']}",
                    f"{stats['std']:.4f} {trend_chars['std']}",
                    f"{stats['min']:.2f} {trend_chars['min']}",
                    f"{stats['max']:.2f} {trend_chars['max']}",
                )

            # --- 3. Update the panel with the new table ---
            # Create a borderless table for perfect alignment
            diagram_table = Table.grid(expand=True, padding=(0, 1))
            diagram_table.add_column(justify="center")
            diagram_table.add_column(justify="center", style="dim")  # For the arrow
            diagram_table.add_column(justify="center")
            diagram_table.add_column(justify="center", style="dim")  # For the arrow
            diagram_table.add_column(justify="left")

            try:
                # Dynamically get component names from config
                obs_shape = getattr(self.config.env, "obs_shape", (46, 9, 9))
                if hasattr(self.config.env, "input_channels"):
                    input_channels = self.config.env.input_channels
                    obs_shape = (input_channels, 9, 9)

                input_shape_str = f"{obs_shape[1]}x{obs_shape[2]}x{obs_shape[0]}"
                core_name_str = f"{self.config.training.model_type.capitalize()} Core"

                # Group the two output heads to stack them vertically
                heads = Group(
                    Text("[Policy Head]", style="bold"),
                    Text("[Value Head]", style="bold"),
                )

                # Add the components as a single row in the table
                diagram_table.add_row(
                    Text(f"[Input: {input_shape_str}]", style="bold"),
                    "->",
                    Text(f"[{core_name_str}]", style="bold"),
                    "->",
                    heads,
                )
                arch_diagram = diagram_table
            except (AttributeError, IndexError):
                # Fallback to a simple table version if config is unavailable
                fallback_table = Table.grid(expand=True, padding=(0, 1))
                fallback_table.add_column(justify="center")
                fallback_table.add_row(Text("[Input] -> [Core] -> [Policy/Value Heads]", style="bold"))
                arch_diagram = fallback_table

            # Group the architecture diagram and the new stats table
            panel_content = Group(arch_diagram, stats_table)
            self.layout["evolution_panel"].update(Panel(panel_content, border_style="magenta", title="Model Evolution"))

            # --- 4. Store a deep copy for the next update (The Bug Fix) ---
            self.previous_model_stats = copy.deepcopy(current_stats)

        if self.elo_component_enabled:
            snap = getattr(trainer, "evaluation_elo_snapshot", None)
            if snap and snap.get("top_ratings") and len(snap["top_ratings"]) >= 2:
                lines = [f"{mid}: {rating:.0f}" for mid, rating in snap["top_ratings"]]
                content = Text("\n".join(lines), style="yellow")
            else:
                content = Text(
                    "Waiting for initial model evaluations...",
                    style="yellow",
                )
            self.layout["elo_panel"].update(Panel(content, border_style="yellow", title="Elo Ratings"))

    def start(self):
        return Live(
            self.layout,
            console=self.rich_console,
            refresh_per_second=self.config.training.refresh_per_second,
            transient=False,
        )
