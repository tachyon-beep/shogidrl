from keisei.config_schema import DisplayConfig
from keisei.training.adaptive_display import AdaptiveDisplayManager
from keisei.training.display_components import Sparkline, ShogiBoard
from keisei.training.elo_rating import EloRatingSystem
from keisei.training.metrics_manager import MetricsHistory
from keisei.shogi.shogi_core_definitions import Color
from rich.console import Console


def test_display_config_defaults():
    config = DisplayConfig()
    assert config.enable_board_display is True
    assert config.sparkline_width == 15
    assert config.elo_initial_rating == 1500.0


def test_metrics_history_trimming():
    history = MetricsHistory(max_history=3)
    for i in range(5):
        history.add_episode_data({"black_win_rate": float(i)})
    assert len(history.win_rates_history) == 3

    for i in range(5):
        history.add_ppo_data({
            "ppo/learning_rate": float(i),
            "ppo/policy_loss": float(i),
        })
    assert len(history.learning_rates) == 3
    assert len(history.policy_losses) == 3



def test_elo_rating_updates():
    elo = EloRatingSystem()
    initial = elo.black_rating
    elo.update_ratings(Color.BLACK)
    assert elo.black_rating > initial


def test_sparkline_generation():
    spark = Sparkline(width=5)
    s = spark.generate([1, 2, 3, 4, 5])
    assert len(s) == 5



def test_adaptive_layout_choice():
    cfg = DisplayConfig()
    manager = AdaptiveDisplayManager(cfg)
    console = Console(width=200, height=50, color_system=None)
    assert manager.choose_layout(console) == "enhanced"
    console_small = Console(width=80, height=20, color_system=None)
    assert manager.choose_layout(console_small) == "compact"


class DummyBoard:
    def __init__(self) -> None:
        self.board = [[None for _ in range(9)] for _ in range(9)]


def test_shogi_board_basic_render():
    board = ShogiBoard()
    panel = board.render(DummyBoard())
    assert panel.title == "Current Position"
