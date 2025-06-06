from keisei.training.metrics_manager import MetricsManager
from unittest.mock import patch


def test_log_episode_metrics_and_rates():
    mm = MetricsManager(history_size=10)
    mm.log_episode_metrics(40, 40, "win", 1.0)
    with patch("time.time") as mock_time:
        mock_time.side_effect = [1000, 1000.01]  # Simulate time progression
        mm.log_episode_metrics(30, 30, "loss", 0.5)
    mm.log_episode_metrics(30, 30, "loss", 0.2)
    rates = mm.get_win_loss_draw_rates(window_size=2)
    assert rates["win"] == 0.5
    assert rates["loss"] == 0.5
    assert mm.get_moves_per_game_trend()[-1] == 30
    assert mm.history.episode_lengths[-1] == 30
    assert mm.history.episode_rewards[-1] == 0.2
