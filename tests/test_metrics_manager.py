from keisei.training.metrics_manager import MetricsManager
import time


def test_log_episode_metrics_and_rates():
    mm = MetricsManager(history_size=10)
    mm.log_episode_metrics(40, 40, "win")
    time.sleep(0.01)
    mm.log_episode_metrics(30, 30, "loss")
    rates = mm.get_win_loss_draw_rates(window_size=2)
    assert rates["win"] == 0.5
    assert rates["loss"] == 0.5
    assert mm.get_moves_per_game_trend()[-1] == 30
