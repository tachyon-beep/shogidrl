"""
Unit tests for train.py (smoke test for main loop).
"""

import importlib

import config  # Import config directly to modify it


def test_train_main_runs():
    """Smoke test: train.py main() runs without error with reduced timesteps."""
    original_total_timesteps = config.TOTAL_TIMESTEPS
    original_steps_per_epoch = config.STEPS_PER_EPOCH
    original_eval_freq = config.EVAL_FREQ_EPISODES  # Changed from EVAL_FREQ_TIMESTEPS
    original_eval_episodes = config.EVAL_NUM_GAMES  # Changed from NUM_EVAL_EPISODES
    original_save_freq = config.SAVE_FREQ_EPISODES  # Added to track original save freq
    original_max_moves_eval = config.MAX_MOVES_PER_GAME_EVAL # Added

    # Override config values for a quick test run
    config.TOTAL_TIMESTEPS = 200 # Further reduced for a quicker smoke test
    config.STEPS_PER_EPOCH = 32  # Ensure learning is triggered
    config.EVAL_FREQ_EPISODES = 5  # Evaluate more frequently with fewer episodes
    config.EVAL_NUM_GAMES = 1 # Crucially, reduce games per evaluation
    config.SAVE_FREQ_EPISODES = 1  # Ensure model saving is triggered
    config.MAX_MOVES_PER_GAME_EVAL = 50 # Reduce max moves in eval games
    # Ensure MINIBATCH_SIZE is not larger than STEPS_PER_EPOCH for the test
    config.MINIBATCH_SIZE = min(config.MINIBATCH_SIZE, config.STEPS_PER_EPOCH)

    train = importlib.import_module("train")

    try:
        importlib.reload(
            train
        )  # Reload train to pick up patched config if it imports config at module level
        train.main()
    finally:
        # Restore original config values
        config.TOTAL_TIMESTEPS = original_total_timesteps
        config.STEPS_PER_EPOCH = original_steps_per_epoch
        config.EVAL_FREQ_EPISODES = original_eval_freq
        config.EVAL_NUM_GAMES = original_eval_episodes  # Changed from NUM_EVAL_EPISODES
        config.SAVE_FREQ_EPISODES = original_save_freq  # Restore original save freq
        config.MAX_MOVES_PER_GAME_EVAL = original_max_moves_eval # Added
        # Reload train again to restore its state if necessary, or rely on subsequent tests re-importing
        importlib.reload(train)
