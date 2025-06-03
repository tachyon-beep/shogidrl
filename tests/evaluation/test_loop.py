"""
Tests for the core evaluation loop functionality.
"""

import torch

from keisei.evaluation.loop import run_evaluation_loop
from keisei.utils import PolicyOutputMapper
from keisei.utils.opponents import SimpleRandomOpponent

from .conftest import INPUT_CHANNELS, MockPPOAgent, make_test_config


def test_run_evaluation_loop_basic(eval_logger_setup):
    """Test that run_evaluation_loop runs games and logs results correctly."""
    logger, log_file_path = eval_logger_setup

    agent_to_eval = MockPPOAgent(
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="PPOAgentToEvaluate",
    )
    opponent = SimpleRandomOpponent(name="TestRandomOpponent")

    num_games = 2
    max_moves = 5  # Keep games short for testing

    results = run_evaluation_loop(agent_to_eval, opponent, num_games, logger, max_moves)

    assert results["games_played"] == num_games
    assert "agent_wins" in results
    assert "opponent_wins" in results
    assert "draws" in results
    assert (
        results["agent_wins"] + results["opponent_wins"] + results["draws"] == num_games
    )
    assert "avg_game_length" in results
    assert results["avg_game_length"] <= max_moves  # Can be less if game ends early

    with open(log_file_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    # The following log is not produced by run_evaluation_loop directly:
    # f"Starting evaluation: {agent_to_eval.name} vs {opponent.name}"
    assert f"Starting evaluation game 1/{num_games}" in log_content
    assert f"Starting evaluation game 2/{num_games}" in log_content
    assert (
        "Agent wins game 1" in log_content
        or "Opponent wins game 1" in log_content
        or "Game 1 is a draw" in log_content
    )
    assert (
        "Agent wins game 2" in log_content
        or "Opponent wins game 2" in log_content
        or "Game 2 is a draw" in log_content
    )
