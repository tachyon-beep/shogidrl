"""
Unit test for evaluate_agent in train.py
"""

import train
from keisei.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper, TrainingLogger


def test_evaluate_agent_runs(tmp_path):
    """Test that the evaluate_agent function runs without errors and logs results."""
    logger = TrainingLogger(str(tmp_path / "eval.log"), also_stdout=False)
    mapper = PolicyOutputMapper()
    agent = PPOAgent(input_channels=46, policy_output_mapper=mapper)
    # Should run without error and log results
    train.evaluate_agent(agent, num_games=2, logger=logger)
    logger.close()
    with open(tmp_path / "eval.log", encoding="utf-8") as f:
        lines = f.readlines()
    assert any("Evaluation Complete:" in line for line in lines)
