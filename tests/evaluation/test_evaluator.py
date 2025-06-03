"""
Tests for the Evaluator class.

This module contains integration tests for the Evaluator class which provides 
a high-level interface for running complete evaluations.
"""

from unittest.mock import MagicMock

import torch

from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.evaluate import Evaluator
from keisei.evaluation.opponents.base_opponent import BaseOpponent


def test_evaluator_class_basic(monkeypatch, tmp_path, policy_mapper):
    """
    Integration test for the Evaluator class: random-vs-agent, no W&B, log to file.
    
    This test verifies that the Evaluator class correctly:
    - Initializes with proper configuration
    - Loads agents and opponents through the evaluation system
    - Executes the evaluation pipeline
    - Produces proper results and logging
    - Works without W&B integration
    """

    # Create dummy agent and opponent for integration testing
    class DummyAgent(PPOAgent):
        def __init__(self):
            from keisei.core.neural_network import ActorCritic
            from keisei.evaluation.policy_output_mapper import PolicyOutputMapper
            from conftest import make_test_config

            config = make_test_config("cpu", 46, PolicyOutputMapper())
            policy_mapper_instance = PolicyOutputMapper()
            mock_model = ActorCritic(
                config.env.input_channels, policy_mapper_instance.get_total_actions()
            )
            super().__init__(
                model=mock_model,
                config=config,
                device=torch.device("cpu"),
                name="DummyAgent",
            )
            self.model = MagicMock()

        def select_action(self, obs, legal_mask, *, is_training=True):
            # Always pick the first legal move, index 0, dummy log_prob and value
            idx = (
                int(legal_mask.nonzero(as_tuple=True)[0][0]) if legal_mask.any() else 0
            )
            return None, idx, 0.0, 0.0

    class DummyOpponent(BaseOpponent):
        def __init__(self):
            super().__init__(name="DummyOpponent")

        def select_move(self, game_instance):
            return game_instance.get_legal_moves()[0]

    # Patch the loading functions to return our dummy implementations
    monkeypatch.setattr(
        "keisei.evaluation.evaluate.load_evaluation_agent",
        lambda *a, **kw: DummyAgent(),
    )
    monkeypatch.setattr(
        "keisei.evaluation.evaluate.initialize_opponent",
        lambda *a, **kw: DummyOpponent(),
    )

    log_file = tmp_path / "evaluator_test.log"

    # Create and run evaluator
    evaluator = Evaluator(
        agent_checkpoint_path="dummy_agent.pth",
        opponent_type="random",
        opponent_checkpoint_path=None,
        num_games=1,
        max_moves_per_game=5,
        device_str="cpu",
        log_file_path_eval=str(log_file),
        policy_mapper=policy_mapper,
        seed=42,
        wandb_log_eval=False,
        wandb_project_eval=None,
        wandb_entity_eval=None,
        wandb_run_name_eval=None,
        logger_also_stdout=False,
    )

    results = evaluator.evaluate()

    # Verify results structure
    assert isinstance(results, dict)
    assert results["games_played"] == 1
    assert "agent_wins" in results 
    assert "opponent_wins" in results 
    assert "draws" in results

    # Verify logging occurred
    with open(log_file, encoding="utf-8") as f:
        log_content = f.read()
    assert "Starting Shogi Agent Evaluation" in log_content
    assert "[Evaluator] Evaluation Summary:" in log_content
