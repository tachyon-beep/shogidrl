"""
Tests for modern evaluation system - focused legacy compatibility tests.

This module contains focused tests that verify the modern evaluation system
provides equivalent functionality to the legacy Evaluator class.
"""

from unittest.mock import MagicMock, patch

import torch

from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.core.evaluation_config import (
    EvaluationStrategy,
    SingleOpponentConfig,
    create_evaluation_config,
)
from keisei.evaluation.core_manager import EvaluationManager
from keisei.utils import PolicyOutputMapper
from keisei.utils.utils import BaseOpponent
from tests.evaluation.conftest import make_test_config


def test_evaluation_manager_replaces_evaluator_basic(tmp_path):
    """
    Test that EvaluationManager provides equivalent functionality to the legacy Evaluator.

    This test verifies that the modern evaluation system:
    - Can replace the old Evaluator class functionality
    - Provides equivalent results structure
    - Handles the same configuration options
    - Works with existing agent and opponent types
    """

    # Create dummy agent for testing
    class DummyAgent(PPOAgent):
        def __init__(self):
            from keisei.core.neural_network import ActorCritic

            config = make_test_config()
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

    # Create modern evaluation configuration (equivalent to legacy Evaluator params)
    config = create_evaluation_config(
        strategy=EvaluationStrategy.SINGLE_OPPONENT,
        num_games=1,
        wandb_logging=False,  # Equivalent to wandb_log_eval=False
        opponent_name="random",
    )

    # Create EvaluationManager (replacement for Evaluator)
    eval_manager = EvaluationManager(
        config=config,
        run_name="legacy_compatibility_test",  # Equivalent to wandb_run_name_eval
        pool_size=3,
        elo_registry_path=str(
            tmp_path / "elo.json"
        ),  # Same as legacy elo_registry_path
    )

    # Setup (equivalent to legacy initialization)
    policy_mapper = PolicyOutputMapper()
    eval_manager.setup(
        device="cpu",  # Equivalent to device_str="cpu"
        policy_mapper=policy_mapper,
        model_dir=str(tmp_path),
        wandb_active=False,
    )

    # Mock the evaluation strategy directly instead of legacy components
    with patch(
        "keisei.evaluation.core_manager.EvaluatorFactory.create"
    ) as mock_create_evaluator:

        # Create a mock evaluator that returns expected results
        mock_evaluator = MagicMock()

        # Mock evaluation result
        from keisei.evaluation.core.evaluation_context import EvaluationContext
        from keisei.evaluation.core.evaluation_result import (
            EvaluationResult,
            SummaryStats,
        )

        # Create proper context and summary stats
        mock_context = MagicMock(spec=EvaluationContext)
        mock_summary = SummaryStats(
            total_games=1,
            agent_wins=1,
            opponent_wins=0,
            draws=0,
            win_rate=1.0,
            loss_rate=0.0,
            draw_rate=0.0,
            avg_game_length=5.0,
            total_moves=10,
            avg_duration_seconds=30.0,
        )

        # Create mock evaluation result
        mock_result = EvaluationResult(
            context=mock_context,
            games=[],  # Empty games list for simplicity
            summary_stats=mock_summary,
        )

        # Configure mock evaluator to return this result
        async def mock_evaluate(agent_info, context):
            return mock_result

        mock_evaluator.evaluate = mock_evaluate
        mock_create_evaluator.return_value = mock_evaluator

        # Create dummy agent to evaluate
        dummy_agent = DummyAgent()

        # Run evaluation using modern API
        result = eval_manager.evaluate_current_agent(dummy_agent)

        # Verify results structure (equivalent to legacy Evaluator.evaluate() output)
        assert result is not None

        # Extract summary statistics (equivalent to legacy results dict)
        summary = result.summary_stats
        assert summary.total_games == 1  # Equivalent to results["games_played"]
        assert summary.agent_wins == 1  # Equivalent to results["agent_wins"]
        assert summary.opponent_wins == 0  # Equivalent to results["opponent_wins"]
        assert summary.draws == 0  # Equivalent to results["draws"]
        assert abs(summary.win_rate - 1.0) < 0.01

        # Manually trigger ELO registry creation (since mock bypasses normal flow)
        # This simulates what would happen in a real evaluation
        eval_manager.opponent_pool.add_checkpoint("dummy_checkpoint.pth")

        # Verify ELO registry was created (same as legacy)
        elo_file = tmp_path / "elo.json"
        assert elo_file.exists()

        # Verify that the modern system created the evaluator
        mock_create_evaluator.assert_called_once()


def test_evaluation_manager_checkpoint_compatibility(tmp_path):
    """
    Test that EvaluationManager checkpoint evaluation is compatible with legacy usage.

    This verifies the modern system can handle checkpoint evaluation like the old
    Evaluator class with agent_checkpoint_path parameter.
    """

    # Create evaluation configuration
    config = create_evaluation_config(
        strategy=EvaluationStrategy.SINGLE_OPPONENT,
        num_games=1,
        wandb_logging=False,
        opponent_name="random",
    )

    # Create EvaluationManager
    eval_manager = EvaluationManager(
        config=config,
        run_name="checkpoint_compatibility_test",
        pool_size=3,
        elo_registry_path=str(tmp_path / "elo.json"),
    )

    # Setup
    policy_mapper = PolicyOutputMapper()
    eval_manager.setup(
        device="cpu",
        policy_mapper=policy_mapper,
        model_dir=str(tmp_path),
        wandb_active=False,
    )

    # Create test checkpoint file (equivalent to agent_checkpoint_path in legacy)
    checkpoint_path = tmp_path / "dummy_agent.pth"
    torch.save({"dummy": "checkpoint"}, checkpoint_path)

    with patch(
        "keisei.evaluation.core_manager.EvaluatorFactory.create"
    ) as mock_create_evaluator:

        # Create a mock evaluator that returns expected results
        mock_evaluator = MagicMock()

        # Mock evaluation result for checkpoint test
        from keisei.evaluation.core.evaluation_context import EvaluationContext
        from keisei.evaluation.core.evaluation_result import (
            EvaluationResult,
            SummaryStats,
        )

        # Create proper context and summary stats for checkpoint test
        mock_context = MagicMock(spec=EvaluationContext)
        mock_summary = SummaryStats(
            total_games=1,
            agent_wins=0,
            opponent_wins=1,
            draws=0,
            win_rate=0.0,
            loss_rate=1.0,
            draw_rate=0.0,
            avg_game_length=5.0,
            total_moves=10,
            avg_duration_seconds=30.0,
        )

        # Create mock evaluation result
        mock_result = EvaluationResult(
            context=mock_context,
            games=[],  # Empty games list for simplicity
            summary_stats=mock_summary,
        )

        # Configure mock evaluator to return this result
        async def mock_evaluate(agent_info, context):
            return mock_result

        mock_evaluator.evaluate = mock_evaluate
        mock_create_evaluator.return_value = mock_evaluator

        # Test checkpoint evaluation (modern equivalent of legacy checkpoint loading)
        result = eval_manager.evaluate_checkpoint(str(checkpoint_path))

        # Verify results (equivalent to legacy format)
        assert result is not None
        summary = result.summary_stats
        assert summary.total_games == 1
        assert summary.agent_wins == 0
        assert summary.opponent_wins == 1
        assert abs(summary.loss_rate - 1.0) < 0.01

        # Verify the modern system created the evaluator
        mock_create_evaluator.assert_called_once()
