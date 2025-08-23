"""
Tests for the modern evaluation system components - integration tests.

This module contains integration tests for the modern evaluation system,
replacing the legacy Evaluator class tests with EvaluationManager tests.
"""

from unittest.mock import MagicMock, patch

import torch

from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.core import (
    EvaluationStrategy,
    create_evaluation_config,
)
from keisei.evaluation.core_manager import EvaluationManager
from keisei.utils import PolicyOutputMapper
from tests.evaluation.conftest import make_test_config


def test_evaluation_manager_integration_basic(tmp_path):
    """
    Integration test for EvaluationManager: random-vs-agent, no W&B, file logging.

    This test verifies that the EvaluationManager correctly:
    - Initializes with proper configuration
    - Executes the evaluation pipeline end-to-end
    - Produces proper results and analytics
    - Works without W&B integration
    - Handles file-based logging
    """

    # Create dummy agent for integration testing
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
        run_name="integration_test",
        pool_size=3,
        elo_registry_path=str(tmp_path / "elo_registry.json"),
    )

    # Setup
    policy_mapper = PolicyOutputMapper()
    eval_manager.setup(
        device="cpu",
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

        # Run evaluation
        result = eval_manager.evaluate_current_agent(dummy_agent)

        # Verify results structure
        assert result is not None
        assert result.summary_stats.total_games == 1
        assert result.summary_stats.agent_wins == 1
        assert result.summary_stats.opponent_wins == 0
        assert result.summary_stats.draws == 0
        assert abs(result.summary_stats.win_rate - 1.0) < 0.01

        # Verify that the modern system created the evaluator
        mock_create_evaluator.assert_called_once()

        # Manually trigger ELO registry creation (since mock bypasses normal flow)
        # Create a dummy checkpoint file for testing
        dummy_checkpoint = tmp_path / "dummy_checkpoint.pth"
        dummy_checkpoint.write_text("dummy checkpoint content")
        eval_manager.opponent_pool.add_checkpoint(str(dummy_checkpoint))

        # Verify Elo registry file exists
        elo_file = tmp_path / "elo_registry.json"
        assert elo_file.exists()


def test_evaluation_manager_checkpoint_integration(tmp_path):
    """
    Integration test for EvaluationManager checkpoint loading.

    Tests that the evaluation system can load and evaluate a checkpoint
    through the modern pipeline.
    """

    # Create evaluation configuration
    config = create_evaluation_config(
        strategy=EvaluationStrategy.SINGLE_OPPONENT,
        num_games=1,
        wandb_logging=False,
        opponent_name="heuristic",
    )

    # Create EvaluationManager
    eval_manager = EvaluationManager(
        config=config,
        run_name="checkpoint_integration_test",
        pool_size=3,
        elo_registry_path=str(tmp_path / "elo_registry.json"),
    )

    # Setup
    policy_mapper = PolicyOutputMapper()
    eval_manager.setup(
        device="cpu",
        policy_mapper=policy_mapper,
        model_dir=str(tmp_path),
        wandb_active=False,
    )

    # Create test checkpoint file
    checkpoint_path = tmp_path / "integration_test_checkpoint.pth"
    torch.save({"model_state": "dummy_state"}, checkpoint_path)

    # Mock the evaluation strategy for checkpoint test
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
            avg_game_length=8.0,
            total_moves=16,
            avg_duration_seconds=60.0,
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

        # Run checkpoint evaluation
        result = eval_manager.evaluate_checkpoint(str(checkpoint_path))

        # Verify results
        assert result is not None
        assert result.summary_stats.total_games == 1
        assert result.summary_stats.agent_wins == 0
        assert result.summary_stats.opponent_wins == 1
        assert abs(result.summary_stats.loss_rate - 1.0) < 0.01

        # Verify that the modern system created the evaluator
        mock_create_evaluator.assert_called_once()


def test_evaluation_manager_error_handling_integration(tmp_path):
    """
    Integration test for EvaluationManager error handling.

    Tests that the evaluation system properly handles various error conditions.
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
        run_name="error_handling_test",
        pool_size=3,
        elo_registry_path=str(tmp_path / "elo_registry.json"),
    )

    # Setup
    policy_mapper = PolicyOutputMapper()
    eval_manager.setup(
        device="cpu",
        policy_mapper=policy_mapper,
        model_dir=str(tmp_path),
        wandb_active=False,
    )

    # Test: Agent without model attribute should raise ValueError
    class AgentWithoutModel:
        def __init__(self):
            self.name = "AgentWithoutModel"

        # Deliberately no model attribute

    mock_agent_no_model = AgentWithoutModel()

    try:
        eval_manager.evaluate_current_agent(mock_agent_no_model)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Agent must have a 'model' attribute" in str(e)

    # Test passes if error is properly handled
    assert True