"""
Tests for the modern evaluation system components.

This module contains tests for EvaluationManager and modern evaluation strategies,
replacing the legacy execute_full_evaluation_run function tests.
"""

from unittest.mock import MagicMock, patch

import pytest

from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.core.evaluation_config import (
    EvaluationStrategy,
    create_evaluation_config,
)
from keisei.evaluation.manager import EvaluationManager
from keisei.utils import PolicyOutputMapper


def test_evaluation_manager_single_opponent_basic(tmp_path):
    """
    Test basic functionality of EvaluationManager with single opponent strategy.

    Verifies the modern evaluation pipeline works correctly including:
    - EvaluationManager initialization
    - Single opponent evaluation strategy
    - Agent evaluation without W&B integration
    - Result aggregation
    """
    # Create evaluation configuration
    config = create_evaluation_config(
        strategy=EvaluationStrategy.SINGLE_OPPONENT,
        num_games=2,
        wandb_logging=False,
        opponent_name="random",
    )

    # Create EvaluationManager
    eval_manager = EvaluationManager(
        config=config,
        run_name="test_run",
        pool_size=3,
        elo_registry_path=str(tmp_path / "elo_registry.json"),
    )

    # Setup with mock components
    policy_mapper = PolicyOutputMapper()
    eval_manager.setup(
        device="cpu",
        policy_mapper=policy_mapper,
        model_dir=str(tmp_path),
        wandb_active=False,
    )

    # Mock the evaluation strategy directly instead of legacy components
    with patch(
        "keisei.evaluation.manager.EvaluatorFactory.create"
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
            total_games=2,
            agent_wins=1,
            opponent_wins=1,
            draws=0,
            win_rate=0.5,
            loss_rate=0.5,
            draw_rate=0.0,
            avg_game_length=25.0,
            total_moves=50,
            avg_duration_seconds=120.0,
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

        # Create test agent
        mock_agent = MagicMock(spec=PPOAgent)
        mock_agent.name = "TestAgent"
        mock_agent.model = MagicMock()

        # Test current agent evaluation (the main API)
        result = eval_manager.evaluate_current_agent(mock_agent)

        # Verify results
        assert result is not None
        assert result.summary_stats.total_games == 2
        assert abs(result.summary_stats.win_rate - 0.5) < 0.01

        # Verify that the modern system created the evaluator
        mock_create_evaluator.assert_called_once()


def test_evaluation_manager_with_checkpoint(tmp_path):
    """
    Test EvaluationManager.evaluate_checkpoint functionality.

    Verifies that checkpoint-based evaluation works correctly.
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
        run_name="test_checkpoint_run",
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

    # Create test checkpoint file (just needs to exist)
    checkpoint_path = tmp_path / "test_checkpoint.pth"
    checkpoint_path.write_text("dummy_checkpoint")

    # Mock the evaluation strategy for checkpoint test
    with patch(
        "keisei.evaluation.manager.EvaluatorFactory.create"
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
            agent_wins=1,
            opponent_wins=0,
            draws=0,
            win_rate=1.0,
            loss_rate=0.0,
            draw_rate=0.0,
            avg_game_length=15.0,
            total_moves=30,
            avg_duration_seconds=90.0,
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

        # Test checkpoint evaluation
        result = eval_manager.evaluate_checkpoint(str(checkpoint_path))

        # Verify results
        assert result is not None
        assert result.summary_stats.total_games == 1
        assert abs(result.summary_stats.win_rate - 1.0) < 0.01

        # Verify that the modern system created the evaluator
        mock_create_evaluator.assert_called_once()


def test_evaluation_manager_current_agent_with_model_check(tmp_path):
    """
    Test EvaluationManager.evaluate_current_agent with model validation.

    Verifies that current agent evaluation works and validates model attribute.
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
        run_name="test_current_agent",
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

    # Create mock current agent
    mock_agent = MagicMock(spec=PPOAgent)
    mock_agent.name = "CurrentAgent"
    mock_agent.model = MagicMock()  # Important: agent must have model attribute

    # Mock the evaluation strategy for current agent test
    with patch(
        "keisei.evaluation.manager.EvaluatorFactory.create"
    ) as mock_create_evaluator:

        # Create a mock evaluator that returns expected results
        mock_evaluator = MagicMock()

        # Mock evaluation result for current agent test
        from keisei.evaluation.core.evaluation_context import EvaluationContext
        from keisei.evaluation.core.evaluation_result import (
            EvaluationResult,
            SummaryStats,
        )

        # Create proper context and summary stats for current agent test
        mock_context = MagicMock(spec=EvaluationContext)
        mock_summary = SummaryStats(
            total_games=1,
            agent_wins=0,
            opponent_wins=1,
            draws=0,
            win_rate=0.0,
            loss_rate=1.0,
            draw_rate=0.0,
            avg_game_length=30.0,
            total_moves=60,
            avg_duration_seconds=180.0,
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

        # Test current agent evaluation
        result = eval_manager.evaluate_current_agent(mock_agent)

        # Verify results
        assert result is not None
        assert result.summary_stats.total_games == 1
        assert abs(result.summary_stats.win_rate - 0.0) < 0.01

        # Verify that the modern system created the evaluator
        mock_create_evaluator.assert_called_once()


def test_evaluation_manager_agent_without_model_raises_error(tmp_path):
    """
    Test that EvaluationManager raises error for agent without model attribute.
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
        run_name="test_error_case",
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

    # Create agent without model attribute
    mock_agent = MagicMock()
    mock_agent.name = "AgentWithoutModel"
    # Explicitly set model to None to trigger the validation error
    mock_agent.model = None

    # Test that evaluation raises ValueError
    with pytest.raises(ValueError, match="Agent must have a 'model' attribute"):
        eval_manager.evaluate_current_agent(mock_agent)


def test_evaluation_manager_tournament_strategy_with_mock(tmp_path):
    """
    Test EvaluationManager with tournament strategy using mocked evaluator.

    Verifies the tournament evaluation strategy works correctly.
    """
    # Create evaluation configuration for tournament
    config = create_evaluation_config(
        strategy=EvaluationStrategy.TOURNAMENT,
        num_games=2,
        wandb_logging=False,
        opponent_pool_config=[
            {"name": "agent1", "type": "ppo", "checkpoint_path": "agent1.pth"},
            {"name": "agent2", "type": "ppo", "checkpoint_path": "agent2.pth"},
        ],
    )

    # Create EvaluationManager
    eval_manager = EvaluationManager(
        config=config,
        run_name="test_tournament",
        pool_size=5,
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

    # Create mock current agent
    mock_agent = MagicMock(spec=PPOAgent)
    mock_agent.name = "TournamentAgent"
    mock_agent.model = MagicMock()

    with patch(
        "keisei.evaluation.core.base_evaluator.EvaluatorFactory.create"
    ) as mock_factory:
        # Setup mock tournament evaluator
        mock_evaluator = MagicMock()
        mock_factory.return_value = mock_evaluator

        # Setup mock tournament results
        from keisei.evaluation.core.evaluation_context import EvaluationContext
        from keisei.evaluation.core.evaluation_result import (
            EvaluationResult,
            SummaryStats,
        )

        mock_result = EvaluationResult(
            summary_stats=SummaryStats(
                total_games=4,  # 2 games against each of 2 opponents
                agent_wins=2,
                opponent_wins=2,
                draws=0,
                win_rate=0.5,
                loss_rate=0.5,
                draw_rate=0.0,
                avg_game_length=20.0,
                total_moves=80,
                avg_duration_seconds=15.0,
            ),
            games=[],
            context=MagicMock(spec=EvaluationContext),
            analytics_data={"tournament_standing": 2},
        )

        # Mock the async evaluate method
        async def mock_evaluate(_agent_info, _context):
            return mock_result

        mock_evaluator.evaluate = mock_evaluate

        # Test tournament evaluation
        result = eval_manager.evaluate_current_agent(mock_agent)

        # Verify results
        assert result is not None
        assert result.summary_stats.total_games == 4
        assert abs(result.summary_stats.win_rate - 0.5) < 0.01
        assert "tournament_standing" in result.analytics_data

        # Verify factory was called
        mock_factory.assert_called_once_with(config)
