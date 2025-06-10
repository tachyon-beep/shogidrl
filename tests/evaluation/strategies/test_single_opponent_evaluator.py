import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import torch


def async_test(coro):
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


from keisei.evaluation.core import (
    AgentInfo,
    EvaluationContext,
    OpponentInfo,
    SingleOpponentConfig,
)
from keisei.evaluation.strategies.single_opponent import SingleOpponentEvaluator


@async_test
async def test_load_agent_instance_direct():
    cfg = SingleOpponentConfig(opponent_name="opp")
    evaluator = SingleOpponentEvaluator(cfg)
    dummy_agent = MagicMock()
    info = AgentInfo(name="agent", metadata={"agent_instance": dummy_agent})

    with patch(
        "keisei.evaluation.strategies.single_opponent.load_evaluation_agent",
        new_callable=AsyncMock,
    ) as mock_load:
        loaded = await evaluator._load_evaluation_entity(info, "cpu", 46)
        mock_load.assert_not_called()
        assert loaded is dummy_agent


@async_test
async def test_evaluate_in_memory_basic():
    """Test basic in-memory evaluation functionality."""
    cfg = SingleOpponentConfig(
        opponent_name="test_opponent", opponent_path="dummy/path", num_games=2
    )
    evaluator = SingleOpponentEvaluator(cfg)

    # Create mock agent info
    agent_info = AgentInfo(name="test_agent", checkpoint_path="dummy/agent/path")

    # Create mock weights
    agent_weights = {"weight1": torch.randn(5, 5)}
    opponent_weights = {"weight1": torch.randn(5, 5)}

    # Mock the internal methods
    with (
        patch.object(evaluator, "validate_agent", return_value=True),
        patch.object(evaluator, "validate_config", return_value=True),
        patch.object(evaluator, "setup_context") as mock_setup_context,
        patch.object(evaluator, "log_evaluation_start"),
        patch.object(evaluator, "log_evaluation_complete"),
        patch.object(
            evaluator, "evaluate_step_in_memory", new_callable=AsyncMock
        ) as mock_evaluate_step,
        patch.object(evaluator, "_calculate_analytics", return_value={}),
    ):

        # Mock context
        mock_context = MagicMock()
        mock_context.session_id = "test_session"
        mock_setup_context.return_value = mock_context

        # Mock game results
        from keisei.evaluation.core import create_game_result

        mock_game_result = create_game_result(
            game_id="test_game_1",
            agent_info=agent_info,
            opponent_info=OpponentInfo(name="test_opponent", type="ppo_agent"),
            winner=0,
            moves_count=50,
            duration_seconds=1.5,
        )
        mock_evaluate_step.return_value = mock_game_result

        # Test in-memory evaluation
        result = await evaluator.evaluate_in_memory(
            agent_info=agent_info,
            agent_weights=agent_weights,
            opponent_weights=opponent_weights,
        )

        # Verify results
        assert result is not None
        assert len(result.games) == 2  # num_games configured
        assert result.summary_stats.total_games == 2

        # Verify that weights were stored during evaluation
        assert mock_evaluate_step.call_count == 2


@async_test
async def test_load_evaluation_entity_in_memory_fallback():
    """Test that in-memory entity loading falls back to regular loading when weights not available."""
    cfg = SingleOpponentConfig(opponent_name="test_opponent")
    evaluator = SingleOpponentEvaluator(cfg)

    # Create agent info without in-memory weights
    agent_info = AgentInfo(name="test_agent", checkpoint_path="dummy/path")

    with patch(
        "keisei.evaluation.strategies.single_opponent.load_evaluation_agent"
    ) as mock_load:
        mock_agent = MagicMock()
        mock_load.return_value = mock_agent

        # Test loading without in-memory weights (should fallback)
        result = await evaluator._load_evaluation_entity_in_memory(
            agent_info, "cpu", 46
        )

        # Should have called the regular loading function
        mock_load.assert_called_once()
        assert result is mock_agent
