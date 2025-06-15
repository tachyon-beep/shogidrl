"""Shared fixtures for tournament evaluation tests."""

from unittest.mock import MagicMock

import pytest

from keisei.evaluation.core import (
    AgentInfo,
    EvaluationContext,
    OpponentInfo,
    TournamentConfig,
)


@pytest.fixture
def mock_tournament_config():
    """Mock tournament configuration for testing."""
    config = MagicMock(spec=TournamentConfig)
    config.name = "TestTournament"
    config.num_games = 10
    config.num_games_per_opponent = 2
    config.opponent_pool_config = []
    config.default_device = "cpu"
    config.input_channels = 46
    config.max_moves_per_game = 100
    config.log_level = "INFO"
    config.evaluation_strategy = "tournament"
    config.output_dir = "/tmp/eval_output"
    config.save_results = True
    config.save_replays = False
    config.custom_params = {}
    return config


@pytest.fixture
def mock_agent_info():
    """Mock agent info for testing."""
    agent_info = MagicMock(spec=AgentInfo)
    agent_info.name = "TestAgent"
    agent_info.checkpoint_path = "/path/to/agent.ptk"
    agent_info.type = "ppo_agent"
    agent_info.to_dict = MagicMock(
        return_value={
            "name": "TestAgent",
            "checkpoint_path": "/path/to/agent.ptk",
            "type": "ppo_agent",
        }
    )
    return agent_info


@pytest.fixture
def mock_opponent_info():
    """Mock opponent info for testing."""
    opponent_info = MagicMock(spec=OpponentInfo)
    opponent_info.name = "TestOpponent1"
    opponent_info.type = "random"
    opponent_info.checkpoint_path = None
    opponent_info.metadata = {}
    opponent_info.to_dict = MagicMock(
        return_value={"name": "TestOpponent1", "type": "random", "metadata": {}}
    )
    return opponent_info


@pytest.fixture
def mock_evaluation_context(mock_tournament_config, mock_agent_info):
    """Mock evaluation context for testing."""
    context = MagicMock(spec=EvaluationContext)
    context.session_id = "test_session_123"
    context.configuration = mock_tournament_config
    context.agent_info = mock_agent_info
    context.timestamp = MagicMock()
    context.environment_info = {}
    context.metadata = {}
    return context
