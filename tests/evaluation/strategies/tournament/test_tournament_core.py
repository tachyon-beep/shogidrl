"""Core tournament evaluator functionality tests.

This module contains tests for the basic tournament evaluator setup,
configuration validation, and core tournament logic.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from keisei.evaluation.core import BaseEvaluator, SummaryStats
from keisei.evaluation.strategies.tournament import TournamentEvaluator
from keisei.utils import PolicyOutputMapper


class TestTournamentCore:
    """Test core tournament evaluator functionality."""

    def test_init(self, mock_tournament_config):
        """Test tournament evaluator initialization."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        assert evaluator.config == mock_tournament_config
        assert (
            evaluator.logger is not None
        ), "Logger should be initialized by BaseEvaluator"
        assert isinstance(evaluator.policy_mapper, PolicyOutputMapper)

    def test_validate_config_valid(self, mock_tournament_config):
        """Test validation of valid tournament configuration."""
        mock_tournament_config.opponent_pool_config = [
            {"name": "Opp1", "type": "random"}
        ]
        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch.object(BaseEvaluator, "validate_config", return_value=True):
            assert evaluator.validate_config() is True

    def test_validate_config_missing_opponent_pool(self, mock_tournament_config):
        """Test validation with missing opponent pool configuration."""
        mock_tournament_config.opponent_pool_config = None
        evaluator = TournamentEvaluator(mock_tournament_config)
        with (
            patch.object(BaseEvaluator, "validate_config", return_value=True),
            patch(
                "keisei.evaluation.strategies.tournament.logger", MagicMock()
            ) as mock_logger,
        ):
            assert evaluator.validate_config() is True
            mock_logger.warning.assert_called_once()
            assert (
                "TournamentConfig.opponent_pool_config is missing or not a list"
                in mock_logger.warning.call_args[0][0]
            )

    def test_validate_config_opponent_pool_not_list(self, mock_tournament_config):
        """Test validation when opponent pool config is not a list."""
        mock_tournament_config.opponent_pool_config = {"not": "a list"}
        evaluator = TournamentEvaluator(mock_tournament_config)
        with (
            patch.object(BaseEvaluator, "validate_config", return_value=True),
            patch(
                "keisei.evaluation.strategies.tournament.logger", MagicMock()
            ) as mock_logger,
        ):
            assert evaluator.validate_config() is True
            mock_logger.warning.assert_called_once()
            assert (
                "TournamentConfig.opponent_pool_config is missing or not a list"
                in mock_logger.warning.call_args[0][0]
            )

    def test_validate_config_base_invalid(self, mock_tournament_config):
        """Test validation when base configuration is invalid."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch.object(BaseEvaluator, "validate_config", return_value=False):
            assert evaluator.validate_config() is False

    @pytest.mark.asyncio
    async def test_evaluate_no_opponents(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test evaluation behavior when no opponents are configured."""
        mock_tournament_config.opponent_pool_config = []
        evaluator = TournamentEvaluator(mock_tournament_config)

        evaluator.setup_context = MagicMock(return_value=mock_evaluation_context)
        evaluator.log_evaluation_start = MagicMock()
        evaluator.log_evaluation_complete = MagicMock()
        evaluator._load_tournament_opponents = AsyncMock(return_value=[])

        with patch(
            "keisei.evaluation.core.SummaryStats.from_games",
            MagicMock(return_value=MagicMock(spec=SummaryStats)),
        ) as mock_summary_stats_from_games:
            result = await evaluator.evaluate(mock_agent_info)

            assert result is not None
            assert result.games == []
            assert result.context == mock_evaluation_context
            mock_summary_stats_from_games.assert_called_once_with([])
            evaluator.setup_context.assert_called_once_with(mock_agent_info)
            evaluator.log_evaluation_start.assert_called_once()
            evaluator.log_evaluation_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_tournament_standings_no_games(
        self, mock_tournament_config, mock_agent_info
    ):
        """Test tournament standings calculation with no games."""
        evaluator = TournamentEvaluator(mock_tournament_config)

        standings = evaluator._calculate_tournament_standings([], [], mock_agent_info)

        assert standings["overall_tournament_stats"]["total_games"] == 0

    @pytest.mark.asyncio
    async def test_calculate_tournament_standings_with_games(
        self, mock_tournament_config, mock_agent_info
    ):
        """Test tournament standings calculation with actual games."""
        evaluator = TournamentEvaluator(mock_tournament_config)

        # Create mock games
        game1 = MagicMock()
        game1.winner = 0  # Agent wins
        game1.agent_color = "sente"
        game1.opponent_name = "Opponent1"

        game2 = MagicMock()
        game2.winner = 1  # Opponent wins
        game2.agent_color = "gote"
        game2.opponent_name = "Opponent1"

        game3 = MagicMock()
        game3.winner = None  # Draw
        game3.agent_color = "sente"
        game3.opponent_name = "Opponent2"

        games = [game1, game2, game3]
        
        # Create mock opponents
        opponents = [
            MagicMock(name="Opponent1"),
            MagicMock(name="Opponent2")
        ]

        standings = evaluator._calculate_tournament_standings(games, opponents, mock_agent_info)

        # Verify standings structure
        assert "overall_tournament_stats" in standings
        assert standings["overall_tournament_stats"]["total_games"] == 3
