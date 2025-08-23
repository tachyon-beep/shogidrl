"""Core tournament evaluator functionality tests.

This module contains tests for the basic tournament evaluator setup,
configuration validation, and core tournament logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        mock_tournament_config.set_strategy_param("opponent_pool_config", [
            {"name": "Opp1", "type": "random"}
        ])
        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch.object(BaseEvaluator, "validate_config", return_value=True):
            assert evaluator.validate_config() is True

    def test_validate_config_missing_opponent_pool(self, mock_tournament_config):
        """Test validation with missing opponent pool configuration."""
        mock_tournament_config.set_strategy_param("opponent_pool_config", None)
        evaluator = TournamentEvaluator(mock_tournament_config)
        with (
            patch.object(BaseEvaluator, "validate_config", return_value=True),
            patch(
                "keisei.evaluation.strategies.tournament.logger", MagicMock()
            ) as mock_logger,
        ):
            assert evaluator.validate_config() is True
            mock_logger.warning.assert_called_once()
            # Updated to match actual message format
            assert (
                "opponent_pool_config is missing or not a list"
                in mock_logger.warning.call_args[0][0]
            )

    def test_validate_config_opponent_pool_not_list(self, mock_tournament_config):
        """Test validation with invalid opponent pool configuration."""
        mock_tournament_config.set_strategy_param("opponent_pool_config", "not_a_list")
        evaluator = TournamentEvaluator(mock_tournament_config)
        with (
            patch.object(BaseEvaluator, "validate_config", return_value=True),
            patch(
                "keisei.evaluation.strategies.tournament.logger", MagicMock()
            ) as mock_logger,
        ):
            assert evaluator.validate_config() is True
            mock_logger.warning.assert_called_once()
            # Updated to match actual message format
            assert (
                "opponent_pool_config is missing or not a list"
                in mock_logger.warning.call_args[0][0]
            )

    def test_validate_config_base_invalid(self, mock_tournament_config):
        """Test validation with invalid base configuration."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch.object(BaseEvaluator, "validate_config", return_value=False):
            assert evaluator.validate_config() is False

    def test_evaluate_no_opponents(self, mock_tournament_config, mock_agent_info):
        """Test evaluation with no opponents configured."""
        mock_tournament_config.set_strategy_param("opponent_pool_config", [])
        mock_tournament_config.set_strategy_param("num_games_per_opponent", 2)
        evaluator = TournamentEvaluator(mock_tournament_config)

        # Create async function that returns empty result
        async def run_evaluation():
            with (
                patch.object(evaluator, "validate_agent", return_value=True),
                patch.object(evaluator, "validate_config", return_value=True),
                patch.object(evaluator, "setup_context") as mock_setup_context,
                patch.object(evaluator, "log_evaluation_start"),
                patch.object(evaluator, "log_evaluation_complete"),
            ):
                mock_context = MagicMock()
                mock_context.session_id = "test_session"
                mock_setup_context.return_value = mock_context

                result = await evaluator.evaluate(mock_agent_info)
                return result

        import asyncio
        result = asyncio.run(run_evaluation())

        # Should return empty result
        assert result is not None
        assert len(result.games) == 0
        assert result.summary_stats.total_games == 0

    def test_calculate_tournament_standings_no_games(self):
        """Test tournament standings calculation with no games."""
        from keisei.evaluation.strategies.tournament import TournamentEvaluator
        from keisei.evaluation.core import create_evaluation_config, AgentInfo
        
        config = create_evaluation_config(strategy="tournament")
        evaluator = TournamentEvaluator(config)
        
        # Method signature requires opponents and agent_info
        agent_info = AgentInfo(name="TestAgent")
        opponents = []
        standings = evaluator._calculate_tournament_standings([], opponents, agent_info)
        
        # Check the actual structure returned
        assert isinstance(standings, dict)
        assert "overall_tournament_stats" in standings
        assert standings["overall_tournament_stats"]["total_games"] == 0

    def test_calculate_tournament_standings_with_games(self):
        """Test tournament standings calculation with games."""
        from keisei.evaluation.strategies.tournament import TournamentEvaluator
        from keisei.evaluation.core import create_evaluation_config, AgentInfo, OpponentInfo, create_game_result
        
        config = create_evaluation_config(strategy="tournament")
        evaluator = TournamentEvaluator(config)
        
        # Create mock games with corrected signature
        agent_info = AgentInfo(name="TestAgent")
        opp1_info = OpponentInfo(name="Opp1", type="random")
        opp2_info = OpponentInfo(name="Opp2", type="random")
        opponents = [opp1_info, opp2_info]
        
        games = [
            create_game_result(
                game_id="game1", 
                winner=0, 
                moves_count=50, 
                duration_seconds=1.0,
                agent_info=agent_info, 
                opponent_info=opp1_info
            ),
            create_game_result(
                game_id="game2", 
                winner=1, 
                moves_count=60, 
                duration_seconds=1.5,
                agent_info=agent_info, 
                opponent_info=opp2_info
            ),
            create_game_result(
                game_id="game3", 
                winner=0, 
                moves_count=55, 
                duration_seconds=1.2,
                agent_info=agent_info, 
                opponent_info=opp1_info
            ),
        ]
        
        standings = evaluator._calculate_tournament_standings(games, opponents, agent_info)
        
        # Verify the actual standings structure
        assert isinstance(standings, dict)
        assert "overall_tournament_stats" in standings
        assert "per_opponent_results" in standings
        
        # Check overall stats
        overall_stats = standings["overall_tournament_stats"]
        assert overall_stats["agent_total_wins"] == 2
        assert overall_stats["agent_total_losses"] == 1
        assert overall_stats["total_games"] == 3
        
        # Check per-opponent results
        per_opponent = standings["per_opponent_results"]
        assert "Opp1" in per_opponent
        assert "Opp2" in per_opponent
        assert per_opponent["Opp1"]["wins"] == 2  # Agent won 2 vs Opp1
        assert per_opponent["Opp2"]["wins"] == 0  # Agent won 0 vs Opp2