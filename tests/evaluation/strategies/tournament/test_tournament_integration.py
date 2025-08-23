"""Tournament integration tests.

This module contains tests for full tournament evaluation workflows,
game execution integration, and end-to-end tournament functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from keisei.evaluation.core import (
    EvaluationResult,
    GameResult,
    OpponentInfo,
    SummaryStats,
)
from keisei.evaluation.strategies.tournament import TournamentEvaluator
from keisei.shogi.shogi_core_definitions import Color


class TestGameExecution:
    """Test game execution integration functionality."""

    @pytest.mark.asyncio
    async def test_play_games_against_opponent(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test playing games against an opponent."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent = OpponentInfo(name="TestOpp", type="random", metadata={})

        # Mock game results
        mock_game_result_1 = MagicMock(spec=GameResult)
        mock_game_result_1.winner = 0  # Agent wins
        mock_game_result_1.agent_color = "sente"
        mock_game_result_1.opponent_name = "TestOpp"

        mock_game_result_2 = MagicMock(spec=GameResult)
        mock_game_result_2.winner = 1  # Opponent wins
        mock_game_result_2.agent_color = "gote"
        mock_game_result_2.opponent_name = "TestOpp"

        evaluator.evaluate_step = AsyncMock(
            side_effect=[mock_game_result_1, mock_game_result_2]
        )

        results, errors = await evaluator._play_games_against_opponent(
            mock_agent_info, mock_opponent, 2, mock_evaluation_context
        )

        assert len(results) == 2
        assert results[0] == mock_game_result_1
        assert results[1] == mock_game_result_2
        assert evaluator.evaluate_step.call_count == 2

        # Check that both games were called with alternating colors
        calls = evaluator.evaluate_step.call_args_list
        assert len(calls) == 2

        # Verify the structure of calls (agent_info, opponent, context)
        agent_info_1, opponent_1, context_1 = calls[0][0]
        assert agent_info_1 == mock_agent_info
        assert isinstance(opponent_1, OpponentInfo)
        assert opponent_1.name == mock_opponent.name
        assert context_1 == mock_evaluation_context

        agent_info_2, opponent_2, context_2 = calls[1][0]
        assert agent_info_2 == mock_agent_info
        assert isinstance(opponent_2, OpponentInfo)
        assert opponent_2.name == mock_opponent.name
        assert context_2 == mock_evaluation_context

        # Verify metadata for alternating colors - check the opponent metadata
        # Game 0 should have agent_plays_sente_in_eval_step = True (agent plays sente)
        # Game 1 should have agent_plays_sente_in_eval_step = False (agent plays gote)
        assert opponent_1.metadata.get("agent_plays_sente_in_eval_step") == True
        assert opponent_2.metadata.get("agent_plays_sente_in_eval_step") == False

    @pytest.mark.asyncio
    async def test_play_games_against_opponent_eval_step_error(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test playing games when evaluate step encounters error."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent = OpponentInfo(name="ErrorOpp", type="problematic")

        evaluator.evaluate_step = AsyncMock(
            side_effect=RuntimeError("Evaluation step failed")
        )

        with patch(
            "keisei.evaluation.strategies.tournament.logger", MagicMock()
        ) as mock_logger:
            results, errors = await evaluator._play_games_against_opponent(
                mock_agent_info, mock_opponent, 2, mock_evaluation_context
            )

            assert len(results) == 0  # No successful games
            assert len(errors) == 2  # Two error messages
            mock_logger.error.assert_called()


class TestEvaluationSteps:
    """Test individual evaluation step functionality."""

    @pytest.mark.asyncio
    async def test_evaluate_step_successful_game_agent_sente(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test successful evaluation step with agent as sente."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent = OpponentInfo(name="TestOpp", type="random")

        # Mock entities
        mock_agent_entity = MagicMock()
        mock_opponent_entity = MagicMock()

        # Mock game and result
        mock_game = MagicMock()
        mock_game.winner = Color(0)  # Agent wins (Sente is 0)
        mock_game.moves_count = 42
        mock_game.game_over = True

        # Mock the internal methods properly
        with patch.object(
            evaluator,
            "_load_evaluation_entity",
            side_effect=[mock_agent_entity, mock_opponent_entity],
        ) as mock_load:
            with patch.object(
                evaluator, "_run_tournament_game_loop", return_value=42
            ) as mock_game_loop:
                result = await evaluator.evaluate_step(
                    mock_agent_info, mock_opponent, mock_evaluation_context
                )

            # Verify the result is a GameResult (not the mock)
            assert isinstance(result, GameResult)
            assert result.moves_count == 42
            assert mock_load.call_count == 2  # Called for agent and opponent
            mock_game_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_step_successful_game_agent_gote(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test successful evaluation step with agent as gote."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent = OpponentInfo(name="TestOpp", type="random")

        # Mock entities
        mock_agent_entity = MagicMock()
        mock_opponent_entity = MagicMock()

        # Mock the internal methods properly
        with patch.object(
            evaluator,
            "_load_evaluation_entity",
            side_effect=[mock_agent_entity, mock_opponent_entity],
        ):
            with patch.object(evaluator, "_run_tournament_game_loop", return_value=35):
                result = await evaluator.evaluate_step(
                    mock_agent_info, mock_opponent, mock_evaluation_context
                )

                assert result is not None
                assert isinstance(result, GameResult)

    @pytest.mark.asyncio
    async def test_evaluate_step_game_loop_error(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test evaluation step when game loop encounters error."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent = OpponentInfo(name="ProblematicOpp", type="error_prone")

        # Mock entities
        mock_agent_entity = MagicMock()
        mock_opponent_entity = MagicMock()

        with patch.multiple(
            evaluator,
            _load_evaluation_entity=MagicMock(
                side_effect=[mock_agent_entity, mock_opponent_entity]
            ),
            _run_tournament_game_loop=AsyncMock(
                side_effect=RuntimeError("Game loop crashed")
            ),
        ):
            with patch(
                "keisei.evaluation.strategies.tournament.logger", MagicMock()
            ) as mock_logger:
                result = await evaluator.evaluate_step(
                    mock_agent_info, mock_opponent, mock_evaluation_context
                )

                # The evaluator returns a GameResult with error information, not None
                assert result is not None
                assert isinstance(result, GameResult)
                assert "error" in result.metadata
                assert "Game loop crashed" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_evaluate_step_load_entity_error(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test evaluation step when entity loading fails."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent = OpponentInfo(name="UnloadableOpp", type="broken")

        evaluator._load_evaluation_entity = MagicMock(
            side_effect=ValueError("Cannot load entity")
        )

        with patch(
            "keisei.evaluation.strategies.tournament.logger", MagicMock()
        ) as mock_logger:
            result = await evaluator.evaluate_step(
                mock_agent_info, mock_opponent, mock_evaluation_context
            )

            # The evaluator returns a GameResult with error information, not None
            assert result is not None
            assert isinstance(result, GameResult)
            assert "error" in result.metadata
            assert "Cannot load entity" in result.metadata["error"]


class TestFullEvaluation:
    """Test full tournament evaluation workflows."""

    @pytest.mark.asyncio
    async def test_evaluate_full_run_calculates_num_games_per_opponent_dynamically(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test full evaluation with dynamic game count calculation."""
        # Configure for dynamic calculation using strategy params
        mock_tournament_config.num_games = 20
        mock_tournament_config.set_strategy_param("num_games_per_opponent", None)

        # Set up 3 opponents
        opponents = [
            OpponentInfo(name="Opp1", type="random"),
            OpponentInfo(name="Opp2", type="random"),
            OpponentInfo(name="Opp3", type="random"),
        ]

        evaluator = TournamentEvaluator(mock_tournament_config)
        evaluator.setup_context = MagicMock(return_value=mock_evaluation_context)
        evaluator.log_evaluation_start = MagicMock()
        evaluator.log_evaluation_complete = MagicMock()
        evaluator._load_tournament_opponents = AsyncMock(return_value=opponents)

        # Mock game results
        mock_games = [MagicMock(spec=GameResult) for _ in range(20)]
        # Set up proper attributes for the mock GameResult objects
        for i, game in enumerate(mock_games):
            game.winner = i % 3  # Some variety in winners (0, 1, None represented as 2)
            if game.winner == 2:
                game.winner = None
            game.opponent_info = MagicMock()
            game.opponent_info.name = (
                f"Opp{(i // 7) + 1}"  # Distribute across opponents
            )
        evaluator._play_games_against_opponent = AsyncMock(
            side_effect=[
                (mock_games[0:7], []),  # (games, errors) tuple for 7 games against Opp1
                (
                    mock_games[7:14],
                    [],
                ),  # (games, errors) tuple for 7 games against Opp2
                (
                    mock_games[14:20],
                    [],
                ),  # (games, errors) tuple for 6 games against Opp3
            ]
        )

        # Mock summary stats
        mock_summary_stats = MagicMock(spec=SummaryStats)
        mock_summary_stats.total_games = 20
        with patch(
            "keisei.evaluation.core.SummaryStats.from_games",
            return_value=mock_summary_stats,
        ):
            result = await evaluator.evaluate(mock_agent_info)

            assert result is not None
            assert len(result.games) == 20
            assert result.summary_stats == mock_summary_stats

            # Verify dynamic game distribution (20 games / 3 opponents = 6-7 games each)
            play_calls = evaluator._play_games_against_opponent.call_args_list
            assert len(play_calls) == 3

            # Check game counts: should be roughly equal distribution
            game_counts = [
                call[0][2] for call in play_calls
            ]  # Third argument is num_games
            assert sum(game_counts) == 20
            for count in game_counts:
                assert 6 <= count <= 7  # Should be 6 or 7 games per opponent

    @pytest.mark.asyncio
    async def test_evaluate_full_run_with_opponents_and_games(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test full evaluation run with configured opponents and games."""
        # Configure specific game count per opponent using strategy params
        mock_tournament_config.set_strategy_param("num_games_per_opponent", 4)

        # Set up 2 opponents
        opponents = [
            OpponentInfo(name="StrongOpp", type="ppo_agent"),
            OpponentInfo(name="WeakOpp", type="random"),
        ]

        evaluator = TournamentEvaluator(mock_tournament_config)
        evaluator.setup_context = MagicMock(return_value=mock_evaluation_context)
        evaluator.log_evaluation_start = MagicMock()
        evaluator.log_evaluation_complete = MagicMock()
        evaluator._load_tournament_opponents = AsyncMock(return_value=opponents)

        # Mock game results (4 games per opponent = 8 total)
        mock_games = [MagicMock(spec=GameResult) for _ in range(8)]
        # Set up proper attributes for the mock GameResult objects
        for i, game in enumerate(mock_games):
            game.winner = i % 3  # Some variety in winners (0, 1, None represented as 2)
            if game.winner == 2:
                game.winner = None
            game.opponent_info = MagicMock()
            game.opponent_info.name = "StrongOpp" if i < 4 else "WeakOpp"
        evaluator._play_games_against_opponent = AsyncMock(
            side_effect=[
                (
                    mock_games[0:4],
                    [],
                ),  # (games, errors) tuple for 4 games against StrongOpp
                (
                    mock_games[4:8],
                    [],
                ),  # (games, errors) tuple for 4 games against WeakOpp
            ]
        )

        # Mock summary stats and standings
        mock_summary_stats = MagicMock(spec=SummaryStats)
        mock_summary_stats.total_games = 8
        mock_standings = {"TestAgent": {"wins": 5, "losses": 3}}

        with patch(
            "keisei.evaluation.core.SummaryStats.from_games",
            return_value=mock_summary_stats,
        ):
            evaluator._calculate_tournament_standings = MagicMock(
                return_value=mock_standings
            )

            result = await evaluator.evaluate(mock_agent_info)

            assert result is not None
            assert len(result.games) == 8
            assert result.summary_stats == mock_summary_stats

            # Verify both opponents were played against
            play_calls = evaluator._play_games_against_opponent.call_args_list
            assert len(play_calls) == 2

            # Each call should be for 4 games
            for call in play_calls:
                assert call[0][2] == 4  # num_games parameter

            evaluator._calculate_tournament_standings.assert_called_once()