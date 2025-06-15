"""Tournament game execution tests.

This module contains tests for game execution mechanics including player action selection,
move validation, turn processing, and game state management.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from keisei.evaluation.strategies.tournament import TournamentEvaluator
from keisei.shogi.shogi_core_definitions import Color
from keisei.shogi.shogi_game import ShogiGame


class TestPlayerActions:
    """Test player action selection functionality."""

    @pytest.mark.asyncio
    async def test_game_get_player_action_ppo_agent(self, mock_tournament_config):
        """Test getting action from PPO agent entity."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_player_entity = MagicMock()
        mock_player_entity.select_action = MagicMock(
            return_value=("move_data", "action_info")
        )
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_observation = MagicMock(return_value="observation_data")
        mock_legal_mask = MagicMock()

        action = await evaluator._get_player_action(
            mock_player_entity, mock_game, [], mock_legal_mask
        )

        assert action == "move_data"
        mock_player_entity.select_action.assert_called_once_with(
            "observation_data", mock_legal_mask, is_training=False
        )
        mock_game.get_observation.assert_called_once()

    @pytest.mark.asyncio
    async def test_game_get_player_action_opponent(self, mock_tournament_config):
        """Test getting action from opponent entity."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent_entity = MagicMock()
        mock_opponent_entity.select_action = None  # Make sure this path isn't taken
        mock_opponent_entity.select_move = MagicMock(return_value="opponent_move")
        mock_game = MagicMock(spec=ShogiGame)
        mock_legal_mask = MagicMock()

        action = await evaluator._get_player_action(
            mock_opponent_entity, mock_game, [], mock_legal_mask
        )

        assert action == "opponent_move"
        mock_opponent_entity.select_move.assert_called_once_with(mock_game)


class TestMoveValidation:
    """Test move validation and execution functionality."""

    @pytest.mark.asyncio
    async def test_validate_and_make_move_valid(self, mock_tournament_config):
        """Test validation and execution of valid move."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.make_move = MagicMock()
        mock_move = "valid_move"
        legal_moves = ["valid_move", "other_move"]
        current_player_color_value = 0
        player_entity_type_name = "test_agent"

        result = await evaluator._validate_and_make_move(
            mock_game,
            mock_move,
            legal_moves,
            current_player_color_value,
            player_entity_type_name,
        )

        assert result is True
        mock_game.make_move.assert_called_once_with(mock_move)

    @pytest.mark.asyncio
    async def test_validate_and_make_move_none_move(self, mock_tournament_config):
        """Test validation when move is None."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_move = None
        legal_moves = ["legal_move"]
        current_player_color_value = 0
        player_entity_type_name = "test_agent"

        result = await evaluator._validate_and_make_move(
            mock_game,
            mock_move,
            legal_moves,
            current_player_color_value,
            player_entity_type_name,
        )

        assert result is False
        assert mock_game.game_over is True
        assert mock_game.winner == Color(1)  # Opponent wins
        assert mock_game.termination_reason == "Illegal/No move"
