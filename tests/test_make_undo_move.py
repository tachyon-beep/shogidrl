import pytest

from keisei.shogi.shogi_core_definitions import Color
from keisei.shogi.shogi_game import ShogiGame

# Helper to serialize critical parts of the game state


def serialize_state(game):
    board_state = tuple(
        tuple(None if p is None else (p.type, p.color, p.is_promoted) for p in row)
        for row in game.board
    )
    hands_state = (
        # Sort by PieceType.value for consistent ordering
        tuple(
            sorted(
                game.hands[Color.BLACK.value].items(), key=lambda item: item[0].value
            )
        ),
        tuple(
            sorted(
                game.hands[Color.WHITE.value].items(), key=lambda item: item[0].value
            )
        ),
    )
    return {
        "board": board_state,
        "hands": hands_state,
        "current_player": game.current_player,
        "move_count": game.move_count,
        "game_over": game.game_over,
        "winner": game.winner,
        "termination_reason": game.termination_reason,
    }


@pytest.fixture
def fresh_game():
    """Creates a new game with default setup"""
    return ShogiGame(max_moves_per_game=512)


# DEPRECATED: All undo move tests have been consolidated into
# 'test_shogi_game_mock_comprehensive.py' and 'test_shogi_game_updated_with_mocks.py'.
# This file is retained for reference only. Do not add new tests here.
