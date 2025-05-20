import pytest

from keisei.shogi.shogi_core_definitions import BoardMoveTuple, Color, Piece, PieceType
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


def test_make_undo_simple_move(fresh_game):
    game = fresh_game
    state_before = serialize_state(game)
    # Simple pawn move from (6,4) to (5,4)
    move: BoardMoveTuple = (6, 4, 5, 4, False)
    # Use make_move and undo_move via ShogiGame interface
    game.make_move(move)
    game.undo_move()
    state_after = serialize_state(game)
    assert state_before == state_after


def test_make_undo_capture_move(fresh_game):
    game = fresh_game
    # Place opponent pawn in front of black pawn to capture
    game.set_piece(5, 4, Piece(PieceType.PAWN, Color.WHITE))
    state_before = serialize_state(game)
    move: BoardMoveTuple = (6, 4, 5, 4, False)
    game.make_move(move)
    game.undo_move()
    state_after = serialize_state(game)
    assert state_before == state_after
