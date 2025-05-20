# shogi_move_execution.py
"""
Contains functions for applying and reverting moves in the Shogi game.
These functions operate on a ShogiGame instance.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    cast,
)

from .shogi_core_definitions import (
    Color,
    MoveTuple,
    Piece,
    PieceType,
    get_unpromoted_types,
    PROMOTED_TO_BASE_TYPE, # Added PROMOTED_TO_BASE_TYPE
)
from . import shogi_rules_logic

if TYPE_CHECKING:
    from .shogi_game import ShogiGame


def apply_move_to_board(
    game: "ShogiGame",
    is_simulation: bool = False,
) -> None:
    """
    Updates the game state after a move has been applied to the board by ShogiGame.make_move.
    Its primary roles here are:
    1. Updating game.current_player.
    2. Incrementing game.move_count.
    3. Checking for game termination conditions (checkmate, stalemate, sennichite, max_moves)
       if not a simulation.

    Args:
        game: The ShogiGame instance.
        is_simulation: If True, indicates the move is part of a simulation
                       and game-ending checks (checkmate, stalemate) should be skipped.
    """
    player_who_made_the_move = game.current_player

    game.current_player = (
        Color.WHITE if player_who_made_the_move == Color.BLACK else Color.BLACK
    )
    game.move_count += 1

    if not is_simulation:
        # Pass the game instance itself to shogi_rules_logic functions
        king_of_current_player_in_check = shogi_rules_logic.is_in_check(
            game, game.current_player
        )
        legal_moves_for_current_player = game.get_legal_moves()

        if not legal_moves_for_current_player:
            if king_of_current_player_in_check:
                game.game_over = True
                game.winner = player_who_made_the_move
                game.termination_reason = "Tsumi"
            else:
                game.game_over = True
                game.winner = None
                game.termination_reason = "Stalemate"
        # Pass the game instance to check_for_sennichite
        elif shogi_rules_logic.check_for_sennichite(game):
            game.game_over = True
            game.winner = None
            game.termination_reason = "Sennichite"
        elif game.move_count >= game.max_moves_per_game:
            game.game_over = True
            game.winner = None
            game.termination_reason = "Max moves reached"


def revert_last_applied_move(game: "ShogiGame") -> None:
    """
    Reverts the last move made in the game.
    Operates on the 'game' (ShogiGame instance).
    """
    if not game.move_history:
        raise RuntimeError("No move to undo")

    last_move_details: Dict[str, Any] = game.move_history.pop()
    move_tuple: MoveTuple = last_move_details["move"]

    if game.board_history:
        game.board_history.pop()

    game.current_player = (
        Color.WHITE if game.current_player == Color.BLACK else Color.BLACK
    )
    player_who_made_the_undone_move: Color = game.current_player

    if last_move_details["is_drop"]:
        dropped_piece_type_any = last_move_details["dropped_piece_type"]
        if not isinstance(dropped_piece_type_any, PieceType):
            raise TypeError(f"Expected PieceType for dropped_piece_type, got {type(dropped_piece_type_any)}")
        dropped_piece_type: PieceType = dropped_piece_type_any
        
        r_to_drop_any = move_tuple[2]
        c_to_drop_any = move_tuple[3]

        if not isinstance(r_to_drop_any, int) or not isinstance(c_to_drop_any, int):
            raise TypeError(
                f"Expected int for r_to_drop and c_to_drop, got {type(r_to_drop_any)} and {type(c_to_drop_any)} respectively."
            )
        r_to_drop: int = r_to_drop_any
        c_to_drop: int = c_to_drop_any

        game.set_piece(r_to_drop, c_to_drop, None)
        if dropped_piece_type in get_unpromoted_types() and dropped_piece_type != PieceType.KING:
            game.hands[player_who_made_the_undone_move.value][dropped_piece_type] = \
                game.hands[player_who_made_the_undone_move.value].get(dropped_piece_type, 0) + 1
        else:
            raise ValueError(f"Attempted to return invalid piece type {dropped_piece_type} to hand during undo of drop.")

    else:  # Board move
        orig_r_from: int = cast(int, move_tuple[0])
        orig_c_from: int = cast(int, move_tuple[1])
        orig_r_to: int = cast(int, move_tuple[2])
        orig_c_to: int = cast(int, move_tuple[3])

        original_type_before_promotion_any = last_move_details["original_type_before_promotion"]
        original_color_of_moved_piece_any = last_move_details["original_color_of_moved_piece"]

        if not isinstance(original_type_before_promotion_any, PieceType):
            raise TypeError(
                f"Expected PieceType for original_type_before_promotion, got {type(original_type_before_promotion_any)}"
            )
        current_original_type_before_promotion: PieceType = original_type_before_promotion_any

        if not isinstance(original_color_of_moved_piece_any, Color):
            raise TypeError(
                f"Expected Color for original_color_of_moved_piece, got {type(original_color_of_moved_piece_any)}"
            )
        current_original_color_of_moved_piece: Color = original_color_of_moved_piece_any

        piece_to_restore_at_from = Piece(current_original_type_before_promotion, current_original_color_of_moved_piece)
        
        game.set_piece(orig_r_from, orig_c_from, piece_to_restore_at_from)

        captured_piece_object: Optional[Piece] = last_move_details.get("captured")

        if captured_piece_object:
            game.set_piece(orig_r_to, orig_c_to, captured_piece_object)
            
            type_to_remove_from_hand: PieceType
            # Use PROMOTED_TO_BASE_TYPE directly from shogi_core_definitions
            if captured_piece_object.type in PROMOTED_TO_BASE_TYPE:
                type_to_remove_from_hand = PROMOTED_TO_BASE_TYPE[captured_piece_object.type]
            else:
                type_to_remove_from_hand = captured_piece_object.type
            
            if type_to_remove_from_hand != PieceType.KING: # Kings are not held in hand
                game.remove_from_hand(type_to_remove_from_hand, player_who_made_the_undone_move)
        else:
            game.set_piece(orig_r_to, orig_c_to, None)

    game.move_count -= 1
    game.game_over = False
    game.winner = None
    game.termination_reason = None
