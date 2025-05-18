# shogi_move_execution.py
"""
Contains functions for applying and reverting moves in the Shogi game.
These functions operate on a ShogiGame instance.
"""

from typing import Any, Dict, TYPE_CHECKING, cast

from .shogi_core_definitions import (
    Piece,
    PieceType,
    Color,
    MoveTuple,
    BASE_TO_PROMOTED_TYPE,
    PIECE_TYPE_TO_HAND_TYPE,
)

if TYPE_CHECKING:
    from .shogi_game import ShogiGame  # For type hinting the 'game' parameter


def apply_move_to_board(game: "ShogiGame", move_tuple: MoveTuple, is_simulation: bool = False) -> None:
    """
    Make a move and update the game state.
    Operates on the 'game' (ShogiGame instance).

    Board move: (r_from, c_from, r_to, c_to, promote_flag: bool)
    Drop move: (None, None, r_to, c_to, piece_type_to_drop: PieceType)

    Args:
        game: The ShogiGame instance.
        move_tuple: The move to apply.
        is_simulation: If True, indicates the move is part of a simulation
                       and game-ending checks (checkmate, stalemate) should be skipped.
    """
    r_from_orig, c_from_orig, r_to_orig, c_to_orig, move_info_orig = move_tuple
    player_making_move = game.current_player
    move_details: Dict[str, Any] = {
        "move": move_tuple,
        "captured": None,
        "is_drop": False,
        "original_type_before_promotion": None,
        "was_promoted_in_move": False,  # Initialize this for board moves
    }

    # Type narrowing for move_tuple variants
    if (
        r_from_orig is None
        and c_from_orig is None
        # and isinstance(move_info_orig, PieceType) # This check is good for robustness
    ):  # Drop move
        # Ensure move_info_orig is PieceType for drop moves
        if not isinstance(move_info_orig, PieceType):
            raise TypeError(
                f"Invalid move_info for drop move: {move_info_orig}. Expected PieceType."
            )

        r_to, c_to, piece_type_to_drop = r_to_orig, c_to_orig, move_info_orig

        if game.hands[player_making_move.value].get(piece_type_to_drop, 0) > 0:
            game.hands[player_making_move.value][piece_type_to_drop] -= 1
            dropped_piece = Piece(piece_type_to_drop, player_making_move)
            game.set_piece(r_to, c_to, dropped_piece)

            move_details["is_drop"] = True
            move_details["dropped_piece_type"] = piece_type_to_drop
        else:
            raise ValueError(
                f"Cannot drop {piece_type_to_drop.name}: not in hand for {player_making_move.name}"
            )
    # Ensure r_from_orig, c_from_orig are not None for board moves, and move_info_orig is bool
    elif (
        r_from_orig is not None
        and c_from_orig is not None
        # and isinstance(move_info_orig, bool) # This check is good for robustness
    ):  # Board move
        # Ensure move_info_orig is bool for board moves
        if not isinstance(move_info_orig, bool):
            raise TypeError(
                f"Invalid move_info for board move: {move_info_orig}. Expected bool."
            )

        r_from, c_from, r_to, c_to, promote_flag = (
            r_from_orig,
            c_from_orig,
            r_to_orig,
            c_to_orig,
            move_info_orig,
        )

        moving_piece = game.get_piece(r_from, c_from)
        if not moving_piece or moving_piece.color != player_making_move:
            raise ValueError(
                f"Invalid move: no piece or wrong color at ({r_from},{c_from}) for {player_making_move.name}"
            )

        captured_target_on_square = game.get_piece(r_to, c_to)
        if captured_target_on_square:
            # Store a copy of the piece data, not the object itself if it might be mutated
            move_details["captured"] = Piece(
                captured_target_on_square.type, captured_target_on_square.color
            )
            game.add_to_hand(captured_target_on_square, player_making_move)

        original_type = moving_piece.type  # Store before potential promotion

        if promote_flag:
            if moving_piece.type in BASE_TO_PROMOTED_TYPE:
                move_details["original_type_before_promotion"] = moving_piece.type
                moving_piece.type = BASE_TO_PROMOTED_TYPE[moving_piece.type]
            # No explicit error here if promotion is invalid, assuming get_legal_moves prevents this.
            # Original code had a pass for Gold/King/already promoted,
            # implying get_legal_moves ensures promote_flag is False for them.

        game.set_piece(r_to, c_to, moving_piece)
        game.set_piece(r_from, c_from, None)
        move_details["was_promoted_in_move"] = original_type != moving_piece.type
    else:
        raise TypeError(
            f"Invalid move_tuple structure received by apply_move_to_board: {move_tuple}"
        )

    # Record state hash *after* piece placement but *before* player switch.
    move_details["state_hash"] = game.get_board_state_hash()
    game.move_history.append(move_details)

    game.move_count += 1
    game.current_player = (
        Color.WHITE if player_making_move == Color.BLACK else Color.BLACK
    )

    # Check for game end conditions for the new current_player
    # Only perform these checks if not in a simulation context to prevent recursion
    if not is_simulation:
        if game.is_sennichite():
            game.game_over = True
            game.winner = None  # Draw by repetition
        elif game.is_checkmate():  # Check for checkmate for the new current_player
            game.game_over = True
            # The player who made the last move (player_making_move) is the winner
            game.winner = player_making_move
        elif game.is_stalemate():  # Check for stalemate for the new current_player
            game.game_over = True
            game.winner = None  # Draw by stalemate

    # The game instance\'s is_sennichite method will be a wrapper
    # if game.is_sennichite(): # This check is now done above
    #     game.game_over = True
    #     game.winner = None  # Draw

    # Check for game over by checkmate or stalemate (no legal moves for the new current_player)
    # is typically handled by the game loop after this function returns. # This is now handled here.


def revert_last_applied_move(game: "ShogiGame") -> None:
    """
    Reverts the last move made in the game.
    Operates on the 'game' (ShogiGame instance).
    """
    if not game.move_history:
        raise RuntimeError("No move to undo")

    last_move_details = game.move_history.pop()
    move_tuple: MoveTuple = last_move_details["move"]

    # Switch current player back first
    game.current_player = (
        Color.WHITE if game.current_player == Color.BLACK else Color.BLACK
    )
    player_who_made_move = game.current_player

    if last_move_details["is_drop"]:
        dropped_piece_type: PieceType = last_move_details["dropped_piece_type"]
        r_to_drop = move_tuple[2]
        c_to_drop = move_tuple[3]

        game.set_piece(r_to_drop, c_to_drop, None)
        game.hands[player_who_made_move.value][dropped_piece_type] += 1
    else:  # Board move
        orig_r_from: int = cast(int, move_tuple[0])
        orig_c_from: int = cast(int, move_tuple[1])
        orig_r_to: int = cast(int, move_tuple[2])
        orig_c_to: int = cast(int, move_tuple[3])
        # promote_flag_original = move_tuple[4] # Stored in last_move_details if needed

        moved_piece = game.get_piece(orig_r_to, orig_c_to)
        if not moved_piece:
            raise RuntimeError(
                f"Inconsistent state: Expected piece at ({orig_r_to}, {orig_c_to}) for undo."
            )

        if last_move_details.get("was_promoted_in_move", False):
            original_type = last_move_details.get("original_type_before_promotion")
            if original_type is not None:
                moved_piece.type = original_type
            else:
                # This implies an inconsistency if was_promoted_in_move is True but no original_type stored.
                pass

        game.set_piece(orig_r_from, orig_c_from, moved_piece)

        captured_piece_data = last_move_details.get("captured")
        if captured_piece_data:  # captured_piece_data is a Piece object
            game.set_piece(orig_r_to, orig_c_to, captured_piece_data)  # Restore captured piece to board

            if captured_piece_data.type != PieceType.KING:
                hand_equivalent_type = PIECE_TYPE_TO_HAND_TYPE.get(captured_piece_data.type)
                
                # Ensure hand_equivalent_type is a valid PieceType before using as a key
                if hand_equivalent_type is None:
                    # This case should ideally not be reached if PIECE_TYPE_TO_HAND_TYPE is comprehensive
                    # for all capturable, non-king pieces.
                    # If it's a base type not in the map (e.g. Gold), it should be the type itself.
                    if captured_piece_data.type in PieceType.get_unpromoted_types() and captured_piece_data.type != PieceType.KING:
                        hand_equivalent_type = captured_piece_data.type
                    else:
                        # This indicates an issue with the piece type or the mapping
                        raise ValueError(f"Cannot determine hand equivalent for captured piece type: {captured_piece_data.type}")

                if game.hands[player_who_made_move.value].get(hand_equivalent_type, 0) > 0:
                    game.hands[player_who_made_move.value][hand_equivalent_type] -= 1
                else:
                    # This would be an inconsistency: a non-King piece was captured,
                    # should have been added to hand, but not found to remove.
                    # Consider logging a warning or raising an error for debugging.
                    pass # Or raise error
        else:
            game.set_piece(orig_r_to, orig_c_to, None)  # Square becomes empty

    game.move_count -= 1
    game.game_over = False
    game.winner = None
    # Repetition history and sennichite status will be naturally re-evaluated by game logic.
