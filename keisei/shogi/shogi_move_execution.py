# shogi_move_execution.py
"""
Contains functions for applying and reverting moves in the Shogi game.
These functions operate on a ShogiGame instance.
"""

from typing import Any, Dict
from typing import TYPE_CHECKING

from .shogi_core_definitions import (
    Piece,
    PieceType,
    Color,
    MoveTuple,
    BASE_TO_PROMOTED_TYPE,
    PROMOTED_TO_BASE_TYPE,
    PIECE_TYPE_TO_HAND_TYPE,
)

if TYPE_CHECKING:
    from .shogi_game import ShogiGame  # For type hinting the 'game' parameter


# No direct numpy or config import needed for these functions based on original code


def apply_move_to_board(game: "ShogiGame", move_tuple: MoveTuple) -> None:
    """
    Make a move and update the game state.
    Operates on the 'game' (ShogiGame instance).

    Board move: (r_from, c_from, r_to, c_to, promote_flag: bool)
    Drop move: (None, None, r_to, c_to, piece_type_to_drop: PieceType)
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

    # The game instance's is_sennichite method will be a wrapper
    if game.is_sennichite():
        game.game_over = True
        game.winner = None  # Draw

    # Check for game over by checkmate or stalemate (no legal moves for the new current_player)
    # is typically handled by the game loop after this function returns.


def revert_last_applied_move(game: "ShogiGame") -> None:
    """
    Reverts the last move made in the game.
    Operates on the 'game' (ShogiGame instance).
    """
    if not game.move_history:
        raise RuntimeError("No move to undo")

    last_move_details = game.move_history.pop()
    move_tuple: MoveTuple = last_move_details[
        "move"
    ]  # Ensure MoveTuple type from details
    # Unpack carefully based on whether it's a BoardMove or DropMove.
    # The original unpacking was generic: (r_from, c_from, r_to, c_to, _ )
    # For drop moves, r_from and c_from are None. For board moves, they are int.
    r_to_original_move = move_tuple[2]  # This is always the destination row
    c_to_original_move = move_tuple[3]  # This is always the destination col

    # Switch current player back first
    game.current_player = (
        Color.WHITE if game.current_player == Color.BLACK else Color.BLACK
    )
    player_who_made_move = (
        game.current_player
    )  # Now refers to the player whose move is being undone

    if last_move_details["is_drop"]:
        dropped_piece_type: PieceType = last_move_details["dropped_piece_type"]
        # Piece that was dropped is on (r_to_original_move, c_to_original_move)
        game.set_piece(r_to_original_move, c_to_original_move, None)
        game.hands[player_who_made_move.value][dropped_piece_type] += 1
    else:  # Board move
        r_from_original_move = move_tuple[0]  # Must be int for board move
        c_from_original_move = move_tuple[1]  # Must be int for board move
        if r_from_original_move is None or c_from_original_move is None:
            raise RuntimeError(
                "Undo error: r_from/c_from is None for a non-drop move during undo."
            )

        # Piece that moved is currently at (r_to_original_move, c_to_original_move)
        moved_piece = game.get_piece(r_to_original_move, c_to_original_move)
        if not moved_piece:
            raise RuntimeError(
                f"Undo error: No piece found at destination {r_to_original_move},{c_to_original_move} for non-drop."
            )

        # Restore piece to its original square
        game.set_piece(r_from_original_move, c_from_original_move, moved_piece)

        # Restore captured piece, if any
        captured_piece_data: Piece | None = last_move_details[
            "captured"
        ]  # type hint Piece or None
        if captured_piece_data:
            restored_captured_piece = Piece(
                captured_piece_data.type, captured_piece_data.color
            )
            game.set_piece(
                r_to_original_move, c_to_original_move, restored_captured_piece
            )
            # Remove from hand of player_who_made_move
            hand_type_of_captured = PIECE_TYPE_TO_HAND_TYPE.get(
                captured_piece_data.type  # Access type from Piece object
            )
            if hand_type_of_captured:
                if (
                    game.hands[player_who_made_move.value].get(hand_type_of_captured, 0)
                    > 0
                ):
                    game.hands[player_who_made_move.value][hand_type_of_captured] -= 1
                else:
                    # This would indicate an inconsistency in game state or undo history
                    print(
                        "Warning: Tried to undo capture of {hand_type_of_captured.name} but not found in hand."
                    )
            else:
                # Should not happen if captured piece was valid (e.g. not a King)
                print(
                    "Warning: Could not determine hand type for captured "
                    "{captured_piece_data.type.name} during undo."
                )
        else:  # No capture, destination was empty
            game.set_piece(r_to_original_move, c_to_original_move, None)

        # Handle demotion
        if last_move_details.get("was_promoted_in_move", False):
            original_base_type: PieceType | None = last_move_details.get(
                "original_type_before_promotion"
            )
            # Ensure moved_piece.type is actually a promoted type that maps back
            if original_base_type and moved_piece.type in PROMOTED_TO_BASE_TYPE:
                # Double check: original_base_type should match PROMOTED_TO_BASE_TYPE[moved_piece.type]
                if PROMOTED_TO_BASE_TYPE[moved_piece.type] == original_base_type:
                    moved_piece.type = original_base_type
                else:
                    print(
                        f"Warning: Mismatch in undo demotion. Original was {original_base_type.name}, "
                        f"current promoted is {moved_piece.type.name} which unpromotes to "
                        f"{PROMOTED_TO_BASE_TYPE[moved_piece.type].name}"
                    )
                    # Fallback to stored original_base_type if different, but log it.
                    moved_piece.type = original_base_type

            elif original_base_type and moved_piece.type == original_base_type:
                # This case might happen if original_type_before_promotion was set
                # but was_promoted_in_move was True erroneously (e.g. piece type didn't change).
                # No actual demotion needed if types are already the same.
                pass
            elif not original_base_type and moved_piece.type in PROMOTED_TO_BASE_TYPE:
                # If original_type_before_promotion wasn't stored but should have been.
                print(
                    f"Warning: Undoing promotion for {moved_piece.type.name} but original_base_type not found. "
                    f"Demoting to standard base type."
                )
                moved_piece.type = PROMOTED_TO_BASE_TYPE[moved_piece.type]

            # No explicit 'else' for cases where demotion isn't possible/logged as error,
            # as per original logic that might rely on valid state from get_legal_moves.

    game.move_count -= 1
    game.game_over = False
    game.winner = None
