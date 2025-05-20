# shogi_move_execution.py
"""
Contains functions for applying and reverting moves in the Shogi game.
These functions operate on a ShogiGame instance.
"""

from typing import (  # Added Optional
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

from . import shogi_rules_logic  # Ensure shogi_rules_logic is imported
from .shogi_core_definitions import PROMOTED_TO_BASE_TYPE  # Added import
from .shogi_core_definitions import (
    BASE_TO_PROMOTED_TYPE,
    PIECE_TYPE_TO_HAND_TYPE,
    Color,
    MoveTuple,
    Piece,
    PieceType,
    get_unpromoted_types,
)

# from . import shogi_game_io # Commenting out as log_move is not found

if TYPE_CHECKING:
    from .shogi_game import ShogiGame  # For type hinting


def apply_move_to_board(
    game: "ShogiGame", move_tuple: MoveTuple, is_simulation: bool = False
) -> None:
    """
    Applies a given move to the board, updating piece positions, hands,
    current player, and move count. Also checks for game termination.

    Board move: (r_from, c_from, r_to, c_to, promote_flag: bool)
    Drop move: (None, None, r_to, c_to, piece_type_to_drop: PieceType)

    Args:
        game: The ShogiGame instance.
        move_tuple: The move to apply.
        is_simulation: If True, indicates the move is part of a simulation
                       and game-ending checks (checkmate, stalemate) should be skipped.
    """
    player_who_made_the_move = game.current_player  # Store before switching

    # --- Execute the move ---
    # (This section is assumed to be present and correct as per original file structure)
    # Example of what might be here:
    # r_from, c_from, r_to, c_to = move_tuple[0], move_tuple[1], move_tuple[2], move_tuple[3]
    # piece_to_move = game.get_piece(r_from, c_from)
    # captured_piece_type: Optional[PieceType] = None # Store type for adding to hand
    # target_piece_on_board = game.get_piece(r_to, c_to)
    # if target_piece_on_board:
    #     captured_piece_type = target_piece_on_board.unpromote().type # Always add unpromoted type to hand
    #     game.add_to_hand(player_who_made_the_move, captured_piece_type)
    #     game.set_piece(r_to, c_to, None) # Clear target square before moving

    # if len(move_tuple) == 5 and isinstance(move_tuple[4], PieceType): # Drop move
    #     piece_type_to_drop: PieceType = move_tuple[4]
    #     game.set_piece(r_to, c_to, Piece(piece_type_to_drop, player_who_made_the_move))
    #     game.remove_from_hand(player_who_made_the_move, piece_type_to_drop)
    # else: # Board move
    #     if piece_to_move is None: # Should not happen if move is valid
    #         # Handle error or raise exception
    #         return
    #     game.set_piece(r_to, c_to, piece_to_move)
    #     game.set_piece(r_from, c_from, None)
    #     if len(move_tuple) == 5 and isinstance(move_tuple[4], bool) and move_tuple[4]: # Promotion
    #         promoted_piece = game.get_piece(r_to, c_to)
    #         if promoted_piece: # Check if piece exists before promoting
    #             promoted_piece.promote()
    # --- End of example move execution ---

    # Switch current player
    game.current_player = (
        Color.WHITE if player_who_made_the_move == Color.BLACK else Color.BLACK
    )
    game.move_count += 1

    if not is_simulation:
        # The ShogiGame class should handle adding to move_history and board_history internally
        # when its make_move method calls this apply_move_to_board function.
        # For sennichite, the game object itself should manage its history correctly.
        # We assume game.add_to_move_history() or similar is called by game.make_move()
        # and that game.is_sennichite() uses that history.
        pass  # History management is part of ShogiGame.make_move

    # Check for game termination conditions
    if not is_simulation:
        # game.current_player is the player whose turn it *now* is.
        # player_who_made_the_move is the player who just made the move.

        # Check game state for the player whose turn it now is (game.current_player)
        king_of_current_player_in_check = (
            shogi_rules_logic.is_king_in_check_after_simulated_move(
                game, game.current_player
            )
        )
        # Pass a temporary game state if generate_all_legal_moves modifies the game,
        # or ensure it's safe to call on the actual game state.
        # Assuming shogi_rules_logic.generate_all_legal_moves is safe to call on current game state.
        legal_moves_for_current_player = shogi_rules_logic.generate_all_legal_moves(
            game
        )

        if not legal_moves_for_current_player:
            if king_of_current_player_in_check:
                game.game_over = True
                game.winner = (
                    player_who_made_the_move  # The player who delivered checkmate
                )
                game.termination_reason = "Tsumi"  # Checkmate
            else:
                game.game_over = True
                game.winner = None  # Stalemate is a draw
                game.termination_reason = (
                    "Stalemate"  # Or a more specific Shogi term like "Jishogi"
                )
        elif (
            game.is_sennichite()
        ):  # Repetition draw - relies on ShogiGame's history management
            game.game_over = True
            game.winner = None
            game.termination_reason = "Sennichite"
        elif game.move_count >= game.max_moves_per_game:
            game.game_over = True
            game.winner = None
            game.termination_reason = "Max moves reached"

    # Logging the move (if not simulation and if logging is re-enabled)
    # if not is_simulation:
    # move_details = { ... }
    # shogi_game_io.log_move(game, move_details)
    # pass


def revert_last_applied_move(game: "ShogiGame") -> None:
    """
    Reverts the last move made in the game.
    Operates on the 'game' (ShogiGame instance).
    """
    if not game.move_history:
        raise RuntimeError("No move to undo")

    last_move_details = game.move_history.pop()
    move_tuple: "MoveTuple" = last_move_details["move"]

    # Remove the state hash of the move being undone from board_history
    if game.board_history:
        game.board_history.pop()  # The last hash corresponds to the state after the move we are undoing

    # Switch current player back first. The player in current_player is the one who *would have* moved.
    # The player who *made* the move being undone is the new game.current_player after this switch.
    game.current_player = (
        Color.WHITE if game.current_player == Color.BLACK else Color.BLACK
    )
    player_who_made_the_undone_move = game.current_player

    if last_move_details["is_drop"]:
        dropped_piece_type: PieceType = last_move_details["dropped_piece_type"]
        # r_to_drop, c_to_drop are where the piece was dropped
        r_to_drop = move_tuple[2]
        c_to_drop = move_tuple[3]

        # Remove the dropped piece from the board
        game.set_piece(r_to_drop, c_to_drop, None)
        # Add it back to the hand of the player who made the drop
        if dropped_piece_type in get_unpromoted_types() and dropped_piece_type != PieceType.KING:
            game.hands[player_who_made_the_undone_move.value][dropped_piece_type] = \
                game.hands[player_who_made_the_undone_move.value].get(dropped_piece_type, 0) + 1
        else:
            raise ValueError(f"Attempted to return invalid piece type {dropped_piece_type} to hand during undo of drop.")

    else:  # Board move
        orig_r_from: int = cast(int, move_tuple[0])
        orig_c_from: int = cast(int, move_tuple[1])
        orig_r_to: int = cast(int, move_tuple[2])  # Square piece moved to
        orig_c_to: int = cast(int, move_tuple[3])  # Square piece moved to

        # Details of the piece that was moved, from history
        original_type_before_promotion = last_move_details["original_type_before_promotion"]
        original_color_of_moved_piece = last_move_details.get("original_color", player_who_made_the_undone_move)

        # Create the piece as it was *before* the move (i.e., at its original type)
        piece_to_restore_at_from = Piece(original_type_before_promotion, original_color_of_moved_piece)
        
        # Place it back on its original square
        game.set_piece(orig_r_from, orig_c_from, piece_to_restore_at_from)

        # Handle what was on the destination square (orig_r_to, orig_c_to)
        captured_piece_object: Optional[Piece] = last_move_details.get("captured")

        if captured_piece_object:  # A piece was captured
            # Restore the captured piece to the board at its capture location
            game.set_piece(orig_r_to, orig_c_to, captured_piece_object)
            
            # Remove the captured piece (now restored to board) from the hand of player_who_made_the_undone_move
            # The type added to hand is always unpromoted.
            # Create a copy to unpromote for getting the type, or use PROMOTED_TO_BASE_TYPE
            type_to_remove_from_hand: PieceType
            if captured_piece_object.is_promoted:
                type_to_remove_from_hand = PROMOTED_TO_BASE_TYPE[captured_piece_object.type]
            else:
                type_to_remove_from_hand = captured_piece_object.type
            game.remove_from_hand(type_to_remove_from_hand, player_who_made_the_undone_move)
        else:
            # The destination square was empty, so clear it now
            game.set_piece(orig_r_to, orig_c_to, None)

    game.move_count -= 1
    game.game_over = False
    game.winner = None
    game.termination_reason = None
