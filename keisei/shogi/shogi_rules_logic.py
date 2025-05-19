"""
Core Shogi game rules, move generation, and validation logic.
Functions in this module operate on a ShogiGame instance.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

# Ensure all necessary types are imported:
from .shogi_core_definitions import (  # Corrected: MoveTuple is defined and used, not Move; Add any other constants from core_definitions that might be used directly here
    BASE_TO_PROMOTED_TYPE, Color, MoveTuple, Piece, PieceType)

if TYPE_CHECKING:
    from .shogi_core_definitions import BoardMoveTuple, DropMoveTuple  # Added for type hinting
    from .shogi_game import ShogiGame  # For type hinting the 'game' parameter


# --- Helper functions for move generation and validation ---


def is_piece_type_sliding(piece_type: PieceType) -> bool:  # Removed 'game' parameter
    """Returns True if the piece type is a sliding piece (Lance, Bishop, Rook or their promoted versions)."""
    return piece_type in (
        PieceType.LANCE,
        PieceType.BISHOP,
        PieceType.ROOK,
        PieceType.PROMOTED_BISHOP,
        PieceType.PROMOTED_ROOK,
    )


def generate_piece_potential_moves(
    game: "ShogiGame", piece: Piece, r_from: int, c_from: int
) -> list[tuple[int, int]]:
    """
    Returns a list of (r_to, c_to) tuples for a piece, considering its
    fundamental movement rules and path-blocking by other pieces.
    This function generates squares a piece *attacks* or can move to if empty.
    It stops at the first piece encountered. If that piece is an opponent,
    the square is included (as a capture). If friendly, it's not included.
    (Formerly ShogiGame.get_individual_piece_moves)
    """
    moves = []
    # Black (Sente, 0) moves towards smaller row indices, White (Gote, 1) towards larger
    forward = (
        -1 if piece.color == Color.BLACK else 1
    )  # Assuming game instance has Color enum

    piece_type = piece.type

    # Define move offsets
    gold_move_offsets = [
        (forward, 0),
        (forward, -1),
        (forward, 1),
        (0, -1),
        (0, 1),
        (-forward, 0),  # Backwards for Gold
    ]
    king_move_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    knight_move_offsets = [(forward * 2, -1), (forward * 2, 1)]
    silver_move_offsets = [
        (forward, 0),
        (forward, -1),
        (forward, 1),
        (-forward, -1),
        (-forward, 1),  # Backwards-diagonal
    ]
    promoted_rook_extra_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    promoted_bishop_extra_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    current_offsets = []
    is_sliding = False

    if piece_type == PieceType.PAWN:
        current_offsets = [(forward, 0)]
    elif piece_type == PieceType.KNIGHT:
        current_offsets = knight_move_offsets
    elif piece_type == PieceType.SILVER:
        current_offsets = silver_move_offsets
    elif (
        piece_type == PieceType.GOLD
        or piece_type == PieceType.PROMOTED_PAWN
        or piece_type == PieceType.PROMOTED_LANCE
        or piece_type == PieceType.PROMOTED_KNIGHT
        or piece_type == PieceType.PROMOTED_SILVER
    ):
        current_offsets = gold_move_offsets
    elif piece_type == PieceType.KING:
        current_offsets = king_move_offsets

    for dr, dc in current_offsets:
        nr, nc = r_from + dr, c_from + dc
        if game.is_on_board(nr, nc):
            target_piece = game.get_piece(nr, nc)
            if target_piece is None or target_piece.color != piece.color:
                moves.append((nr, nc))

    sliding_directions = []
    if piece_type == PieceType.LANCE:
        is_sliding = True
        sliding_directions = [(forward, 0)]
    elif piece_type == PieceType.BISHOP or piece_type == PieceType.PROMOTED_BISHOP:
        is_sliding = True
        sliding_directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        if piece_type == PieceType.PROMOTED_BISHOP:
            for dr, dc in promoted_bishop_extra_offsets:
                nr, nc = r_from + dr, c_from + dc
                if game.is_on_board(nr, nc):
                    target_piece = game.get_piece(nr, nc)
                    if target_piece is None or target_piece.color != piece.color:
                        moves.append((nr, nc))
    elif piece_type == PieceType.ROOK or piece_type == PieceType.PROMOTED_ROOK:
        is_sliding = True
        sliding_directions.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])
        if piece_type == PieceType.PROMOTED_ROOK:
            for dr, dc in promoted_rook_extra_offsets:
                nr, nc = r_from + dr, c_from + dc
                if game.is_on_board(nr, nc):
                    target_piece = game.get_piece(nr, nc)
                    if target_piece is None or target_piece.color != piece.color:
                        moves.append((nr, nc))

    if is_sliding:
        for dr_slide, dc_slide in sliding_directions:
            for i in range(1, 9):  # Max 8 steps
                nr, nc = r_from + dr_slide * i, c_from + dc_slide * i
                if not game.is_on_board(nr, nc):
                    break
                target_piece = game.get_piece(nr, nc)
                if target_piece is None:
                    moves.append((nr, nc))
                else:
                    if target_piece.color != piece.color:
                        moves.append((nr, nc))
                    break
    return list(set(moves))  # Remove duplicates


def check_for_nifu(game: "ShogiGame", color: Color, col: int) -> bool:
    """
    Checks for two unpromoted pawns of the same color on the given file.
    (Formerly ShogiGame.is_nifu)
    """
    for r in range(9):
        p = game.get_piece(r, col)
        if (
            p
            and p.type == PieceType.PAWN
            and p.color == color
            # and not p.is_promoted # This check is redundant if type is PAWN
        ):
            # Found one pawn, need to check for a *second* one.
            # The original logic implies if *any* pawn is found, it's nifu,
            # which is incorrect. Nifu = "two pawns".
            # This function as written in original code is actually "is_pawn_on_file"
            # For a correct nifu, one pawn must already be on the file.
            # This function is used when *dropping* a pawn. So it checks if a pawn *already exists*.
            return True  # A pawn of that color already exists on this file.
    return False


def check_if_square_is_attacked(
    game: "ShogiGame", row: int, col: int, attacker_color: Color
) -> bool:
    """
    Returns True if the square (row, col) is attacked by any piece of attacker_color.
    (Formerly ShogiGame._is_square_attacked)
    """
    for r_attacker in range(9):
        for c_attacker in range(9):
            p_attacker = game.get_piece(r_attacker, c_attacker)
            if p_attacker and p_attacker.color == attacker_color:
                attacked_squares = generate_piece_potential_moves(
                    game, p_attacker, r_attacker, c_attacker
                )
                if (row, col) in attacked_squares:
                    return True
    return False


def check_for_uchi_fu_zume(
    game: "ShogiGame", drop_row: int, drop_col: int, color: Color
) -> bool:
    """
    Returns True if dropping a pawn at (drop_row, drop_col) by 'color'
    results in immediate checkmate (Uchi Fu Zume).
    (Formerly ShogiGame.is_uchi_fu_zume)
    """
    if game.get_piece(drop_row, drop_col) is not None:
        return False  # Square occupied

    opp_color = Color.WHITE if color == Color.BLACK else Color.BLACK
    king_pos: Optional[Tuple[int, int]] = None
    for r_k in range(9):
        for c_k in range(9):
            p_k = game.get_piece(r_k, c_k)
            if p_k and p_k.type == PieceType.KING and p_k.color == opp_color:
                king_pos = (r_k, c_k)
                break
        if king_pos:
            break
    if not king_pos:
        # This should ideally not happen in a valid game state.
        # If opponent's king is not on board, it's already a game-ending condition.
        # For uchi-fu-zume context, this means no king to checkmate.
        return False

    king_r, king_c = king_pos

    # Check if dropped pawn delivers check
    pawn_attack_row = drop_row + (-1 if color == Color.BLACK else 1)
    if not (king_r == pawn_attack_row and king_c == drop_col):
        return False  # Pawn drop doesn't check the king

    # Temporarily place the pawn
    # The `Piece` class used here should be the one from shogi_core_definitions
    dropped_pawn = Piece(PieceType.PAWN, color)
    game.set_piece(drop_row, drop_col, dropped_pawn)

    # Temporarily switch player to simulate opponent's turn for legal move check
    original_player = game.current_player
    game.current_player = opp_color  # Temporarily set for the check

    # Check if opponent has any legal moves to escape checkmate
    opponent_has_legal_move = False

    # Generate all legal moves for the opponent.
    # This requires a lightweight version or careful use of the full get_legal_moves
    # to avoid infinite recursion if get_legal_moves itself calls is_uchi_fu_zume.
    # For uchi_fu_zume, the key is that the *king* has no escape,
    # and the checking pawn cannot be captured, or if it can, king is still mated.

    # Simplified check: can the king move to a safe square, or can the pawn be captured?
    # More robust: generate opponent's legal moves.
    # We need a temporary ShogiGame state or a way for generate_all_legal_moves
    # to work with the current (modified) board state of 'game'.

    # If we call generate_all_legal_moves(game), it will use game.current_player (which is now opp_color).
    # It will also use the current board state (with the dropped pawn).
    # The generate_all_legal_moves function will internally handle checks to ensure
    # the opponent does not move into check.

    # Critical: Ensure generate_all_legal_moves does not recursively call check_for_uchi_fu_zume
    # if it's part of pawn drop validation within generate_all_legal_moves.
    # Assuming check_for_uchi_fu_zume is called *before* adding a pawn drop to legal moves.

    # Option A: A simplified check focused on king escape / pawn capture
    # (as in the original provided code for ShogiGame.is_uchi_fu_zume)

    # 1. Can the king escape?
    king_piece_at_kr_kc = game.get_piece(king_r, king_c)
    if king_piece_at_kr_kc:  # Should be the king
        king_potential_moves = generate_piece_potential_moves(
            game, king_piece_at_kr_kc, king_r, king_c
        )
        for kr_new, kc_new in king_potential_moves:
            # Simulate king move
            original_piece_at_escape_sq = game.get_piece(kr_new, kc_new)
            game.set_piece(king_r, king_c, None)
            game.set_piece(kr_new, kc_new, king_piece_at_kr_kc)

            if not check_if_square_is_attacked(
                game, kr_new, kc_new, color
            ):  # Is king safe after move?
                opponent_has_legal_move = True

            # Undo simulated king move
            game.set_piece(king_r, king_c, king_piece_at_kr_kc)
            game.set_piece(kr_new, kc_new, original_piece_at_escape_sq)
            if opponent_has_legal_move:
                break

    if opponent_has_legal_move:
        game.set_piece(drop_row, drop_col, None)  # Remove test pawn
        game.current_player = original_player  # Restore player
        return False  # Not uchi_fu_zume, king can escape

    # 2. Can the pawn be captured by any opponent piece (resulting in king not being in check)?
    # Or can any piece block? (Original focused on capture of the pawn)
    for r_att in range(9):
        if opponent_has_legal_move:
            break
        for c_att in range(9):
            if opponent_has_legal_move:
                break
            attacker_piece = game.get_piece(r_att, c_att)
            if attacker_piece and attacker_piece.color == opp_color:
                possible_attacker_moves = generate_piece_potential_moves(
                    game, attacker_piece, r_att, c_att
                )
                if (
                    drop_row,
                    drop_col,
                ) in possible_attacker_moves:  # Can this piece capture the pawn?
                    # Simulate capture
                    original_attacker_on_board = attacker_piece
                    game.set_piece(
                        drop_row, drop_col, original_attacker_on_board
                    )  # Pawn is captured
                    game.set_piece(r_att, c_att, None)

                    if not check_if_square_is_attacked(
                        game, king_r, king_c, color
                    ):  # Is king safe?
                        opponent_has_legal_move = True

                    # Undo simulated capture
                    game.set_piece(r_att, c_att, original_attacker_on_board)
                    game.set_piece(
                        drop_row, drop_col, dropped_pawn
                    )  # Put test pawn back
                    if opponent_has_legal_move:
                        break

    # Restore board and current player fully
    game.set_piece(drop_row, drop_col, None)
    game.current_player = original_player

    return (
        not opponent_has_legal_move
    )  # If no legal move for opponent, it's Uchi Fu Zume


def is_king_in_check_after_simulated_move(
    game: "ShogiGame", player_color: Color
) -> bool:
    """
    Checks if the king of 'player_color' is in check on the current board.
    Assumes the board reflects the state *after* a move has been made.
    (Formerly ShogiGame._king_in_check_after_move)
    """
    king_pos = None
    for r_k in range(9):
        for c_k in range(9):
            p = game.get_piece(r_k, c_k)
            if p and p.type == PieceType.KING and p.color == player_color:
                king_pos = (r_k, c_k)
                break
        if king_pos:
            break

    if not king_pos:
        # This implies king was captured or somehow removed.
        # In Shogi, a move that results in your own king being captured is illegal.
        # If king is missing, it's an invalid state for this check, effectively means check.
        return True

    opponent_color = Color.WHITE if player_color == Color.BLACK else Color.BLACK
    return check_if_square_is_attacked(game, king_pos[0], king_pos[1], opponent_color)


def can_promote_specific_piece(
    game: "ShogiGame", piece: Piece, r_from: int, r_to: int
) -> bool:
    """
    Checks if a piece *can* be promoted given its type and move.
    (Formerly ShogiGame.can_promote_piece)
    """
    if piece.type in [PieceType.GOLD, PieceType.KING] or piece.is_promoted:
        return False

    if piece.type not in BASE_TO_PROMOTED_TYPE:  # Not a base type that can promote
        return False

    # Must start in, end in, or cross promotion zone
    # Assumes game instance has is_in_promotion_zone(row, color) method
    if game.is_in_promotion_zone(r_from, piece.color) or game.is_in_promotion_zone(
        r_to, piece.color
    ):
        return True
    return False


def must_promote_specific_piece(
    piece: Piece, r_to: int
) -> bool:  # Removed 'game' parameter
    """Checks if a piece *must* promote when moving to r_to."""
    # Pawns and Lances must promote if they reach the last rank.
    if piece.type == PieceType.PAWN or piece.type == PieceType.LANCE:
        if (piece.color == Color.BLACK and r_to == 0) or (
            piece.color == Color.WHITE and r_to == 8
        ):
            return True

    # Knights on the last two ranks
    if piece.type == PieceType.KNIGHT:
        if (piece.color == Color.BLACK and r_to <= 1) or (
            piece.color == Color.WHITE and r_to >= 7
        ):
            return True
    return False


def can_drop_specific_piece(
    game: "ShogiGame", piece_type: PieceType, r_to: int, c_to: int, color: Color
) -> bool:
    """
    Checks if a specific piece_type can be legally dropped by 'color' at (r_to, c_to).
    This function checks rules like:
    - Square must be empty.
    - Nifu (two pawns on the same file).
    - Piece cannot be dropped where it has no further moves (Pawn, Lance on last rank; Knight on last two ranks).
    - Uchi Fu Zume (dropping a pawn for an immediate checkmate that cannot be escaped).
    It does NOT check if the drop leaves the current player's king in check;
    that is handled by the calling function (e.g., generate_all_legal_moves).
    """
    if game.get_piece(r_to, c_to) is not None:
        return False  # Square must be empty

    # Determine player-specific "forward" direction and promotion zone boundaries
    # Black (Sente, 0) moves towards smaller row indices (ranks 8 to 0)
    # White (Gote, 1) moves towards larger row indices (ranks 0 to 8)
    # Black's last rank is 0, White's last rank is 8.
    last_rank = 0 if color == Color.BLACK else 8
    second_last_rank = 1 if color == Color.BLACK else 7

    if piece_type == PieceType.PAWN:
        # 1. Nifu check: Cannot drop a pawn on a file that already contains an unpromoted pawn of the same color.
        if check_for_nifu(game, color, c_to):
            return False
        # 2. Cannot drop a pawn on the last rank (it would have no moves).
        if r_to == last_rank:
            return False
        # 3. Uchi Fu Zume check: Cannot drop a pawn to give immediate checkmate if that checkmate has no escape.
        #    The check_for_uchi_fu_zume function returns True if it *is* uchi_fu_zume.
        if check_for_uchi_fu_zume(game, r_to, c_to, color):
            return False
    elif piece_type == PieceType.LANCE:
        # Cannot drop a lance on the last rank.
        if r_to == last_rank:
            return False
    elif piece_type == PieceType.KNIGHT:
        # Cannot drop a knight on the last two ranks.
        if r_to == last_rank or r_to == second_last_rank:
            return False

    # Other pieces (Gold, Silver, Bishop, Rook, King) can be dropped on any empty square
    # provided other conditions (like not leaving king in check) are met by the caller.
    # King is not a droppable piece type, but included for completeness if logic changes.
    # For now, hands will only contain P, L, N, S, G, B, R.
    return True


def generate_all_legal_moves(game: "ShogiGame") -> List[MoveTuple]:
    """
    Generates all legal moves for the current player.
    A move is legal if:
    1. It follows the piece's movement rules.
    2. For drops: it follows drop rules (Nifu, Uchi Fu Zume, no-move squares).
    3. It does not leave the current player's king in check.
    (Formerly ShogiGame.get_legal_moves)
    """
    legal_moves: List[MoveTuple] = []
    original_player_color = game.current_player
    # opponent_color = Color.WHITE if original_player_color == Color.BLACK else Color.BLACK

    # I. Generate Board Moves (moving a piece already on the board)
    for r_from in range(9):
        for c_from in range(9):
            piece = game.get_piece(r_from, c_from)
            if piece and piece.color == original_player_color:
                # Get squares the piece attacks or can move to if empty
                potential_squares = generate_piece_potential_moves(
                    game, piece, r_from, c_from
                )

                for r_to, c_to in potential_squares:
                    # Check for promotion possibilities
                    can_promote = can_promote_specific_piece(game, piece, r_from, r_to)
                    must_promote = must_promote_specific_piece(
                        piece, r_to
                    )  # Removed game argument

                    current_move_tuples_to_check: List['BoardMoveTuple']  # Declare type here
                    if must_promote:
                        # Only one move: promotion is forced
                        current_move_tuples_to_check = [
                            (r_from, c_from, r_to, c_to, True)
                        ]
                    elif can_promote:
                        # Two possibilities: promote or not promote
                        current_move_tuples_to_check = [
                            (r_from, c_from, r_to, c_to, True),
                            (r_from, c_from, r_to, c_to, False),
                        ]
                    else:
                        # Only one move: no promotion possible or allowed
                        current_move_tuples_to_check = [
                            (r_from, c_from, r_to, c_to, False)
                        ]

                    for board_move_tuple in current_move_tuples_to_check:
                        # Simulate the move
                        game.make_move(
                            board_move_tuple, is_simulation=True
                        )  # Pass is_simulation=True
                        # Check if the current player's king is NOT in check after this move
                        if not is_king_in_check_after_simulated_move(
                            game, original_player_color
                        ):
                            legal_moves.append(board_move_tuple)
                        game.undo_move()  # Revert the simulated move

    # II. Generate Drop Moves
    for piece_type_to_drop in game.hands[original_player_color.value]:
        if game.hands[original_player_color.value][piece_type_to_drop] > 0:
            for r_to in range(9):
                for c_to in range(9):
                    if game.get_piece(r_to, c_to) is None:  # Square must be empty
                        # Check basic drop legality (Nifu, Uchi Fu Zume, no-move squares)
                        if can_drop_specific_piece(
                            game, piece_type_to_drop, r_to, c_to, original_player_color
                        ):
                            drop_move_tuple: 'DropMoveTuple' = (
                                None,
                                None,
                                r_to,
                                c_to,
                                piece_type_to_drop,
                            )
                            # Simulate the drop
                            game.make_move(
                                drop_move_tuple, is_simulation=True
                            )  # Pass is_simulation=True
                            # Check if the current player's king is NOT in check after this drop
                            if not is_king_in_check_after_simulated_move(
                                game, original_player_color
                            ):
                                legal_moves.append(drop_move_tuple)
                            game.undo_move()  # Revert the simulated drop
    return legal_moves


def check_for_sennichite(game: "ShogiGame") -> bool:
    """
    Returns True if the current board state has occurred four times (Sennichite).
    (Formerly ShogiGame.is_sennichite)
    Relies on game.move_history and game._board_state_hash().
    The hash in move_history is for the state *after* a move, including whose turn it was *before* switching players.
    Sennichite rule states: same game position (pieces on board, pieces in hand, and player to move)
    has appeared for the fourth time.
    """
    # The state hash to check for repetition should represent (board, hands, current_player_to_move)
    # In the original make_move:
    # 1. move pieces
    # 2. state_hash = _board_state_hash() (current_player is still P_old who made the move) -> stored in history
    # 3. current_player is switched to P_new
    # 4. if is_sennichite(): ... is called.
    # Inside is_sennichite (this function):
    # game.current_player is P_new.
    # So, state_to_check_for_repetition = game._board_state_hash() will use P_new.
    # This means we are checking if the state (board, hands, P_old_who_just_moved) has repeated.
    # The history stores (board, hands, P_old_who_just_moved). This comparison is subtle.

    # The problem description (II.8) implies the hash in history is the one to count.
    # "The state_hash stored in move_history by make_move is for the board
    # state *after* the move and includes self.current_player *before* it's switched."
    # "is_sennichite is called *after* self.current_player is switched."
    # "If current make_move appends hash for state *after* move & *before* player switch,
    # then is_sennichite is called *after* player switch. The hashes won't match."

    # The original `is_sennichite` code in the prompt:
    # final_state_hash_of_move = (
    #         game.move_history[-1].get("state_hash") if game.move_history else None
    # ) # This is (board_after_P_old_move, hands, P_old_turn)
    # ...
    # count_of_this_state = 0
    # for record in game.move_history:
    #     if record.get("state_hash") == final_state_hash_of_move:
    #         count_of_this_state += 1
    # return count_of_this_state >= 4
    # This means sennichite is declared if the state *just achieved by the previous player*
    # (which includes that player as the one whose turn it was for that hash) has appeared 4 times.
    # This is a common interpretation for how repetition is tracked.

    if not game.move_history:
        return False

    # This hash represents the board state achieved by the *previous* player's move,
    # and importantly, the game._board_state_hash() includes whose turn it *was* when that state was recorded.
    last_recorded_state_hash = game.move_history[-1].get("state_hash")
    if not last_recorded_state_hash:
        return False  # Should not happen if history is populated correctly

    count = 0
    for move_record in game.move_history:
        if move_record.get("state_hash") == last_recorded_state_hash:
            count += 1

    # If this state (board, hands, player_who_just_moved) has now occurred 4 times.
    return count >= 4
