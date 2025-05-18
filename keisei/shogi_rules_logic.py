"""
Core Shogi game rules, move generation, and validation logic.
Functions in this module operate on a ShogiGame instance.
"""

from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING, Union

# Ensure all necessary types are imported:
from .shogi_core_definitions import (
    Piece,
    Color,
    PieceType,
    MoveTuple,
    BoardMove,
    DropMove,
    BASE_TO_PROMOTED_TYPE,
    PROMOTED_TO_BASE_TYPE,
    PROMOTED_TYPES_SET,
    # Add any other constants from core_definitions that might be used directly here
)

if TYPE_CHECKING:
    from .shogi_game import ShogiGame  # For type hinting the 'game' parameter


def is_piece_type_sliding(game: 'ShogiGame', piece_type: PieceType) -> bool:
    """Helper to identify sliding pieces (including promoted)."""
    # 'game' instance is not strictly needed here if PieceType is self-contained,
    # but kept for consistency if other game state might be relevant in future.
    return piece_type in [
        PieceType.LANCE,
        PieceType.BISHOP,
        PieceType.ROOK,
        PieceType.PROMOTED_BISHOP,
        PieceType.PROMOTED_ROOK,
    ]


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


def must_promote_specific_piece(game: "ShogiGame", piece: Piece, r_to: int) -> bool:
    """
    Checks if a piece *must* be promoted on reaching r_to.
    (Formerly ShogiGame.must_promote_piece)
    """
    if piece.is_promoted:
        return False

    # Pawns and Lances on the very last rank
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
    game: "ShogiGame", piece_type: PieceType, row: int, col: int, color: Color
) -> bool:
    """
    Checks if a piece of given base type can be legally dropped.
    (Formerly ShogiGame.can_drop_piece)
    """
    if game.get_piece(row, col) is not None:  # Square must be empty
        return False
    # Player must have the piece in hand
    if game.hands[color.value].get(piece_type, 0) <= 0:
        return False

    # Pawn specific checks
    if piece_type == PieceType.PAWN:
        # Nifu check: is there already an unpromoted pawn of the same color on this file?
        if check_for_nifu(game, color, col):  # Uses the refactored check_for_nifu
            return False
        # Cannot drop pawn on the last rank where it has no moves
        if (color == Color.BLACK and row == 0) or (color == Color.WHITE and row == 8):
            return False
        # Uchi Fu Zume (pawn drop checkmate) check
        if check_for_uchi_fu_zume(
            game, row, col, color
        ):  # Uses the refactored check_for_uchi_fu_zume
            return False

    # Lance specific checks (cannot drop on last rank)
    if piece_type == PieceType.LANCE:
        if (color == Color.BLACK and row == 0) or (color == Color.WHITE and row == 8):
            return False

    # Knight specific checks (cannot drop on last two ranks)
    if piece_type == PieceType.KNIGHT:
        if (color == Color.BLACK and row <= 1) or (color == Color.WHITE and row >= 7):
            return False

    return True


def generate_all_legal_moves(game: "ShogiGame") -> List[MoveTuple]:
    """
    Generate all legal moves for the current player.
    (Formerly ShogiGame.get_legal_moves)
    Move tuple format:
    - Board move: (r_from, c_from, r_to, c_to, promote_flag: bool)
    - Drop move: (None, None, r_to, c_to, piece_type_to_drop: PieceType)
    """
    legal_moves: List[MoveTuple] = []
    current_p_color = game.current_player

    # 1. Generate board moves
    for r_from in range(9):
        for c_from in range(9):
            piece = game.get_piece(r_from, c_from)
            if not piece or piece.color != current_p_color:
                continue

            possible_tos = generate_piece_potential_moves(game, piece, r_from, c_from)

            for r_to, c_to in possible_tos:
                can_promote_opt = can_promote_specific_piece(game, piece, r_from, r_to)
                must_promote_now = must_promote_specific_piece(game, piece, r_to)

                # Option 1: Move without promotion
                if not must_promote_now:
                    move_tuple_no_promo: MoveTuple = (r_from, c_from, r_to, c_to, False)
                    captured_piece_sim = game.get_piece(
                        r_to, c_to
                    )  # Store for restoration
                    original_piece_type_sim = piece.type  # Store for restoration

                    # Simulate move
                    game.set_piece(r_to, c_to, piece)
                    game.set_piece(r_from, c_from, None)

                    if not is_king_in_check_after_simulated_move(game, current_p_color):
                        # II.10 Stranded piece rule:
                        # Player can only decline an optional promotion if the piece, in its unpromoted state,
                        # would still have legal moves from the destination square.
                        is_stranded_if_no_promo = False
                        if (
                            can_promote_opt
                            and piece.type not in [PieceType.GOLD, PieceType.KING]
                            and piece.type not in PROMOTED_TYPES_SET
                        ):
                            # Create a temporary piece representing the unpromoted state for checking future moves
                            # This check is about the piece *at the destination square* r_to, c_to
                            # The piece is already at (r_to, c_to) in its unpromoted form due to simulation.
                            # Piece type must be its original unpromoted type if it was promoted for a check.
                            # In this path (no promotion), piece.type is already unpromoted.

                            # If piece.type is already promoted (e.g. +P), then it can't be "unpromoted further"
                            # This stranded rule applies to base pieces choosing not to promote.

                            # Check if this unpromoted piece at (r_to, c_to) has any potential moves.
                            # This doesn't need full legality check (king safety), just physical moves.
                            potential_next_moves = generate_piece_potential_moves(
                                game, piece, r_to, c_to
                            )

                            # A piece is considered stranded if it has no moves AND it *could* have promoted
                            # because it entered/moved within the promotion zone.
                            # The must_promote_now condition already covers some forms of strandedness (Pawn/Lance on last rank, Knight on last two).
                            # This II.10 rule is for cases where promotion is optional but declining strands it.
                            if not potential_next_moves and (
                                game.is_in_promotion_zone(r_from, current_p_color)
                                or game.is_in_promotion_zone(r_to, current_p_color)
                            ):
                                is_stranded_if_no_promo = True

                        if not is_stranded_if_no_promo:
                            legal_moves.append(move_tuple_no_promo)

                    # Undo simulation
                    piece.type = original_piece_type_sim  # Restore piece type
                    game.set_piece(r_from, c_from, piece)
                    game.set_piece(r_to, c_to, captured_piece_sim)

                # Option 2: Move with promotion
                if can_promote_opt:  # This also covers must_promote_now implicitly
                    move_tuple_promo: MoveTuple = (r_from, c_from, r_to, c_to, True)
                    promoted_type = BASE_TO_PROMOTED_TYPE.get(piece.type)

                    if promoted_type:  # Ensure piece is promotable
                        captured_piece_sim = game.get_piece(r_to, c_to)
                        original_type_sim = piece.type

                        piece.type = promoted_type  # Temporarily promote
                        game.set_piece(r_to, c_to, piece)
                        game.set_piece(r_from, c_from, None)

                        if not is_king_in_check_after_simulated_move(
                            game, current_p_color
                        ):
                            legal_moves.append(move_tuple_promo)

                        # Undo simulation
                        piece.type = original_type_sim  # Revert type
                        game.set_piece(r_from, c_from, piece)
                        game.set_piece(r_to, c_to, captured_piece_sim)

    # 2. Generate drop moves
    player_hand = game.hands[current_p_color.value]
    # Use PieceType.get_unpromoted_types() or similar from game/core_definitions for iteration order if specific
    # hand_piece_order = PieceType.get_unpromoted_types() # if order matters for policy output
    # for piece_type_to_drop_enum in hand_piece_order:
    #    count = player_hand.get(piece_type_to_drop_enum, 0)
    # Original iterates dict items, order might not be guaranteed but usually fine.
    for piece_type_to_drop_enum, count in player_hand.items():
        if count > 0:
            for r_to_drop in range(9):
                for c_to_drop in range(9):
                    if (
                        game.get_piece(r_to_drop, c_to_drop) is None
                    ):  # Can only drop on empty square
                        if can_drop_specific_piece(
                            game,
                            piece_type_to_drop_enum,
                            r_to_drop,
                            c_to_drop,
                            current_p_color,
                        ):
                            move_tuple_drop: MoveTuple = (
                                None,
                                None,
                                r_to_drop,
                                c_to_drop,
                                piece_type_to_drop_enum,
                            )

                            # Simulate drop
                            game.hands[current_p_color.value][
                                piece_type_to_drop_enum
                            ] -= 1
                            dropped_p_sim = Piece(
                                piece_type_to_drop_enum, current_p_color
                            )
                            game.set_piece(r_to_drop, c_to_drop, dropped_p_sim)

                            if not is_king_in_check_after_simulated_move(
                                game, current_p_color
                            ):
                                legal_moves.append(move_tuple_drop)

                            # Undo simulation
                            game.set_piece(r_to_drop, c_to_drop, None)
                            game.hands[current_p_color.value][
                                piece_type_to_drop_enum
                            ] += 1

    return list(set(legal_moves))  # Ensure uniqueness


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
    # This means we are checking if the state (board, hands, P_new_to_move) has repeated.
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
