"""
Core Shogi game rules, move generation, and validation logic.
Functions in this module operate on a ShogiGame instance.
"""

from copy import deepcopy  # ensure deep copy is available
from typing import TYPE_CHECKING, List, Optional, Tuple

# Ensure all necessary types are imported:
from .shogi_core_definitions import (  # Corrected: MoveTuple is defined and used, not Move; Add any other constants from core_definitions that might be used directly here
    BASE_TO_PROMOTED_TYPE,
    Color,
    DropMoveTuple,
    MoveTuple,
    Piece,
    PieceType,
)

if TYPE_CHECKING:
    from .shogi_core_definitions import BoardMoveTuple  # Added for type hinting
    from .shogi_game import ShogiGame  # For type hinting the 'game' parameter


# --- Helper functions for move generation and validation ---

# ADDED: Centralized king finding function
def find_king(game: "ShogiGame", color: Color) -> Optional[Tuple[int, int]]:
    """Finds the king of the specified color on the board."""
    for r_k in range(9):
        for c_k in range(9):
            p = game.get_piece(r_k, c_k)
            if p and p.type == PieceType.KING and p.color == color:
                return (r_k, c_k)
    return None

# ADDED: is_in_check function to be called by ShogiGame.is_in_check
def is_in_check(game: "ShogiGame", player_color: Color, debug_recursion: bool = False) -> bool:
    """Checks if the king of 'player_color' is in check."""
    king_pos = find_king(game, player_color)

    if not king_pos:
        if debug_recursion:
            # Try to get SFEN, but game object might be in an intermediate state for this print
            sfen_str = "unavailable (game object might be partial)"
            try:
                sfen_str = game.to_sfen_string()
            except Exception: # pylint: disable=broad-except
                pass # Keep sfen_str as unavailable
            print(f"DEBUG_IS_IN_CHECK: King of color {player_color} not found. Game state (SFEN): {sfen_str}. Returning True (check).")
        return True  # King not found implies a lost/invalid state, effectively in check.

    opponent_color = Color.WHITE if player_color == Color.BLACK else Color.BLACK
    
    if debug_recursion:
        print(f"DEBUG_IS_IN_CHECK: [{player_color}] King at {king_pos}. Checking if attacked by {opponent_color}. Debug on.")
        # Detailed print for attack check will come from check_if_square_is_attacked

    return check_if_square_is_attacked(
        game, king_pos[0], king_pos[1], opponent_color, debug=debug_recursion
    )


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
    game: "ShogiGame",
    r_target: int,
    c_target: int,
    attacker_color: Color,
    debug: bool = False, # ADDED debug flag
) -> bool:
    """
    Checks if the square (r_target, c_target) is attacked by any piece of attacker_color.
    """ # Corrected string literal
    if debug:
        print(f"DEBUG_CHECK_SQ_ATTACKED: Checking if ({r_target},{c_target}) is attacked by {attacker_color}")
        print(f"DEBUG_CHECK_SQ_ATTACKED: Game state for check: {game.to_sfen_string()}")

    for r_attacker in range(9):
        for c_attacker in range(9):
            piece = game.get_piece(r_attacker, c_attacker)
            if piece and piece.color == attacker_color:
                if debug:
                    print(f"DEBUG_CHECK_SQ_ATTACKED: Checking attacker {piece.type.name} ({piece.color.name}) at ({r_attacker},{c_attacker}) against target ({r_target},{c_target})")

                potential_moves_of_attacker = generate_piece_potential_moves(
                    game, piece, r_attacker, c_attacker
                )
                if (r_target, c_target) in potential_moves_of_attacker:
                    if debug:
                        print(f"DEBUG_CHECK_SQ_ATTACKED: YES, {piece.type.name} ({piece.color.name}) at ({r_attacker},{c_attacker}) attacks ({r_target},{c_target}). Potential moves: {potential_moves_of_attacker}")
                    return True
    if debug:
        print(f"DEBUG_CHECK_SQ_ATTACKED: NO, ({r_target},{c_target}) is NOT attacked by {attacker_color}")
    return False


def check_for_uchi_fu_zume(
    game: "ShogiGame", drop_row: int, drop_col: int, color: Color
) -> bool:
    """
    Returns True if dropping a pawn at (drop_row, drop_col) by 'color'
    results in immediate, unescapable checkmate for the opponent.
    This is an illegal move by Shogi rules.
    This function does NOT check if the drop itself leaves 'color's king in check;
    that is handled by the main move generation logic.
    """
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Entered for color {color}, drop PAWN at ({drop_row}, {drop_col})")
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Initial game state (SFEN): {game.to_sfen_string()}")
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Current player in initial game: {game.current_player}")

    opp_color = Color.WHITE if color == Color.BLACK else Color.BLACK

    # Create a deep copy to simulate the pawn drop
    temp_game = deepcopy(game)
    # print(f"DEBUG_UCHI_FU_ZUME_DETAILED: temp_game.current_player after deepcopy: {temp_game.current_player}")


    # Ensure the square is empty (pre-condition, but good for robustness if called directly)
    if temp_game.get_piece(drop_row, drop_col) is not None:
        print("DEBUG_UCHI_FU_ZUME_DETAILED: Target square not empty. Returning False.")
        # This should ideally be caught by can_drop_specific_piece before calling this
        return False

    # Simulate the pawn drop
    # Ensure the piece type is PAWN and player has it in hand
    if not (
        PieceType.PAWN in temp_game.hands[color.value]
        and temp_game.hands[color.value][PieceType.PAWN] > 0
    ):
        print("DEBUG_UCHI_FU_ZUME_DETAILED: No pawn in hand to drop. Returning False.")
        return False  # Not uchi_fu_zume if no pawn to drop or not in hand

    temp_game.set_piece(drop_row, drop_col, Piece(PieceType.PAWN, color))
    temp_game.hands[color.value][PieceType.PAWN] -= 1
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Game state after pawn drop (SFEN): {temp_game.to_sfen_string()}")
    # Note: temp_game.current_player is still the original game\\'s current_player at this point.
    # This is important for context if any functions rely on it, though most here use explicit color.
    # print(f"DEBUG_UCHI_FU_ZUME_DETAILED: temp_game.current_player after pawn drop, before player switch for G_A_L_M: {temp_game.current_player}")


    # Find the opponent\\'s king
    opp_king_pos = find_king(temp_game, opp_color) # MODIFIED: Use find_king helper
    
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Opponent color: {opp_color}, Opponent king pos: {opp_king_pos}")

    if not opp_king_pos:
        print("DEBUG_UCHI_FU_ZUME_DETAILED: Opponent king not found. Returning False.")
        # Opponent's king not found (e.g., captured in a hypothetical scenario not relevant to uchi_fu_zume)
        return False

    # 1. Check if the drop delivers check to the opponent's king.
    #    check_if_square_is_attacked function takes attacker_color explicitly.
    drop_delivers_check = check_if_square_is_attacked(
        temp_game, opp_king_pos[0], opp_king_pos[1], color # color is the one who dropped the pawn
    )
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Drop by {color} delivers check to {opp_color} king: {drop_delivers_check}")

    if not drop_delivers_check:
        print("DEBUG_UCHI_FU_ZUME_DETAILED: Drop does not deliver check. Returning False.")
        return False  # Not uchi_fu_zume if the drop itself doesn't give check

    # Original logic for checking opponent's escape moves:
    # 2. Check if the opponent's king has any legal moves to escape the check.
    #    Temporarily switch current player in temp_game to opponent to generate their legal moves.
    original_player_in_temp_game = temp_game.current_player
    temp_game.current_player = opp_color # Now it's opponent's turn to find escapes
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Switched temp_game.current_player to {temp_game.current_player} (opp_color) for opponent move generation.")
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Calling generate_all_legal_moves for {opp_color} with is_uchi_fu_zume_check=True. SFEN: {temp_game.to_sfen_string()}")
    # Pass a flag to prevent recursive uchi_fu_zume checks within this call
    opponent_legal_moves = generate_all_legal_moves(
        temp_game, is_uchi_fu_zume_check=True # Restore this flag
    )
    temp_game.current_player = original_player_in_temp_game  # Restore current player
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Restored temp_game.current_player to {temp_game.current_player}.")
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Opponent ({opp_color}) legal moves found: {len(opponent_legal_moves)} moves: {opponent_legal_moves}")

    # If opponent_legal_moves is empty, it means the opponent is checkmated by the pawn drop.
    # Therefore, it IS uchi_fu_zume.
    result = not opponent_legal_moves
    print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Returning {result} (is uchi_fu_zume if True)")
    return result


def is_king_in_check_after_simulated_move(
    game: "ShogiGame", player_color: Color
) -> bool:
    """
    Checks if the king of 'player_color' is in check on the current board.
    Assumes the board reflects the state *after* a move has been made.
    (Formerly ShogiGame._king_in_check_after_move)
    """
    # MODIFIED: Use find_king helper
    king_pos = find_king(game, player_color)

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
    game: "ShogiGame", piece_type: PieceType, r_to: int, c_to: int, color: Color,
    is_escape_check: bool = False  # Added for uchi_fu_zume fix
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
    # print(f"DEBUG_CAN_DROP: Checking drop of {piece_type} by {color} at ({r_to},{c_to}), is_escape_check={is_escape_check}") # Basic log
    if game.get_piece(r_to, c_to) is not None:
        # print(f"DEBUG_CAN_DROP: Square ({r_to},{c_to}) not empty. Returning False.") # Basic log
        return False  # Square must be empty

    # Determine player-specific "forward" direction and promotion zone boundaries
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
        #    This check is skipped if we are evaluating an escape move during an uchi_fu_zume check.
        if not is_escape_check:  # <-- MODIFIED: Only check uchi_fu_zume if not an escape check
            # print(f"DEBUG_CAN_DROP: PAWN drop, is_escape_check is False. Calling check_for_uchi_fu_zume for {color} at ({r_to},{c_to}).") # Basic log
            if check_for_uchi_fu_zume(game, r_to, c_to, color):
                print(f"DEBUG_CAN_DROP: uchi_fu_zume check for {color} dropping PAWN at ({r_to},{c_to}) returned True. Preventing drop.") # Debug print
                return False
        # else: # Basic log
            # print(f"DEBUG_CAN_DROP: PAWN drop, is_escape_check is True. Skipping uchi_fu_zume check for {color} at ({r_to},{c_to}).") # Basic log
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


def generate_all_legal_moves(
    game: "ShogiGame", is_uchi_fu_zume_check: bool = False
) -> List[MoveTuple]:
    """
    Generates all legal moves for the current player.
    A move is legal if:
    1. It follows the piece\'s movement rules.
    2. For drops: it follows drop rules (Nifu, Uchi Fu Zume, no-move squares).
    3. It does not leave the current player\'s king in check.
    (Formerly ShogiGame.get_legal_moves)
    """
    if is_uchi_fu_zume_check:
        print(f"DEBUG_GALM_UCHI_CHECK: Entered for player {game.current_player}. SFEN: {game.to_sfen_string()}")
        print(f"DEBUG_GALM_UCHI_CHECK: Hands for {game.current_player}: {game.hands[game.current_player.value]}")
        # For uchi_fu_zume check, we only care if the current player (whose escape moves are being generated) has a king.
        current_player_king_exists = find_king(game, game.current_player) is not None
        if not current_player_king_exists:
            print(f"DEBUG_GALM_UCHI_CHECK: Exiting early because current player {game.current_player} (opponent) has no king. SFEN: {game.to_sfen_string()}")
            return []
    else:
        # Original check for normal move generation: both kings must be present.
        # Refactored to use find_king for consistency.
        has_black_king = find_king(game, Color.BLACK) is not None
        has_white_king = find_king(game, Color.WHITE) is not None
        if not (has_black_king and has_white_king):
            # Added SFEN to this log path as well for better debugging if it ever triggers.
            print(f"DEBUG_GALM_NORMAL_CHECK: Exiting early because not (has_black_king and has_white_king). BK: {has_black_king}, WK: {has_white_king}. SFEN: {game.to_sfen_string()}")
            return []

    legal_moves: List[MoveTuple] = []
    original_player_color = game.current_player
    # opponent_color = Color.WHITE if original_player_color == Color.BLACK else Color.BLACK

    # I. Generate Board Moves (moving a piece already on the board)
    if is_uchi_fu_zume_check:
        print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Starting board move generation.")
    for r_from in range(9):
        for c_from in range(9):
            piece = game.get_piece(r_from, c_from)
            if piece and piece.color == original_player_color:
                # Get squares the piece attacks or can move to if empty
                potential_squares = generate_piece_potential_moves(
                    game, piece, r_from, c_from
                )
                if is_uchi_fu_zume_check and potential_squares:
                    print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Piece {piece.type.name} at ({r_from},{c_from}) potential squares: {potential_squares}")


                for r_to, c_to in potential_squares:
                    # Check for promotion possibilities
                    can_promote = can_promote_specific_piece(game, piece, r_from, r_to)
                    must_promote = must_promote_specific_piece(
                        piece, r_to
                    )  # Removed game argument

                    current_move_tuples_to_check: List[
                        "BoardMoveTuple"
                    ]  # Declare type here
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
                        # Simulate the move on a game copy via deepcopy
                        temp_game = deepcopy(game)
                        # print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Simulating board move: {board_move_tuple}, current player in temp_game before make_move: {temp_game.current_player}")
                        temp_game.make_move(board_move_tuple, is_simulation=True)
                        # print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] After simulated board move {board_move_tuple}, current player in temp_game: {temp_game.current_player}. SFEN: {temp_game.to_sfen_string()}")
                        # Check if the current player's king is NOT in check after this move
                        king_in_check_after_sim = temp_game.is_in_check(original_player_color, debug_recursion=is_uchi_fu_zume_check) # Pass through based on top-level flag
                        if is_uchi_fu_zume_check:
                            print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Board Move {board_move_tuple}: King in check after sim: {king_in_check_after_sim}")
                            if king_in_check_after_sim:
                                king_pos_after_move = temp_game.find_king(original_player_color) # Correct: temp_game has find_king
                                print(f"DEBUG_GALM_UCHI_CHECK_DETAIL: King {original_player_color} at {king_pos_after_move} considered in check after move {board_move_tuple}.")
                                print(f"DEBUG_GALM_UCHI_CHECK_DETAIL: Board state (SFEN): {temp_game.to_sfen_string()}")
                                opponent_color_of_current_player = Color.BLACK if original_player_color == Color.WHITE else Color.WHITE
                                if king_pos_after_move: # Ensure king_pos is not None
                                    # Call with debug=True to get detailed output from check_if_square_is_attacked
                                    is_attacked_detail = check_if_square_is_attacked(temp_game, king_pos_after_move[0], king_pos_after_move[1], opponent_color_of_current_player, debug=True)
                                    print(f"DEBUG_GALM_UCHI_CHECK_DETAIL: check_if_square_is_attacked for king at {king_pos_after_move} by {opponent_color_of_current_player} returned: {is_attacked_detail}")


                        if not king_in_check_after_sim:
                            legal_moves.append(board_move_tuple)

    # II. Generate Drop Moves
    if is_uchi_fu_zume_check:
        print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Starting drop move generation. Hand: {game.hands[original_player_color.value]}")
    for piece_type_to_drop_val, count in game.hands[original_player_color.value].items():
        piece_type_to_drop = PieceType(piece_type_to_drop_val) # Ensure it's PieceType enum
        if count > 0:
            if is_uchi_fu_zume_check:
                print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Considering dropping {piece_type_to_drop.name}")
            for r_to in range(9):
                for c_to in range(9):
                    if game.get_piece(r_to, c_to) is None:  # Square must be empty
                        # Check basic drop legality (Nifu, Uchi Fu Zume, no-move squares)
                        # Pass is_uchi_fu_zume_check to can_drop_specific_piece's new is_escape_check parameter
                        if is_uchi_fu_zume_check:
                            print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Checking can_drop_specific_piece for {piece_type_to_drop.name} at ({r_to},{c_to}), is_escape_check={is_uchi_fu_zume_check}")
                        
                        can_drop = can_drop_specific_piece(
                            game, # Use original game state for can_drop checks
                            piece_type_to_drop,
                            r_to,
                            c_to,
                            original_player_color,
                            is_escape_check=is_uchi_fu_zume_check # <-- MODIFIED
                        )
                        if is_uchi_fu_zume_check:
                            print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] can_drop_specific_piece for {piece_type_to_drop.name} at ({r_to},{c_to}) returned: {can_drop}")


                        if can_drop:
                            # Create the drop move tuple
                            drop_move_tuple: DropMoveTuple = (
                                None,
                                None,
                                r_to,
                                c_to,
                                piece_type_to_drop,
                            )
                            # Simulate the drop on a game copy
                            temp_game = deepcopy(game)
                            # print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Simulating drop move: {drop_move_tuple}, current player in temp_game before make_move: {temp_game.current_player}")
                            temp_game.make_move(drop_move_tuple, is_simulation=True)
                            # print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] After simulated drop move {drop_move_tuple}, current player in temp_game: {temp_game.current_player}. SFEN: {temp_game.to_sfen_string()}")
                            # Check if the current player's king is NOT in check after this drop
                            king_in_check_after_sim = temp_game.is_in_check(original_player_color, debug_recursion=is_uchi_fu_zume_check) # Pass through
                            if is_uchi_fu_zume_check:
                                print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Drop Move {drop_move_tuple}: King in check after sim: {king_in_check_after_sim}")
                                if king_in_check_after_sim:
                                    king_pos_after_move = temp_game.find_king(original_player_color) # Correct: temp_game has find_king
                                    print(f"DEBUG_GALM_UCHI_CHECK_DETAIL: King {original_player_color} at {king_pos_after_move} considered in check after drop {drop_move_tuple}.")
                                    print(f"DEBUG_GALM_UCHI_CHECK_DETAIL: Board state (SFEN): {temp_game.to_sfen_string()}")
                                    opponent_color_of_current_player = Color.BLACK if original_player_color == Color.WHITE else Color.WHITE
                                    if king_pos_after_move: # Ensure king_pos is not None
                                        is_attacked_detail = check_if_square_is_attacked(temp_game, king_pos_after_move[0], king_pos_after_move[1], opponent_color_of_current_player, debug=True)
                                        print(f"DEBUG_GALM_UCHI_CHECK_DETAIL: check_if_square_is_attacked for king at {king_pos_after_move} by {opponent_color_of_current_player} returned: {is_attacked_detail}")

                            if not king_in_check_after_sim:
                                if is_uchi_fu_zume_check:
                                    print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] ADDING legal drop move: {drop_move_tuple}")
                                legal_moves.append(drop_move_tuple)
    if is_uchi_fu_zume_check:
        print(f"DEBUG_GALM_UCHI_CHECK: [{original_player_color}] Finished. Total legal moves found: {len(legal_moves)}")
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
