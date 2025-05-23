# shogi_game_io.py

import datetime  # For KIF Date header
import re  # Import the re module
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..utils import PolicyOutputMapper  # Assuming this path is correct
from .shogi_core_definitions import (  # Observation plane constants
    KIF_PIECE_SYMBOLS,
    OBS_CURR_PLAYER_HAND_START,
    OBS_CURR_PLAYER_INDICATOR,
    OBS_CURR_PLAYER_PROMOTED_START,
    OBS_CURR_PLAYER_UNPROMOTED_START,
    OBS_MOVE_COUNT,
    OBS_OPP_PLAYER_HAND_START,
    OBS_OPP_PLAYER_PROMOTED_START,
    OBS_OPP_PLAYER_UNPROMOTED_START,
    OBS_PROMOTED_ORDER,
    OBS_UNPROMOTED_ORDER,
    SYMBOL_TO_PIECE_TYPE,
    Color,
    MoveTuple,
    Piece,
    PieceType,
    TerminationReason,
    get_unpromoted_types,
)

if TYPE_CHECKING:
    from .shogi_game import ShogiGame  # For type hinting the 'game' parameter


def generate_neural_network_observation(game: "ShogiGame") -> np.ndarray:
    """
    Returns the current board state as a (Channels, 9, 9) NumPy array for RL input.
    Channels:
    Player planes (current player POV):
     0-7: Unpromoted pieces (P, L, N, S, G, B, R, K)
     8-13: Promoted pieces (+P, +L, +N, +S, +B, +R)
    Opponent planes:
     14-21: Unpromoted pieces
     22-27: Promoted pieces
    Hand planes:
     28-34: Current player's hand (P, L, N, S, G, B, R) - 7 types
     35-41: Opponent's hand (P, L, N, S, G, B, R) - 7 types
    Meta planes:
     42: Current player (all 1.0 if Black, all 0.0 if White playing as Black's opponent view)
     43: Move count (normalized)
     44: Reserved for potential future features like repetition count (currently all zeros)
     45: Reserved for potential future features like game phase indicators (currently all zeros)
    Total: 46 channels (14 player board + 14 opponent board + 7 player hand + 7 opponent hand + 4 meta).
    """
    obs = np.zeros((46, 9, 9), dtype=np.float32)
    is_black_perspective = game.current_player == Color.BLACK

    # Map PieceType to its index in OBS_UNPROMOTED_ORDER or OBS_PROMOTED_ORDER
    unpromoted_map: Dict[PieceType, int] = {
        pt: i for i, pt in enumerate(OBS_UNPROMOTED_ORDER)
    }
    promoted_map: Dict[PieceType, int] = {
        pt: i for i, pt in enumerate(OBS_PROMOTED_ORDER)
    }

    for r in range(9):
        for c in range(9):
            # For white's perspective, we need to mirror the board coordinates
            flipped_r = r if is_black_perspective else 8 - r
            flipped_c = c if is_black_perspective else 8 - c

            p: Optional[Piece] = game.board[r][c]
            if p is None:
                continue

            is_current_player_piece: bool = p.color == game.current_player
            channel_offset: int = -1

            if p.is_promoted:
                if p.type in promoted_map:
                    # Offset for promoted planes
                    promoted_block_offset = (
                        OBS_CURR_PLAYER_PROMOTED_START
                        if is_current_player_piece
                        else OBS_OPP_PLAYER_PROMOTED_START
                    )
                    channel_offset = promoted_block_offset + promoted_map[p.type]
            else:  # Unpromoted or non-promotable (King, Gold)
                if p.type in unpromoted_map:
                    # Offset for unpromoted planes
                    unpromoted_block_offset = (
                        OBS_CURR_PLAYER_UNPROMOTED_START
                        if is_current_player_piece
                        else OBS_OPP_PLAYER_UNPROMOTED_START
                    )
                    channel_offset = unpromoted_block_offset + unpromoted_map[p.type]

            if channel_offset != -1:
                # Use the flipped coordinates for setting the observation plane
                obs[channel_offset, flipped_r, flipped_c] = 1.0

    # Pieces in hand (7 channels per player: P,L,N,S,G,B,R)
    hand_piece_order: List[
        PieceType
    ] = get_unpromoted_types()  # Use the imported function

    # Current player's hand
    for i, piece_type_enum_player in enumerate(hand_piece_order):
        player_hand_count: int = game.hands[game.current_player.value].get(
            piece_type_enum_player, 0
        )
        if player_hand_count > 0:
            player_ch: int = OBS_CURR_PLAYER_HAND_START + i
            obs[player_ch, :, :] = (
                player_hand_count / 18.0
            )  # Normalize (e.g., by max pawns)

    # Opponent's hand
    opponent_color_val: int = (
        Color.WHITE.value if game.current_player == Color.BLACK else Color.BLACK.value
    )
    for i, piece_type_enum_opponent in enumerate(hand_piece_order):
        opponent_hand_count: int = game.hands[opponent_color_val].get(
            piece_type_enum_opponent, 0
        )
        if opponent_hand_count > 0:
            opponent_ch: int = OBS_OPP_PLAYER_HAND_START + i
            obs[opponent_ch, :, :] = opponent_hand_count / 18.0

    # Current player indicator plane
    obs[OBS_CURR_PLAYER_INDICATOR, :, :] = (
        1.0 if game.current_player == Color.BLACK else 0.0
    )

    # Move count plane (normalized)
    max_moves = float(game.max_moves_per_game)
    obs[OBS_MOVE_COUNT, :, :] = game.move_count / max_moves if max_moves > 0 else 0.0

    # Planes for OBS_RESERVED_1 and OBS_RESERVED_2 remain zeros.
    return obs


def convert_game_to_text_representation(game: "ShogiGame") -> str:
    """
    Returns a string representation of the Shogi game board and state.
    """
    lines = []
    for r_idx, row_data in enumerate(game.board):
        line_str: str = f"{9-r_idx} "  # Shogi board rank numbers (9 down to 1)
        line_pieces: List[str] = []
        for (
            p_cell
        ) in (
            row_data
        ):  # Renamed p to p_cell to avoid conflict with p in outer scope if any
            if p_cell:
                symbol: str = p_cell.symbol()
                if len(symbol) == 1:  # e.g., P
                    line_pieces.append(f" {symbol} ")  # Results in " P "
                else:  # e.g., +P
                    line_pieces.append(f"{symbol} ")  # Results in "+P "
            else:
                line_pieces.append(" . ")  # Consistent 3-char width
        lines.append(line_str + "".join(line_pieces))
    # Add file numbers at the bottom with consistent single-space formatting
    lines.append(
        "a b c d e f g h i"  # Single space between column labels as expected by tests
    )

    # Add player turn and move info first, then hands info
    lines.append(f"Turn: {game.current_player.name}, Move: {game.move_count+1}")

    black_hand_dict: Dict[str, int] = {
        pt.name: count
        for pt, count in game.hands[Color.BLACK.value].items()
        if count > 0
    }
    lines.append(f"Black's hand: {black_hand_dict}")

    white_hand_dict: Dict[str, int] = {
        pt.name: count
        for pt, count in game.hands[Color.WHITE.value].items()
        if count > 0
    }
    lines.append(f"White's hand: {white_hand_dict}")
    return "\n".join(lines)


def game_to_kif(
    game: "ShogiGame",
    filename: Optional[str] = None,
    sente_player_name: str = "Sente",
    gote_player_name: str = "Gote",
) -> Optional[str]:
    """
    Converts a game to a KIF file or string representation.
    Uses standard KIF piece notation (+FU, -FU, etc.) and includes more headers.

    Args:
        game: The ShogiGame to convert
        filename: If provided, the KIF will be written to this file
        sente_player_name: Name of the black/sente player
        gote_player_name: Name of the white/gote player

    Returns:
        If filename is None, returns a string representation of the KIF.
        Otherwise returns None after writing to the file.
    """

    # Helper function to create KIF content
    def create_kif_content():
        lines = []
        # --- KIF Headers ---
        lines.append("#KIF version=2.0 encoding=UTF-8")
        lines.append("*Event: Casual Game")
        lines.append("*Site: Local Machine")
        lines.append(f"*Date: {datetime.date.today().strftime('%Y/%m/%d')}")
        lines.append(f"*Player Sente: {sente_player_name}")
        lines.append(f"*Player Gote: {gote_player_name}")
        lines.append("*Handicap: HIRATE")

        # Standard HIRATE starting position
        lines.append("P1-KY-KE-GI-KI-OU-KI-GI-KE-KY")
        lines.append("P2 * -HI * * * * * -KA * ")
        lines.append("P3-FU-FU-FU-FU-FU-FU-FU-FU-FU")
        lines.append("P4 * * * * * * * * * ")
        lines.append("P5 * * * * * * * * * ")
        lines.append("P6 * * * * * * * * * ")
        lines.append("P7+FU+FU+FU+FU+FU+FU+FU+FU+FU")
        lines.append("P8 * +KA * * * * * +HI * ")
        lines.append("P9+KY+KE+GI+KI+OU+KI+GI+KE+KY")

        # --- Initial Hands (KIF format: P+00FU00KY... for Sente, P-00FU00KY... for Gote) ---
        # This assumes starting with empty hands for a standard game from initial board setup.
        sente_hand_str: str = "P+"
        gote_hand_str: str = "P-"
        hand_order_for_kif: List[PieceType] = [
            PieceType.ROOK,
            PieceType.BISHOP,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
            PieceType.PAWN,
        ]  # Common KIF hand order

        initial_sente_hand: Dict[PieceType, int] = game.hands[Color.BLACK.value]
        initial_gote_hand: Dict[PieceType, int] = game.hands[Color.WHITE.value]

        for pt in hand_order_for_kif:
            sente_hand_str += (
                f"{initial_sente_hand.get(pt, 0):02d}{KIF_PIECE_SYMBOLS.get(pt, '??')}"
            )
            gote_hand_str += (
                f"{initial_gote_hand.get(pt, 0):02d}{KIF_PIECE_SYMBOLS.get(pt, '??')}"
            )
        lines.append(f"{sente_hand_str}")
        lines.append(f"{gote_hand_str}")

        # --- Player to move first ---
        lines.append(
            f"{'+' if game.current_player == Color.BLACK else '-'}"
        )  # + for Sente, - for Gote

        lines.append("moves")  # Start of the moves section

        # --- Moves ---
        mapper = PolicyOutputMapper()
        for i, move_entry in enumerate(game.move_history):
            move_obj: Optional[MoveTuple] = move_entry.get(
                "move"
            )  # Your internal move object/tuple
            if not move_obj:
                continue

            usi_move_str: str = mapper.shogi_move_to_usi(move_obj)
            lines.append(f"{i+1} {usi_move_str}")

        # --- Game Termination ---
        if game.game_over:
            termination_map: Dict[str, str] = {
                "Tsumi": "詰み",
                "Toryo": "投了",
                "Sennichite": "千日手",
                "Stalemate": "持将棋",
                "Max moves reached": "持将棋",  # Or "最大手数" or similar
                # Add other mappings for values set in game.termination_reason
            }
            reason_str: Optional[str] = game.termination_reason
            kif_termination_reason_display: str

            if reason_str is None:
                kif_termination_reason_display = ""  # No reason string if None
            else:
                # Now reason_str is str, so termination_map.get(str, str) is used
                kif_termination_reason_display = termination_map.get(
                    reason_str, reason_str
                )

            if kif_termination_reason_display:
                lines.append(kif_termination_reason_display)

            # Append the RESULT line based on winner
            if game.winner == Color.BLACK:
                lines.append("RESULT:SENTE_WIN")
            elif game.winner == Color.WHITE:
                lines.append("RESULT:GOTE_WIN")
            elif game.winner is None:  # Draw conditions
                # More specific draw reasons could be mapped to KIF draw results
                if game.termination_reason in [
                    TerminationReason.REPETITION.value,
                    TerminationReason.IMPASSE.value,
                    TerminationReason.MAX_MOVES_EXCEEDED.value,
                ]:
                    lines.append("RESULT:DRAW")  # Or HIKIWAKE etc.

        lines.append("*EOF")  # Standard KIF end marker
        return "\n".join(lines)

    # Generate KIF content
    kif_content = create_kif_content()

    # Either write to file or return as string
    if filename:
        with open(filename, "w", encoding="utf-8") as kif_file:
            kif_file.write(kif_content)
        return None
    else:
        return kif_content


# --- SFEN Move Parsing ---


def _parse_sfen_square(sfen_sq: str) -> Tuple[int, int]:
    """
    Converts an SFEN square string (e.g., "7g", "5e") to 0-indexed (row, col).
    File (1-9) maps to column (8-0). Rank (a-i) maps to row (0-8).
    Example: "9a" -> (0,0), "1i" -> (8,8)
    """
    if not (
        len(sfen_sq) == 2 and "1" <= sfen_sq[0] <= "9" and "a" <= sfen_sq[1] <= "i"
    ):
        raise ValueError(f"Invalid SFEN square format: {sfen_sq}")

    file_char: str = sfen_sq[0]
    rank_char: str = sfen_sq[1]

    col: int = 9 - int(file_char)
    row: int = ord(rank_char) - ord("a")

    return row, col


def _get_piece_type_from_sfen_char(char: str) -> PieceType:
    """
    Converts an SFEN piece character (e.g., 'P', 'L', 'B') to a PieceType enum.
    SFEN uses uppercase single letters for standard pieces.
    """
    if char in SYMBOL_TO_PIECE_TYPE:
        pt = SYMBOL_TO_PIECE_TYPE[char]
        # Ensure it's a basic piece type that can be dropped
        if (
            pt in get_unpromoted_types() and pt != PieceType.KING
        ):  # King cannot be dropped
            return pt
    raise ValueError(f"Invalid SFEN piece character for drop: {char}")


def sfen_to_move_tuple(sfen_move_str: str) -> MoveTuple:
    """
    Parses an SFEN move string (e.g., "7g7f", "P*5e", "2b3a+")
    and converts it into an internal MoveTuple using regular expressions.
    """
    sfen_move_str = sfen_move_str.strip()

    # Regex for drop moves: e.g., "P*5e"
    # Group 1: Piece character (P, L, N, S, G, B, R)
    # Group 2: Square (e.g., 5e)
    drop_move_pattern = re.compile(r"^([PLNSGBR])\*([1-9][a-i])$")

    # Regex for board moves: e.g., "7g7f", "2b3a+"
    # Group 1: From square (e.g., 7g)
    # Group 2: To square (e.g., 7f)
    # Group 3: Optional promotion character (+)
    board_move_pattern = re.compile(r"^([1-9][a-i])([1-9][a-i])(\+)?$")

    drop_match = drop_move_pattern.match(sfen_move_str)
    if drop_match:
        piece_char: str = drop_match.group(1)
        sfen_sq_to: str = drop_match.group(2)

        try:
            piece_to_drop: PieceType = _get_piece_type_from_sfen_char(piece_char)
            r_to, c_to = _parse_sfen_square(sfen_sq_to)
            return (None, None, r_to, c_to, piece_to_drop)
        except ValueError as e:
            raise ValueError(
                f"Error parsing SFEN drop move '{sfen_move_str}': {e}"
            ) from e

    board_match = board_move_pattern.match(sfen_move_str)
    if board_match:
        sfen_sq_from_str: str = board_match.group(1)
        sfen_sq_to_str: str = board_match.group(2)
        promote_flag: bool = board_match.group(3) is not None

        try:
            r_from, c_from = _parse_sfen_square(sfen_sq_from_str)
            r_to, c_to = _parse_sfen_square(sfen_sq_to_str)
            return (r_from, c_from, r_to, c_to, promote_flag)
        except ValueError as e:
            raise ValueError(
                f"Error parsing SFEN board move '{sfen_move_str}': {e}"
            ) from e

    raise ValueError(f"Invalid SFEN move format: {sfen_move_str}")


# TODO: Consider adding kif_to_game and sfen_to_game functions if needed.
# These would involve more complex parsing of full game states or move sequences.
