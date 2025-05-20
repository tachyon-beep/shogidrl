# shogi_game_io.py

from typing import TYPE_CHECKING, Tuple
import numpy as np
import datetime  # For KIF Date header

from ..utils import PolicyOutputMapper  # Assuming this path is correct
from .shogi_core_definitions import (
    OBS_PROMOTED_ORDER,
    OBS_UNPROMOTED_ORDER,
    Color,
    Piece,
    PieceType,
    TerminationReason,
    get_piece_type_from_symbol,
    get_unpromoted_types,
    KIF_PIECE_SYMBOLS,
    MoveTuple,
    BoardMoveTuple,
    DropMoveTuple,
    SYMBOL_TO_PIECE_TYPE,
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
     42: Current player (all 1.0 if Black, all 0.0 if White playing as Black\'\'\'s opponent view)
     43: Move count (normalized)
     44, 45: Reserved (all zeros) - Kept for potential future features like repetition counts or specific game phase indicators.
    Total: 14 (player) + 14 (opponent) + 7 (player hand) + 7 (opp hand) + 2 (meta) + 2 (reserved) = 46 channels.
    The original code specifies 46 channels, and these two are explicitly reserved.
    """
    obs = np.zeros((46, 9, 9), dtype=np.float32)

    # Map PieceType to its index in OBS_UNPROMOTED_ORDER or OBS_PROMOTED_ORDER
    unpromoted_map = {pt: i for i, pt in enumerate(OBS_UNPROMOTED_ORDER)}
    promoted_map = {pt: i for i, pt in enumerate(OBS_PROMOTED_ORDER)}

    for r in range(9):
        for c in range(9):
            p = game.board[r][c]
            if p is None:
                continue

            is_current_player_piece = p.color == game.current_player
            # player_base_channel = ( # This was defined in original but not used directly here
            # 0 if is_current_player_piece else 14
            # )

            channel_offset = -1

            if p.is_promoted:
                if p.type in promoted_map:
                    # Offset for promoted planes (e.g., 8 for current player, 22 for opponent)
                    promoted_block_offset = (
                        8 if is_current_player_piece else (14 + 8)
                    )  # 14 is base for opponent, 8 for promoted group
                    channel_offset = promoted_block_offset + promoted_map[p.type]
            else:  # Unpromoted or non-promotable (King, Gold)
                if p.type in unpromoted_map:
                    # Offset for unpromoted planes (e.g., 0 for current player, 14 for opponent)
                    unpromoted_block_offset = 0 if is_current_player_piece else 14
                    channel_offset = unpromoted_block_offset + unpromoted_map[p.type]

            if channel_offset != -1:
                obs[channel_offset, r, c] = 1.0

    # Pieces in hand (7 channels per player: P,L,N,S,G,B,R)
    hand_piece_order = get_unpromoted_types()  # Use the imported function

    # Current player's hand (channels 28-34)
    current_player_hand_start_ch = 28
    for i, piece_type_enum in enumerate(hand_piece_order):
        count = game.hands[game.current_player.value].get(piece_type_enum, 0)
        if count > 0:
            ch = current_player_hand_start_ch + i
            obs[ch, :, :] = count / 18.0  # Normalize (e.g., by max pawns)

    # Opponent's hand (channels 35-41)
    opponent_hand_start_ch = 35
    opponent_color_val = (
        Color.WHITE.value if game.current_player == Color.BLACK else Color.BLACK.value
    )
    for i, piece_type_enum in enumerate(hand_piece_order):
        count = game.hands[opponent_color_val].get(piece_type_enum, 0)
        if count > 0:
            ch = opponent_hand_start_ch + i
            obs[ch, :, :] = count / 18.0

    # Current player plane (42)
    obs[42, :, :] = 1.0 if game.current_player == Color.BLACK else 0.0

    # Move count plane (43)
    # Use game._max_moves_this_game for consistency
    max_moves = float(game.max_moves_per_game)
    obs[43, :, :] = game.move_count / max_moves if max_moves > 0 else 0.0

    # Planes 44, 45 are reserved for potential future features (e.g., repetition count, game phase indicators)
    # and remain zeros by default.
    return obs


def convert_game_to_text_representation(game: "ShogiGame") -> str:
    """
    Returns a string representation of the Shogi game board and state.
    """
    lines = []
    for r_idx, row_data in enumerate(game.board):
        line_str = f"{9-r_idx} "  # Shogi board rank numbers (9 down to 1)
        line_pieces = []
        for p in row_data:
            if p:
                symbol = p.symbol()
                if len(symbol) == 1:  # e.g., P
                    line_pieces.append(f" {symbol} ")  # Results in " P "
                else:  # e.g., +P
                    line_pieces.append(f"{symbol} ")  # Results in "+P "
            else:
                line_pieces.append(" . ")  # Consistent 3-char width
        lines.append(line_str + "".join(line_pieces))
    # Add file numbers at the bottom, ensuring 3-char spacing
    lines.append(
        "  a  b  c  d  e  f  g  h  i"  # Adjusted spacing: 2 spaces before 'a', then 2 between letters
    )
    # Add hands
    black_hand_dict = {
        pt.name: count
        for pt, count in game.hands[Color.BLACK.value].items()
        if count > 0
    }
    lines.append(f"Black's hand: {black_hand_dict}")
    white_hand_dict = {
        pt.name: count
        for pt, count in game.hands[Color.WHITE.value].items()
        if count > 0
    }
    lines.append(f"White's hand: {white_hand_dict}")
    lines.append(f"Current player: {game.current_player.name}")
    return "\n".join(lines)


def game_to_kif(
    game: "ShogiGame",
    filename: str,
    sente_player_name: str = "Sente",
    gote_player_name: str = "Gote",
) -> None:
    """
    Converts a game to a KIF file.
    Uses standard KIF piece notation (+FU, -FU, etc.) and includes more headers.
    """
    with open(filename, "w", encoding="utf-8") as kif_file:
        # --- KIF Headers ---
        kif_file.write("#KIF version=2.0 encoding=UTF-8\n")
        kif_file.write("*Event: Casual Game\n")  # Placeholder, was f-string
        kif_file.write("*Site: Local Machine\n")  # Placeholder, was f-string
        kif_file.write(f"*Date: {datetime.date.today().strftime('%Y/%m/%d')}\n")
        kif_file.write(f"*Player Sente: {sente_player_name}\n")
        kif_file.write(f"*Player Gote: {gote_player_name}\n")
        kif_file.write("*Handicap: HIRATE\n")  # Or NONE. Was f-string

        # Standard HIRATE starting position
        kif_file.write("P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n")
        kif_file.write("P2 * -HI * * * * * -KA * \n")
        kif_file.write("P3-FU-FU-FU-FU-FU-FU-FU-FU-FU\n")
        kif_file.write("P4 * * * * * * * * * \n")
        kif_file.write("P5 * * * * * * * * * \n")
        kif_file.write("P6 * * * * * * * * * \n")
        kif_file.write("P7+FU+FU+FU+FU+FU+FU+FU+FU+FU\n")
        kif_file.write("P8 * +KA * * * * * +HI * \n")
        kif_file.write("P9+KY+KE+GI+KI+OU+KI+GI+KE+KY\n")

        # --- Initial Hands (KIF format: P+00FU00KY... for Sente, P-00FU00KY... for Gote) ---
        # This assumes starting with empty hands for a standard game from initial board setup.
        sente_hand_str = "P+"
        gote_hand_str = "P-"
        hand_order_for_kif = [
            PieceType.ROOK,
            PieceType.BISHOP,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
            PieceType.PAWN,
        ]  # Common KIF hand order

        initial_sente_hand = game.hands[Color.BLACK.value]
        initial_gote_hand = game.hands[Color.WHITE.value]

        for pt in hand_order_for_kif:
            sente_hand_str += (
                f"{initial_sente_hand.get(pt, 0):02d}{KIF_PIECE_SYMBOLS.get(pt, '??')}"
            )
            gote_hand_str += (
                f"{initial_gote_hand.get(pt, 0):02d}{KIF_PIECE_SYMBOLS.get(pt, '??')}"
            )
        kif_file.write(f"{sente_hand_str}\n")
        kif_file.write(f"{gote_hand_str}\n")

        # --- Player to move first ---
        kif_file.write(
            f"{'+' if game.current_player == Color.BLACK else '-'}\n"
        )  # + for Sente, - for Gote

        kif_file.write("moves\n")  # Start of the moves section

        # --- Moves ---
        mapper = PolicyOutputMapper()
        for i, move_entry in enumerate(game.move_history):
            move_obj = move_entry.get("move")  # Your internal move object/tuple
            if not move_obj:
                continue

            usi_move_str = mapper.shogi_move_to_usi(move_obj)
            kif_file.write(f"{i+1} {usi_move_str}\n")

        # --- Game Termination ---
        if game.game_over:
            if game.termination_reason == TerminationReason.CHECKMATE.value:
                kif_file.write("Tsumi\n")
            elif game.termination_reason == TerminationReason.RESIGNATION.value:
                kif_file.write("Toryo\n")
            elif game.termination_reason == TerminationReason.MAX_MOVES_EXCEEDED.value:
                kif_file.write("Jishogi\n")
            elif game.termination_reason == TerminationReason.REPETITION.value:
                kif_file.write("Sennichite\n")
            elif game.termination_reason == TerminationReason.ILLEGAL_MOVE.value:
                kif_file.write("Illegal move\n")
            elif game.termination_reason == TerminationReason.TIME_FORFEIT.value:
                kif_file.write("Time_up\n")

        kif_file.write("*EOF\n")  # Standard KIF end marker


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

    file_char = sfen_sq[0]
    rank_char = sfen_sq[1]

    col = 9 - int(file_char)
    row = ord(rank_char) - ord("a")

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
    and converts it into an internal MoveTuple.
    """
    sfen_move_str = sfen_move_str.strip()

    if "*" in sfen_move_str:
        parts = sfen_move_str.split("*")
        if len(parts) != 2 or len(parts[0]) != 1 or len(parts[1]) != 2:
            raise ValueError(f"Invalid SFEN drop move format: {sfen_move_str}")

        piece_char = parts[0]
        sfen_sq_to = parts[1]

        try:
            piece_to_drop = _get_piece_type_from_sfen_char(piece_char)
            to_r, to_c = _parse_sfen_square(sfen_sq_to)
        except ValueError as e:
            raise ValueError(
                f"Error parsing SFEN drop move '{sfen_move_str}': {e}"
            ) from e

        return (None, None, to_r, to_c, piece_to_drop)
    else:
        promote = False
        if sfen_move_str.endswith("+"):
            promote = True
            sfen_move_str = sfen_move_str[:-1]

        if len(sfen_move_str) != 4:
            raise ValueError(f"Invalid SFEN board move format: {sfen_move_str}")

        sfen_sq_from = sfen_move_str[:2]
        sfen_sq_to = sfen_move_str[2:]

        try:
            from_r, from_c = _parse_sfen_square(sfen_sq_from)
            to_r, to_c = _parse_sfen_square(sfen_sq_to)
        except ValueError as e:
            raise ValueError(
                f"Error parsing SFEN board move '{sfen_move_str}': {e}"
            ) from e

        return (from_r, from_c, to_r, to_c, promote)
