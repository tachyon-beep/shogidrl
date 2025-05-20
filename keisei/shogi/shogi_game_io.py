# shogi_game_io.py

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils import PolicyOutputMapper
from .shogi_core_definitions import (
    OBS_PROMOTED_ORDER,
    OBS_UNPROMOTED_ORDER,
    Color,
    Piece,
    PieceType,
    TerminationReason,
    get_piece_type_from_symbol,
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
     44, 45: Reserved (all zeros)
    Total: 14 (player) + 14 (opponent) + 7 (player hand) + 7 (opp hand) + 2 (meta) = 44 channels.
    Original code specifies 46 channels, so 44 and 45 are explicitly kept.
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

    # Planes 44, 45 are reserved and remain zeros.
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
                    line_pieces.append(f" {symbol} ") # Results in " P "
                else:  # e.g., +P
                    line_pieces.append(f"{symbol} ")  # Results in "+P "
            else:
                line_pieces.append(" . ") # Consistent 3-char width
        lines.append(line_str + "".join(line_pieces))
    # Add file numbers at the bottom, ensuring 3-char spacing
    lines.append(
        "  a  b  c  d  e  f  g  h  i" # Adjusted spacing: 2 spaces before 'a', then 2 between letters
    )
    # Add hands
    black_hand_dict = {pt.name: count for pt, count in game.hands[Color.BLACK.value].items() if count > 0}
    lines.append(f"Black's hand: {black_hand_dict}")
    white_hand_dict = {pt.name: count for pt, count in game.hands[Color.WHITE.value].items() if count > 0}
    lines.append(f"White's hand: {white_hand_dict}")
    lines.append(f"Current player: {game.current_player.name}")
    return "\n".join(lines)


def game_to_kif(game: 'ShogiGame', filename: str) -> None:
    """Converts a game to a KIF file."""
    # Use game.max_moves_per_game property
    max_moves = float(game.max_moves_per_game)
    with open(filename, "w", encoding="utf-8") as kif_file:
        # Basic KIF headers (example, adapt as needed)
        # These might come from game.config or other game properties if available
        # For now, using placeholders or omitting if not directly on game object
        # kif_file.write(f"*Player Sente: {game.player_names[0]}\\n")
        # kif_file.write(f"*Player Gote: {game.player_names[1]}\\n")
        # kif_file.write(f"*Initial setup: {game.initial_setup}\\n") # Or a representation
        # kif_file.write(f"*Handicap: {game.handicap}\\n")

        kif_file.write("*KIF version: 2.0\\n")
        kif_file.write("*Game type: Shogi\\n")
        # if hasattr(game, 'start_time') and game.start_time:
        #     kif_file.write(f"*Start time: {game.start_time.isoformat()}\\n")

        # Board state
        kif_file.write("P1" + "".join([str(i) for i in range(9, 0, -1)]) + "\\n")
        for r_idx, row in enumerate(game.board):
            line = f"P{r_idx+1}"
            for c_idx, p in enumerate(row): # c_idx is used by some KIF variants, keep for now
                if p is None:
                    line += " . " # KIF uses spaces for empty squares, not '+'
                else:
                    # KIF piece representation: Color (P for Black, p for White) + Piece Type (e.g., FU, KY)
                    # This needs a mapping from PieceType to KIF piece strings.
                    # For now, using the existing symbol() method which might not be KIF standard.
                    # Standard KIF is like: P+FU, P-GI, etc. or just FU, GI with player context.
                    # The current Piece.symbol() returns things like "P", "p", "+P", "+p".
                    # We need to ensure it aligns with KIF expectations or adapt.
                    # A common KIF style is just the piece type (e.g., FU, KY, HI, KA, OU, KI, GI, KE, TO, NY, NK, NG, UM, RY)
                    # and the player is implicit from whose turn it is or explicit in headers.
                    # Another style uses e.g. " FU" for black pawn, " fu" for white pawn.
                    # Let's assume for now that Piece.symbol() gives a suitable representation
                    # and that KIF readers can interpret it or it's a simplified KIF.
                    # A more robust solution would map PieceType to standard KIF piece names.
                    piece_char = p.symbol() # e.g. P, p, +P, +p
                    # KIF usually expects two characters for a piece, e.g., "FU", "KY".
                    # If symbol() returns one char (e.g. "P"), pad with space. If two (e.g. "+P"), use as is.
                    # This is a simplification.
                    if len(piece_char) == 1:
                        line += f" {piece_char} " # e.g. " P "
                    elif len(piece_char) == 2: # e.g. "+P" or "p " if symbol() was adapted
                        if piece_char.startswith("+") or piece_char.startswith("-"):
                            line += f"{piece_char} " # e.g. "+P "
                        else: # Should not happen with current symbol()
                            line += f"{piece_char}" # e.g. "FU"
                    else: # Should not happen
                        line += " ??"

            kif_file.write(line + "\\n")

        kif_file.write("moves\\n")

        # Instantiate PolicyOutputMapper to use its shogi_move_to_usi method
        mapper = PolicyOutputMapper()

        for i, move_entry in enumerate(game.move_history):
            move_obj = move_entry.get("move")
            if not move_obj:
                continue

            kif_file.write(f"{i+1} {mapper.shogi_move_to_usi(move_obj)}\\n") # Use mapper instance

        # Game termination
        if game.game_over:
            if game.termination_reason == TerminationReason.CHECKMATE.value:
                kif_file.write(
                    "Tsumi\\n" # Checkmate
                )
            elif game.termination_reason == TerminationReason.RESIGNATION.value:
                kif_file.write(
                    "Toryo\\n" # Resignation
                )
            elif game.termination_reason == TerminationReason.MAX_MOVES_EXCEEDED.value:
                kif_file.write(
                    "Jishogi\\n" # Draw by max moves (can be more specific)
                )
            elif game.termination_reason == TerminationReason.REPETITION.value: # Sennichite
                kif_file.write(
                    "Sennichite\\n"
                )
            # Add other termination reasons as needed

        # if hasattr(game, 'end_time') and game.end_time:
        #     kif_file.write(f"*End time: {game.end_time.isoformat()}\\n")
        # if hasattr(game, 'result') and game.result:
        #     kif_file.write(f"*Result: {game.result}\\n")
        # if hasattr(game, 'comment') and game.comment:
        #     kif_file.write(f"*Comment: {game.comment}\\n")
        kif_file.write("*EOF\\n")
