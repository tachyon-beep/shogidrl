# shogi_game_io.py

from typing import TYPE_CHECKING

import numpy as np

import config  # For MAX_MOVES_PER_GAME

from .shogi_core_definitions import (
    OBS_PROMOTED_ORDER,
    OBS_UNPROMOTED_ORDER,
    Color,
    PieceType,
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
    hand_piece_order = PieceType.get_unpromoted_types()

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
    max_moves = float(getattr(config, "MAX_MOVES_PER_GAME", 512))
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
                # Ensure consistent width for alignment
                if len(symbol) == 1:  # P, L, N, S, G, B, R, K (and lowercase)
                    line_pieces.append(" " + symbol)
                else:  # +P, +L, etc. (and lowercase)
                    line_pieces.append(symbol)
            else:
                line_pieces.append(" . ")  # Add spaces around dot for alignment
        lines.append(line_str + "".join(line_pieces))
    # Add file numbers at the bottom
    lines.append(
        "   a  b  c  d  e  f  g  h  i"
    )  # Shogi board file letters, matching original printout spacing
    # Add hands
    black_hand_dict = {pt.name: count for pt, count in game.hands[Color.BLACK.value].items() if count > 0}
    lines.append(f"Black's hand: {black_hand_dict}")
    white_hand_dict = {pt.name: count for pt, count in game.hands[Color.WHITE.value].items() if count > 0}
    lines.append(f"White's hand: {white_hand_dict}")
    lines.append(f"Current player: {game.current_player.name}")
    return "\n".join(lines)
