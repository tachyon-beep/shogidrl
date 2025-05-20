"""
shogi_game.py: Main ShogiGame class for DRL Shogi Client.
Orchestrates game state and delegates complex logic to helper modules.
"""

import copy  # Added for __deepcopy__
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple  # Added Any

import numpy as np

# Import helper modules
from . import shogi_game_io, shogi_move_execution, shogi_rules_logic

# Import types and fundamental classes from shogi_core_definitions
from .shogi_core_definitions import (
    Color,
    PieceType,
    MoveTuple,  # Already imported above
    BoardMoveTuple,  # Added for clarity if used directly, though MoveTuple covers it
    DropMoveTuple,  # Added for clarity
    Piece,
    get_unpromoted_types,
    PROMOTED_TYPES_SET,  # For SFEN serialization
    PIECE_TYPE_TO_HAND_TYPE,  # Used in add_to_hand
    BASE_TO_PROMOTED_TYPE,  # For SFEN deserialization
    SYMBOL_TO_PIECE_TYPE,  # Added for SFEN parsing
    PROMOTED_TO_BASE_TYPE,  # Potentially useful for SFEN parsing/validation
)

if TYPE_CHECKING:
    # from .shogi_core_definitions import MoveTuple # Already imported above
    pass


class ShogiGame:
    """
    Represents the Shogi game state, board, and operations.
    Delegates complex rule logic, I/O, and move execution to helper modules.
    """

    _SFEN_BOARD_CHARS: Dict[PieceType, str] = {
        PieceType.PAWN: "P",
        PieceType.LANCE: "L",
        PieceType.KNIGHT: "N",
        PieceType.SILVER: "S",
        PieceType.GOLD: "G",
        PieceType.BISHOP: "B",
        PieceType.ROOK: "R",
        PieceType.KING: "K",
        PieceType.PROMOTED_PAWN: "P",  # Note: SFEN uses the base piece char for promoted pieces on board, promotion is indicated by '+'
        PieceType.PROMOTED_LANCE: "L",
        PieceType.PROMOTED_KNIGHT: "N",
        PieceType.PROMOTED_SILVER: "S",
        PieceType.PROMOTED_BISHOP: "B",
        PieceType.PROMOTED_ROOK: "R",
    }

    def __init__(
        self, max_moves_per_game: int = 500
    ) -> None:  # Added default for max_moves_per_game
        self.board: List[List[Optional[Piece]]]
        self.hands: Dict[int, Dict[PieceType, int]]
        self.current_player: Color = Color.BLACK
        self.move_count: int = 0
        self.game_over: bool = False
        self.winner: Optional[Color] = None
        self.termination_reason: Optional[str] = None
        self.move_history: List[Dict[str, Any]] = []
        self.board_history: List[Tuple] = []  # Added board_history
        self._max_moves_this_game = max_moves_per_game
        self._initial_board_setup_done = False
        self.reset()

    @property
    def max_moves_per_game(self) -> int:
        return self._max_moves_this_game

    def _setup_initial_board(self):
        """Sets up the board to the standard initial Shogi position."""
        self.board = [[None for _ in range(9)] for _ in range(9)]

        for i in range(9):
            self.board[2][i] = Piece(PieceType.PAWN, Color.WHITE)
            self.board[6][i] = Piece(PieceType.PAWN, Color.BLACK)

        self.board[0][0] = Piece(PieceType.LANCE, Color.WHITE)
        self.board[0][1] = Piece(PieceType.KNIGHT, Color.WHITE)
        self.board[0][2] = Piece(PieceType.SILVER, Color.WHITE)
        self.board[0][3] = Piece(PieceType.GOLD, Color.WHITE)
        self.board[0][4] = Piece(PieceType.KING, Color.WHITE)
        self.board[0][5] = Piece(PieceType.GOLD, Color.WHITE)
        self.board[0][6] = Piece(PieceType.SILVER, Color.WHITE)
        self.board[0][7] = Piece(PieceType.KNIGHT, Color.WHITE)
        self.board[0][8] = Piece(PieceType.LANCE, Color.WHITE)

        self.board[1][1] = Piece(PieceType.BISHOP, Color.WHITE)
        self.board[1][7] = Piece(PieceType.ROOK, Color.WHITE)

        self.board[8][0] = Piece(PieceType.LANCE, Color.BLACK)
        self.board[8][1] = Piece(PieceType.KNIGHT, Color.BLACK)
        self.board[8][2] = Piece(PieceType.SILVER, Color.BLACK)
        self.board[8][3] = Piece(PieceType.GOLD, Color.BLACK)
        self.board[8][4] = Piece(PieceType.KING, Color.BLACK)
        self.board[8][5] = Piece(PieceType.GOLD, Color.BLACK)
        self.board[8][6] = Piece(PieceType.SILVER, Color.BLACK)
        self.board[8][7] = Piece(PieceType.KNIGHT, Color.BLACK)
        self.board[8][8] = Piece(PieceType.LANCE, Color.BLACK)

        self.board[7][1] = Piece(PieceType.ROOK, Color.BLACK)
        self.board[7][7] = Piece(PieceType.BISHOP, Color.BLACK)

    def reset(self) -> np.ndarray:  # MODIFIED: Return np.ndarray
        """Resets the game to the initial state and returns the observation."""  # MODIFIED: Docstring
        self._setup_initial_board()
        self.hands = {
            Color.BLACK.value: {ptype: 0 for ptype in get_unpromoted_types()},
            Color.WHITE.value: {ptype: 0 for ptype in get_unpromoted_types()},
        }
        self.current_player = Color.BLACK
        self.move_count = 0
        self.game_over = False
        self.winner = None
        self.termination_reason = None
        self.move_history = []
        self.board_history = [
            self._board_state_hash()
        ]  # Initialize with starting position hash
        self._initial_board_setup_done = True
        return self.get_observation() # MODIFIED: Return observation

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """Returns the piece at the specified position, or None if empty or out of bounds."""
        if self.is_on_board(row, col):
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        """Sets or removes a piece at the specified position on the board."""
        if self.is_on_board(row, col):
            self.board[row][col] = piece

    def to_string(self) -> str:
        """Returns a string representation of the current board state."""
        return shogi_game_io.convert_game_to_text_representation(self)

    def is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < 9 and 0 <= col < 9

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.board = copy.deepcopy(self.board, memo)
        result.hands = copy.deepcopy(self.hands, memo)
        result.current_player = self.current_player
        result.move_count = self.move_count
        result.game_over = self.game_over
        result.winner = self.winner
        result.termination_reason = self.termination_reason
        result.move_history = []
        result._max_moves_this_game = self._max_moves_this_game
        result._initial_board_setup_done = self._initial_board_setup_done
        # Ensure all attributes are set on result before calling _board_state_hash
        result.board_history = [result._board_state_hash()]

        return result

    def _is_sliding_piece_type(self, piece_type: PieceType) -> bool:
        return shogi_rules_logic.is_piece_type_sliding(piece_type)

    def get_individual_piece_moves(
        self, piece: Piece, r_from: int, c_from: int
    ) -> list[tuple[int, int]]:
        return shogi_rules_logic.generate_piece_potential_moves(
            self, piece, r_from, c_from
        )

    def get_observation(self) -> np.ndarray:
        """
        Generates the neural network observation for the current game state.

        The observation is a multi-channel NumPy array representing the board,
        hands, and other game metadata from the current player's perspective.
        For detailed structure, see `shogi_game_io.generate_neural_network_observation`.

        Returns:
            np.ndarray: The observation array.
        """
        return shogi_game_io.generate_neural_network_observation(self)

    def is_nifu(self, color: Color, col: int) -> bool:
        return shogi_rules_logic.check_for_nifu(self, color, col)

    def is_uchi_fu_zume(self, drop_row: int, drop_col: int, color: Color) -> bool:
        return shogi_rules_logic.check_for_uchi_fu_zume(self, drop_row, drop_col, color)

    def _is_square_attacked(self, row: int, col: int, attacker_color: Color) -> bool:
        return shogi_rules_logic.check_if_square_is_attacked(
            self, row, col, attacker_color
        )

    def get_legal_moves(self) -> List["MoveTuple"]:
        return shogi_rules_logic.generate_all_legal_moves(self)

    def _king_in_check_after_move(self, player_color: Color) -> bool:
        return shogi_rules_logic.is_king_in_check_after_simulated_move(
            self, player_color
        )

    def find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        """Finds the king of the specified color on the board."""
        # Delegate to the rules logic function
        return shogi_rules_logic.find_king(self, color)

    def is_in_check(self, color: Color, debug_recursion: bool = False) -> bool: # Added debug_recursion
        """Checks if the specified player is in check."""
        # Delegate to the rules logic function
        return shogi_rules_logic.is_in_check(self, color, debug_recursion=debug_recursion) # Pass debug flag

    # --- SFEN Encoding Helper Methods ---
    def _sfen_sq(self, r: int, c: int) -> str:
        """Converts 0-indexed (row, col) to SFEN square string (e.g., (0,0) -> "9a")."""
        if not (0 <= r <= 8 and 0 <= c <= 8):
            raise ValueError(f"Invalid Shogi coordinate for SFEN: row {r}, col {c}")
        file = str(9 - c)
        rank = chr(ord("a") + r)
        return file + rank

    def _get_sfen_drop_char(self, piece_type: PieceType) -> str:
        """Helper to get the uppercase SFEN character for a droppable piece type."""
        if piece_type == PieceType.PAWN:
            return "P"
        if piece_type == PieceType.LANCE:
            return "L"
        if piece_type == PieceType.KNIGHT:
            return "N"
        if piece_type == PieceType.SILVER:
            return "S"
        if piece_type == PieceType.GOLD:
            return "G"
        if piece_type == PieceType.BISHOP:
            return "B"
        if piece_type == PieceType.ROOK:
            return "R"
        raise ValueError(
            f"PieceType {piece_type.name if hasattr(piece_type, 'name') else piece_type} "
            f"is not a standard droppable piece for SFEN notation or is invalid."
        )

    def sfen_encode_move(self, move_tuple: "MoveTuple") -> str:
        """
        Encodes a move in SFEN (Shogi Forsyth-Edwards Notation) format.
        Board move: (from_r, from_c, to_r, to_c, promote_bool) -> e.g., "7g7f" or "2b3a+"
        Drop move: (None, None, to_r, to_c, piece_type) -> e.g., "P*5e"
        """
        if (
            len(move_tuple) == 5
            and isinstance(move_tuple[0], int)
            and isinstance(move_tuple[1], int)
            and isinstance(move_tuple[2], int)
            and isinstance(move_tuple[3], int)
            and isinstance(move_tuple[4], bool)
        ):
            from_r, from_c, to_r, to_c, promote = (
                move_tuple[0],
                move_tuple[1],
                move_tuple[2],
                move_tuple[3],
                move_tuple[4],
            )
            from_sq_str = self._sfen_sq(from_r, from_c)
            to_sq_str = self._sfen_sq(to_r, to_c)
            promo_char = "+" if promote else ""
            return f"{from_sq_str}{to_sq_str}{promo_char}"
        elif (
            len(move_tuple) == 5
            and move_tuple[0] is None
            and move_tuple[1] is None
            and isinstance(move_tuple[2], int)
            and isinstance(move_tuple[3], int)
            and isinstance(move_tuple[4], PieceType)
        ):
            _, _, to_r, to_c, piece_to_drop = (
                move_tuple[0],
                move_tuple[1],
                move_tuple[2],
                move_tuple[3],
                move_tuple[4],
            )
            piece_char = self._get_sfen_drop_char(piece_to_drop)
            to_sq_str = self._sfen_sq(to_r, to_c)
            return f"{piece_char}*{to_sq_str}"
        else:
            element_types = [type(el).__name__ for el in move_tuple]
            element_values = [str(el) for el in move_tuple]
            raise ValueError(
                f"Invalid MoveTuple format for SFEN conversion: {move_tuple}. "
                f"Types: {element_types}. Values: {element_values}."
            )

    # --- SFEN Game State Serialization ---
    def _get_sfen_board_char(self, piece: Piece) -> str:
        """Helper to get the SFEN character for a piece on the board."""
        if not isinstance(piece, Piece):
            raise TypeError(f"Expected a Piece object, got {type(piece)}")

        # Get the base character (e.g., 'P' for PAWN and PROMOTED_PAWN)
        base_char = self._SFEN_BOARD_CHARS.get(piece.type)
        if base_char is None:
            # This handles cases like KING which don't have a separate entry for a promoted type
            # but are in _SFEN_BOARD_CHARS. It mainly catches truly unknown types.
            raise ValueError(
                f"Unknown piece type for SFEN board character: {piece.type}"
            )

        sfen_char = ""
        if piece.type in PROMOTED_TYPES_SET:
            sfen_char += "+"
        sfen_char += base_char

        if piece.color == Color.WHITE:
            return sfen_char.lower()
        return sfen_char

    def to_sfen_string(self) -> str:
        """
        Serializes the current game state to an SFEN string.
        Format: <board> <turn> <hands> <move_number>
        Example: lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
        """
        sfen_ranks = []
        # Board: ranks 1-9 (board[0] to board[8]), files 1-9 (col[8] to col[0])
        for r in range(9):  # board row 0 (SFEN rank 1) to board row 8 (SFEN rank 9)
            empty_squares_count = 0
            sfen_rank_str = ""
            # Iterate from file 9 (board column 0) down to file 1 (board column 8) for SFEN
            # However, our internal board is board[row][col] where col 0 is file 9 (leftmost from Black's view)
            # So we iterate columns from 0 to 8 for our board, which corresponds to SFEN files 9 to 1.
            # SFEN standard is ranks from top (1) to bottom (9), files from left (9) to right (1) from Black's perspective.
            # Our board[r][c] where r=0 is rank 1, c=0 is file 9.
            # So for SFEN, we need to read board[r][c] where c goes from 0 to 8 (file 9 to 1)
            # The example lnsgkgsnl/1r5b1/... means rank 1 is lnsgkgsnl.
            # 'l' is file 9, 'n' is file 8, ..., 'l' is file 1.
            # Our board[0] = [L, N, S, G, K, G, S, N, L] (assuming Piece objects)
            # So the inner loop should be `for c in range(9):` if we build the string left-to-right for SFEN.
            # Let's re-verify SFEN board representation: files are 9->1 (left to right for Black).
            # Example: 9  8  7  6  5  4  3  2  1 (file)
            #          l  n  s  g  k  g  s  n  l (rank 1)
            # Our board[0] = [L, N, S, G, K, G, S, N, L] (assuming Piece objects)
            # So, board[0][0] is file 9, board[0][8] is file 1.
            # The loop `for c in range(8, -1, -1)` means c goes 8, 7, ..., 0.
            # This means board[r][8] (file 1), then board[r][7] (file 2), ... board[r][0] (file 9).
            # This order is correct for constructing the SFEN rank string.

            for c in range(9):  # Iterate through columns 0 to 8 (SFEN files 9 to 1)
                piece = self.board[r][c]
                if piece:
                    if empty_squares_count > 0:
                        sfen_rank_str += str(empty_squares_count)
                        empty_squares_count = 0
                    sfen_rank_str += self._get_sfen_board_char(piece)
                else:
                    empty_squares_count += 1

            if empty_squares_count > 0: # Append any trailing empty count for the rank
                sfen_rank_str += str(empty_squares_count)
            sfen_ranks.append(sfen_rank_str)
        board_sfen = "/".join(sfen_ranks)

        # Player turn
        turn_sfen = "b" if self.current_player == Color.BLACK else "w"

        # Hands
        # Standard SFEN hand piece order: R, B, G, S, N, L, P (but any order is fine as long as counts are correct)
        # For consistency, let's use a defined order.
        # Uppercase for Black, lowercase for White.
        # If a player has multiple pieces of the same type, the number precedes the piece character (e.g., 2P).
        # If no pieces in hand, it's "-".
        SFEN_HAND_PIECE_ORDER = [
            PieceType.ROOK,
            PieceType.BISHOP,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
            PieceType.PAWN,
        ]
        hand_sfen_parts = []

        # Black's hand (uppercase)
        black_hand = self.hands[Color.BLACK.value]
        for piece_type in SFEN_HAND_PIECE_ORDER:
            count = black_hand.get(piece_type, 0)
            if count > 0:
                sfen_char = self._SFEN_BOARD_CHARS[piece_type]  # e.g., "P", "L"
                if count > 1:
                    hand_sfen_parts.append(str(count))
                hand_sfen_parts.append(sfen_char.upper())  # Ensure uppercase for Black

        # White's hand (lowercase)
        white_hand = self.hands[Color.WHITE.value]
        for piece_type in SFEN_HAND_PIECE_ORDER:
            count = white_hand.get(piece_type, 0)
            if count > 0:
                sfen_char_upper = self._SFEN_BOARD_CHARS[piece_type]
                sfen_char = sfen_char_upper.lower()  # Ensure lowercase for White
                if count > 1:
                    hand_sfen_parts.append(str(count))
                hand_sfen_parts.append(sfen_char)

        hands_sfen = "".join(hand_sfen_parts) if hand_sfen_parts else "-"

        # Move number (1-indexed)
        # self.move_count is 0 for the first move to be made, so SFEN move number is move_count + 1
        move_num_sfen = str(self.move_count + 1)

        return f"{board_sfen} {turn_sfen} {hands_sfen} {move_num_sfen}"

    # --- SFEN Game State Deserialization ---

    @staticmethod
    def _parse_sfen_board_piece(
        sfen_char_or_plus: str, next_char_if_plus: Optional[str] = None
    ) -> Tuple[PieceType, Color, bool]:
        """
        Parses an SFEN board piece character (e.g., 'P', 'l', '+R')
        into (PieceType, Color, is_promoted_flag).
        `next_char_if_plus` is used to check for promotion if `sfen_char_or_plus` is '+'.
        Returns (PieceType, Color, was_explicitly_promoted_char_present)
        """
        promoted_prefix_present = False
        actual_char_to_lookup = sfen_char_or_plus

        if sfen_char_or_plus == "+":
            if next_char_if_plus is None:
                raise ValueError(
                    "SFEN board: '+' must be followed by a piece character."
                )
            promoted_prefix_present = True
            actual_char_to_lookup = next_char_if_plus

        color = Color.BLACK if "A" <= actual_char_to_lookup <= "Z" else Color.WHITE

        # Find the PieceType from _SFEN_BOARD_CHARS using the uppercase version of the char
        # This map contains base piece chars: e.g., "P" for PAWN, "L" for LANCE.
        # It also maps PROMOTED_PAWN to "P", PROMOTED_LANCE to "L" etc.
        # We need to find the *base* unpromoted PieceType first if not promoted_prefix_present,
        # or the promoted type if promoted_prefix_present.

        target_char_upper = actual_char_to_lookup.upper()
        found_piece_type = None

        # Prioritize finding the exact match (promoted or unpromoted) based on prefix
        if promoted_prefix_present:
            # We are looking for a PieceType that IS a promoted type and matches target_char_upper via _SFEN_BOARD_CHARS
            # e.g., if we have +P, target_char_upper is P. We need PROMOTED_PAWN.
            for pt, char_val in ShogiGame._SFEN_BOARD_CHARS.items():
                if char_val == target_char_upper and pt in PROMOTED_TYPES_SET:
                    found_piece_type = pt
                    break
            if found_piece_type is None:
                # Fallback: if +R, target_char_upper is R. PROMOTED_ROOK maps to R.
                # This can happen if PROMOTED_TYPES_SET isn't perfectly aligned or char is for non-promotable like G, K
                # Check if the base type can be promoted
                base_type_for_char = None
                for pt_b, char_val_b in ShogiGame._SFEN_BOARD_CHARS.items():
                    if (
                        char_val_b == target_char_upper
                        and pt_b not in PROMOTED_TYPES_SET
                    ):
                        base_type_for_char = pt_b
                        break
                if base_type_for_char and base_type_for_char in BASE_TO_PROMOTED_TYPE:
                    found_piece_type = BASE_TO_PROMOTED_TYPE[base_type_for_char]
                else:
                    raise ValueError(
                        f"Invalid promoted SFEN piece: +{target_char_upper}"
                    )
        else:
            # Not promoted_prefix_present. We are looking for an UNPROMOTED PieceType.
            # e.g., if we have P, target_char_upper is P. We need PAWN.
            # e.g., if we have K, target_char_upper is K. We need KING.
            for pt, char_val in ShogiGame._SFEN_BOARD_CHARS.items():
                if char_val == target_char_upper and pt not in PROMOTED_TYPES_SET:
                    found_piece_type = pt
                    break
            if found_piece_type is None:
                # This could happen if only a promoted type maps to this char in _SFEN_BOARD_CHARS
                # e.g. if _SFEN_BOARD_CHARS only had PROMOTED_PAWN: "P" but not PAWN: "P"
                # This would be an issue with _SFEN_BOARD_CHARS definition or an invalid SFEN char.
                raise ValueError(
                    f"Invalid unpromoted SFEN piece character: {target_char_upper}"
                )

        return found_piece_type, color, promoted_prefix_present

    @classmethod
    def from_sfen(
        cls, sfen_str: str, max_moves_for_game_instance: int = 500
    ) -> "ShogiGame":
        """Loads a game state from an SFEN string."""
        parts = sfen_str.strip().split()
        if len(parts) != 4:
            raise ValueError(
                f"Invalid SFEN string: Expected 4 parts, got {len(parts)}. SFEN: '{sfen_str}'"
            )

        board_sfen, turn_sfen, hands_sfen, move_num_sfen = parts

        game = cls(max_moves_per_game=max_moves_for_game_instance)  # Pass max_moves
        # game.reset() is called by __init__, so board/hands are already initialized.
        # We will overwrite them based on SFEN.

        # 1. Parse Board
        game.board = [
            [None for _ in range(9)] for _ in range(9)
        ]  # Clear board for SFEN state
        sfen_ranks = board_sfen.split("/")
        if len(sfen_ranks) != 9:
            raise ValueError(
                f"Invalid SFEN board: Expected 9 ranks, got {len(sfen_ranks)}."
            )

        for r, rank_str in enumerate(sfen_ranks):
            c = 0  # Current column on our 0-8 board
            rank_idx = 0
            current_sfen_char_in_rank = ""  # For error reporting if rank is malformed
            while rank_idx < len(rank_str) and c < 9:
                current_sfen_char_in_rank = rank_str[rank_idx]
                if "1" <= current_sfen_char_in_rank <= "9":
                    c += int(current_sfen_char_in_rank)
                    rank_idx += 1
                else:
                    next_char_for_plus = (
                        rank_str[rank_idx + 1]
                        if current_sfen_char_in_rank == "+"
                        and rank_idx + 1 < len(rank_str)
                        else None
                    )
                    piece_type, color, _ = cls._parse_sfen_board_piece(
                        current_sfen_char_in_rank, next_char_for_plus
                    )
                    game.board[r][c] = Piece(piece_type, color)
                    if current_sfen_char_in_rank == "+":
                        rank_idx += 2  # Consumed '+' and the piece char
                    else:
                        rank_idx += 1  # Consumed the piece char
                    c += 1
            # After iterating through rank_str, check if columns filled is exactly 9
            # unless the last char was a number that completed the rank implicitly
            if c != 9:
                # If c > 9, it means a number overfilled the rank. This is an error unless it was the last char.
                # If c < 9, it means the rank_str did not describe enough pieces/empty squares.
                is_last_char_a_digit = "1" <= current_sfen_char_in_rank <= "9"
                if not (is_last_char_a_digit and c > 9 and rank_idx == len(rank_str)):
                    raise ValueError(
                        f"Invalid SFEN rank {r+1}: '{rank_str}'. Columns described: {c}, expected 9."
                    )

        # 2. Parse Turn
        if turn_sfen == "b":
            game.current_player = Color.BLACK
        elif turn_sfen == "w":
            game.current_player = Color.WHITE
        else:
            raise ValueError(f"Invalid SFEN turn indicator: {turn_sfen}")

        # 3. Parse Hands - game.hands is already initialized by reset() called in __init__
        # to the correct structure: Dict[int, Dict[PieceType, int]]
        # Clear existing counts before populating from SFEN
        for color_val in game.hands:
            for p_type in game.hands[color_val]:
                game.hands[color_val][p_type] = 0

        if hands_sfen != "-":
            count_str = ""
            for char_hand in hands_sfen:
                if "0" <= char_hand <= "9":  # Allow multi-digit counts e.g. 11P
                    count_str += char_hand
                else:
                    num_pieces = int(count_str) if count_str else 1
                    count_str = ""

                    hand_piece_color = (
                        Color.BLACK if "A" <= char_hand <= "Z" else Color.WHITE
                    )
                    hand_piece_char_upper = char_hand.upper()

                    hand_piece_type = None
                    # In SFEN hands, pieces are always their base types (P, L, N, S, G, B, R)
                    # SYMBOL_TO_PIECE_TYPE maps these uppercase chars to their PieceType (e.g., "P" -> PieceType.PAWN)
                    if hand_piece_char_upper in SYMBOL_TO_PIECE_TYPE:
                        pt_candidate = SYMBOL_TO_PIECE_TYPE[hand_piece_char_upper]
                        # Ensure it's a type that can be in hand (unpromoted, not King)
                        if (
                            pt_candidate in get_unpromoted_types()
                            and pt_candidate != PieceType.KING
                        ):
                            hand_piece_type = pt_candidate

                    if hand_piece_type is None:
                        raise ValueError(
                            f"Invalid piece character '{char_hand}' in SFEN hands: {hands_sfen}"
                        )

                    game.hands[hand_piece_color.value][hand_piece_type] += num_pieces

        # 4. Parse Move Number
        try:
            parsed_move_num = int(move_num_sfen)
            if parsed_move_num <= 0:
                raise ValueError("SFEN move number must be positive.")
            game.move_count = (
                parsed_move_num - 1
            )  # SFEN is 1-indexed, game.move_count is 0-indexed
        except ValueError as e:
            raise ValueError(
                f"Invalid SFEN move number: {move_num_sfen}. Error: {e}"
            ) from e

        game.game_over = False
        game.termination_reason = None
        game.move_history = []
        game.board_history = [
            game._board_state_hash()
        ]  # Reset history with current SFEN position

        return game

    def _board_state_hash(self) -> tuple:
        """
        Returns a hashable representation of the current board state, hands, and player to move.
        Used for checking for repetition (sennichite).
        """
        board_tuple = tuple(
            tuple((p.type.value, p.color.value) if p else None for p in row)
            for row in self.board
        )
        hands_tuple = (
            tuple(
                sorted(
                    (pt.value, count)
                    for pt, count in self.hands[Color.BLACK.value].items()
                    if count > 0
                )
            ),
            tuple(
                sorted(
                    (pt.value, count)
                    for pt, count in self.hands[Color.WHITE.value].items()
                    if count > 0
                )
            ),
        )
        return (board_tuple, hands_tuple, self.current_player.value)

    def get_board_state_hash(self) -> tuple:
        """
        Public interface to access the board state hash.
        """
        return self._board_state_hash()

    def is_sennichite(self) -> bool:
        """
        Checks if the current position has occurred four times, resulting in a draw.
        """
        return shogi_rules_logic.check_for_sennichite(self)

    def make_move(self, move_tuple: MoveTuple, is_simulation: bool = False) -> None:
        """
        Applies a move, updates history, and delegates to shogi_move_execution.
        This is the primary method for making a move.
        """
        if self.game_over and not is_simulation:
            # print(\\"Game is over. No more moves allowed.\\")
            return

        player_who_made_the_move = self.current_player # Define this early

        # Validate move_tuple structure early
        if not (
            isinstance(move_tuple, tuple)
            and len(move_tuple) == 5
            and (
                (  # Board move
                    isinstance(move_tuple[0], int)
                    and isinstance(move_tuple[1], int)
                    and isinstance(move_tuple[2], int)
                    and isinstance(move_tuple[3], int)
                    and isinstance(move_tuple[4], bool)
                )
                or (  # Drop move
                    move_tuple[0] is None
                    and move_tuple[1] is None
                    and isinstance(move_tuple[2], int)
                    and isinstance(move_tuple[3], int)
                    and isinstance(move_tuple[4], PieceType)
                )
            )
        ):
            raise ValueError(f"Invalid move_tuple format: {move_tuple}")

        r_from, c_from, r_to, c_to = (
            move_tuple[0],
            move_tuple[1],
            move_tuple[2],
            move_tuple[3],
        )

        move_details_for_history: Dict[str, Any] = {
            "move": move_tuple,
            "is_drop": False,  # Default to False, override for drops
            "captured": None,
            "was_promoted_in_move": False,
            "original_type_before_promotion": None, # For board moves
            "dropped_piece_type": None, # For drop moves
            "original_color": None, # For board moves, to aid undo
        }

        # --- Part 1: Gather details for history & perform initial piece manipulation ---
        # This part happens *before* calling apply_move_to_board,
        # so we have the state *before* the piece is moved/dropped.

        if r_from is None:  # Drop move
            move_details_for_history["is_drop"] = True
            piece_type_to_drop: PieceType = move_tuple[4] # Already validated as PieceType
            move_details_for_history["dropped_piece_type"] = piece_type_to_drop
            # Actual board/hand update will be in apply_move_to_board or directly here if preferred
            # For now, let\\'s assume apply_move_to_board handles the set_piece and remove_from_hand for drops
        else:  # Board move
            piece_to_move = self.get_piece(r_from, c_from)
            if piece_to_move is None:
                raise ValueError(
                    f"Invalid move: No piece at source ({r_from},{c_from})"
                )
            if piece_to_move.color != player_who_made_the_move:
                raise ValueError(
                    f"Invalid move: Piece at ({r_from},{c_from}) does not belong to current player."
                )

            move_details_for_history["original_type_before_promotion"] = piece_to_move.type
            move_details_for_history["original_color"] = piece_to_move.color

            target_piece_on_board = self.get_piece(r_to, c_to)
            if target_piece_on_board:
                if target_piece_on_board.color == player_who_made_the_move:
                    raise ValueError(
                        f"Invalid move: Cannot capture own piece at ({r_to},{c_to})"
                    )
                # Store a deepcopy of the captured piece for history
                move_details_for_history["captured"] = copy.deepcopy(target_piece_on_board)

            promote_flag = move_tuple[4]  # This is a bool for board moves
            if promote_flag:
                if not shogi_rules_logic.can_promote_specific_piece(
                    self, piece_to_move, r_from, r_to
                ):
                    raise ValueError("Invalid promotion.")
                move_details_for_history["was_promoted_in_move"] = True
        
        # --- Part 2: Execute the move on the board (directly or via shogi_move_execution) ---
        # For simplicity and to ensure consistency with how apply_move_to_board was designed,
        # let\\'s perform the direct board manipulations here for the forward move,
        # and shogi_move_execution.apply_move_to_board will primarily handle player switching,
        # move count, and game termination checks.

        if move_details_for_history["is_drop"]:
            ptd = move_details_for_history["dropped_piece_type"]
            self.set_piece(r_to, c_to, Piece(ptd, player_who_made_the_move))
            self.remove_from_hand(ptd, player_who_made_the_move) # Corrected arguments
        else: # Board move
            # piece_to_move is already fetched and validated
            piece_to_move_on_board = self.get_piece(r_from, c_from) # Should be same as piece_to_move
            if piece_to_move_on_board is None: # Should not happen due to prior checks
                raise RuntimeError("Consistency check failed: piece_to_move_on_board is None")

            # Handle capture by adding to hand
            if move_details_for_history["captured"]:
                captured_p: Piece = move_details_for_history["captured"]
                self.add_to_hand(captured_p, player_who_made_the_move) # Corrected call
            
            self.set_piece(r_to, c_to, piece_to_move_on_board) # Move the piece
            self.set_piece(r_from, c_from, None) # Clear original square

            if move_details_for_history["was_promoted_in_move"]:
                piece_at_dest = self.get_piece(r_to, c_to)
                if piece_at_dest: # Should exist
                    piece_at_dest.promote()
                else: # Should not happen
                    raise RuntimeError("Consistency check failed: piece_at_dest is None after move for promotion")

        # --- Part 3: Update history and game state (delegating parts to shogi_move_execution) ---
        # Store state hash *after* the move is made on the board, but *before* player switch.
        # The hash should reflect the board, hands, and the player *who just made the move*.
        current_state_hash = self._board_state_hash() # Corrected call
        move_details_for_history["state_hash"] = current_state_hash

        if not is_simulation:
            self.move_history.append(move_details_for_history)
            # board_history is for sennichite and should store the hash of the state
            # *after* the move, associated with the player who made it.
            self.board_history.append(current_state_hash)

        # Call apply_move_to_board to switch player, increment move count, and check game end.
        # Pass the original move_tuple as it might be used by apply_move_to_board for its logic,
        # though we\\'ve handled direct board changes here.
        shogi_move_execution.apply_move_to_board(self, move_tuple, is_simulation)

    def undo_move(self):
        """
        Reverts the last move made, restoring the previous game state.
        """
        shogi_move_execution.revert_last_applied_move(self)

    def add_to_hand(self, captured_piece: Piece, capturing_player_color: Color) -> None:
        """
        Adds a captured piece (as unpromoted) to the capturing player's hand.
        """
        if captured_piece.type == PieceType.KING:
            return

        hand_piece_type = PIECE_TYPE_TO_HAND_TYPE.get(captured_piece.type)
        if hand_piece_type is None:
            raise ValueError(
                f"Invalid piece type {captured_piece.type} to add to hand."
            )

        self.hands[capturing_player_color.value][hand_piece_type] = (
            self.hands[capturing_player_color.value].get(hand_piece_type, 0) + 1
        )

    def remove_from_hand(self, piece_type: PieceType, color: Color) -> bool: # Corrected signature
        """Removes one piece of piece_type from the specified color's hand."""
        if piece_type not in get_unpromoted_types():
            # Attempting to remove a promoted type from hand, which is invalid.
            # Or, piece_type is KING, which cannot be in hand.
            # print(f\"Warning: Attempted to remove invalid piece type '{piece_type}' from hand.\")
            return False # Or raise error

        hand_to_modify = self.hands[color.value]
        if hand_to_modify.get(piece_type, 0) > 0:
            hand_to_modify[piece_type] -= 1
            return True
        # print(f\"Warning: Attempted to remove {piece_type} from {color}\'s hand, but not available.\")
        return False

    def get_pieces_in_hand(self, color: Color) -> Dict[PieceType, int]:
        """
        Returns a copy of the pieces in hand for the specified player.
        """
        return self.hands[color.value].copy()

    def is_in_promotion_zone(self, row: int, color: Color) -> bool:
        """
        Checks if the specified row is in the promotion zone for the given color.
        """
        if color == Color.BLACK:
            return 0 <= row <= 2
        return 6 <= row <= 8

    def can_drop_piece(
        self, piece_type: PieceType, row: int, col: int, color: Color
    ) -> bool:
        """
        Checks if a piece of the specified type can be legally dropped at the given position.
        """
        return shogi_rules_logic.can_drop_specific_piece(
            self, piece_type, row, col, color
        )

    def can_promote_piece(self, piece: Piece, r_from: int, r_to: int) -> bool:
        """
        Checks if a piece can be promoted when moving from one position to another.
        """
        return shogi_rules_logic.can_promote_specific_piece(self, piece, r_from, r_to)

    def must_promote_piece(self, piece: Piece, r_to: int) -> bool:
        """
        Checks if a piece must be promoted when moving to the specified position.
        """
        return shogi_rules_logic.must_promote_specific_piece(piece, r_to)

    def is_checkmate(self) -> bool:
        """
        Checks if the current player is checkmated.
        """
        if not self.is_in_check(self.current_player):
            return False

        legal_moves = self.get_legal_moves()
        return not legal_moves

    def is_stalemate(self) -> bool:
        """
        Checks if the current player is stalemated.
        """
        if self.is_in_check(self.current_player):
            return False

        legal_moves = self.get_legal_moves()
        return not legal_moves
