"""
shogi_game.py: Main ShogiGame class for DRL Shogi Client.
Orchestrates game state and delegates complex logic to helper modules.
"""

import copy  # Added for __deepcopy__
import re  # Added for SFEN parsing
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

# Import helper modules
from . import shogi_game_io, shogi_move_execution, shogi_rules_logic

# Import types and fundamental classes from shogi_core_definitions
from .shogi_core_definitions import BASE_TO_PROMOTED_TYPE  # For SFEN deserialization
from .shogi_core_definitions import PIECE_TYPE_TO_HAND_TYPE  # Used in add_to_hand
from .shogi_core_definitions import PROMOTED_TYPES_SET  # For SFEN serialization
from .shogi_core_definitions import SYMBOL_TO_PIECE_TYPE  # Added for SFEN parsing
from .shogi_core_definitions import MoveTuple  # Already imported above
from .shogi_core_definitions import (
    Color,
    Piece,
    PieceType,
    get_unpromoted_types,
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

        self.board[1][1] = Piece(PieceType.ROOK, Color.WHITE)  # Corrected: Was BISHOP
        self.board[1][7] = Piece(PieceType.BISHOP, Color.WHITE)  # Corrected: Was ROOK

        self.board[8][0] = Piece(PieceType.LANCE, Color.BLACK)
        self.board[8][1] = Piece(PieceType.KNIGHT, Color.BLACK)
        self.board[8][2] = Piece(PieceType.SILVER, Color.BLACK)
        self.board[8][3] = Piece(PieceType.GOLD, Color.BLACK)
        self.board[8][4] = Piece(PieceType.KING, Color.BLACK)
        self.board[8][5] = Piece(PieceType.GOLD, Color.BLACK)
        self.board[8][6] = Piece(PieceType.SILVER, Color.BLACK)
        self.board[8][7] = Piece(PieceType.KNIGHT, Color.BLACK)
        self.board[8][8] = Piece(PieceType.LANCE, Color.BLACK)

        self.board[7][1] = Piece(PieceType.BISHOP, Color.BLACK)  # Corrected: Was ROOK
        self.board[7][7] = Piece(PieceType.ROOK, Color.BLACK)  # Corrected: Was BISHOP

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
        return self.get_observation()  # MODIFIED: Return observation

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

    def __deepcopy__(self, memo: Dict[int, Any]):  # Added type hint for memo
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
    ) -> List[Tuple[int, int]]:  # Changed from list[tuple[int, int]]
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

    def get_legal_moves(self) -> List[MoveTuple]:  # Changed from List["MoveTuple"]
        return shogi_rules_logic.generate_all_legal_moves(self)

    def _king_in_check_after_move(self, player_color: Color) -> bool:
        return shogi_rules_logic.is_king_in_check_after_simulated_move(
            self, player_color
        )

    def find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        """Finds the king of the specified color on the board."""
        # Delegate to the rules logic function
        return shogi_rules_logic.find_king(self, color)

    def is_in_check(
        self, color: Color, debug_recursion: bool = False
    ) -> bool:  # Added debug_recursion
        """Checks if the specified player is in check."""
        # Delegate to the rules logic function
        return shogi_rules_logic.is_in_check(
            self, color, debug_recursion=debug_recursion
        )  # Pass debug flag

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
        # Using a mapping for clarity and directness
        sfen_char_map: Dict[PieceType, str] = {
            PieceType.PAWN: "P",
            PieceType.LANCE: "L",
            PieceType.KNIGHT: "N",
            PieceType.SILVER: "S",
            PieceType.GOLD: "G",
            PieceType.BISHOP: "B",
            PieceType.ROOK: "R",
        }
        char = sfen_char_map.get(piece_type)
        if char is None:
            raise ValueError(
                f"PieceType {piece_type.name if hasattr(piece_type, 'name') else piece_type} "
                f"is not a standard droppable piece for SFEN notation or is invalid."
            )
        return char

    def sfen_encode_move(
        self, move_tuple: MoveTuple
    ) -> str:  # Changed from "MoveTuple"
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
            # Our internal board is board[row][col] where col 0 is file 9 (leftmost from Black's view)
            # So we iterate columns from 0 to 8 for our board, which corresponds to SFEN files 9 to 1.
            for c in range(9):  # Iterate through columns 0 to 8 (SFEN files 9 to 1)
                piece = self.board[r][c]
                if piece:
                    if empty_squares_count > 0:
                        sfen_rank_str += str(empty_squares_count)
                        empty_squares_count = 0
                    sfen_rank_str += self._get_sfen_board_char(piece)
                else:
                    empty_squares_count += 1

            if empty_squares_count > 0:  # Append any trailing empty count for the rank
                sfen_rank_str += str(empty_squares_count)
            sfen_ranks.append(sfen_rank_str)
        board_sfen = "/".join(sfen_ranks)

        # Player turn
        turn_sfen = "b" if self.current_player == Color.BLACK else "w"

        # Hands
        # SFEN standard hand piece order: R, B, G, S, N, L, P.
        # Uppercase for Black, lowercase for White.
        # If a player has multiple pieces of the same type, the number precedes the piece character (e.g., 2P).
        # If no pieces in hand, it's "-".

        # Canonical order for pieces in hand as per many SFEN implementations/expectations.
        # Though the spec might be flexible, tests often expect this order.
        SFEN_HAND_PIECE_CANONICAL_ORDER = [
            PieceType.ROOK,
            PieceType.BISHOP,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
            PieceType.PAWN,
        ]

        hand_sfen_parts = []
        has_black_pieces = False
        # Black's hand (uppercase)
        for piece_type in SFEN_HAND_PIECE_CANONICAL_ORDER:
            count = self.hands[Color.BLACK.value].get(piece_type, 0)
            if count > 0:
                has_black_pieces = True
                sfen_char = self._SFEN_BOARD_CHARS[
                    piece_type
                ]  # Should be uppercase by convention from _SFEN_BOARD_CHARS
                if count > 1:
                    hand_sfen_parts.append(str(count))
                hand_sfen_parts.append(sfen_char)

        has_white_pieces = False
        # White's hand (lowercase)
        for piece_type in SFEN_HAND_PIECE_CANONICAL_ORDER:
            count = self.hands[Color.WHITE.value].get(piece_type, 0)
            if count > 0:
                has_white_pieces = True
                sfen_char_upper = self._SFEN_BOARD_CHARS[piece_type]
                sfen_char = sfen_char_upper.lower()  # Ensure lowercase for White
                if count > 1:
                    hand_sfen_parts.append(str(count))
                hand_sfen_parts.append(sfen_char)

        hands_sfen = (
            "".join(hand_sfen_parts) if (has_black_pieces or has_white_pieces) else "-"
        )

        # Move number (1-indexed)
        # self.move_count is 0 for the first move to be made, so SFEN move number is move_count + 1
        move_num_sfen = str(self.move_count + 1)

        return f"{board_sfen} {turn_sfen} {hands_sfen} {move_num_sfen}"

    # --- SFEN Game State Deserialization ---

    @staticmethod
    def _parse_sfen_board_piece(
        sfen_char_on_board: str, is_promoted_sfen_token: bool
    ) -> Tuple[PieceType, Color]:
        """
        Parses an SFEN board piece character (e.g., 'P', 'l', 'R') and promotion status
        into (PieceType, Color).
        `sfen_char_on_board` is the actual piece letter (e.g., 'P' from "+P", or 'K').
        `is_promoted_sfen_token` is True if a '+' preceded this character in SFEN.
        Returns (PieceType, Color)
        """
        color = Color.BLACK if "A" <= sfen_char_on_board <= "Z" else Color.WHITE

        base_char_upper = sfen_char_on_board.upper()
        base_piece_type = SYMBOL_TO_PIECE_TYPE.get(base_char_upper)

        if base_piece_type is None:
            raise ValueError(
                f"Invalid SFEN piece character for board: {sfen_char_on_board}"
            )

        if is_promoted_sfen_token:
            if base_piece_type in BASE_TO_PROMOTED_TYPE:
                final_piece_type = BASE_TO_PROMOTED_TYPE[base_piece_type]
            elif (
                base_piece_type in PROMOTED_TYPES_SET
            ):  # Already a promoted type, e.g. if SYMBOL_TO_PIECE_TYPE mapped +P directly
                final_piece_type = base_piece_type
            else:
                # '+' was applied to a non-promotable piece like King or Gold
                raise ValueError(
                    f"Invalid promotion: SFEN token '+' applied to non-promotable piece type {base_piece_type.name} (from char '{sfen_char_on_board}')"
                )
        else:  # Not a promoted token
            # If the base_piece_type itself is a promoted type (e.g. if _SFEN_BOARD_CHARS had +P: P and no plain P:P)
            # this would be an issue, but SYMBOL_TO_PIECE_TYPE should map to base types.
            if base_piece_type in PROMOTED_TYPES_SET:
                raise ValueError(
                    f"Invalid SFEN: Character '{sfen_char_on_board}' (mapped to {base_piece_type.name}) implies promotion, but no '+' prefix found."
                )
            final_piece_type = base_piece_type

        return final_piece_type, color

    @classmethod
    def from_sfen(
        cls, sfen_str: str, max_moves_for_game_instance: int = 500
    ) -> "ShogiGame":
        """Loads a game state from an SFEN string."""
        sfen_pattern = re.compile(r"^\s*([^ ]+)\s+([bw])\s+([^ ]+)\s+(\d+)\s*$")
        match = sfen_pattern.match(sfen_str.strip())
        if not match:
            raise ValueError(f"Invalid SFEN string structure: '{sfen_str}'")

        board_sfen, turn_sfen, hands_sfen, move_number_sfen = match.groups()
        current_player = Color.BLACK if turn_sfen == "b" else Color.WHITE

        try:
            move_number = int(move_number_sfen)
            if move_number < 1:
                raise ValueError("SFEN move number must be positive")
        except ValueError as e:
            if "SFEN move number must be positive" in str(e):
                raise
            raise ValueError(
                f"Invalid move number in SFEN: '{move_number_sfen}'"
            ) from e

        game = cls()
        game.current_player = current_player
        game._max_moves_this_game = max_moves_for_game_instance
        game.move_count = move_number - 1
        game.board = [[None for _ in range(9)] for _ in range(9)]
        game.hands = {
            Color.BLACK.value: {ptype: 0 for ptype in get_unpromoted_types()},
            Color.WHITE.value: {ptype: 0 for ptype in get_unpromoted_types()},
        }
        game.move_history = []
        game.board_history = []

        # Parse board
        rows = board_sfen.split("/")
        if len(rows) != 9:
            raise ValueError("Expected 9 ranks")

        for r, row_str in enumerate(rows):
            c = 0
            promoted_flag_active = False
            while c < 9 and row_str:
                char_sfen = row_str[0]
                row_str = row_str[1:]
                if char_sfen == "+":
                    if promoted_flag_active:
                        raise ValueError(
                            "Invalid piece character sequence starting with '+'"
                        )
                    promoted_flag_active = True
                    continue
                elif char_sfen.isdigit():
                    if promoted_flag_active:
                        raise ValueError("Invalid SFEN piece character for board: 0")
                    if char_sfen == "0":
                        raise ValueError("Invalid SFEN piece character for board: 0")
                    empty_squares = int(char_sfen)
                    if not (1 <= empty_squares <= 9 - c):
                        raise ValueError(
                            f"Row {r+1} ('{row_str}') describes {c+empty_squares} columns, expected 9"
                        )
                    c += empty_squares
                else:
                    piece_char_upper = char_sfen.upper()
                    base_piece_type = SYMBOL_TO_PIECE_TYPE.get(piece_char_upper)
                    if base_piece_type is None:
                        raise ValueError(
                            f"Invalid SFEN piece character for board: {char_sfen}"
                        )
                    piece_color = Color.BLACK if char_sfen.isupper() else Color.WHITE
                    final_piece_type = base_piece_type
                    is_actually_promoted = False
                    if promoted_flag_active:
                        if base_piece_type in BASE_TO_PROMOTED_TYPE:
                            final_piece_type = BASE_TO_PROMOTED_TYPE[base_piece_type]
                            is_actually_promoted = True
                        elif base_piece_type in PROMOTED_TYPES_SET:
                            raise ValueError(
                                f"Invalid promotion: SFEN token '+' applied to non-promotable piece type {base_piece_type.name}"
                            )
                        else:
                            raise ValueError(
                                f"Invalid promotion: SFEN token '+' applied to non-promotable piece type {base_piece_type.name}"
                            )
                    elif (
                        not promoted_flag_active
                        and base_piece_type in PROMOTED_TYPES_SET
                    ):
                        raise ValueError(
                            f"Invalid SFEN piece character for board: {char_sfen}"
                        )
                    game.board[r][c] = Piece(final_piece_type, piece_color)
                    if is_actually_promoted:
                        piece_on_board = game.board[r][c]
                        if piece_on_board is not None:
                            piece_on_board.is_promoted = True
                    c += 1
                    promoted_flag_active = False
            if c != 9:
                raise ValueError(
                    f"Row {r+1} ('{rows[r]}') describes {c} columns, expected 9"
                )

        # Parse hands
        if hands_sfen != "-":
            hand_segment_pattern = re.compile(r"(\d*)([PLNSGBRplnsgbr])")
            pos = 0
            while pos < len(hands_sfen):
                match_hand = hand_segment_pattern.match(hands_sfen, pos)
                if not match_hand:
                    if hands_sfen[pos:].startswith("K") or hands_sfen[pos:].startswith(
                        "k"
                    ):
                        raise ValueError(
                            "Invalid piece character 'K' or non-droppable piece type in SFEN hands"
                        )
                    raise ValueError("Invalid character sequence in SFEN hands")
                count_str, piece_char = match_hand.groups()
                count = int(count_str) if count_str else 1
                try:
                    piece_type_in_hand = SYMBOL_TO_PIECE_TYPE[piece_char.upper()]
                    hand_color = Color.BLACK if piece_char.isupper() else Color.WHITE
                except KeyError as e:
                    raise ValueError("Invalid character sequence in SFEN hands") from e
                if piece_type_in_hand == PieceType.KING:
                    raise ValueError(
                        "Invalid piece character 'K' or non-droppable piece type in SFEN hands"
                    )
                if piece_type_in_hand in PROMOTED_TYPES_SET:
                    raise ValueError("Invalid character sequence in SFEN hands")
                current_hand_for_color = game.hands[hand_color.value]
                current_hand_for_color[piece_type_in_hand] = (
                    current_hand_for_color.get(piece_type_in_hand, 0) + count
                )
                pos = match_hand.end()
        game._initial_board_setup_done = True
        game.board_history.append(game._board_state_hash())
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

    def make_move(
        self, move_tuple: MoveTuple, is_simulation: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Applies a move, updates history, and delegates to shogi_move_execution.
        This is the primary method for making a move.
        Returns move details if is_simulation is True, otherwise None.
        """
        if self.game_over and not is_simulation:
            # print("Game is over. No more moves allowed.")
            return None

        player_who_made_the_move = self.current_player
        move_count_before_move = self.move_count

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
                    move_tuple[0] is None  # Allow None for drop move
                    and move_tuple[1] is None  # Allow None for drop move
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
            "original_type_before_promotion": None,  # For board moves
            "dropped_piece_type": None,  # For drop moves
            "original_color_of_moved_piece": None,  # For board moves, to aid undo
            "player_who_made_the_move": player_who_made_the_move,  # RESTORED COMMA HERE
            "move_count_before_move": move_count_before_move,
        }

        # --- Part 1: Gather details for history & perform initial piece manipulation ---
        # This part happens *before* calling apply_move_to_board,
        # so we have the state *before* the piece is moved/dropped.

        if r_from is None:  # Drop move
            move_details_for_history["is_drop"] = True
            if isinstance(move_tuple[4], PieceType):
                drop_piece_type_for_move = move_tuple[4]
                move_details_for_history["dropped_piece_type"] = (
                    drop_piece_type_for_move
                )
                # The actual board update and hand removal will happen in Part 2 for consistency
            else:
                raise ValueError(
                    f"Invalid drop move: move_tuple[4] is not a PieceType: {move_tuple[4]}"
                )
        else:  # Board move
            if (
                r_from is not None and c_from is not None
            ):  # Should always be true for board move
                piece_to_move = self.get_piece(r_from, c_from)
            else:
                # This case should ideally be caught by the tuple validation earlier
                raise ValueError("Invalid board move: r_from or c_from is None")

            if piece_to_move is None:
                raise ValueError(
                    f"Invalid move: No piece at source ({r_from},{c_from})"
                )
            if piece_to_move.color != player_who_made_the_move:
                raise ValueError(
                    f"Invalid move: Piece at ({r_from},{c_from}) does not belong to current player."
                )
            move_details_for_history["original_type_before_promotion"] = (
                piece_to_move.type
            )
            move_details_for_history["original_color_of_moved_piece"] = (
                piece_to_move.color
            )

            if r_to is not None and c_to is not None:  # Should always be true
                target_piece_on_board = self.get_piece(r_to, c_to)
            else:
                # This case should ideally be caught by the tuple validation earlier
                raise ValueError("Invalid board move: r_to or c_to is None")

            if target_piece_on_board:
                if target_piece_on_board.color == player_who_made_the_move:
                    raise ValueError(
                        f"Invalid move: Cannot capture own piece at ({r_to},{c_to})"
                    )
                move_details_for_history["captured"] = copy.deepcopy(
                    target_piece_on_board
                )

            promote_flag = move_tuple[4]
            if (
                isinstance(promote_flag, bool) and promote_flag
            ):  # Ensure promote_flag is bool for board moves
                if not shogi_rules_logic.can_promote_specific_piece(
                    self, piece_to_move, r_from, r_to
                ):
                    raise ValueError("Invalid promotion.")
                move_details_for_history["was_promoted_in_move"] = True
            elif not isinstance(promote_flag, bool):
                raise ValueError(
                    f"Invalid promotion flag type for board move: {type(promote_flag)}"
                )

        # --- Part 2: Execute the move on the board ---
        if move_details_for_history["is_drop"]:
            if isinstance(move_tuple[4], PieceType):
                drop_piece_type = move_tuple[4]
                if (
                    r_to is not None and c_to is not None
                ):  # Should always be true for drop
                    self.set_piece(
                        r_to, c_to, Piece(drop_piece_type, player_who_made_the_move)
                    )
                self.remove_from_hand(drop_piece_type, player_who_made_the_move)
            # Error case for invalid type already handled in Part 1
        else:  # Board move
            # piece_to_move was fetched in Part 1 and validated
            # r_from, c_from, r_to, c_to are validated to be not None for board moves
            if r_from is None or c_from is None:  # Add assertion for type checker
                raise RuntimeError(
                    "r_from and c_from should not be None for a board move at this stage."
                )
            current_piece_to_move = self.get_piece(
                r_from, c_from
            )  # Get it again, as it might be needed for promotion logic
            if current_piece_to_move is None:  # Should not happen due to prior checks
                raise RuntimeError(
                    f"Consistency check failed: piece at ({r_from},{c_from}) disappeared before move execution"
                )

            # Handle capture by adding to hand
            if move_details_for_history["captured"]:
                captured_p: Piece = move_details_for_history["captured"]
                self.add_to_hand(captured_p, player_who_made_the_move)

            # Move the piece
            # Ensure r_to and c_to are not None (already validated by move_tuple structure check)
            if r_to is None or c_to is None:  # Should ideally not be reached
                raise ValueError(
                    "Invalid board move: r_to or c_to is None during piece placement."
                )
            self.set_piece(r_to, c_to, current_piece_to_move)  # type: ignore

            # Clear original square
            # Ensure r_from and c_from are not None (already validated)
            if r_from is None or c_from is None:  # Should ideally not be reached
                raise ValueError(
                    "Invalid board move: r_from or c_from is None during piece removal."
                )
            self.set_piece(r_from, c_from, None)

            # Handle promotion
            if move_details_for_history["was_promoted_in_move"]:
                # r_to, c_to are known to be not None for board moves
                piece_at_dest = self.get_piece(r_to, c_to)
                if piece_at_dest:  # Should exist as we just placed it
                    piece_at_dest.promote()
                else:  # Should not happen
                    raise RuntimeError(
                        "Consistency check failed: piece_at_dest is None after move for promotion"
                    )

        # --- Part 3: Update history and game state (delegating parts to shogi_move_execution) ---
        # Store state hash *after* the move is made on the board, but *before* player switch.
        # The hash should reflect the board, hands, and the player *who just made the move*.
        current_state_hash = self._board_state_hash()
        move_details_for_history["state_hash"] = current_state_hash

        if not is_simulation:
            self.move_history.append(move_details_for_history)
            # board_history is for sennichite and should store the hash of the state
            # *after* the move, associated with the player who made it.
            self.board_history.append(current_state_hash)

        # Call apply_move_to_board to switch player, increment move count, and check game end.
        # Pass the original move_tuple as it might be used by apply_move_to_board for its logic,
        # though we\\'ve handled direct board changes here.
        shogi_move_execution.apply_move_to_board(self, is_simulation)

        if is_simulation:
            return move_details_for_history
        return None

    def undo_move(
        self, simulation_undo_details: Optional[Dict[str, Any]] = None
    ) -> None:  # Added return type hint & param
        """
        Reverts the last move made, restoring the previous game state.
        Can use simulation_undo_details to undo a simulated move not in history.
        """
        shogi_move_execution.revert_last_applied_move(self, simulation_undo_details)

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

    def remove_from_hand(
        self, piece_type: PieceType, color: Color
    ) -> bool:  # Corrected signature
        """Removes one piece of piece_type from the specified color's hand."""
        if piece_type not in get_unpromoted_types():
            # Attempting to remove a promoted type from hand, which is invalid.
            # Or, piece_type is KING, which cannot be in hand.
            # print(f\"Warning: Attempted to remove invalid piece type '{piece_type}' from hand.\")
            return False  # Or raise error

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
        Checks if a piece of the specified type can be legally dropped on the given square.
        Delegates to shogi_rules_logic.can_drop_specific_piece for all rule checks.
        """
        return shogi_rules_logic.can_drop_specific_piece(
            self, piece_type, row, col, color, is_escape_check=False
        )

    def __repr__(self):
        return f"<ShogiGame move_count={self.move_count} current_player={self.current_player}>"
