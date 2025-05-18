"""
shogi_game.py: Main ShogiGame class for DRL Shogi Client.
Orchestrates game state and delegates complex logic to helper modules.
"""

from typing import Optional, List, Any, Dict, Tuple, Set, Union
import numpy as np
import config  # MAX_MOVES_PER_GAME and other constants (used by externalized get_observation)

# Import types and fundamental classes from shogi_core_definitions
from .shogi_core_definitions import (
    Color,
    PieceType,
    Piece,
    MoveTuple,
    BoardMove,  # Needed for type hints if specific variants are checked
    DropMove,  # Needed for type hints if specific variants are checked
    PIECE_TYPE_TO_HAND_TYPE,  # Used in add_to_hand
    # PROMOTED_TO_BASE_TYPE, # Not directly used by methods remaining in ShogiGame
    # BASE_TO_PROMOTED_TYPE, # Not directly used by methods remaining in ShogiGame
    # PROMOTED_TYPES_SET, # Not directly used by methods remaining in ShogiGame
)

# Import helper modules
from . import shogi_rules_logic
from . import shogi_game_io
from . import shogi_move_execution


class ShogiGame:
    """
    Represents the Shogi game state, board, and operations.
    Delegates complex rule logic, I/O, and move execution to helper modules.
    """

    def __init__(self) -> None:
        self.board: List[List[Optional[Piece]]] = [
            [None for _ in range(9)] for _ in range(9)
        ]
        # Key is PieceType (unpromoted), value is count
        self.hands: List[Dict[PieceType, int]] = [
            {},
            {},
        ]  # [Black's hand, White's hand]
        self.move_count: int = 0
        self.current_player: Color = Color.BLACK
        self.move_history: list = []
        self.game_over: bool = False
        self.winner: Optional[Color] = None
        self.reset()

    def reset(self) -> None:
        """
        Initializes the board to the standard Shogi starting position.
        """
        self.board = [[None for _ in range(9)] for _ in range(9)]
        # Initialize empty hands for both players (using unpromoted PieceType enums)
        self.hands = [
            {pt: 0 for pt in PieceType.get_unpromoted_types()},
            {pt: 0 for pt in PieceType.get_unpromoted_types()},
        ]

        # Piece types for the back rank (Lance, Knight, Silver, Gold, King)
        back_rank_types = [
            PieceType.LANCE,
            PieceType.KNIGHT,
            PieceType.SILVER,
            PieceType.GOLD,
            PieceType.KING,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
        ]

        # White pieces (Gote, player 1, top rows 0-2)
        for c, pt_enum in enumerate(back_rank_types):
            self.board[0][c] = Piece(pt_enum, Color.WHITE)
        self.board[1][1] = Piece(PieceType.ROOK, Color.WHITE)
        self.board[1][7] = Piece(PieceType.BISHOP, Color.WHITE)
        for c in range(9):
            self.board[2][c] = Piece(PieceType.PAWN, Color.WHITE)

        # Black pieces (Sente, player 0, bottom rows 6-8)
        for c in range(9):
            self.board[6][c] = Piece(PieceType.PAWN, Color.BLACK)
        self.board[7][1] = Piece(PieceType.BISHOP, Color.BLACK)
        self.board[7][7] = Piece(PieceType.ROOK, Color.BLACK)
        for c, pt_enum in enumerate(back_rank_types):
            self.board[8][c] = Piece(pt_enum, Color.BLACK)

        # Empty middle
        for r in range(3, 6):
            for c in range(9):
                self.board[r][c] = None

        self.move_count = 0
        self.current_player = Color.BLACK
        self.move_history = []
        self.game_over = False
        self.winner = None

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        if self.is_on_board(row, col):
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        if self.is_on_board(row, col):
            self.board[row][col] = piece

    def to_string(self) -> str:
        return shogi_game_io.convert_game_to_text_representation(self)

    def is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < 9 and 0 <= col < 9

    def _is_sliding_piece_type(self, piece_type: PieceType) -> bool:
        return shogi_rules_logic.is_piece_type_sliding(self, piece_type)

    def get_individual_piece_moves(
        self, piece: Piece, r_from: int, c_from: int
    ) -> list[tuple[int, int]]:
        return shogi_rules_logic.generate_piece_potential_moves(
            self, piece, r_from, c_from
        )

    def get_observation(self) -> np.ndarray:
        return shogi_game_io.generate_neural_network_observation(self)

    def is_nifu(self, color: Color, col: int) -> bool:
        return shogi_rules_logic.check_for_nifu(self, color, col)

    def is_uchi_fu_zume(self, drop_row: int, drop_col: int, color: Color) -> bool:
        return shogi_rules_logic.check_for_uchi_fu_zume(self, drop_row, drop_col, color)

    def _is_square_attacked(self, row: int, col: int, attacker_color: Color) -> bool:
        return shogi_rules_logic.check_if_square_is_attacked(
            self, row, col, attacker_color
        )

    def get_legal_moves(self) -> List[MoveTuple]:
        return shogi_rules_logic.generate_all_legal_moves(self)

    def _king_in_check_after_move(self, player_color: Color) -> bool:
        return shogi_rules_logic.is_king_in_check_after_simulated_move(
            self, player_color
        )

    def _board_state_hash(self) -> tuple:
        """Returns a hashable representation of board, hands, and current player."""
        # This method is kept here as it's fundamental to the game state representation
        # and used by externalized sennichite check and make_move.
        board_tuple = tuple(
            tuple((p.type.value, p.color.value) if p else None for p in row)
            for row in self.board
        )
        hands_tuple = (
            tuple(
                sorted(
                    (pt.value, count)
                    for pt, count in self.hands[Color.BLACK.value].items()
                )
            ),
            tuple(
                sorted(
                    (pt.value, count)
                    for pt, count in self.hands[Color.WHITE.value].items()
                )
            ),
        )
        return (board_tuple, hands_tuple, self.current_player.value)

    def is_sennichite(self) -> bool:
        return shogi_rules_logic.check_for_sennichite(self)

    def make_move(self, move_tuple: MoveTuple):
        shogi_move_execution.apply_move_to_board(self, move_tuple)

    def undo_move(self):
        shogi_move_execution.revert_last_applied_move(self)

    def is_in_check(self, player_color: Color) -> bool:
        """Checks if the specified player_color is currently in check."""
        king_pos = None
        for r_k in range(9):
            for c_k in range(9):
                p = self.get_piece(r_k, c_k)
                if p and p.type == PieceType.KING and p.color == player_color:
                    king_pos = (r_k, c_k)
                    break
            if king_pos:
                break
        if not king_pos:
            # No king found is a critical error, or means game might have ended.
            # Depending on exact rules for this state, can be True or raise error.
            # Original code implies True for safety.
            return True

        opponent_color = Color.WHITE if player_color == Color.BLACK else Color.BLACK
        # Uses the wrapper for _is_square_attacked
        return self._is_square_attacked(king_pos[0], king_pos[1], opponent_color)

    def sfen_encode_move(self, move_tuple: MoveTuple) -> str:  # TODO: Implement fully
        raise NotImplementedError("sfen_encode_move not yet implemented")

    def add_to_hand(self, captured_piece: Piece, capturing_player_color: Color) -> None:
        """Adds a captured piece (as unpromoted) to the capturing player's hand."""
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

    def remove_from_hand(self, piece_type: PieceType, color: Color) -> bool:
        """Removes a piece from hand. piece_type must be the base unpromoted type."""
        if piece_type not in PieceType.get_unpromoted_types():
            return False

        if self.hands[color.value].get(piece_type, 0) > 0:
            self.hands[color.value][piece_type] -= 1
            return True
        return False

    def get_pieces_in_hand(self, color: Color) -> Dict[PieceType, int]:
        return self.hands[color.value].copy()

    def is_in_promotion_zone(self, row: int, color: Color) -> bool:
        if color == Color.BLACK:  # Moves towards row 0
            return 0 <= row <= 2
        else:  # Color.WHITE, moves towards row 8
            return 6 <= row <= 8

    def can_drop_piece(
        self, piece_type: PieceType, row: int, col: int, color: Color
    ) -> bool:
        return shogi_rules_logic.can_drop_specific_piece(
            self, piece_type, row, col, color
        )

    def can_promote_piece(self, piece: Piece, r_from: int, r_to: int) -> bool:
        return shogi_rules_logic.can_promote_specific_piece(self, piece, r_from, r_to)

    def must_promote_piece(self, piece: Piece, r_to: int) -> bool:
        return shogi_rules_logic.must_promote_specific_piece(self, piece, r_to)
