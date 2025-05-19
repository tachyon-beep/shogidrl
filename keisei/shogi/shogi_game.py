"""
shogi_game.py: Main ShogiGame class for DRL Shogi Client.
Orchestrates game state and delegates complex logic to helper modules.
"""

from typing import Dict, List, Optional, TYPE_CHECKING  # Added TYPE_CHECKING

import numpy as np

# Import helper modules
from . import shogi_game_io, shogi_move_execution, shogi_rules_logic

# Import types and fundamental classes from shogi_core_definitions
from .shogi_core_definitions import PIECE_TYPE_TO_HAND_TYPE  # Used in add_to_hand
from .shogi_core_definitions import Color, Piece, PieceType, get_unpromoted_types # Added get_unpromoted_types

if TYPE_CHECKING:
    from .shogi_core_definitions import MoveTuple


class ShogiGame:
    """
    Represents the Shogi game state, board, and operations.
    Delegates complex rule logic, I/O, and move execution to helper modules.
    """

    def __init__(self, max_moves_per_game: int) -> None:  # Added max_moves_per_game
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
        self.termination_reason: Optional[str] = None  # Added to store game termination reason
        self.max_moves_per_game = max_moves_per_game  # Store max_moves_per_game
        self.reset()

    def reset(self) -> np.ndarray:  # Modified return type
        """
        Initializes the board to the standard Shogi starting position.
        Returns the initial observation.  # Added this line
        """
        self.board = [[None for _ in range(9)] for _ in range(9)]
        # Initialize empty hands for both players (using unpromoted PieceType enums)
        self.hands = [
            {pt: 0 for pt in get_unpromoted_types()},
            {pt: 0 for pt in get_unpromoted_types()},
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
        self.termination_reason = None  # Reset termination reason
        return self.get_observation()  # Added this line

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
        """Checks if the given coordinates are within the 9x9 Shogi board."""
        return 0 <= row < 9 and 0 <= col < 9

    def _is_sliding_piece_type(self, piece_type: PieceType) -> bool:
        """Returns True if the piece type is a sliding piece (Lance, Bishop, Rook or their promoted versions)."""
        return shogi_rules_logic.is_piece_type_sliding(piece_type)

    def get_individual_piece_moves(
        self, piece: Piece, r_from: int, c_from: int
    ) -> list[tuple[int, int]]:
        """Returns a list of valid destination coordinates for the specified piece."""
        return shogi_rules_logic.generate_piece_potential_moves(
            self, piece, r_from, c_from
        )

    def get_observation(self) -> np.ndarray:
        """Returns a neural network-friendly observation representation of the current game state."""
        return shogi_game_io.generate_neural_network_observation(self)

    def is_nifu(self, color: Color, col: int) -> bool:
        """Checks if there already exists a pawn of the specified color on the given file."""
        return shogi_rules_logic.check_for_nifu(self, color, col)

    def is_uchi_fu_zume(self, drop_row: int, drop_col: int, color: Color) -> bool:
        """
        Checks if dropping a pawn at the specified position would result in
        immediate checkmate (uchi-fu-zume), which is illegal in Shogi.
        """
        return shogi_rules_logic.check_for_uchi_fu_zume(self, drop_row, drop_col, color)

    def _is_square_attacked(self, row: int, col: int, attacker_color: Color) -> bool:
        """Returns True if the specified square is attacked by any piece of the attacker's color."""
        return shogi_rules_logic.check_if_square_is_attacked(
            self, row, col, attacker_color
        )

    def get_legal_moves(self) -> List['MoveTuple']:  # Quoted
        """Returns a list of all legal moves for the current player."""
        return shogi_rules_logic.generate_all_legal_moves(self)

    def _king_in_check_after_move(self, player_color: Color) -> bool:
        """
        Checks if the king of the specified color would be in check after a move.
        Used for checking if a move is legal (a player cannot make a move that leaves their king in check).
        """
        return shogi_rules_logic.is_king_in_check_after_simulated_move(
            self, player_color
        )

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
                    if count > 0  # Only include pieces with count > 0
                )
            ),
            tuple(
                sorted(
                    (pt.value, count)
                    for pt, count in self.hands[Color.WHITE.value].items()
                    if count > 0  # Only include pieces with count > 0
                )
            ),
        )
        return (board_tuple, hands_tuple, self.current_player.value)

    def get_board_state_hash(self) -> tuple:
        """
        Public interface to access the board state hash.
        Returns a hashable representation of the current board state, hands, and player to move.
        Used for checking for repetition (sennichite).

        This method is primarily used by move execution functions.
        """
        return self._board_state_hash()

    def is_sennichite(self) -> bool:
        """
        Checks if the current position has occurred four times, resulting in a draw.
        """
        return shogi_rules_logic.check_for_sennichite(self)

    def make_move(
        self, move_tuple: 'MoveTuple', is_simulation: bool = False  # Quoted
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Executes a move on the board and updates game state.
        The move can be either a board move or a drop move.

        Args:
            move_tuple: The move to make.
            is_simulation: True if this move is part of a simulation (e.g., for legal move generation).
                           If True, game-ending checks like checkmate/stalemate might be skipped.

        Returns:
            A tuple (next_observation, reward, done, info_dict).
        """
        # Store current player before it's switched by apply_move_to_board
        player_who_moved = self.current_player

        shogi_move_execution.apply_move_to_board(self, move_tuple, is_simulation)

        next_observation = self.get_observation()
        done = self.game_over
        reward = 0.0
        info = {
            "termination_reason": self.termination_reason,
            "game_length": self.move_count,
        }

        if done and not is_simulation:
            if self.winner is not None:
                if self.winner == player_who_moved:  # Player who made the move won
                    reward = 1.0
                elif self.winner != player_who_moved and self.winner is not None:  # Player who made the move lost
                    reward = -1.0
                # if winner is None, it's a draw, reward remains 0.0
            else:  # Draw (e.g. sennichite, max moves if not handled by winner)
                reward = 0.0

        # The self.termination_reason should be set within shogi_move_execution.apply_move_to_board
        # or by specific game ending condition checks called from there or before.
        # For example, if apply_move_to_board sets self.game_over = True and self.winner,
        # it should also set self.termination_reason.

        return next_observation, reward, done, info

    def undo_move(self):
        """
        Reverts the last move made, restoring the previous game state.
        """
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
            # No king found is a critical error, indicates an impossible game state.
            raise RuntimeError(
                f"King of color {player_color} not found on the board. This indicates an illegal game state."
            )

        opponent_color = Color.WHITE if player_color == Color.BLACK else Color.BLACK
        # Uses the wrapper for _is_square_attacked
        return self._is_square_attacked(king_pos[0], king_pos[1], opponent_color)

    def sfen_encode_move(self, move_tuple: 'MoveTuple') -> str:  # Quoted
        """
        Encodes a move in SFEN (Shogi Forsyth-Edwards Notation) format.
        This is a standard notation for recording Shogi positions and moves.
        """
        # Implementation to be completed in the future
        raise NotImplementedError("sfen_encode_move not yet implemented")

    def add_to_hand(self, captured_piece: Piece, capturing_player_color: Color) -> None:
        """
        Adds a captured piece (as unpromoted) to the capturing player's hand.

        Args:
            captured_piece: The piece that was captured
            capturing_player_color: The color of the player who captured the piece
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

    def remove_from_hand(self, piece_type: PieceType, color: Color) -> bool:
        """
        Removes a piece from a player's hand.

        Args:
            piece_type: The type of piece to remove (must be an unpromoted type)
            color: The color of the player whose hand to modify

        Returns:
            bool: True if the piece was successfully removed, False otherwise
        """
        if piece_type not in get_unpromoted_types(): # Use imported function
            return False

        if self.hands[color.value].get(piece_type, 0) > 0:
            self.hands[color.value][piece_type] -= 1
            return True
        return False

    def get_pieces_in_hand(self, color: Color) -> Dict[PieceType, int]:
        """
        Returns a copy of the pieces in hand for the specified player.

        Args:
            color: The color of the player

        Returns:
            A dictionary mapping piece types to counts
        """
        return self.hands[color.value].copy()

    def is_in_promotion_zone(self, row: int, color: Color) -> bool:
        """
        Checks if the specified row is in the promotion zone for the given color.

        Args:
            row: The row index to check
            color: The color to check the promotion zone for

        Returns:
            bool: True if the row is in the promotion zone, False otherwise
        """
        if color == Color.BLACK:  # Moves towards row 0
            return 0 <= row <= 2
        # Color.WHITE, moves towards row 8
        return 6 <= row <= 8

    def can_drop_piece(
        self, piece_type: PieceType, row: int, col: int, color: Color
    ) -> bool:
        """
        Checks if a piece of the specified type can be legally dropped at the given position.

        Args:
            piece_type: The type of piece to drop (must be unpromoted)
            row: The target row for the drop
            col: The target column for the drop
            color: The color of the player making the drop

        Returns:
            bool: True if the drop is legal, False otherwise
        """
        return shogi_rules_logic.can_drop_specific_piece(
            self, piece_type, row, col, color
        )

    def can_promote_piece(self, piece: Piece, r_from: int, r_to: int) -> bool:
        """
        Checks if a piece can be promoted when moving from one position to another.

        Args:
            piece: The piece to check for promotion eligibility
            r_from: The starting row of the piece
            r_to: The destination row of the piece

        Returns:
            bool: True if the piece can be promoted, False otherwise
        """
        return shogi_rules_logic.can_promote_specific_piece(self, piece, r_from, r_to)

    def must_promote_piece(self, piece: Piece, r_to: int) -> bool:
        """
        Checks if a piece must be promoted when moving to the specified position.
        This happens for pawns and lances on the last rank, and knights on the last two ranks.

        Args:
            piece: The piece to check for mandatory promotion
            r_to: The destination row of the piece

        Returns:
            bool: True if the piece must be promoted, False otherwise
        """
        return shogi_rules_logic.must_promote_specific_piece(piece, r_to)

    def is_checkmate(self) -> bool:
        """
        Checks if the current player is checkmated.
        A player is checkmated if they are in check and have no legal moves.
        """
        if not self.is_in_check(self.current_player):
            return False  # Not in check, so not checkmate

        legal_moves = self.get_legal_moves()
        return not legal_moves  # Checkmate if in check and no legal moves

    def is_stalemate(self) -> bool:
        """
        Checks if the current player is stalemated.
        A player is stalemated if they are not in check but have no legal moves.
        """
        if self.is_in_check(self.current_player):
            return False  # In check, so not stalemate (could be checkmate)

        legal_moves = self.get_legal_moves()
        return not legal_moves  # Stalemate if not in check and no legal moves
