"""
shogi_game.py: Main ShogiGame class for DRL Shogi Client.
Orchestrates game state and delegates complex logic to helper modules.
"""

# pylint: disable=too-many-lines

import copy  # Added for __deepcopy__
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Create logger for this module
logger = logging.getLogger(__name__)

# Import helper modules
from . import shogi_game_io, shogi_move_execution, shogi_rules_logic

# Import types and fundamental classes from shogi_core_definitions
from .shogi_core_definitions import PIECE_TYPE_TO_HAND_TYPE  # Used in add_to_hand
from .shogi_core_definitions import MoveTuple  # Already imported above
from .shogi_core_definitions import (
    Color,
    Piece,
    PieceType,
    get_unpromoted_types,
)


class ShogiGame:
    """
    Represents the Shogi game state, board, and operations.
    Delegates complex rule logic, I/O, and move execution to helper modules.
    """

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

        self.board[1][1] = Piece(PieceType.ROOK, Color.WHITE)
        self.board[1][7] = Piece(PieceType.BISHOP, Color.WHITE)

        self.board[8][0] = Piece(PieceType.LANCE, Color.BLACK)
        self.board[8][1] = Piece(PieceType.KNIGHT, Color.BLACK)
        self.board[8][2] = Piece(PieceType.SILVER, Color.BLACK)
        self.board[8][3] = Piece(PieceType.GOLD, Color.BLACK)
        self.board[8][4] = Piece(PieceType.KING, Color.BLACK)
        self.board[8][5] = Piece(PieceType.GOLD, Color.BLACK)
        self.board[8][6] = Piece(PieceType.SILVER, Color.BLACK)
        self.board[8][7] = Piece(PieceType.KNIGHT, Color.BLACK)
        self.board[8][8] = Piece(PieceType.LANCE, Color.BLACK)

        self.board[7][1] = Piece(PieceType.BISHOP, Color.BLACK)
        self.board[7][7] = Piece(PieceType.ROOK, Color.BLACK)

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

    def get_state(self) -> np.ndarray:
        """
        Alias for get_observation() for backward compatibility.

        Returns:
            np.ndarray: The observation array.
        """
        return self.get_observation()

    def is_nifu(self, color: Color, col: int) -> bool:
        return shogi_rules_logic.check_for_nifu(self, color, col)

    def is_uchi_fu_zume(self, drop_row: int, drop_col: int, color: Color) -> bool:
        return shogi_rules_logic.check_for_uchi_fu_zume(self, drop_row, drop_col, color)

    def _is_square_attacked(self, row: int, col: int, attacker_color: Color) -> bool:
        return shogi_rules_logic.check_if_square_is_attacked(
            self, row, col, attacker_color
        )

    def get_legal_moves(self) -> List[MoveTuple]:  # Changed from List["MoveTuple"]
        moves = shogi_rules_logic.generate_all_legal_moves(self)
        return moves

    def _king_in_check_after_move(self, player_color: Color) -> bool:
        return shogi_rules_logic.is_king_in_check_after_simulated_move(
            self, player_color
        )

    def find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        """Finds the king of the specified color on the board."""
        # Delegate to the rules logic function
        return shogi_rules_logic.find_king(self, color)

    def get_king_legal_moves(self, color: Color) -> int:
        """Return the number of legal moves available to the king of ``color``."""
        original_player = self.current_player
        self.current_player = color
        try:
            king_pos = self.find_king(color)
            if not king_pos:
                return 0
            legal_moves = self.get_legal_moves()
            count = 0
            for move in legal_moves:
                if move[0] is None or move[1] is None:
                    continue
                piece = self.get_piece(move[0], move[1])
                if piece and piece.type == PieceType.KING:
                    count += 1
            return count
        finally:
            self.current_player = original_player

    def is_in_check(
        self, color: Color, debug_recursion: bool = False
    ) -> bool:  # Added debug_recursion
        """Checks if the specified player is in check."""
        # Delegate to the rules logic function
        return shogi_rules_logic.is_in_check(
            self, color, debug_recursion=debug_recursion
        )  # Pass debug flag

    # --- SFEN methods now delegate to shogi_game_io.py ---

    def sfen_encode_move(
        self, move_tuple: MoveTuple
    ) -> str:  # Changed from "MoveTuple"
        """
        Encodes a move in SFEN (Shogi Forsyth-Edwards Notation) format.
        Board move: (from_r, from_c, to_r, to_c, promote_bool) -> e.g., "7g7f" or "2b3a+"
        Drop move: (None, None, to_r, to_c, piece_type) -> e.g., "P*5e"
        """
        return shogi_game_io.encode_move_to_sfen_string(move_tuple)

    def to_sfen_string(self) -> str:
        """
        Serializes the current game state to an SFEN string.
        Format: <board> <turn> <hands> <move_number>
        Example: lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
        """
        return shogi_game_io.convert_game_to_sfen_string(self)

    def to_sfen(self) -> str:
        """Alias for to_sfen_string() for convenience."""
        return self.to_sfen_string()

    @classmethod
    def from_sfen(
        cls, sfen_str: str, max_moves_for_game_instance: int = 500
    ) -> "ShogiGame":
        """Loads a game state from an SFEN string."""
        # Parse SFEN string components
        board_sfen, turn_sfen, hands_sfen, move_number_sfen = shogi_game_io.parse_sfen_string_components(sfen_str)
        
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

        # Create game instance and initialize
        game = cls(max_moves_per_game=max_moves_for_game_instance)
        game.current_player = current_player
        game.move_count = move_number - 1
        game.board = [[None for _ in range(9)] for _ in range(9)]
        game.hands = {
            Color.BLACK.value: {ptype: 0 for ptype in get_unpromoted_types()},
            Color.WHITE.value: {ptype: 0 for ptype in get_unpromoted_types()},
        }
        game.move_history = []
        game.board_history = []

        # Populate board from SFEN
        shogi_game_io.populate_board_from_sfen_segment(game.board, board_sfen)

        # Populate hands from SFEN
        shogi_game_io.populate_hands_from_sfen_segment(game.hands, hands_sfen)
        
        # Finalize setup
        game._initial_board_setup_done = True
        game.board_history.append(game._board_state_hash())

        # Evaluate termination conditions for the position
        # Similar to what's done in apply_move_to_board but without actually making a move
        # This is needed because from_sfen doesn't call apply_move_to_board
        king_in_check = shogi_rules_logic.is_in_check(game, game.current_player)
        legal_moves = shogi_rules_logic.generate_all_legal_moves(game)

        if not legal_moves:
            if king_in_check:
                game.game_over = True
                # In checkmate (tsumi), the opponent wins
                game.winner = (
                    Color.WHITE if game.current_player == Color.BLACK else Color.BLACK
                )
                game.termination_reason = "Tsumi"
            else:
                game.game_over = True
                game.winner = None  # Stalemate means no winner
                game.termination_reason = "Stalemate"
        elif shogi_rules_logic.check_for_sennichite(game):
            game.game_over = True
            game.winner = None
            game.termination_reason = "Sennichite"

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
    ) -> Union[Dict[str, Any], Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
        """
        Applies a move, updates history, and delegates to shogi_move_execution.
        This is the primary method for making a move.

        Args:
            move_tuple: The move to apply
            is_simulation: If True, this is a simulation move for legal move checking

        Returns:
            If is_simulation is True: move details dictionary
            Otherwise: A 4-tuple (observation, reward, done, info) for RL training
        """
        if self.game_over and not is_simulation:
            # Game is over, return the current state, reward 0, done True, and termination info
            next_obs = self.get_observation()
            reward = 0.0  # No reward for trying to move in a completed game
            done = True
            info = {
                "reason": (
                    self.termination_reason
                    if self.termination_reason
                    else "Game already over"
                )
            }
            return next_obs, reward, done, info

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

            # --- ADDED: Hard-fail for illegal movement pattern ---
            potential_squares = shogi_rules_logic.generate_piece_potential_moves(
                self, piece_to_move, r_from, c_from
            )
            if (r_to, c_to) not in potential_squares:
                raise ValueError(
                    f"Illegal movement pattern: {piece_to_move.type.name} at "
                    f"({r_from},{c_from}) cannot move to ({r_to},{c_to}). "
                    f"Potential squares: {potential_squares}"
                )

            # Set these fields for all board moves (simulation or not) since undo needs them
            move_details_for_history["original_type_before_promotion"] = (
                piece_to_move.type
            )
            move_details_for_history["original_color_of_moved_piece"] = (
                piece_to_move.color
            )

            # Capture detection (needed for undo even in simulation)
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

            # Promotion logic (needed for undo even in simulation)
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

        # --- STEP 1 FIX: Strict legal move validation ---
        # Only allow moves that are in the legal moves list (unless simulation)
        if not is_simulation:
            legal_moves = self.get_legal_moves()
            if move_tuple not in legal_moves:
                raise ValueError(
                    f"Illegal move: {move_tuple} is not in the list of legal moves. "
                    f"Legal moves: {legal_moves}"
                )
            # --- END ADDED ---

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

        # Store who made the move before we switch players
        player_who_made_the_move = self.current_player

        # Call apply_move_to_board to switch player, increment move count, and check game end.
        # Pass the original move_tuple as it might be used by apply_move_to_board for its logic.
        shogi_move_execution.apply_move_to_board(self, is_simulation)

        if is_simulation:
            return move_details_for_history

        # For training, return a 4-tuple (observation, reward, done, info)
        next_obs = self.get_observation()
        reward = self.get_reward(
            player_who_made_the_move
        )  # Get reward from perspective of the player who moved
        done = self.game_over
        info = {"reason": self.termination_reason} if self.termination_reason else {}
        if move_details_for_history.get("captured"):
            captured_piece = move_details_for_history["captured"]
            try:
                info["captured_piece_type"] = captured_piece.type.name
            except AttributeError:
                info["captured_piece_type"] = str(captured_piece)

        return next_obs, reward, done, info

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

    # --- Reward Function ---
    def get_reward(self, player_color: Optional[Color] = None) -> float:
        """Calculates the reward for the player_color based on the game outcome."""
        if not self.game_over:
            return 0.0

        perspective_player = player_color
        if perspective_player is None:
            # If no specific player perspective, use the player whose turn it would have been
            # if the game hadn't ended, or the winner if clear.
            # This logic might need refinement based on how rewards are assigned post-game.
            # For now, let's assume if a winner exists, it's from their perspective.
            # If stalemate, it's neutral.
            if self.winner is not None:
                perspective_player = self.winner  # Win is +1 for winner
            else:  # Stalemate or other draw
                return 0.0

        if self.winner == perspective_player:
            return 1.0  # Win
        if self.winner is not None and self.winner != perspective_player:
            return -1.0  # Loss
        return 0.0  # Draw or game not over from this perspective

    def seed(self, seed_value=None):
        """Seed the game environment for reproducibility.

        While standard Shogi is deterministic, this method provides:
        - A hook for future stochastic variants
        - Debugging support for reproducibility testing
        - Environment interface contract completion

        Args:
            seed_value: Random seed value for reproducibility

        Returns:
            Self for method chaining
        """
        # Store seed value for debugging and future use
        self._seed_value = seed_value

        # Log seeding operation for debugging
        logger.debug(f"Game seeded with value: {seed_value}")

        return self
