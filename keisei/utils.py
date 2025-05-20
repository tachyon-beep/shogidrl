from __future__ import annotations

"""
utils.py: Contains PolicyOutputMapper and TrainingLogger.
"""
import datetime
import sys
from typing import TYPE_CHECKING, Dict, List, TextIO

import torch

# Ensure these imports are correct based on your project structure
from keisei.shogi.shogi_core_definitions import (
    get_unpromoted_types,
)  # Import the standalone function
from keisei.shogi.shogi_core_definitions import BoardMoveTuple, DropMoveTuple, PieceType

if TYPE_CHECKING:
    from keisei.shogi.shogi_core_definitions import MoveTuple


class PolicyOutputMapper:
    """Maps Shogi moves to/from policy network output indices."""

    def __init__(self) -> None:
        """Initializes the PolicyOutputMapper by generating all possible move representations."""
        self.idx_to_move: List["MoveTuple"] = []
        self.move_to_idx: Dict["MoveTuple", int] = {}
        current_idx = 0

        # Generate Board Moves: (from_r, from_c, to_r, to_c, promote_flag: bool)
        for r_from in range(9):
            for c_from in range(9):
                for r_to in range(9):
                    for c_to in range(9):
                        if r_from == r_to and c_from == c_to:
                            continue  # Skip null moves

                        # Move without promotion
                        move_no_promo: BoardMoveTuple = (
                            r_from,
                            c_from,
                            r_to,
                            c_to,
                            False,
                        )
                        self.idx_to_move.append(move_no_promo)
                        self.move_to_idx[move_no_promo] = current_idx
                        current_idx += 1

                        # Move with promotion
                        move_promo: BoardMoveTuple = (r_from, c_from, r_to, c_to, True)
                        self.idx_to_move.append(move_promo)
                        self.move_to_idx[move_promo] = current_idx
                        current_idx += 1

        # Define droppable piece types (standard 7, excluding King)
        droppable_piece_types: List[PieceType] = (
            get_unpromoted_types()
        )  # Use the imported function

        # Generate Drop Moves: (None, None, to_r, to_c, piece_type_to_drop: PieceType)
        for r_to in range(9):
            for c_to in range(9):
                for piece_type in droppable_piece_types:
                    drop_move: DropMoveTuple = (None, None, r_to, c_to, piece_type)
                    self.idx_to_move.append(drop_move)
                    self.move_to_idx[drop_move] = current_idx
                    current_idx += 1

        self.total_actions = current_idx

    def get_total_actions(self) -> int:
        """Return the total number of possible actions."""
        return self.total_actions

    def shogi_move_to_policy_index(self, move: "MoveTuple") -> int:
        """Convert a Shogi MoveTuple to its policy index."""
        idx = self.move_to_idx.get(move)
        if idx is None:
            # Attempt to handle cases where the exact tuple might not match due to PieceType enum identity
            # This is a fallback, ideally the move objects should match directly.
            if len(move) == 5 and move[0] is None:  # Potential DropMoveTuple
                # Reconstruct DropMoveTuple with PieceType from self.idx_to_move if a similar one exists
                # This is a bit of a heuristic and might need refinement
                for stored_move in self.move_to_idx.keys():
                    if (
                        len(stored_move) == 5
                        and stored_move[0] is None
                        and stored_move[2] == move[2]
                        and stored_move[3] == move[3]
                        and stored_move[4].value == move[4].value
                    ):  # Compare PieceType by value
                        idx = self.move_to_idx.get(stored_move)
                        break
            if idx is None:  # If still not found after heuristic
                raise ValueError(
                    f"Move {move} (type: {type(move)}, element types: {[type(el) for el in move]}) "
                    f"not found in PolicyOutputMapper's known moves. Known keys example: {list(self.move_to_idx.keys())[0]}"
                )
        return idx

    def policy_index_to_shogi_move(self, idx: int) -> "MoveTuple":
        """Convert a policy index back to its Shogi MoveTuple."""
        if 0 <= idx < self.total_actions:
            return self.idx_to_move[idx]
        raise IndexError(
            f"Policy index {idx} is out of bounds (0-{self.total_actions - 1})."
        )

    def get_legal_mask(
        self, legal_shogi_moves: List["MoveTuple"], device: torch.device
    ) -> torch.Tensor:
        """Converts a list of legal Shogi moves to a boolean mask tensor."""
        mask = torch.zeros(self.total_actions, dtype=torch.bool, device=device)
        if not legal_shogi_moves:
            # If there are no legal moves (e.g., game over), return an all-false mask.
            # This is important because PPOAgent.select_action checks if legal_mask.any() is false.
            return mask

        for move in legal_shogi_moves:
            try:
                idx = self.shogi_move_to_policy_index(move)
                mask[idx] = True
            except ValueError as e:
                # Log a prominent warning if a move from ShogiGame is not in the mapper
                # This indicates a potential desync between game logic and policy mapping.
                warning_msg = (
                    f"[PolicyOutputMapper Warning] Encountered a Shogi move not recognized by the policy mapper: "
                    f"{move}. Error: {e}. This move will be treated as illegal by the agent. "
                    f"This could indicate an issue with move generation in ShogiGame or an incomplete PolicyOutputMapper."
                )
                print(warning_msg, file=sys.stderr)  # Print to stderr for visibility
                # Optionally, could use logging module if a logger is available here.
                # The `pass` behavior is effectively maintained as the mask for this move remains False.
        return mask

    # --- NEW METHODS FOR USI CONVERSION ---
    def _usi_sq(self, r: int, c: int) -> str:
        """Converts 0-indexed (row, col) to USI square string (e.g., (0,0) -> "9a")."""
        if not (0 <= r <= 8 and 0 <= c <= 8):
            raise ValueError(f"Invalid Shogi coordinate for USI: row {r}, col {c}")
        file = str(9 - c)  # Column 0 is file 9, column 8 is file 1
        rank = chr(ord("a") + r)  # Row 0 is rank 'a', row 8 is rank 'i'
        return file + rank

    def _get_usi_char_for_drop(self, piece_type: PieceType) -> str:
        """Helper to get the uppercase USI character for a droppable piece type."""
        # Standard USI drop piece characters (uppercase)
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
            f"is not a standard droppable piece for USI notation or is invalid."
        )

    def shogi_move_to_usi(self, move_tuple: "MoveTuple") -> str:
        """
        Converts an internal MoveTuple representation to a USI string.
        Board move: (from_r, from_c, to_r, to_c, promote_bool) -> e.g., "7g7f" or "2b3a+"
        Drop move: (None, None, to_r, to_c, piece_type) -> e.g., "P*5e"
        """
        # Check for BoardMoveTuple structure: (int, int, int, int, bool)
        if (
            len(move_tuple) == 5
            and isinstance(move_tuple[0], int)
            and isinstance(move_tuple[1], int)  # These ensure they are not None
            and isinstance(move_tuple[2], int)
            and isinstance(move_tuple[3], int)
            and isinstance(move_tuple[4], bool)
        ):
            # Type assertion for mypy after checks
            board_move: BoardMoveTuple = move_tuple  # type: ignore
            from_r, from_c, to_r, to_c, promote = board_move
            from_sq_str = self._usi_sq(from_r, from_c)
            to_sq_str = self._usi_sq(to_r, to_c)
            promo_char = "+" if promote else ""
            return f"{from_sq_str}{to_sq_str}{promo_char}"
        # Check for DropMoveTuple structure: (None, None, int, int, PieceType)
        elif (
            len(move_tuple) == 5
            and move_tuple[0] is None
            and move_tuple[1] is None
            and isinstance(move_tuple[2], int)
            and isinstance(move_tuple[3], int)
            and isinstance(move_tuple[4], PieceType)
        ):
            drop_move: DropMoveTuple = move_tuple  # type: ignore
            _, _, to_r, to_c, piece_to_drop = drop_move
            piece_char = self._get_usi_char_for_drop(piece_to_drop)
            to_sq_str = self._usi_sq(to_r, to_c)
            return f"{piece_char}*{to_sq_str}"
        else:
            # For debugging, print the type of each element if it doesn't match
            # This can help if a move_tuple has an unexpected structure
            # e.g. print([type(x) for x in move_tuple])
            raise ValueError(
                f"Invalid MoveTuple format for USI conversion: {move_tuple}"
            )

    def usi_to_shogi_move(self, usi_move: str) -> "MoveTuple":
        """
        Converts a USI string to an internal MoveTuple.
        e.g., "7g7f" -> (2,2,3,2,False) [assuming 7g=(2,2), 7f=(3,2)]
        e.g., "2b3a+" -> (7,7,8,6,True) [assuming 2b=(7,7), 3a=(8,6)]
        e.g., "P*5e" -> (None,None,4,4,PieceType.PAWN) [assuming 5e=(4,4)]
        """
        if not isinstance(usi_move, str):
            raise TypeError(f"USI move must be a string, got {type(usi_move)}")

        if "*" in usi_move:  # Drop move
            parts = usi_move.split("*")
            if len(parts) != 2 or len(parts[0]) != 1 or len(parts[1]) != 2:
                raise ValueError(f"Invalid USI drop move format: {usi_move}")

            piece_char = parts[0]
            to_sq_str = parts[1]

            piece_type: PieceType
            if piece_char == "P":
                piece_type = PieceType.PAWN
            elif piece_char == "L":
                piece_type = PieceType.LANCE
            elif piece_char == "N":
                piece_type = PieceType.KNIGHT
            elif piece_char == "S":
                piece_type = PieceType.SILVER
            elif piece_char == "G":
                piece_type = PieceType.GOLD
            elif piece_char == "B":
                piece_type = PieceType.BISHOP
            elif piece_char == "R":
                piece_type = PieceType.ROOK
            else:
                raise ValueError(f"Invalid piece character for USI drop: {piece_char}")

            if not (
                len(to_sq_str) == 2
                and "1" <= to_sq_str[0] <= "9"
                and "a" <= to_sq_str[1] <= "i"
            ):
                raise ValueError(f"Invalid USI square format for drop: {to_sq_str}")

            to_c = 9 - int(to_sq_str[0])
            to_r = ord(to_sq_str[1]) - ord("a")
            return (None, None, to_r, to_c, piece_type)

        else:  # Board move
            if not (4 <= len(usi_move) <= 5):
                raise ValueError(f"Invalid USI board move format: {usi_move}")

            from_sq_str = usi_move[0:2]
            to_sq_str = usi_move[2:4]
            promote = len(usi_move) == 5 and usi_move[4] == "+"

            if not (
                len(from_sq_str) == 2
                and "1" <= from_sq_str[0] <= "9"
                and "a" <= from_sq_str[1] <= "i"
            ):
                raise ValueError(f"Invalid USI source square: {from_sq_str}")
            if not (
                len(to_sq_str) == 2
                and "1" <= to_sq_str[0] <= "9"
                and "a" <= to_sq_str[1] <= "i"
            ):
                raise ValueError(f"Invalid USI destination square: {to_sq_str}")

            from_c = 9 - int(from_sq_str[0])
            from_r = ord(from_sq_str[1]) - ord("a")
            to_c = 9 - int(to_sq_str[0])
            to_r = ord(to_sq_str[1]) - ord("a")

            return (from_r, from_c, to_r, to_c, promote)


class TrainingLogger:
    """Logs messages to a file and optionally to stdout."""

    def __init__(self, log_file_path: str, also_stdout: bool = True):
        """Initialize the logger.

        Args:
            log_file_path: Path to the log file.
            also_stdout: If True, also print log messages to stdout.
        """
        self.log_file_path = log_file_path
        self.also_stdout = also_stdout
        self.file_handle: TextIO | None = None

    def __enter__(self) -> "TrainingLogger":
        self.file_handle = open(self.log_file_path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def log(self, message: str) -> None:
        """Log a message to the file and optionally to stdout."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        if self.file_handle:
            self.file_handle.write(log_entry + "\n")
            self.file_handle.flush()  # Ensure it's written immediately

        if self.also_stdout:
            print(log_entry, file=sys.stdout)  # Explicitly use sys.stdout

    def close(self) -> None:
        """Close the log file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
