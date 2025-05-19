"""
utils.py: Contains PolicyOutputMapper and TrainingLogger.
"""
import os
import datetime
from typing import Dict, List

import torch

# Ensure these imports are correct based on your project structure
# The user's provided file had these, which look good:
from keisei.shogi.shogi_core_definitions import (
    BoardMove,
    DropMove,
    MoveTuple,
    PieceType,
)


class PolicyOutputMapper:
    """Maps Shogi moves to/from policy network output indices."""

    def __init__(self) -> None:
        """Initializes the PolicyOutputMapper by generating all possible move representations."""
        self.idx_to_move: List[MoveTuple] = []
        self.move_to_idx: Dict[MoveTuple, int] = {}
        current_idx = 0

        # Generate Board Moves: (from_r, from_c, to_r, to_c, promote_flag: bool)
        for r_from in range(9):
            for c_from in range(9):
                for r_to in range(9):
                    for c_to in range(9):
                        if r_from == r_to and c_from == c_to:
                            continue  # Skip null moves

                        # Move without promotion
                        move_no_promo: BoardMove = (r_from, c_from, r_to, c_to, False)
                        self.idx_to_move.append(move_no_promo)
                        self.move_to_idx[move_no_promo] = current_idx
                        current_idx += 1

                        # Move with promotion
                        move_promo: BoardMove = (r_from, c_from, r_to, c_to, True)
                        self.idx_to_move.append(move_promo)
                        self.move_to_idx[move_promo] = current_idx
                        current_idx += 1

        # Define droppable piece types (standard 7, excluding King)
        droppable_piece_types: List[PieceType] = [
            PieceType.PAWN, PieceType.LANCE, PieceType.KNIGHT, PieceType.SILVER,
            PieceType.GOLD, PieceType.BISHOP, PieceType.ROOK,
        ]

        # Generate Drop Moves: (None, None, to_r, to_c, piece_type_to_drop: PieceType)
        for r_to in range(9):
            for c_to in range(9):
                for piece_type in droppable_piece_types:
                    drop_move: DropMove = (None, None, r_to, c_to, piece_type)
                    self.idx_to_move.append(drop_move)
                    self.move_to_idx[drop_move] = current_idx
                    current_idx += 1

        self.total_actions = current_idx

    def get_total_actions(self) -> int:
        """Return the total number of possible actions."""
        return self.total_actions

    def shogi_move_to_policy_index(self, move: MoveTuple) -> int:
        """Convert a Shogi MoveTuple to its policy index."""
        idx = self.move_to_idx.get(move)
        if idx is None:
            raise ValueError(
                f"Move {move} not found in PolicyOutputMapper's known moves."
            )
        return idx

    def policy_index_to_shogi_move(self, idx: int) -> MoveTuple:
        """Convert a policy index back to its Shogi MoveTuple."""
        if 0 <= idx < self.total_actions:
            return self.idx_to_move[idx]
        raise IndexError(
            f"Policy index {idx} is out of bounds (0-{self.total_actions - 1})."
        )

    def get_legal_mask(
        self, legal_shogi_moves: List[MoveTuple], device: torch.device
    ) -> torch.Tensor:
        """
        Create a boolean mask tensor for legal actions.
        """
        mask = torch.zeros(self.total_actions, dtype=torch.bool, device=device)
        for move in legal_shogi_moves:
            try:
                idx = self.shogi_move_to_policy_index(move)
                mask[idx] = True
            except ValueError:
                # This can happen if ShogiGame generates a move tuple format
                # that doesn't exactly match what PolicyOutputMapper expects
                # or if a move is somehow illegal yet generated.
                # Consider logging this for debugging:
                # print(f"Warning: Legal move {move} from ShogiGame not found in PolicyOutputMapper.")
                pass
        return mask

    # --- NEW METHODS FOR USI CONVERSION ---
    def _usi_sq(self, r: int, c: int) -> str:
        """Converts 0-indexed (row, col) to USI square string (e.g., (0,0) -> "9a")."""
        if not (0 <= r <= 8 and 0 <= c <= 8):
            raise ValueError(f"Invalid Shogi coordinate for USI: row {r}, col {c}")
        file = str(9 - c)  # Column 0 is file 9, column 8 is file 1
        rank = chr(ord('a') + r)  # Row 0 is rank 'a', row 8 is rank 'i'
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

        raise ValueError(f"PieceType {piece_type.name if hasattr(piece_type, 'name') else piece_type} "
                         f"is not a standard droppable piece for USI notation or is invalid.")

    def shogi_move_to_usi(self, move_tuple: MoveTuple) -> str:
        """
        Converts an internal MoveTuple representation to a USI string.
        Board move: (from_r, from_c, to_r, to_c, promote_bool) -> e.g., "7g7f" or "2b3a+"
        Drop move: (None, None, to_r, to_c, piece_type) -> e.g., "P*5e"
        """
        if move_tuple[0] is not None and move_tuple[1] is not None:  # Board move
            from_r, from_c, to_r, to_c = int(move_tuple[0]), int(move_tuple[1]), int(move_tuple[2]), int(move_tuple[3])
            promote = bool(move_tuple[4])

            from_sq_str = self._usi_sq(from_r, from_c)
            to_sq_str = self._usi_sq(to_r, to_c)
            promo_char = "+" if promote else ""
            return f"{from_sq_str}{to_sq_str}{promo_char}"

        elif move_tuple[0] is None and move_tuple[1] is None and isinstance(move_tuple[4], PieceType):  # Drop move
            to_r, to_c = int(move_tuple[2]), int(move_tuple[3])
            piece_to_drop: PieceType = move_tuple[4]

            piece_char = self._get_usi_char_for_drop(piece_to_drop)
            to_sq_str = self._usi_sq(to_r, to_c)
            return f"{piece_char}*{to_sq_str}"
        else:
            raise ValueError(f"Invalid MoveTuple format for USI conversion: {move_tuple}")
    # --- END OF NEW METHODS ---


class TrainingLogger:
    """Simple logger for training and evaluation."""

    def __init__(self, log_path: str, also_stdout: bool = True) -> None:
        """Initializes the TrainingLogger."""
        self.log_path = log_path
        self.also_stdout = also_stdout
        # Ensure directory for log_path exists if it's not just a filename
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = open(log_path, "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        """Logs a message with a timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        self.log_file.write(line)
        self.log_file.flush() # Ensure it's written immediately
        if self.also_stdout:
            print(line.strip()) # Use print for stdout, strip newline as print adds one

    def close(self) -> None:
        """Closes the log file."""
        if self.log_file and not self.log_file.closed:
            self.log_file.close()
