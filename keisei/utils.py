"""
PolicyOutputMapper: Maps Shogi moves to/from policy network output indices.
"""

import torch
from typing import List, Dict, Tuple, Any # Will change Any to MoveTuple once imported
from keisei.shogi.shogi_core_definitions import PieceType, MoveTuple, BoardMove, DropMove
import datetime


class PolicyOutputMapper:
    """Maps Shogi moves to/from policy network output indices."""

    def __init__(self):
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
        # Assuming PieceType enum has these members
        droppable_piece_types: List[PieceType] = [
            PieceType.PAWN,
            PieceType.LANCE,
            PieceType.KNIGHT,
            PieceType.SILVER,
            PieceType.GOLD,
            PieceType.BISHOP,
            PieceType.ROOK,
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
            raise ValueError(f"Move {move} not found in PolicyOutputMapper's known moves.")
        return idx

    def policy_index_to_shogi_move(self, idx: int) -> MoveTuple:
        """Convert a policy index back to its Shogi MoveTuple."""
        if 0 <= idx < self.total_actions:
            return self.idx_to_move[idx]
        raise IndexError(f"Policy index {idx} is out of bounds (0-{self.total_actions - 1}).")

    def get_legal_mask(self, legal_shogi_moves: List[MoveTuple], device: torch.device) -> torch.Tensor:
        """
        Create a boolean mask tensor for legal actions.

        Args:
            legal_shogi_moves: A list of legal MoveTuple objects from ShogiGame.
            device: The torch device to create the tensor on.

        Returns:
            A boolean tensor of shape (total_actions,) where True indicates a legal action.
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
                # For robustness, we can log this or handle as per requirements.
                # For now, we'll assume legal_shogi_moves are always mappable.
                # If not, it indicates a discrepancy between ShogiGame's move generation
                # and PolicyOutputMapper's understanding of moves.
                pass # Or print a warning: print(f"Warning: Legal move {move} not found in mapper.")
        return mask


class TrainingLogger:
    """Simple logger for training and evaluation."""

    def __init__(self, log_path: str, also_stdout: bool = True):
        """Initializes the TrainingLogger."""
        self.log_path = log_path
        self.also_stdout = also_stdout
        self.log_file = open(log_path, "a", encoding="utf-8")

    def log(self, msg: str):
        """Logs a message with a timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        self.log_file.write(line)
        self.log_file.flush()
        if self.also_stdout:
            print(line, end="")

    def close(self):
        """Closes the log file."""
        self.log_file.close()
