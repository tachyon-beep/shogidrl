"""
PolicyOutputMapper: Maps Shogi moves to/from policy network output indices.
"""
from typing import Tuple, Dict, Any

class PolicyOutputMapper:
    def __init__(self):
        # Expanded: map a few basic moves for demonstration
        self.idx_to_shogi_move_spec = [
            (6, 0, 5, 0, False),  # Black pawn forward
            (6, 1, 5, 1, False),  # Black pawn forward (file 2)
            (6, 2, 5, 2, False),  # Black pawn forward (file 3)
            (7, 1, 5, 1, False),  # Black bishop moves two forward (illegal in real game, for demo)
            (2, 0, 3, 0, False),  # White pawn forward
            (2, 1, 3, 1, False),  # White pawn forward (file 2)
            (2, 2, 3, 2, False),  # White pawn forward (file 3)
            (1, 1, 2, 2, False),  # White bishop moves (demo)
            (None, None, 4, 4, 'drop_pawn_black'),  # Black drops pawn at 4,4
            (None, None, 4, 4, 'drop_pawn_white'),  # White drops pawn at 4,4
        ]
        self.shogi_move_spec_to_idx = {move: idx for idx, move in enumerate(self.idx_to_shogi_move_spec)}
        self.total_actions = 3159

    def get_total_actions(self) -> int:
        return self.total_actions

    def shogi_move_to_policy_index(self, move: Any) -> int:
        if move in self.shogi_move_spec_to_idx:
            return self.shogi_move_spec_to_idx[move]
        raise NotImplementedError("Move not mapped in demo PolicyOutputMapper.")

    def policy_index_to_shogi_move(self, idx: int) -> Any:
        if idx < len(self.idx_to_shogi_move_spec):
            return self.idx_to_shogi_move_spec[idx]
        raise NotImplementedError("Index not mapped in demo PolicyOutputMapper.")
