"""
shogi_core_definitions.py: Core type definitions, enums, constants,
and the Piece class for the Shogi game engine.
"""

from enum import Enum
from typing import Dict, List, Set, Tuple, Union


# --- Enums and Constants ---
class Color(Enum):
    BLACK = 0  # Sente
    WHITE = 1  # Gote


class PieceType(Enum):
    PAWN = 0
    LANCE = 1
    KNIGHT = 2
    SILVER = 3
    GOLD = 4
    BISHOP = 5
    ROOK = 6
    KING = 7
    PROMOTED_PAWN = 8  # Tokin
    PROMOTED_LANCE = 9  # Promoted Lance
    PROMOTED_KNIGHT = 10  # Promoted Knight
    PROMOTED_SILVER = 11  # Promoted Silver
    # Gold does not promote
    PROMOTED_BISHOP = 12  # Horse
    PROMOTED_ROOK = 13  # Dragon
    # King does not promote


# --- Custom Types for Moves ---
# BoardMoveTuple: (from_row, from_col, to_row, to_col, promote_flag)
BoardMoveTuple = Tuple[int, int, int, int, bool]

# DropMoveTuple: (None, None, to_row, to_col, piece_type_to_drop)
# Using None for from_row, from_col to distinguish from board moves.
# piece_type_to_drop should be one of the unpromoted PieceType enums.
DropMoveTuple = Tuple[None, None, int, int, PieceType]

# MoveTuple is a union of the two types of moves.
MoveTuple = Union[BoardMoveTuple, DropMoveTuple]


# Helper for unpromoted hand types
def get_unpromoted_types() -> List[PieceType]:
    return [
        PieceType.PAWN,
        PieceType.LANCE,
        PieceType.KNIGHT,
        PieceType.SILVER,
        PieceType.GOLD,
        PieceType.BISHOP,
        PieceType.ROOK,
    ]


PROMOTED_TYPES_SET: Set[PieceType] = {
    PieceType.PROMOTED_PAWN,
    PieceType.PROMOTED_LANCE,
    PieceType.PROMOTED_KNIGHT,
    PieceType.PROMOTED_SILVER,
    PieceType.PROMOTED_BISHOP,
    PieceType.PROMOTED_ROOK,
}

BASE_TO_PROMOTED_TYPE: Dict[PieceType, PieceType] = {
    PieceType.PAWN: PieceType.PROMOTED_PAWN,
    PieceType.LANCE: PieceType.PROMOTED_LANCE,
    PieceType.KNIGHT: PieceType.PROMOTED_KNIGHT,
    PieceType.SILVER: PieceType.PROMOTED_SILVER,
    PieceType.BISHOP: PieceType.PROMOTED_BISHOP,
    PieceType.ROOK: PieceType.PROMOTED_ROOK,
}

PROMOTED_TO_BASE_TYPE: Dict[PieceType, PieceType] = {
    v: k for k, v in BASE_TO_PROMOTED_TYPE.items()
}

# For II.7: add_to_hand()
PIECE_TYPE_TO_HAND_TYPE: Dict[PieceType, PieceType] = {
    PieceType.PAWN: PieceType.PAWN,
    PieceType.LANCE: PieceType.LANCE,
    PieceType.KNIGHT: PieceType.KNIGHT,
    PieceType.SILVER: PieceType.SILVER,
    PieceType.GOLD: PieceType.GOLD,
    PieceType.BISHOP: PieceType.BISHOP,
    PieceType.ROOK: PieceType.ROOK,
    PieceType.PROMOTED_PAWN: PieceType.PAWN,
    PieceType.PROMOTED_LANCE: PieceType.LANCE,
    PieceType.PROMOTED_KNIGHT: PieceType.KNIGHT,
    PieceType.PROMOTED_SILVER: PieceType.SILVER,
    PieceType.PROMOTED_BISHOP: PieceType.BISHOP,
    PieceType.PROMOTED_ROOK: PieceType.ROOK,
    # Kings (PieceType.KING) and Gold (PieceType.GOLD) don't change type when captured.
    # King is handled separately in add_to_hand. Gold is already its base type.
}

# For observation channels (II.2)
# Unpromoted: P, L, N, S, G, B, R, K (8 types)
# Promoted: +P, +L, +N, +S, +B, +R (6 types)
# Total board piece planes per player = 8 (unpromoted) + 6 (promoted) = 14
# Total board piece planes = 14 * 2 = 28
# Hands: 7 piece types * 2 players = 14 planes
# Other: current player, move count = 2 planes
# Total expected channels for current obs structure: 28 (board) + 14 (hands) + 2 (meta) = 44
# Original code uses 46 channels, planes 44 and 45 are "reserved".
# We will use 8 planes for unpromoted (0-7), 6 for promoted (8-13) for the current player section
# And similarly for opponent.

# Unpromoted pieces for observation channels
OBS_UNPROMOTED_ORDER = [
    PieceType.PAWN,
    PieceType.LANCE,
    PieceType.KNIGHT,
    PieceType.SILVER,
    PieceType.GOLD,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.KING,
]
# Promoted pieces for observation channels
OBS_PROMOTED_ORDER = [
    PieceType.PROMOTED_PAWN,
    PieceType.PROMOTED_LANCE,
    PieceType.PROMOTED_KNIGHT,
    PieceType.PROMOTED_SILVER,
    PieceType.PROMOTED_BISHOP,
    PieceType.PROMOTED_ROOK,
]


class Piece:
    """
    Represents a Shogi piece with type and color.
    Promotion status is derived from its type.
    """

    def __init__(self, piece_type: PieceType, color: Color):
        self.type: PieceType = piece_type
        self.color: Color = color

    @property
    def is_promoted(self) -> bool:
        """Returns True if the piece type is a promoted type."""
        return self.type in PROMOTED_TYPES_SET

    def symbol(self) -> str:
        """
        Returns a character representation of the piece for display/logging.
        Relies on self.type being the canonical representation (e.g., PROMOTED_PAWN for a tokin).
        """
        # Adjusted to use PieceType Enum values as keys
        base_symbols = {
            PieceType.PAWN: "P",
            PieceType.LANCE: "L",
            PieceType.KNIGHT: "N",
            PieceType.SILVER: "S",
            PieceType.GOLD: "G",
            PieceType.BISHOP: "B",
            PieceType.ROOK: "R",
            PieceType.KING: "K",
            PieceType.PROMOTED_PAWN: "+P",
            PieceType.PROMOTED_LANCE: "+L",
            PieceType.PROMOTED_KNIGHT: "+N",
            PieceType.PROMOTED_SILVER: "+S",
            PieceType.PROMOTED_BISHOP: "+B",
            PieceType.PROMOTED_ROOK: "+R",
        }
        s = base_symbols.get(self.type, "?")
        if self.color == Color.WHITE:
            s = s.lower()  # Lowercase for Gote (White)
        return s

    def __repr__(self):
        return f"Piece(type={self.type.name}, color={self.color.name})"
