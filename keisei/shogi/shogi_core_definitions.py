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


# KIF Piece Symbol Mapping (Standard two-letter KIF symbols)
KIF_PIECE_SYMBOLS: Dict[PieceType, str] = {
    PieceType.PAWN: "FU",
    PieceType.LANCE: "KY",
    PieceType.KNIGHT: "KE",
    PieceType.SILVER: "GI",
    PieceType.GOLD: "KI",
    PieceType.BISHOP: "KA",
    PieceType.ROOK: "HI",
    PieceType.KING: "OU",  # Or "GY" for Gyoku, "OU" is common for King
    PieceType.PROMOTED_PAWN: "TO",  # Tokin
    PieceType.PROMOTED_LANCE: "NY",  # Promoted Lance (Nari-Kyo)
    PieceType.PROMOTED_KNIGHT: "NK",  # Promoted Knight (Nari-Kei)
    PieceType.PROMOTED_SILVER: "NG",  # Promoted Silver (Nari-Gin)
    PieceType.PROMOTED_BISHOP: "UM",  # Horse (Uma)
    PieceType.PROMOTED_ROOK: "RY",  # Dragon (Ryu)
}


class TerminationReason(Enum):
    CHECKMATE = "checkmate"
    RESIGNATION = "resignation"
    MAX_MOVES_EXCEEDED = "max_moves_exceeded"
    REPETITION = "repetition"  # Sennichite
    IMPASSE = "impasse"  # Jishogi (by points, declaration, etc.)
    ILLEGAL_MOVE = "illegal_move"
    TIME_FORFEIT = "time_forfeit"
    NO_CONTEST = "no_contest"  # E.g. server error, mutual agreement for no result


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


# Symbol to PieceType mapping (inverse of parts of Piece.symbol)
# Uses uppercase symbols as canonical representation.
SYMBOL_TO_PIECE_TYPE: Dict[str, PieceType] = {
    "P": PieceType.PAWN,
    "L": PieceType.LANCE,
    "N": PieceType.KNIGHT,
    "S": PieceType.SILVER,
    "G": PieceType.GOLD,
    "B": PieceType.BISHOP,
    "R": PieceType.ROOK,
    "K": PieceType.KING,
    "+P": PieceType.PROMOTED_PAWN,
    "+L": PieceType.PROMOTED_LANCE,
    "+N": PieceType.PROMOTED_KNIGHT,
    "+S": PieceType.PROMOTED_SILVER,
    "+B": PieceType.PROMOTED_BISHOP,
    "+R": PieceType.PROMOTED_ROOK,
}


def get_piece_type_from_symbol(symbol: str) -> PieceType:
    """
    Converts a piece symbol string (e.g., "P", "+R") to a PieceType enum.
    Assumes canonical (uppercase) symbols.
    """
    # Normalize to uppercase in case lowercase symbols are passed,
    # though canonical form is uppercase (e.g. "p" -> "P")
    # For promoted pieces, the '+' sign is significant.
    # If a symbol like "p" is passed, it should be treated as "P".
    # If "+p" is passed, it should be treated as "+P".

    # Simple case: direct match
    if symbol in SYMBOL_TO_PIECE_TYPE:
        return SYMBOL_TO_PIECE_TYPE[symbol]

    # Handle potentially lowercase symbols (e.g. "p" for "P", but "+p" for "+P")
    # The SYMBOL_TO_PIECE_TYPE map uses uppercase, so we should convert input.
    # If symbol is like "p", convert to "P". If "+p", convert to "+P".

    # Check if it's a promoted piece symbol like "+p"
    if len(symbol) == 2 and symbol.startswith("+") and symbol[1].islower():
        # Convert to uppercase promoted symbol, e.g., "+p" -> "+P"
        upper_symbol = "+" + symbol[1].upper()
        if upper_symbol in SYMBOL_TO_PIECE_TYPE:
            return SYMBOL_TO_PIECE_TYPE[upper_symbol]
    elif len(symbol) == 1 and symbol.islower():
        # Convert to uppercase unpromoted symbol, e.g., "p" -> "P"
        upper_symbol = symbol.upper()
        if upper_symbol in SYMBOL_TO_PIECE_TYPE:
            return SYMBOL_TO_PIECE_TYPE[upper_symbol]

    raise ValueError(f"Unknown piece symbol: {symbol}")


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


# --- Piece Class ---
class Piece:
    """Represents a single Shogi piece on the board."""

    def __init__(self, piece_type: PieceType, color: Color):
        if not isinstance(piece_type, PieceType):
            raise TypeError("piece_type must be an instance of PieceType")
        if not isinstance(color, Color):
            raise TypeError("color must be an instance of Color")

        self.type: PieceType = piece_type
        self.color: Color = color
        self.is_promoted: bool = piece_type in PROMOTED_TYPES_SET

    def symbol(self) -> str:
        """Returns a 1 or 2 character string symbol for the piece (e.g., P, +P, K)."""
        base_symbol: str
        if self.type == PieceType.PAWN:
            base_symbol = "P"
        elif self.type == PieceType.LANCE:
            base_symbol = "L"
        elif self.type == PieceType.KNIGHT:
            base_symbol = "N"
        elif self.type == PieceType.SILVER:
            base_symbol = "S"
        elif self.type == PieceType.GOLD:
            base_symbol = "G"
        elif self.type == PieceType.BISHOP:
            base_symbol = "B"
        elif self.type == PieceType.ROOK:
            base_symbol = "R"
        elif self.type == PieceType.KING:
            base_symbol = "K"
        elif self.type == PieceType.PROMOTED_PAWN:
            base_symbol = "+P"
        elif self.type == PieceType.PROMOTED_LANCE:
            base_symbol = "+L"
        elif self.type == PieceType.PROMOTED_KNIGHT:
            base_symbol = "+N"
        elif self.type == PieceType.PROMOTED_SILVER:
            base_symbol = "+S"
        elif self.type == PieceType.PROMOTED_BISHOP:
            base_symbol = "+B"
        elif self.type == PieceType.PROMOTED_ROOK:
            base_symbol = "+R"
        else:
            raise ValueError(f"Unknown piece type: {self.type}")

        return base_symbol.lower() if self.color == Color.WHITE else base_symbol

    def promote(self) -> None:
        """Promotes the piece if it is promotable and not already promoted."""
        if self.type in BASE_TO_PROMOTED_TYPE and not self.is_promoted:
            self.type = BASE_TO_PROMOTED_TYPE[self.type]
            self.is_promoted = True
        # else: No change if not promotable or already promoted

    def unpromote(self) -> None:
        """Unpromotes the piece if it is promoted."""
        if self.is_promoted and self.type in PROMOTED_TO_BASE_TYPE:
            self.type = PROMOTED_TO_BASE_TYPE[self.type]
            self.is_promoted = False
        # else: No change if not promoted

    def __repr__(self) -> str:
        return f"Piece({self.type.name}, {self.color.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Piece):
            return NotImplemented
        return self.type == other.type and self.color == other.color

    def __hash__(self) -> int:
        return hash((self.type, self.color))

    def __deepcopy__(self, memo: Dict[int, 'Piece']) -> 'Piece':
        # Create a new Piece instance without calling __init__ again if not necessary,
        # or simply create a new one.
        # Since Piece is simple, creating a new one is fine.
        new_piece = Piece(self.type, self.color)
        # self.is_promoted is derived, so no need to copy explicitly
        memo[id(self)] = new_piece
        return new_piece
