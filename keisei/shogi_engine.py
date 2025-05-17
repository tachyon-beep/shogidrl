"""
shogi_engine.py: Core Shogi game mechanics for DRL Shogi Client.
"""
from typing import Optional, List
import numpy as np
import config



class Piece:
    """
    Represents a Shogi piece with type, color, and promotion status.
    """

    def __init__(self, piece_type: int, color: int, is_promoted: bool = False):
        self.type = piece_type  # Integer code for piece type
        self.color = color  # 0 = Black (Sente), 1 = White (Gote)
        self.is_promoted = is_promoted

    def symbol(self) -> str:
        """
        Returns a character representation of the piece for display/logging.
        """
        # Example mapping, to be expanded for all types
        base_symbols = {
            0: "P",
            1: "L",
            2: "N",
            3: "S",
            4: "G",
            5: "B",
            6: "R",
            7: "K",
            8: "+P",
            9: "+L",
            10: "+N",
            11: "+S",
            12: "+B",
            13: "+R",
        }
        s = base_symbols.get(self.type, "?")
        if self.color == 1:
            s = s.lower()
        return s

    def __repr__(self):
        return (
            f"Piece(type={self.type}, color={self.color}, promoted={self.is_promoted})"
        )


class ShogiGame:
    """
    Represents the Shogi game state, board, and basic operations.
    """

    def __init__(self) -> None:
        # 9x9 board: rows 0-8, cols 0-8. Each cell is a Piece or None.
        self.board: List[List[Optional[Piece]]] = [
            [None for _ in range(9)] for _ in range(9)
        ]
        self.move_count: int = 0
        self.current_player: int = 0  # 0 = Black (Sente), 1 = White (Gote)
        self.move_history: list = []
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.reset()

    def reset(self) -> None:
        """
        Initializes the board to the standard Shogi starting position.
        """
        self.board = [[None for _ in range(9)] for _ in range(9)]
        # White pieces
        for c, t in enumerate([1, 2, 3, 4, 7, 4, 3, 2, 1]):
            self.board[0][c] = Piece(t, 1)
        self.board[1][1] = Piece(6, 1)
        self.board[1][7] = Piece(5, 1)
        for c in range(9):
            self.board[2][c] = Piece(0, 1)
        # Black pieces
        for c in range(9):
            self.board[6][c] = Piece(0, 0)
        self.board[7][1] = Piece(5, 0)
        self.board[7][7] = Piece(6, 0)
        for c, t in enumerate([1, 2, 3, 4, 7, 4, 3, 2, 1]):
            self.board[8][c] = Piece(t, 0)
        self.move_count = 0
        self.current_player = 0
        self.move_history = []
        self.game_over = False
        self.winner = None

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """
        Returns the Piece at (row, col), or None.
        """
        if 0 <= row < 9 and 0 <= col < 9:
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        """
        Sets the Piece at (row, col) to the given piece (or None).
        """
        if 0 <= row < 9 and 0 <= col < 9:
            self.board[row][col] = piece

    def to_string(self) -> str:
        """
        Returns a simple text representation of the board for debugging.
        """
        rows = []
        for r in range(9):
            row = []
            for c in range(9):
                p = self.board[r][c]
                row.append(p.symbol() if p else ".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def is_on_board(self, row: int, col: int) -> bool:
        """
        Returns True if (row, col) is a valid board coordinate.
        """
        return 0 <= row < 9 and 0 <= col < 9

    def get_individual_piece_moves(
        self, piece: Piece, r_from: int, c_from: int
    ) -> list[tuple[int, int]]:
        """
        Returns a list of (r_to, c_to) tuples for a piece, considering only its
        fundamental movement rules. Does not check for board boundaries, captures, or
        checks. Handles promoted pieces.
        """
        moves = []
        # Direction: Black (0) moves -1 in row, White (1) moves +1 in row
        forward = -1 if piece.color == 0 else 1
        # Piece type mapping: 0=P, 1=L, 2=N, 3=S, 4=G, 5=B, 6=R, 7=K
        t = piece.type
        promoted = piece.is_promoted or t >= 8
        # Pawn
        if t == 0 or (promoted and t == 8):
            # Promoted pawn moves as gold
            if promoted:
                gold_moves = [
                    (forward, 0),
                    (0, -1),
                    (0, 1),
                    (0, 0),
                    (-forward, -1),
                    (-forward, 1),
                ]
                for dr, dc in gold_moves:
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
            else:
                nr, nc = r_from + forward, c_from
                if self.is_on_board(nr, nc):
                    moves.append((nr, nc))
        # King
        elif t == 7:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
        # Lance
        elif t == 1 or (promoted and t == 9):
            if promoted:
                # Promoted lance moves as gold
                gold_moves = [
                    (forward, 0),
                    (0, -1),
                    (0, 1),
                    (0, 0),
                    (-forward, -1),
                    (-forward, 1),
                ]
                for dr, dc in gold_moves:
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
            else:
                # Lance moves any number of squares forward
                for i in range(1, 9):
                    nr, nc = r_from + i * forward, c_from
                    if not self.is_on_board(nr, nc):
                        break
                    moves.append((nr, nc))
        # Knight
        elif t == 2 or (promoted and t == 10):
            if promoted:
                # Promoted knight moves as gold
                gold_moves = [
                    (forward, 0),
                    (0, -1),
                    (0, 1),
                    (0, 0),
                    (-forward, -1),
                    (-forward, 1),
                ]
                for dr, dc in gold_moves:
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
            else:
                # Knight moves in an L shape: two forward, one left/right
                for dc in [-1, 1]:
                    nr, nc = r_from + 2 * forward, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
        # Silver
        elif t == 3 or (promoted and t == 11):
            if promoted:
                # Promoted silver moves as gold
                gold_moves = [
                    (forward, 0),
                    (0, -1),
                    (0, 1),
                    (0, 0),
                    (-forward, -1),
                    (-forward, 1),
                ]
                for dr, dc in gold_moves:
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
            else:
                # Silver: forward, forward-diagonals, backward-diagonals
                for dr, dc in [
                    (forward, 0),
                    (forward, -1),
                    (forward, 1),
                    (-forward, -1),
                    (-forward, 1),
                ]:
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
        # Gold
        elif t == 4:
            # Gold: forward, left, right, and three backward-diagonals
            gold_moves = [
                (forward, 0),
                (0, -1),
                (0, 1),
                (-forward, -1),
                (-forward, 1),
                (0, 0),
            ]
            for dr, dc in gold_moves:
                nr, nc = r_from + dr, c_from + dc
                if self.is_on_board(nr, nc):
                    moves.append((nr, nc))
        # Bishop
        elif t == 5 or (promoted and t == 12):
            # Bishop: diagonal moves
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                for i in range(1, 9):
                    nr, nc = r_from + dr * i, c_from + dc * i
                    if not self.is_on_board(nr, nc):
                        break
                    moves.append((nr, nc))
            if promoted:
                # Promoted bishop: add king moves (orthogonal)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
        # Rook
        elif t == 6 or (promoted and t == 13):
            # Rook: orthogonal moves
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for i in range(1, 9):
                    nr, nc = r_from + dr * i, c_from + dc * i
                    if not self.is_on_board(nr, nc):
                        break
                    moves.append((nr, nc))
            if promoted:
                # Promoted rook: add king moves (diagonal)
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
        return moves

    def get_observation(self) -> np.ndarray:
        """
        Returns the current board state as a (Channels, 9, 9) NumPy array for RL input.
        Channels:
            0-6: Current player's unpromoted pieces (Pawn, Lance, Knight, Silver, Gold, Bishop, Rook)
            7-13: Current player's promoted pieces (+Pawn, +Lance, +Knight, +Silver, +Bishop, +Rook)
            14-20: Opponent's unpromoted pieces
            21-27: Opponent's promoted pieces
            28-41: Pieces in hand (14 channels: 7 types x 2 players)
            42: Current player (all ones if Black, zeros if White)
            43: Move count (normalized, all cells set to move_count / MAX_MOVES_PER_GAME)
        """
        obs = np.zeros((44, 9, 9), dtype=np.float32)
        # Board pieces
        for r in range(9):
            for c in range(9):
                p = self.board[r][c]
                if p is None:
                    continue
                base = 0 if p.color == self.current_player else 14
                t = p.type
                promoted = p.is_promoted or t >= 8
                if promoted:
                    ch = base + 7 + (t % 7)
                else:
                    ch = base + (t % 7)
                # Only fill if within 0-27 (board planes)
                if 0 <= ch < 28:
                    obs[ch, r, c] = 1.0
        # Pieces in hand (14 channels: 7 types x 2 players)
        # Placeholder: hands not yet implemented, so skip for now
        # Current player plane
        obs[42, :, :] = 1.0 if self.current_player == 0 else 0.0
        # Move count plane (normalized)
        obs[43, :, :] = self.move_count / float(
            getattr(config, "MAX_MOVES_PER_GAME", 512)
        )
        return obs
