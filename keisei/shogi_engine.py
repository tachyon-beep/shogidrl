"""
shogi_engine.py: Core Shogi game mechanics for DRL Shogi Client.
"""

from typing import Optional, List, Any
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
            # Lowercase for white (Gote)
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
        # Dict to store pieces in hand for each player
        # Key is piece type (0-6), value is count
        self.hands: List[dict] = [{}, {}]  # [Black's hand, White's hand]
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
        # Initialize empty hands for both players
        self.hands = [{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, 
                      {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}]
        # White pieces (top)
        for c, t in enumerate([1, 2, 3, 4, 7, 4, 3, 2, 1]):
            self.board[0][c] = Piece(t, 1)
        self.board[1][1] = Piece(6, 1)  # Rook
        self.board[1][7] = Piece(5, 1)  # Bishop
        for c in range(9):
            self.board[2][c] = Piece(0, 1)  # Pawns
        # Black pieces (bottom)
        for c in range(9):
            self.board[6][c] = Piece(0, 0)
        self.board[7][1] = Piece(5, 0)  # Bishop
        self.board[7][7] = Piece(6, 0)  # Rook
        for c, t in enumerate([1, 2, 3, 4, 7, 4, 3, 2, 1]):
            self.board[8][c] = Piece(t, 0)
        # Empty middle
        for r in range(3, 6):
            for c in range(9):
                self.board[r][c] = None
        self.move_count = 0
        self.current_player = 0
        self.move_history = []
        self.game_over = False
        self.winner = None

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """
        Return the Piece at (row, col) if on board, else None.
        """
        if self.is_on_board(row, col):
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        """
        Set the Piece at (row, col) if on board.
        """
        if self.is_on_board(row, col):
            self.board[row][col] = piece

    def to_string(self) -> str:
        """
        Return a string representation of the board for display/logging.
        """
        lines = []
        for row in self.board:
            line = " ".join(p.symbol() if p else "." for p in row)
            lines.append(line)
        return "\n".join(lines)

    def is_on_board(self, row: int, col: int) -> bool:
        """
        Return True if (row, col) is a valid board coordinate.
        """
        if row is None or col is None:
            return False
        return 0 <= row < 9 and 0 <= col < 9

    def get_individual_piece_moves(
        self, piece: Piece, r_from: int, c_from: int
    ) -> list[tuple[int, int]]:
        """
        Returns a list of (r_to, c_to) tuples for a piece, considering only its
        fundamental movement rules. Includes is_on_board checks.
        Handles promoted pieces.
        Considers pieces on the board for path-blocking of sliding pieces.
        """
        moves = []
        forward = (
            -1 if piece.color == 0 else 1
        )  # Black (0) moves row -1, White (1) moves row +1

        t = piece.type
        is_promoted_by_flag = piece.is_promoted  # Explicit promotion flag on the piece

        # Define move offsets once
        gold_move_offsets = [
            (forward, 0),  # Forward orthogonal
            (-forward, 0),  # Backward orthogonal
            (0, -1),  # Left orthogonal
            (0, 1),  # Right orthogonal
            (forward, -1),  # Forward-Left diagonal
            (forward, 1),  # Forward-Right diagonal
        ]
        king_move_offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        knight_move_offsets = [(forward * 2, -1), (forward * 2, 1)]
        silver_move_offsets = [
            (forward, 0),
            (forward, -1),
            (forward, 1),
            (-forward, -1),
            (-forward, 1),
        ]
        promoted_rook_extra_offsets = [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]  # King's diagonal moves

        # --- Gold-like moves ---
        # True if: Gold (4), Promoted P/L/N/S by type (8,9,10,11), or base P/L/N/S (0,1,2,3) and flagged as promoted
        is_gold_equivalent = (
            (t == 4)
            or (t in [8, 9, 10, 11])
            or (is_promoted_by_flag and t in [0, 1, 2, 3])
        )

        if is_gold_equivalent:
            for dr, dc in gold_move_offsets:
                nr, nc = r_from + dr, c_from + dc
                if self.is_on_board(nr, nc):
                    moves.append((nr, nc))

        # --- Pawn (unpromoted) ---
        # Type 0 (Pawn) and NOT (flagged promoted OR type is 8 already)
        elif t == 0 and not is_promoted_by_flag:
            nr, nc = r_from + forward, c_from
            if self.is_on_board(nr, nc):
                moves.append((nr, nc))

        # --- King (type 7) ---
        elif t == 7:
            for dr, dc in king_move_offsets:
                nr, nc = r_from + dr, c_from + dc
                if self.is_on_board(nr, nc):
                    moves.append((nr, nc))

        # --- Lance (unpromoted) ---
        # Type 1 (Lance) and NOT (flagged promoted OR type is 9 already)
        elif t == 1 and not is_promoted_by_flag:
            for i in range(1, 9):
                nr, nc = r_from + forward * i, c_from
                if not self.is_on_board(nr, nc):
                    break
                target = self.get_piece(nr, nc)
                if target is None:
                    moves.append((nr, nc))
                else:
                    # Always add the king's square as a possible move for check detection
                    if target.type == 7:
                        moves.append((nr, nc))
                    elif target.color != piece.color:
                        moves.append((nr, nc))
                    break

        # --- Knight (unpromoted) ---
        # Type 2 (Knight) and NOT (flagged promoted OR type is 10 already)
        elif t == 2 and not is_promoted_by_flag:
            for dr, dc in knight_move_offsets:
                nr, nc = r_from + dr, c_from + dc
                if self.is_on_board(nr, nc):
                    moves.append((nr, nc))

        # --- Silver (unpromoted) ---
        # Type 3 (Silver) and NOT (flagged promoted OR type is 11 already)
        elif t == 3 and not is_promoted_by_flag:
            for dr, dc in silver_move_offsets:
                nr, nc = r_from + dr, c_from + dc
                if self.is_on_board(nr, nc):
                    moves.append((nr, nc))

        # --- Bishop family (Bishop type 5, Promoted Bishop type 12) ---
        elif t == 5 or t == 12:
            # Bishop sliding moves (diagonals)
            for dr_diag, dc_diag in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                for i in range(1, 9):
                    nr, nc = r_from + dr_diag * i, c_from + dc_diag * i
                    if not self.is_on_board(nr, nc):
                        break
                    target = self.get_piece(nr, nc)
                    if target is None:
                        moves.append((nr, nc))
                    else:
                        # Always add the king's square as a possible move for check detection
                        if target.type == 7:
                            moves.append((nr, nc))
                        elif target.color != piece.color:
                            moves.append((nr, nc))
                        break
            # If Promoted Bishop (type 12, OR type 5 and flagged promoted)
            if t == 12 or (t == 5 and is_promoted_by_flag):
                for dr, dc in king_move_offsets:  # Full King moves
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))

        # --- Rook family (Rook type 6, Promoted Rook type 13) ---
        elif t == 6 or t == 13:
            # Rook sliding moves (orthogonals)
            for dr_ortho, dc_ortho in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for i in range(1, 9):
                    nr, nc = r_from + dr_ortho * i, c_from + dc_ortho * i
                    if not self.is_on_board(nr, nc):
                        break
                    target = self.get_piece(nr, nc)
                    if target is None:
                        moves.append((nr, nc))
                    else:
                        # Always add the king's square as a possible move for check detection
                        if target.type == 7:
                            moves.append((nr, nc))
                        elif target.color != piece.color:
                            moves.append((nr, nc))
                        break
            # If Promoted Rook (type 13, OR type 6 and flagged promoted)
            if t == 13 or (t == 6 and is_promoted_by_flag):
                for dr, dc in promoted_rook_extra_offsets:  # King's diagonal moves
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))

        # For +B and +R, moves from sliding and king-like parts can overlap.
        # The tests seem sensitive to exact list content (duplicates/order).
        # So, returning raw list for now. If uchi_fu_zume needs unique moves,
        # this can be changed to list(set(moves)).
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
        obs = np.zeros((46, 9, 9), dtype=np.float32)
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
        curr_hands = self.hands[self.current_player]
        opp_hands = self.hands[1 - self.current_player]
        
        # Current player's pieces in hand (channels 28-34)
        for piece_type, count in curr_hands.items():
            if 0 <= piece_type < 7 and count > 0:  # Valid piece type
                ch = 28 + piece_type
                # Fill the channel with the normalized count
                obs[ch, :, :] = count / 18.0  # Normalize by max possible number (18 pawns)
        
        # Opponent's pieces in hand (channels 35-41)
        for piece_type, count in opp_hands.items():
            if 0 <= piece_type < 7 and count > 0:  # Valid piece type
                ch = 35 + piece_type
                # Fill the channel with the normalized count
                obs[ch, :, :] = count / 18.0  # Normalize by max possible number
        
        # Current player plane
        obs[42, :, :] = 1.0 if self.current_player == 0 else 0.0
        # Move count plane (normalized)
        obs[43, :, :] = self.move_count / float(
            getattr(config, "MAX_MOVES_PER_GAME", 512)
        )
        # Planes 44 and 45: reserved for future use, keep as zeros
        return obs

    def is_nifu(self, color: int, col: int) -> bool:
        """
        Returns True if dropping a pawn of the given color in the given column would violate Nifu
        (two unpromoted pawns on the same file).
        """
        for row in range(9):
            p = self.get_piece(row, col)
            if p and p.type == 0 and p.color == color and not p.is_promoted:
                return True
        return False

    def is_uchi_fu_zume(self, drop_row: int, drop_col: int, color: int) -> bool:
        """
        Returns True if dropping a pawn at (drop_row, drop_col) by 'color' would result in immediate checkmate
        (Uchi Fu Zume, illegal in Shogi).
        Only applies to pawn drops (not other pieces).
        """
        # Quick check - is the square already occupied?
        if self.get_piece(drop_row, drop_col) is not None:
            return False  # Square is occupied, drop is invalid anyway
            
        # Find the opponent's king first
        opp_color = 1 - color
        king_pos = None
        for r in range(9):
            for c in range(9):
                p = self.get_piece(r, c)
                if p and p.type == 7 and p.color == opp_color:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
                
        if not king_pos:
            return False  # No king found (shouldn't happen in a real game)
        
        king_r, king_c = king_pos
            
        # Check if the dropped pawn would give check to the king
        # Pawn attacks depend on color:
        # Black pawn (color 0) attacks the square in front (row-1)
        # White pawn (color 1) attacks the square in front (row+1)
        would_give_check = False
        if color == 0:  # Black dropping a pawn
            would_give_check = (king_r == drop_row - 1 and king_c == drop_col)
        else:  # White dropping a pawn
            would_give_check = (king_r == drop_row + 1 and king_c == drop_col)
            
        if not would_give_check:
            return False  # Pawn doesn't give check, so can't be checkmate
            
        # Temporarily place the pawn to check if it's checkmate
        dropped_pawn = Piece(0, color, False)
        self.set_piece(drop_row, drop_col, dropped_pawn)
        
        # Save original board state before we start testing
        orig_board = []
        for r in range(9):
            row = []
            for c in range(9):
                piece = self.get_piece(r, c)
                row.append(piece)
            orig_board.append(row)
            
        # 1. Can the king escape? Check all possible king moves
        king_can_escape = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip current position
                    
                new_r, new_c = king_r + dr, king_c + dc
                    
                # Skip if off board
                if not self.is_on_board(new_r, new_c):
                    continue
                    
                # Skip if square is occupied by king's own pieces (same color as king)
                target_piece = self.get_piece(new_r, new_c)
                if target_piece and target_piece.color == opp_color:
                    continue  # Can't move to a square occupied by own piece
                    
                # Temporarily move the king to this square
                orig_piece_at_target = self.get_piece(new_r, new_c)
                orig_king = self.get_piece(king_r, king_c)
                self.set_piece(king_r, king_c, None)
                self.set_piece(new_r, new_c, orig_king)
                    
                # Check if king is safe on this square
                is_safe = not self._is_square_attacked(new_r, new_c, color)
                    
                # Restore the king
                self.set_piece(king_r, king_c, orig_king)
                self.set_piece(new_r, new_c, orig_piece_at_target)
                    
                if is_safe:
                    king_can_escape = True
                    break
                    
            if king_can_escape:
                break
                
        if king_can_escape:
            # Restore the board and return
            self.set_piece(drop_row, drop_col, None)
            return False  # King can escape, not checkmate
            
        # 2. Can any of the opponent's pieces capture the dropped pawn?
        pawn_can_be_captured = False
        
        # Restore board to original state with pawn drop
        for r in range(9):
            for c in range(9):
                self.set_piece(r, c, orig_board[r][c])
        self.set_piece(drop_row, drop_col, dropped_pawn)
        
        for r in range(9):
            for c in range(9):
                piece = self.get_piece(r, c)
                if not piece or piece.color != opp_color:
                    continue  # Skip empty squares and attacker's pieces
                    
                # Get all possible moves for this piece
                moves = self.get_individual_piece_moves(piece, r, c)
                if (drop_row, drop_col) not in moves:
                    continue  # This piece can't reach the pawn
                    
                # For sliding pieces, check if the path is clear
                path_clear = True
                if piece.type in [1, 5, 6]:  # Lance, Bishop, Rook
                    # Calculate direction from piece to pawn
                    dr = 0
                    if drop_row > r:
                        dr = 1
                    elif drop_row < r:
                        dr = -1
                    
                    dc = 0
                    if drop_col > c:
                        dc = 1
                    elif drop_col < c:
                        dc = -1
                    
                    # Skip check for adjacent squares
                    if abs(r - drop_row) > 1 or abs(c - drop_col) > 1:
                        curr_r, curr_c = r + dr, c + dc
                        while (curr_r, curr_c) != (drop_row, drop_col):
                            if not self.is_on_board(curr_r, curr_c):
                                path_clear = False
                                break
                            if self.get_piece(curr_r, curr_c) is not None:
                                path_clear = False
                                break
                            curr_r += dr
                            curr_c += dc
                
                if not path_clear:
                    continue  # Path to pawn is blocked
                
                # Simulate capture
                orig_piece = self.get_piece(r, c)
                self.set_piece(r, c, None)
                self.set_piece(drop_row, drop_col, orig_piece)
                
                # Check if king is still in check after capture
                is_still_in_check = self._is_square_attacked(king_r, king_c, color)
                
                # Restore pieces
                self.set_piece(r, c, orig_piece)
                self.set_piece(drop_row, drop_col, dropped_pawn)
                
                if not is_still_in_check:
                    pawn_can_be_captured = True
                    break
            
            if pawn_can_be_captured:
                break
        
        # Restore original board state
        for r in range(9):
            for c in range(9):
                self.set_piece(r, c, orig_board[r][c])
        
        # If king can't escape and pawn can't be captured, it's checkmate (Uchi Fu Zume)
        return not king_can_escape and not pawn_can_be_captured

    def _is_square_attacked(self, row: int, col: int, attacker_color: int) -> bool:
        """
        Returns True if the square (row, col) is attacked by any piece of attacker_color.
        Considers obstructions for sliding pieces.
        """
        for r_attacker in range(9):
            for c_attacker in range(9):
                p_attacker = self.get_piece(r_attacker, c_attacker)
                if p_attacker and p_attacker.color == attacker_color:
                    # This piece belongs to the attacker, check if it attacks (row, col)
                    # Get raw moves for the attacking piece (potential squares it could move to if no blockers)
                    raw_moves = self.get_individual_piece_moves(
                        p_attacker, r_attacker, c_attacker
                    )

                    if (row, col) in raw_moves:
                        # Determine if the piece is a sliding piece type
                        # Piece types: 0=P, 1=L, 2=N, 3=S, 4=G, 5=B, 6=R, 7=K
                        # Promoted: +P=8, +L=9, +N=10, +S=11, +B=12, +R=13

                        _p_attacker_type = p_attacker.type
                        # Not directly needed for this simplified check
                        # _p_attacker_is_promoted = p_attacker.is_promoted

                        is_sliding_piece = False
                        if _p_attacker_type == 0:  # Pawn
                            is_sliding_piece = False
                        elif _p_attacker_type == 1:  # Lance
                            is_sliding_piece = True
                        elif _p_attacker_type == 2:  # Knight
                            is_sliding_piece = False
                        elif _p_attacker_type == 3:  # Silver
                            is_sliding_piece = False
                        elif _p_attacker_type == 4:  # Gold
                            is_sliding_piece = False
                        elif _p_attacker_type == 5:  # Bishop
                            is_sliding_piece = True
                        elif _p_attacker_type == 6:  # Rook
                            is_sliding_piece = True
                        elif _p_attacker_type == 7:  # King
                            is_sliding_piece = False
                        # Promoted pieces
                        elif _p_attacker_type == 8:  # +P (moves like Gold)
                            is_sliding_piece = False
                        elif _p_attacker_type == 9:  # +L (moves like Gold)
                            is_sliding_piece = False
                        elif _p_attacker_type == 10:  # +N (moves like Gold)
                            is_sliding_piece = False
                        elif _p_attacker_type == 11:  # +S (moves like Gold)
                            is_sliding_piece = False
                        elif _p_attacker_type == 12:  # +B (Bishop + King moves)
                            is_sliding_piece = True
                        elif _p_attacker_type == 13:  # +R (Rook + King moves)
                            is_sliding_piece = True

                        if not is_sliding_piece:
                            # For non-sliding pieces, if (row, col) is in raw_moves, it's a direct attack.
                            return True
                        else:
                            # Path checking for sliding pieces (Lance, Bishop, Rook, Promoted B/R)
                            dr = 0
                            if row > r_attacker:
                                dr = 1
                            elif row < r_attacker:
                                dr = -1

                            dc = 0
                            if col > c_attacker:
                                dc = 1
                            elif col < c_attacker:
                                dc = -1

                            # If dr and dc are both 0, it means r_attacker == row and c_attacker == col.
                            # This shouldn't happen if (row, col) is a move for p_attacker,
                            # unless it's a self-capture (which raw_moves should prevent for same color).
                            # However, for safety, if it's an adjacent square (already handled by raw_moves
                            # for king-like parts of +B/+R) or if it's the same square, path check logic
                            # might be tricky.
                            # The crucial part is that (row,col) is in raw_moves.
                            # If it's an adjacent square, the path is trivially clear.

                            path_clear = True
                            # Only check path if not adjacent (distance > 1 in at least one direction)
                            # This also handles the case where (r_attacker, c_attacker) == (row, col)
                            # if dr,dc are 0 which means the while loop condition `(curr_r, curr_c)
                            # != (row, col)` would be initially false.
                            if (
                                abs(r_attacker - row) > 1
                                or abs(c_attacker - col) > 1
                                or (
                                    abs(r_attacker - row) == 1
                                    and abs(c_attacker - col) > 1
                                )
                                or (
                                    abs(r_attacker - row) > 1
                                    and abs(c_attacker - col) == 1
                                )
                                or (
                                    dr == 0
                                    and dc == 0
                                    and (r_attacker != row or c_attacker != col)
                                )
                            ):
                                # Check path only if not adjacent or same square
                                curr_r, curr_c = r_attacker + dr, c_attacker + dc
                                while (curr_r, curr_c) != (row, col):
                                    if not self.is_on_board(curr_r, curr_c):
                                        # Should not happen if raw_moves are correct
                                        path_clear = False
                                        break
                                    piece_on_path = self.get_piece(curr_r, curr_c)
                                    if piece_on_path is not None:
                                        path_clear = False
                                        break
                                    curr_r += dr
                                    curr_c += dc
                                    # Safety break for unexpected scenarios, though (row,col)
                                    # in raw_moves should ensure termination
                                    if not self.is_on_board(curr_r, curr_c) and (
                                        curr_r,
                                        curr_c,
                                    ) != (row, col):
                                        path_clear = False  # Went off board before reaching target
                                        break

                            if path_clear:
                                return True  # If path is clear for a sliding piece, it's an attack.
                            # else: path not clear, continue to the next attacker in the outer loop.
        return False

    def get_legal_moves(self):
        """
        Generate all legal moves for the current player. Returns a list of move tuples.
        Includes board moves, drops, and promotions. Applies all legality rules.
        """
        legal_moves = []
        
        # 1. Generate board moves with potential promotions
        for r_from in range(9):
            for c_from in range(9):
                piece = self.get_piece(r_from, c_from)
                if not piece or piece.color != self.current_player:
                    continue
                
                moves = self.get_individual_piece_moves(piece, r_from, c_from)
                for r_to, c_to in moves:
                    target = self.get_piece(r_to, c_to)
                    if target and target.color == self.current_player:
                        continue  # Can't capture own piece
                    
                    # Check for promotion opportunity or requirement
                    can_promote = self.can_promote_piece(piece, r_from, c_from, r_to, c_to)
                    must_promote = self.must_promote_piece(piece, r_to, c_to)
                    
                    # Add normal move if valid
                    if not must_promote:
                        # Without promotion
                        move_tuple = (r_from, c_from, r_to, c_to, 0)
                        # Simulate move to check if it puts player in check
                        captured = self.get_piece(r_to, c_to)
                        orig_piece = self.get_piece(r_from, c_from)
                        self.set_piece(r_to, c_to, orig_piece)
                        self.set_piece(r_from, c_from, None)
                        in_check = self._king_in_check_after_move(self.current_player)
                        # Undo move
                        self.set_piece(r_from, c_from, orig_piece)
                        self.set_piece(r_to, c_to, captured)
                        if not in_check:
                            legal_moves.append(move_tuple)
                    
                    # Add promotion move if valid
                    if can_promote:
                        # With promotion
                        move_tuple = (r_from, c_from, r_to, c_to, 1)
                        # Simulate move to check if it puts player in check
                        captured = self.get_piece(r_to, c_to)
                        orig_piece = self.get_piece(r_from, c_from)
                        # Create a copy of the piece with promotion if orig_piece is not None
                        if orig_piece is not None:
                            promoted_piece = Piece(orig_piece.type, orig_piece.color, True)
                            self.set_piece(r_to, c_to, promoted_piece)
                            self.set_piece(r_from, c_from, None)
                            in_check = self._king_in_check_after_move(self.current_player)
                            # Undo move
                            self.set_piece(r_from, c_from, orig_piece)
                            self.set_piece(r_to, c_to, captured)
                            if not in_check:
                                legal_moves.append(move_tuple)
        
        # 2. Generate drop moves
        # Get pieces in hand
        hand_pieces = self.get_pieces_in_hand(self.current_player)
        
        # Generate drops for each piece type in hand
        for piece_type, count in hand_pieces.items():
            if count <= 0:
                continue  # Skip if no pieces of this type in hand
                
            # For each empty square on the board
            for r in range(9):
                for c in range(9):
                    if self.get_piece(r, c) is not None:
                        continue  # Skip occupied squares
                        
                    # Check if this drop is legal
                    if self.can_drop_piece(piece_type, r, c, self.current_player):
                        # Create drop string notation
                        piece_type_names = ["pawn", "lance", "knight", "silver", "gold", "bishop", "rook"]
                        piece_color_names = ["black", "white"]
                        drop_str = f"drop_{piece_type_names[piece_type]}_{piece_color_names[self.current_player]}"
                        
                        # Create drop move tuple
                        move_tuple = (None, None, r, c, drop_str)
                        
                        # Simulate drop to check if it puts player in check
                        dropped_piece = Piece(piece_type, self.current_player, False)
                        self.set_piece(r, c, dropped_piece)
                        in_check = self._king_in_check_after_move(self.current_player)
                        # Undo drop
                        self.set_piece(r, c, None)
                        
                        if not in_check:
                            legal_moves.append(move_tuple)
        
        return legal_moves

    def _king_in_check_after_move(self, color):
        # Helper: after a move, is the given color's king in check?
        king_pos = None
        for r in range(9):
            for c in range(9):
                p = self.get_piece(r, c)
                if p and p.type == 7 and p.color == color:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if not king_pos:
            return True  # No king found, treat as in check
        return self._is_square_attacked(king_pos[0], king_pos[1], 1 - color)

    def _board_state_hash(self):
        """
        Returns a hashable representation of the current board, hands, and player to move.
        """
        board_tuple = tuple(
            tuple((p.type, p.color, p.is_promoted) if p else None for p in row)
            for row in self.board
        )
        # Include hands in the hash
        hands_tuple = (
            tuple((piece_type, count) for piece_type, count in sorted(self.hands[0].items())),
            tuple((piece_type, count) for piece_type, count in sorted(self.hands[1].items()))
        )
        return (board_tuple, hands_tuple, self.current_player)

    def is_sennichite(self) -> bool:
        """
        Returns True if the current board state has occurred four times (Sennichite).
        """
        current_state_hash = self._board_state_hash()
        
        # Count occurrences of the current state in the move history
        state_counts: dict[Any, int] = {}
        
        # Count the occurrences of each state in the move history
        for move in self.move_history:
            state = move.get("state_hash")
            if state:
                if state in state_counts:
                    state_counts[state] += 1
                else:
                    state_counts[state] = 1
        
        # Check if the current state has occurred 4 or more times
        if current_state_hash in state_counts and state_counts[current_state_hash] >= 3:
            # Add 1 for the current state which isn't in the history yet
            return True
        
        return False

    def make_move(self, move_tuple):
        """
        Make a move and update the game state. Handles board moves, captures,
        drops, and promotions. Records board state for Sennichite.
        
        move_tuple format:
        - Board move: (r_from, c_from, r_to, c_to, promote_flag)
        - Drop move: (None, None, r_to, c_to, piece_type_and_drop_info)
        
        Where promote_flag is 0 (no promotion) or 1 (promote).
        For drops, piece_type_and_drop_info is a string like "drop_pawn_black".
        """
        r_from, c_from, r_to, c_to, move_info = move_tuple
        
        # Handle drops (when r_from and c_from are None)
        if r_from is None and c_from is None and isinstance(move_info, str) and "drop_" in move_info:
            # Parse drop info (e.g., "drop_pawn_black")
            parts = move_info.split("_")
            piece_type_name = parts[1]
            piece_color_name = parts[2]
            
            # Convert piece type name to integer
            piece_type_map = {"pawn": 0, "lance": 1, "knight": 2, "silver": 3, 
                              "gold": 4, "bishop": 5, "rook": 6}
            piece_type = piece_type_map.get(piece_type_name, 0)
            
            # Convert color name to integer
            color = 0 if piece_color_name == "black" else 1
            
            # Check if player has this piece in hand
            if self.hands[color][piece_type] > 0:
                # Remove piece from hand
                self.remove_from_hand(piece_type, color)
                
                # Add piece to board
                dropped_piece = Piece(piece_type, color, False)
                self.set_piece(r_to, c_to, dropped_piece)
                
                # Record move for undo
                state_hash = self._board_state_hash()
                self.move_history.append({
                    "move": move_tuple, 
                    "captured": None, 
                    "state_hash": state_hash,
                    "is_drop": True,
                    "piece_type": piece_type
                })
                
                # Update state
                self.move_count += 1
                self.current_player = 1 - self.current_player
                
                # Check for Sennichite
                if self.is_sennichite():
                    self.game_over = True
                    self.winner = None  # Draw
                
                return
        
        # Handle board moves
        moving_piece = self.get_piece(r_from, c_from)
        captured_piece = self.get_piece(r_to, c_to)
        
        # Capture logic - add captured piece to hand if there is one
        if captured_piece:
            self.add_to_hand(captured_piece, self.current_player)
        
        # Promotion logic
        promote = bool(move_info) if isinstance(move_info, int) else False
        if promote and moving_piece:
            moving_piece.is_promoted = True
        
        # Move the piece
        self.set_piece(r_to, c_to, moving_piece)
        self.set_piece(r_from, c_from, None)
        
        # Record move for undo
        state_hash = self._board_state_hash()
        self.move_history.append({
            "move": move_tuple, 
            "captured": captured_piece, 
            "state_hash": state_hash,
            "is_drop": False,
            "was_promoted": promote
        })
        
        # Update state
        self.move_count += 1
        self.current_player = 1 - self.current_player
        
        # Check for Sennichite - set game state appropriately
        if self.is_sennichite():
            self.game_over = True
            self.winner = None  # Draw

    def undo_move(self):
        """
        Undo the last move (for search or testing). Supports board moves, captures, drops, and promotions.
        """
        if not self.move_history:
            raise RuntimeError("No move to undo")
        
        last = self.move_history.pop()
        move_tuple = last["move"]
        captured_piece = last["captured"]
        r_from, c_from, r_to, c_to, _ = move_tuple
        
        # Handle drops
        if last.get("is_drop", False):
            # Get the piece from the board
            dropped_piece = self.get_piece(r_to, c_to)
            
            if dropped_piece:
                # Add piece back to hand
                piece_type = last.get("piece_type", 0)
                self.hands[self.current_player][piece_type] += 1
                
                # Remove from board
                self.set_piece(r_to, c_to, None)
        
        # Handle board moves
        else:
            moving_piece = self.get_piece(r_to, c_to)
            
            # Handle promotion undo
            if last.get("was_promoted", False) and moving_piece:
                moving_piece.is_promoted = False
            
            # Move piece back
            self.set_piece(r_from, c_from, moving_piece)
            self.set_piece(r_to, c_to, captured_piece)
            
            # If there was a capture, remove from hand
            if captured_piece:
                base_type = captured_piece.type % 7
                self.hands[1 - self.current_player][base_type] -= 1
        
        # Update game state
        self.current_player = 1 - self.current_player
        self.move_count -= 1
        self.game_over = False
        self.winner = None

    def is_in_check(self, player_color_int):
        """
        Return True if the given player's king is in check.
        """
        # Find the king's position
        king_pos = None
        for r in range(9):
            for c in range(9):
                p = self.get_piece(r, c)
                if p and p.type == 7 and p.color == player_color_int:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if not king_pos:
            return True  # No king found, treat as in check
        # Is the king's square attacked by the opponent?
        return self._is_square_attacked(king_pos[0], king_pos[1], 1 - player_color_int)

    def sfen_encode_move(self, move_tuple):
        """
        Convert a move tuple to SFEN/USI string for logging.
        """
        raise NotImplementedError("sfen_encode_move not yet implemented")

    def add_to_hand(self, piece: Piece, color: int) -> None:
        """
        Add a captured piece to the hand of the given color.
        Ensures the piece is unpromoted as per Shogi rules.
        """
        if not piece:
            return
            
        # Get the base piece type (unpromoted)
        base_type = piece.type % 7  # Convert promoted types (8-13) to base types (1-6)
        
        # Kings (type 7) should not be captured, but for safety:
        if base_type == 7:
            return
            
        # Add to the capturing player's hand
        self.hands[color][base_type] += 1
        
    def remove_from_hand(self, piece_type: int, color: int) -> bool:
        """
        Remove a piece of the given type from the hand of the given color.
        Returns True if successful, False if no such piece in hand.
        """
        # Ensure we're using the base piece type (0-6)
        base_type = piece_type % 7
        
        # Kings (type 7) are not in hand
        if base_type == 7:
            return False
            
        # Check if the piece is in hand
        if self.hands[color][base_type] <= 0:
            return False
            
        # Remove from hand
        self.hands[color][base_type] -= 1
        return True
        
    def get_pieces_in_hand(self, color: int) -> dict:
        """
        Returns a dictionary of piece types and counts in the given player's hand.
        """
        return self.hands[color].copy()
        
    def is_in_promotion_zone(self, row: int, col: int, color: int) -> bool:
        """
        Returns True if the position (row, col) is in the promotion zone for the given color.
        For Black (color 0), promotion zone is rows 0-2.
        For White (color 1), promotion zone is rows 6-8.
        """
        # col is unused but kept for the function signature
        if color == 0:  # Black
            return 0 <= row <= 2
        else:  # White
            return 6 <= row <= 8

    def can_drop_piece(self, piece_type: int, row: int, col: int, color: int) -> bool:
        """
        Check if a piece of the given type can be legally dropped at (row, col) by the player of given color.
        """
        # Check if the square is empty
        if self.get_piece(row, col) is not None:
            return False
        
        # Check if the player has the piece in hand
        if self.hands[color][piece_type] <= 0:
            return False
        
        # Pawn-specific checks
        if piece_type == 0:  # Pawn
            # Check for Nifu (two unpromoted pawns on the same file)
            if self.is_nifu(color, col):
                return False
            
            # Check for drop on the last rank (where pawn can't move)
            if (color == 0 and row == 0) or (color == 1 and row == 8):
                return False
            
            # Check for Uchi Fu Zume (pawn drop checkmate)
            if self.is_uchi_fu_zume(row, col, color):
                return False
                
        # Lance-specific checks
        if piece_type == 1:  # Lance
            # Check for drop on the last rank (where lance can't move)
            if (color == 0 and row == 0) or (color == 1 and row == 8):
                return False
                
        # Knight-specific checks
        if piece_type == 2:  # Knight
            # Check for drop on the last two ranks (where knight can't move)
            if (color == 0 and row <= 1) or (color == 1 and row >= 7):
                return False
        
        # All other pieces can be dropped anywhere
        return True
        
    def can_promote_piece(self, piece: Piece, r_from: int, c_from: int, r_to: int, c_to: int) -> bool:
        """
        Check if a piece can be promoted when moving from (r_from, c_from) to (r_to, c_to).
        """
        if not piece:
            return False
            
        # Kings and Gold Generals can't be promoted
        if piece.type == 4 or piece.type == 7:  # Gold or King
            return False
            
        # Already promoted pieces can't be promoted again
        if piece.is_promoted or piece.type >= 8:
            return False
            
        # Can only promote if move starts in, ends in, or crosses the promotion zone
        if self.is_in_promotion_zone(r_from, c_from, piece.color) or self.is_in_promotion_zone(r_to, c_to, piece.color):
            return True
            
        return False
        
    def must_promote_piece(self, piece: Piece, r_to: int, c_to: int) -> bool:
        """
        Check if a piece must be promoted on the given move due to being unable to move afterward.
        """
        # c_to is unused but kept for the function signature
        if not piece:
            return False
            
        # Already promoted pieces don't need forced promotion
        if piece.is_promoted or piece.type >= 8:
            return False
            
        # Pawn or Lance reaching the last rank must promote
        if (piece.type == 0 or piece.type == 1) and (
            (piece.color == 0 and r_to == 0) or (piece.color == 1 and r_to == 8)
        ):
            return True
            
        # Knight reaching the last two ranks must promote
        if piece.type == 2 and (
            (piece.color == 0 and r_to <= 1) or (piece.color == 1 and r_to >= 7)
        ):
            return True
            
        return False
