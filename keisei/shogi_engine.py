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
        # Placeholder: hands not yet implemented, so skip for now
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
        if self.get_piece(drop_row, drop_col) is not None:
            return False  # Should not happen if called after checking drop validity

        orig_piece_at_drop_sq = self.get_piece(drop_row, drop_col)  # Should be None
        dropped_pawn = Piece(0, color, False)
        self.set_piece(drop_row, drop_col, dropped_pawn)

        opp_color = 1 - color
        king_pos = None
        for r_king_search in range(9):
            for c_king_search in range(9):
                p = self.get_piece(r_king_search, c_king_search)
                if p and p.type == 7 and p.color == opp_color:  # King piece type is 7
                    king_pos = (r_king_search, c_king_search)
                    break
            if king_pos:
                break

        mate = False  # Initialize mate status

        if king_pos:
            # Check if the dropped pawn itself delivers check to the opponent's king
            in_check_by_dropped_pawn = False
            # Pawn attack for Black (color 0) is one step forward (row decreases)
            if color == 0 and king_pos[0] == drop_row - 1 and king_pos[1] == drop_col:
                in_check_by_dropped_pawn = True
            # Pawn attack for White (color 1) is one step forward (row increases)
            elif color == 1 and king_pos[0] == drop_row + 1 and king_pos[1] == drop_col:
                in_check_by_dropped_pawn = True

            if in_check_by_dropped_pawn:
                # King is in check by the dropped pawn. Now check if it's mate (king has no legal moves).
                can_king_escape = False
                king_piece_for_simulation = Piece(7, opp_color)  # Opponent's king

                for dr_king_move in [-1, 0, 1]:
                    for dc_king_move in [-1, 0, 1]:
                        if dr_king_move == 0 and dc_king_move == 0:
                            continue  # King must move

                        king_escape_r, king_escape_c = (
                            king_pos[0] + dr_king_move,
                            king_pos[1] + dc_king_move,
                        )

                        if self.is_on_board(king_escape_r, king_escape_c):
                            piece_at_escape_sq = self.get_piece(
                                king_escape_r, king_escape_c
                            )

                            # King can move to an empty square or capture an opponent's piece
                            # (same color as 'color' who dropped the pawn)
                            if (
                                not piece_at_escape_sq
                                or piece_at_escape_sq.color == color
                            ):
                                # Simulate king moving to the escape square
                                original_piece_at_king_pos = self.get_piece(
                                    king_pos[0], king_pos[1]
                                )  # Should be the king
                                self.set_piece(
                                    king_pos[0], king_pos[1], None
                                )  # Remove king from original square
                                self.set_piece(
                                    king_escape_r,
                                    king_escape_c,
                                    king_piece_for_simulation,
                                )  # Place king on escape square

                                # Is the new king square attacked by 'color' (the player who dropped the pawn)?
                                is_escape_square_attacked = self._is_square_attacked(
                                    king_escape_r, king_escape_c, color
                                )

                                # Restore board to state before king's hypothetical move (pawn is still dropped)
                                self.set_piece(
                                    king_pos[0], king_pos[1], original_piece_at_king_pos
                                )  # Restore king to original pos
                                self.set_piece(
                                    king_escape_r, king_escape_c, piece_at_escape_sq
                                )  # Restore original piece at escape square

                                if not is_escape_square_attacked:
                                    can_king_escape = True
                                    break  # Found an escape, break from dc_king_move loop
                    if can_king_escape:
                        break  # Break from dr_king_move loop as well

                if not can_king_escape:
                    mate = True
                else:  # can_king_escape is True
                    mate = False  # Already false, but for clarity
            else:  # Not in check by the dropped pawn
                mate = False
        else:  # King not found
            mate = False

        # Restore the board to its state before the pawn drop simulation
        self.set_piece(drop_row, drop_col, orig_piece_at_drop_sq)
        return mate

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
                        if _p_attacker_type == 0: # Pawn
                            is_sliding_piece = False
                        elif _p_attacker_type == 1: # Lance
                            is_sliding_piece = True
                        elif _p_attacker_type == 2: # Knight
                            is_sliding_piece = False
                        elif _p_attacker_type == 3: # Silver
                            is_sliding_piece = False
                        elif _p_attacker_type == 4: # Gold
                            is_sliding_piece = False
                        elif _p_attacker_type == 5: # Bishop
                            is_sliding_piece = True
                        elif _p_attacker_type == 6: # Rook
                            is_sliding_piece = True
                        elif _p_attacker_type == 7: # King
                            is_sliding_piece = False
                        # Promoted pieces
                        elif _p_attacker_type == 8: # +P (moves like Gold)
                            is_sliding_piece = False
                        elif _p_attacker_type == 9: # +L (moves like Gold)
                            is_sliding_piece = False
                        elif _p_attacker_type == 10: # +N (moves like Gold)
                            is_sliding_piece = False
                        elif _p_attacker_type == 11: # +S (moves like Gold)
                            is_sliding_piece = False
                        elif _p_attacker_type == 12: # +B (Bishop + King moves)
                            is_sliding_piece = True
                        elif _p_attacker_type == 13: # +R (Rook + King moves)
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
                            if abs(r_attacker - row) > 1 or abs(c_attacker - col) > 1 or \
                               (abs(r_attacker - row) == 1 and abs(c_attacker - col) > 1) or \
                               (abs(r_attacker - row) > 1 and abs(c_attacker - col) == 1) or \
                               (dr == 0 and dc == 0 and (r_attacker != row or c_attacker !=col )):
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
                                    if not self.is_on_board(curr_r, curr_c) and (curr_r, curr_c) != (row, col):
                                        path_clear = False # Went off board before reaching target
                                        break


                            if path_clear:
                                return True # If path is clear for a sliding piece, it's an attack.
                            # else: path not clear, continue to the next attacker in the outer loop.
        return False

    def get_legal_moves(self):
        """
        Generate all legal moves for the current player. Returns a list of move tuples.
        Only board moves (no drops) and no advanced rules for this first step.
        """
        legal_moves = []
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
                    # For now, ignore promotion and drops
                    move_tuple = (r_from, c_from, r_to, c_to, 0)
                    # Simulate move
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

    def make_move(self, move_tuple):
        """
        Make a move (board move only) and update the game state.
        """
        r_from, c_from, r_to, c_to, _ = move_tuple
        moving_piece = self.get_piece(r_from, c_from)
        captured_piece = self.get_piece(r_to, c_to)
        # Move the piece
        self.set_piece(r_to, c_to, moving_piece)
        self.set_piece(r_from, c_from, None)
        # No hand/capture logic yet
        self.move_count += 1
        self.current_player = 1 - self.current_player
        # No game over/checkmate logic yet
        # Record move for undo (minimal)
        self.move_history.append((move_tuple, captured_piece))

    def undo_move(self):
        """
        Undo the last move (for search or testing). Only supports board moves (no drops/captures yet).
        """
        if not self.move_history:
            raise RuntimeError("No move to undo")
        move_tuple, captured_piece = self.move_history.pop()
        r_from, c_from, r_to, c_to, _ = move_tuple
        moving_piece = self.get_piece(r_to, c_to)
        # Move piece back
        self.set_piece(r_from, c_from, moving_piece)
        self.set_piece(r_to, c_to, captured_piece)
        self.current_player = 1 - self.current_player
        self.move_count -= 1
        # No hand/capture logic yet
        # No game over/winner logic yet

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
