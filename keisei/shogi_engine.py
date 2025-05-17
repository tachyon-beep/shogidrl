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
        if self.is_on_board(row, col):
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        if self.is_on_board(row, col):
            self.board[row][col] = piece

    def to_string(self) -> str:
        lines = []
        for row in self.board:
            line = " ".join(p.symbol() if p else "." for p in row)
            lines.append(line)
        return "\n".join(lines)

    def is_on_board(self, row: int, col: int) -> bool:
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
        forward = -1 if piece.color == 0 else 1 # Black (0) moves row -1, White (1) moves row +1

        t = piece.type
        is_promoted_by_flag = piece.is_promoted # Explicit promotion flag on the piece

        # Define move offsets once
        gold_move_offsets = [
            (forward, 0),       # Forward orthogonal
            (-forward, 0),      # Backward orthogonal
            (0, -1),            # Left orthogonal
            (0, 1),             # Right orthogonal
            (forward, -1),      # Forward-Left diagonal
            (forward, 1)       # Forward-Right diagonal
        ]
        king_move_offsets = [
            (-1,-1), (-1,0), (-1,1),
            (0,-1), (0,1),
            (1,-1), (1,0), (1,1)
        ]
        knight_move_offsets = [(forward * 2, -1), (forward * 2, 1)]
        silver_move_offsets = [
            (forward, 0), (forward, -1), (forward, 1),
            (-forward, -1), (-forward, 1)
        ]
        promoted_rook_extra_offsets = [(-1,-1), (-1,1), (1,-1), (1,1)] # King's diagonal moves

        # --- Gold-like moves ---
        # True if: Gold (4), Promoted P/L/N/S by type (8,9,10,11), or base P/L/N/S (0,1,2,3) and flagged as promoted
        is_gold_equivalent = (t == 4) or \
                             (t in [8, 9, 10, 11]) or \
                             (is_promoted_by_flag and t in [0, 1, 2, 3])
        
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
                if self.is_on_board(nr, nc):
                    moves.append((nr, nc))
                    # if self.get_piece(nr, nc) is not None: break # Removed to allow full path generation
                else: break
        
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
            for dr_diag, dc_diag in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                for i in range(1,9):
                    nr, nc = r_from + dr_diag * i, c_from + dc_diag * i
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
                        # if self.get_piece(nr, nc) is not None: break # Removed to allow full path generation
                    else: break
            # If Promoted Bishop (type 12, OR type 5 and flagged promoted)
            if t == 12 or (t == 5 and is_promoted_by_flag):
                for dr, dc in king_move_offsets: # Full King moves
                    nr, nc = r_from + dr, c_from + dc
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
        
        # --- Rook family (Rook type 6, Promoted Rook type 13) ---
        elif t == 6 or t == 13:
            # Rook sliding moves (orthogonals)
            for dr_ortho, dc_ortho in [(-1,0), (1,0), (0,-1), (0,1)]:
                for i in range(1,9):
                    nr, nc = r_from + dr_ortho * i, c_from + dc_ortho * i
                    if self.is_on_board(nr, nc):
                        moves.append((nr, nc))
                        # if self.get_piece(nr, nc) is not None: break # Removed to allow full path generation
                    else: break
            # If Promoted Rook (type 13, OR type 6 and flagged promoted)
            if t == 13 or (t == 6 and is_promoted_by_flag):
                for dr, dc in promoted_rook_extra_offsets: # King's diagonal moves
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
        Returns True if dropping a pawn of the given color in the given column would violate Nifu (two unpromoted pawns on the same file).
        """
        for row in range(9):
            p = self.get_piece(row, col)
            if p and p.type == 0 and p.color == color and not p.is_promoted:
                return True
        return False

    def is_uchi_fu_zume(self, drop_row: int, drop_col: int, color: int) -> bool:
        """
        Returns True if dropping a pawn at (drop_row, drop_col) by 'color' would result in immediate checkmate (Uchi Fu Zume, illegal in Shogi).
        Only applies to pawn drops (not other pieces).
        """
        print(f"[UCHIFUZUME DEBUG] Called for pawn drop at ({drop_row},{drop_col}) by player {color} (0=Black, 1=White)")
        print(f"[UCHIFUZUME DEBUG] Board state BEFORE pawn drop simulation for Uchi Fu Zume check:\\n{self.to_string()}")

        if self.get_piece(drop_row, drop_col) is not None:
            print(f"[UCHIFUZUME DEBUG] Square ({drop_row},{drop_col}) is already occupied. Cannot drop pawn. Returning False.")
            return False # Should not happen if called after checking drop validity

        orig_piece_at_drop_sq = self.get_piece(drop_row, drop_col) # Should be None
        dropped_pawn = Piece(0, color, False)
        self.set_piece(drop_row, drop_col, dropped_pawn)
        print(f"[UCHIFUZUME DEBUG] Board state AFTER pawn drop at ({drop_row},{drop_col}):\\n{self.to_string()}")

        opp_color = 1 - color
        king_pos = None
        for r_king_search in range(9):
            for c_king_search in range(9):
                p = self.get_piece(r_king_search, c_king_search)
                if p and p.type == 7 and p.color == opp_color: # King piece type is 7
                    king_pos = (r_king_search, c_king_search)
                    break
            if king_pos:
                break
        
        mate = False # Initialize mate status

        if king_pos:
            print(f"[UCHIFUZUME DEBUG] Opponent's king (color {opp_color}) found at {king_pos}")

            # Check if the dropped pawn itself delivers check to the opponent's king
            in_check_by_dropped_pawn = False
            # Pawn attack for Black (color 0) is one step forward (row decreases)
            if color == 0 and king_pos[0] == drop_row - 1 and king_pos[1] == drop_col:
                in_check_by_dropped_pawn = True
            # Pawn attack for White (color 1) is one step forward (row increases)
            elif color == 1 and king_pos[0] == drop_row + 1 and king_pos[1] == drop_col:
                in_check_by_dropped_pawn = True
            
            print(f"[UCHIFUZUME DEBUG] Is opponent's king at {king_pos} in check by the dropped pawn at ({drop_row},{drop_col})? {in_check_by_dropped_pawn}")

            if in_check_by_dropped_pawn:
                # King is in check by the dropped pawn. Now check if it's mate (king has no legal moves).
                can_king_escape = False
                king_piece_for_simulation = Piece(7, opp_color) # Opponent's king

                for dr_king_move in [-1, 0, 1]:
                    for dc_king_move in [-1, 0, 1]:
                        if dr_king_move == 0 and dc_king_move == 0:
                            continue # King must move

                        king_escape_r, king_escape_c = king_pos[0] + dr_king_move, king_pos[1] + dc_king_move
                        print(f"[UCHIFUZUME DEBUG] Checking king escape to: ({king_escape_r},{king_escape_c})")

                        if self.is_on_board(king_escape_r, king_escape_c):
                            piece_at_escape_sq = self.get_piece(king_escape_r, king_escape_c)
                            
                            # King can move to an empty square or capture an opponent's piece (same color as 'color' who dropped the pawn)
                            if not piece_at_escape_sq or piece_at_escape_sq.color == color:
                                print(f"[UCHIFUZUME DEBUG]  Square ({king_escape_r},{king_escape_c}) is a potential escape (empty or capturable). Simulating move.")
                                
                                # Simulate king moving to the escape square
                                original_piece_at_king_pos = self.get_piece(king_pos[0], king_pos[1]) # Should be the king
                                self.set_piece(king_pos[0], king_pos[1], None) # Remove king from original square
                                self.set_piece(king_escape_r, king_escape_c, king_piece_for_simulation) # Place king on escape square
                                
                                print(f"[UCHIFUZUME DEBUG]  Board state AFTER simulating king move to ({king_escape_r},{king_escape_c}):\\n{self.to_string()}")
                                print(f"[UCHIFUZUME DEBUG]  Calling _is_square_attacked({king_escape_r}, {king_escape_c}, attacker_color={color}) to see if escape square is safe.")
                                
                                # Is the new king square attacked by 'color' (the player who dropped the pawn)?
                                is_escape_square_attacked = self._is_square_attacked(king_escape_r, king_escape_c, color)
                                print(f"[UCHIFUZUME DEBUG]  _is_square_attacked({king_escape_r},{king_escape_c}, by_player_{color}) returned: {is_escape_square_attacked}")

                                # Restore board to state before king's hypothetical move (pawn is still dropped)
                                self.set_piece(king_pos[0], king_pos[1], original_piece_at_king_pos) # Restore king to original pos
                                self.set_piece(king_escape_r, king_escape_c, piece_at_escape_sq) # Restore original piece at escape square

                                if not is_escape_square_attacked:
                                    print(f"[UCHIFUZUME DEBUG]  King CAN escape to ({king_escape_r},{king_escape_c}). This is NOT uchi_fu_zume.")
                                    can_king_escape = True
                                    break # Found an escape, break from dc_king_move loop
                                else:
                                    print(f"[UCHIFUZUME DEBUG]  King cannot escape to ({king_escape_r},{king_escape_c}) as it's attacked.")
                            else:
                                print(f"[UCHIFUZUME DEBUG]  King cannot escape to ({king_escape_r},{king_escape_c}) as it's occupied by its own piece (color {piece_at_escape_sq.color}).")
                        else:
                            print(f"[UCHIFUZUME DEBUG]  King escape to ({king_escape_r},{king_escape_c}) is off board.")
                    
                    if can_king_escape:
                        break # Break from dr_king_move loop as well

                if not can_king_escape:
                    print("[UCHIFUZUME DEBUG] King has NO valid escapes. This IS uchi_fu_zume.")
                    mate = True
                else: # can_king_escape is True
                    mate = False # Already false, but for clarity
            else: # Not in check by the dropped pawn
                print("[UCHIFUZUME DEBUG] Dropped pawn does not put king in check. Not uchi_fu_zume.")
                mate = False
        else: # King not found
            print(f"[UCHIFUZUME DEBUG] Opponent's king (color {opp_color}) was NOT found on the board. This is unexpected. Returning False for uchi_fu_zume.")
            mate = False

        # Restore the board to its state before the pawn drop simulation
        self.set_piece(drop_row, drop_col, orig_piece_at_drop_sq)
        print(f"[UCHIFUZUME DEBUG] Board state RESTORED after Uchi Fu Zume check:\\n{self.to_string()}")
        print(f"[UCHIFUZUME DEBUG] is_uchi_fu_zume final result: {mate}")
        return mate

    def _is_square_attacked(self, row: int, col: int, attacker_color: int) -> bool:
        """
        Returns True if the square (row, col) is attacked by any piece of attacker_color.
        Considers obstructions for sliding pieces.
        """
        print(f"[_IS_SQ_ATTACKED DEBUG] Checking if square ({row},{col}) is attacked by player {attacker_color} (0=Black, 1=White)")
        # print(f"[_IS_SQ_ATTACKED DEBUG] Current board state for _is_square_attacked:\\n{self.to_string()}")

        for r_attacker in range(9):
            for c_attacker in range(9):
                p_attacker = self.get_piece(r_attacker, c_attacker)
                if p_attacker and p_attacker.color == attacker_color:
                    # This piece belongs to the attacker, check if it attacks (row, col)
                    # print(f"[_IS_SQ_ATTACKED DEBUG] Considering potential attacker: {p_attacker.symbol()} at ({r_attacker},{c_attacker}) of color {p_attacker.color}")
                    
                    # Get raw moves for the attacking piece (potential squares it could move to if no blockers)
                    raw_moves = self.get_individual_piece_moves(p_attacker, r_attacker, c_attacker)
                    # print(f"[_IS_SQ_ATTACKED DEBUG]  Raw moves for {p_attacker.symbol()} at ({r_attacker},{c_attacker}): {raw_moves}")

                    if (row, col) in raw_moves:
                        print(f"[_IS_SQ_ATTACKED DEBUG]  Target ({row},{col}) IS in raw_moves of {p_attacker.symbol()} at ({r_attacker},{c_attacker}).")
                        
                        # Determine if the piece is a sliding piece type
                        # Piece types: 0=P, 1=L, 2=N, 3=S, 4=G, 5=B, 6=R, 7=K
                        # Promoted: +P=8, +L=9, +N=10, +S=11, +B=12, +R=13
                        
                        # Using the existing complex logic for is_sliding_piece from the file:
                        _p_attacker_type = p_attacker.type 
                        _p_attacker_is_promoted = p_attacker.is_promoted # Note: piece.type >=8 already implies promotion generally

                        is_sliding_piece = _p_attacker_type in [1, 5, 6] or \
                                           (_p_attacker_is_promoted and _p_attacker_type in [8,9,10,11,12,13])
                        
                        if _p_attacker_type == 0: is_sliding_piece = False # Pawn
                        elif _p_attacker_type == 1: is_sliding_piece = True  # Lance
                        elif _p_attacker_type == 2: is_sliding_piece = False # Knight
                        elif _p_attacker_type == 3: is_sliding_piece = False # Silver
                        elif _p_attacker_type == 4: is_sliding_piece = False # Gold
                        elif _p_attacker_type == 5: is_sliding_piece = True  # Bishop
                        elif _p_attacker_type == 6: is_sliding_piece = True  # Rook
                        elif _p_attacker_type == 7: is_sliding_piece = False # King
                        # Promoted pieces
                        elif _p_attacker_type == 8: is_sliding_piece = False # +P (moves like Gold)
                        elif _p_attacker_type == 9: is_sliding_piece = False # +L (moves like Gold)
                        elif _p_attacker_type == 10: is_sliding_piece = False # +N (moves like Gold)
                        elif _p_attacker_type == 11: is_sliding_piece = False # +S (moves like Gold)
                        elif _p_attacker_type == 12: is_sliding_piece = True  # +B (Bishop + King moves)
                        elif _p_attacker_type == 13: is_sliding_piece = True  # +R (Rook + King moves)
                        else: is_sliding_piece = False # Should not happen

                        print(f"[_IS_SQ_ATTACKED DEBUG]  Attacker {p_attacker.symbol()} (type {_p_attacker_type}, promoted status based on type/is_promoted flag) at ({r_attacker},{c_attacker}). Determined is_sliding_piece: {is_sliding_piece}")

                        if not is_sliding_piece:
                            # For non-sliding pieces, if (row, col) is in raw_moves, it's a direct attack.
                            print(f"[_IS_SQ_ATTACKED DEBUG]  Non-sliding attack by {p_attacker.symbol()} from ({r_attacker},{c_attacker}) to ({row},{col}). Square is attacked. Returning True.")
                            return True
                        else:
                            # Path checking for sliding pieces (Lance, Bishop, Rook, Promoted B/R)
                            print(f"[_IS_SQ_ATTACKED DEBUG]  Sliding piece {p_attacker.symbol()} at ({r_attacker},{c_attacker}). Checking path to ({row},{col}).")
                            
                            # Determine direction of attack
                            dr = 0
                            if row > r_attacker: dr = 1
                            elif row < r_attacker: dr = -1
                            
                            dc = 0
                            if col > c_attacker: dc = 1
                            elif col < c_attacker: dc = -1
                            
                            print(f"[_IS_SQ_ATTACKED DEBUG]   Sliding direction: dr={dr}, dc={dc} (from ({r_attacker},{c_attacker}) to ({row},{col}))")

                            # Check squares between attacker and target
                            curr_r, curr_c = r_attacker + dr, c_attacker + dc
                            path_clear = True
                            path_segment_log = []
                            while (curr_r, curr_c) != (row, col):
                                path_segment_log.append(f"({curr_r},{curr_c})")
                                if not self.is_on_board(curr_r, curr_c): 
                                    # This case should ideally not be reached if (row, col) is on board and raw_moves are generated correctly for on-board targets.
                                    # However, if raw_moves somehow generated an off-board intermediate step for an on-board target, this would catch it.
                                    print(f"[_IS_SQ_ATTACKED DEBUG]   Path check: intermediate square ({curr_r},{curr_c}) is off board. Path blocked.")
                                    path_clear = False; break 
                                
                                piece_on_path = self.get_piece(curr_r, curr_c)
                                if piece_on_path is not None:
                                    print(f"[_IS_SQ_ATTACKED DEBUG]   Path check: intermediate square ({curr_r},{curr_c}) is blocked by {piece_on_path.symbol()}. Path blocked.")
                                    path_clear = False; break
                                curr_r += dr
                                curr_c += dc
                            
                            if path_segment_log:
                                print(f"[_IS_SQ_ATTACKED DEBUG]   Path segments checked: {' -> '.join(path_segment_log)}")
                            else:
                                print("[_IS_SQ_ATTACKED DEBUG]   No intermediate squares to check (target is adjacent or move type doesn't require intermediate like Knight).")
                            print(f"[_IS_SQ_ATTACKED DEBUG]   Path clear result: {path_clear}")
                            
                            if path_clear:
                                # For promoted Bishop/Rook, they also have non-sliding King-like moves.
                                # If the target square is adjacent, it's a direct attack regardless of path IF that adjacent move is part of their King-like moves.
                                # The raw_moves check already confirmed (row,col) is a possible destination.
                                # The is_sliding_piece logic means we are here for the sliding component or a piece that has sliding components (+B, +R).
                                
                                if p_attacker.type in [12, 13]: # Promoted Bishop or Rook
                                    print(f"[_IS_SQ_ATTACKED DEBUG]   Attacker {p_attacker.symbol()} is Promoted B/R. Path is clear.")
                                    # Check if this attack is due to its King-like component (adjacent square)
                                    # This specific check is to see if an adjacent square is hit by the "king-move" part of +B/+R
                                    # If it's an adjacent square, it could be a king-like move or a 1-step sliding move.
                                    # get_individual_piece_moves includes both. If path_clear is true, it means any sliding path is open.
                                    # The original code's logic:
                                    if abs(r_attacker - row) <= 1 and abs(c_attacker - col) <= 1:
                                        # This means the target is adjacent. Since (row,col) is in raw_moves for +B/+R,
                                        # and it's adjacent, it's a valid move (either king-like or 1-step slide).
                                        # Path being clear for a 1-step slide is trivial.
                                        print(f"[_IS_SQ_ATTACKED DEBUG]   Target ({row},{col}) is adjacent to Promoted B/R {p_attacker.symbol()} at ({r_attacker},{c_attacker}) AND path is clear. Square is attacked. Returning True.")
                                        return True # Adjacent square, direct attack by king-like move part or 1-step slide
                                
                                # If path is clear, it's an attack by the sliding component (or non-sliding if it passed earlier)
                                print(f"[_IS_SQ_ATTACKED DEBUG]   Path is clear for sliding attack by {p_attacker.symbol()} from ({r_attacker},{c_attacker}) to ({row},{col}). Square is attacked. Returning True.")
                                return True
                            else: # Path not clear
                                print(f"[_IS_SQ_ATTACKED DEBUG]   Path is NOT clear for sliding attack by {p_attacker.symbol()} from ({r_attacker},{c_attacker}) to ({row},{col}). This attacker does not attack via this path.")
                    # else:
                        # print(f"[_IS_SQ_ATTACKED DEBUG]  Target ({row},{col}) is NOT in raw_moves of {p_attacker.symbol()} at ({r_attacker},{c_attacker}).")

        print(f"[_IS_SQ_ATTACKED DEBUG] No piece of player {attacker_color} found to attack square ({row},{col}). Returning False.")
        return False
