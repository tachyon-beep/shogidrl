from typing import List, Set

import torch

from keisei.shogi.shogi_core_definitions import Color, MoveTuple, PieceType
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper

# Removed unused pytest import
# Removed unused numpy import


# Helper to create a device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLegalMaskGeneration:  # pylint: disable=too-many-public-methods
    """Test suite for legal mask generation."""

    def test_initial_position_legal_mask(self):  # pylint: disable=too-many-statements
        """Tests the legal move mask for the initial game position."""
        game = ShogiGame()
        mapper = PolicyOutputMapper()

        # Standard Shogi game has 13527 possible move encodings in this mapper
        assert mapper.get_total_actions() == 13527

        legal_moves_tuples: List[MoveTuple] = game.get_legal_moves()

        # Black has 30 legal moves in the initial standard Shogi position.
        assert (
            len(legal_moves_tuples) == 30
        ), f"Expected 30 legal moves, got {len(legal_moves_tuples)}. Moves: {legal_moves_tuples}"
        legal_mask = mapper.get_legal_mask(legal_moves_tuples, device=DEVICE)
        assert (
            legal_mask.sum().item() == 30
        ), f"Expected mask sum 30, got {legal_mask.sum().item()}"

    def test_king_in_check_mask(self):  # pylint: disable=too-many-statements
        """Tests the legal move mask when the king is in check."""
        # SFEN: Black king e1 (my (8,4)), White Rook e8 (my (1,4)), White King e9 (my (0,4)). Black to move.
        # White Rook 'r' on e8 (board (1,4)) checks Black King 'K' on e1 (board (8,4)).
        # White King 'k' on e9 (board (0,4)).
        sfen_check = "4k4/4r4/9/9/9/9/9/9/4K4 b - 1"
        game = ShogiGame.from_sfen(sfen_check)
        mapper = PolicyOutputMapper()  # Moved mapper instantiation here
        assert game.current_player == Color.BLACK
        assert game.is_in_check(Color.BLACK) is True

        legal_moves_tuples = game.get_legal_moves()
        legal_mask = mapper.get_legal_mask(legal_moves_tuples, device=DEVICE)

        # Expected legal moves for Black King at (8,4) checked by White Rook at (1,4):
        # King can move to (8,3) [d1], (8,5) [f1], (7,3) [d2], (7,5) [f2]. (4 moves)
        # Moving to (7,4) [e2] is ILLEGAL as it's on the same file as the attacking rook.
        expected_king_moves_in_check: Set[MoveTuple] = {
            (8, 4, 8, 3, False),
            (8, 4, 8, 5, False),  # K to d1, f1
            (8, 4, 7, 3, False),
            (8, 4, 7, 5, False),  # K to d2, f2
        }
        assert (
            len(legal_moves_tuples) == 4
        ), f"Expected 4 legal moves, got {len(legal_moves_tuples)}. Moves: {legal_moves_tuples}"
        assert legal_mask.sum().item() == 4

        # Verify that these specific moves are in the mask
        for move in expected_king_moves_in_check:
            idx = mapper.shogi_move_to_policy_index(move)
            assert (
                legal_mask[idx].item() is True
            ), f"Expected move {move} to be legal, but it's not in the mask."

        # Verify that moving into check (e.g. K to e2 (7,4)) is not legal
        illegal_move_into_check: MoveTuple = (8, 4, 7, 4, False)  # K to e2
        idx_illegal = mapper.shogi_move_to_policy_index(illegal_move_into_check)
        assert (
            legal_mask[idx_illegal].item() is False
        ), "Move into check (K to e2) should be illegal."

    def test_checkmate_mask(self):  # pylint: disable=too-many-statements
        """Test that no moves are legal in a checkmate position."""
        # SFEN for a checkmate position (Black to move, Black is in checkmate)
        # Black King K at e1 (my (8,4)). White Rook r at e2 (my (7,4)). White Gold g at e3 (my (6,4)).
        sfen_checkmate = "9/9/9/9/9/4G4/4r4/4g4/4K4 b - 1"
        game = ShogiGame.from_sfen(sfen_checkmate)
        mapper = PolicyOutputMapper()

        assert game.current_player == Color.BLACK
        assert (
            game.is_in_check(Color.BLACK) is True
        ), "Black should be in check in this position."
        assert game.game_over is True, "Game should be over due to checkmate."

        legal_moves_tuples = game.get_legal_moves()
        assert (
            not legal_moves_tuples
        ), f"Expected no legal moves in checkmate, got {legal_moves_tuples}"

        legal_mask = mapper.get_legal_mask(legal_moves_tuples, device=DEVICE)
        assert legal_mask.sum().item() == 0, "Mask sum should be 0 in checkmate."

    def test_drop_moves_mask(
        self,
    ):  # pylint: disable=too-many-locals, too-many-statements # Disabled for now
        """Tests the legal move mask for positions involving drop moves."""
        mapper = PolicyOutputMapper()

        # Scenario 1: Black has a pawn in hand, board allows pawn drops
        # Black King at e9 (my (0,4)), White King at e1 (my (8,4)).
        # SFEN: Black King at e9 (4,0), White King at e1 (4,8)
        # My coordinates: Black King at (0,4), White King at (8,4)
        # The test description says: Black King at e5 (my (4,4)), White King at e1 (my (8,4)).
        # Let's use the description's king positions for clarity.
        # SFEN for Black King at (4,4) [e5], White King at (8,4) [e1]
        sfen_drop_scenario = "9/9/9/9/4K4/9/9/9/4k4 b P 1"
        game_black_can_drop = ShogiGame.from_sfen(sfen_drop_scenario)
        assert game_black_can_drop.current_player == Color.BLACK
        assert game_black_can_drop.hands[Color.BLACK.value].get(PieceType.PAWN, 0) == 1

        black_legal_moves = game_black_can_drop.get_legal_moves()
        legal_mask_black = mapper.get_legal_mask(black_legal_moves, device=DEVICE)

        # Expected moves for Black King at (4,4) [e5]:
        # (3,3), (3,4), (3,5)
        # (4,3),       (4,5)
        # (5,3), (5,4), (5,5)
        # Total 8 king moves.
        # Pawn drops:
        # 81 total squares.
        # 2 squares occupied by kings. -> 79 empty squares.
        # Black cannot drop a pawn on the last rank (row 0 for Black). There are 9 such squares.
        # However, these squares might also be occupied or be one of the 79 empty squares.
        # Empty squares not in the last rank (row 0):
        # Total empty squares = 79.
        # Empty squares in row 0:
        # King k is at (4,4). King K is at (8,4).
        # Row 0 is all empty. So 9 squares in row 0 are empty.
        # Valid pawn drop squares = Total empty squares - Empty squares in row 0 for pawn drops
        # = 79 - 9 = 70.
        # Total legal moves = 5 (king moves) + 70 (pawn drops) = 78.
        assert (
            len(black_legal_moves) == 78
        ), f"Expected 78 legal moves, got {len(black_legal_moves)}. Moves: {black_legal_moves}"
        assert (
            legal_mask_black.sum().item() == 78
        ), f"Expected mask sum 78, got {legal_mask_black.sum().item()}"

        # Verify one legal king move (e.g., Black King from (4,4) to (3,4))
        # Original SFEN: 9/9/9/9/4K4/9/9/9/4k4 b P 1 -> Black King at (4,4)
        king_move: MoveTuple = (4, 4, 3, 4, False)
        idx_king_move = mapper.shogi_move_to_policy_index(king_move)
        assert (
            legal_mask_black[idx_king_move].item() is True
        ), f"Legal king move {king_move} not in mask"

        # Verify one legal pawn drop (e.g., to (1,0) - not last rank)
        legal_pawn_drop: MoveTuple = (None, None, 1, 0, PieceType.PAWN)  # Drop to (1,0)
        idx_legal_pawn_drop = mapper.shogi_move_to_policy_index(legal_pawn_drop)
        assert (
            legal_mask_black[idx_legal_pawn_drop].item() is True
        ), f"Legal pawn drop {legal_pawn_drop} not in mask"

        # Verify an illegal pawn drop (on last rank for Black - row 0)
        illegal_pawn_drop_last_rank: MoveTuple = (
            None,
            None,
            0,
            0,
            PieceType.PAWN,
        )  # Drop to (0,0)
        idx_illegal_drop_last_rank = mapper.shogi_move_to_policy_index(
            illegal_pawn_drop_last_rank
        )
        assert (
            legal_mask_black[idx_illegal_drop_last_rank].item() is False
        ), "Illegal pawn drop (last rank) should not be in mask"

        # Verify an illegal pawn drop (on an occupied square - Black King at (4,4))
        illegal_pawn_drop_occupied: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
        idx_illegal_drop_occupied = mapper.shogi_move_to_policy_index(
            illegal_pawn_drop_occupied
        )
        assert (
            legal_mask_black[idx_illegal_drop_occupied].item() is False
        ), "Illegal pawn drop (occupied) should not be in mask"

        # Scenario 2: Nifu (two pawns on the same file)
        # Black pawn 'P' at (3,0) (SFEN: a6). Black king 'k' at (4,4) (SFEN: e5). White King 'K' at (8,4) (SFEN: e1).
        # Hands: Black has one Pawn 'P'.
        sfen_nifu_scenario = (
            "P8/9/9/9/4k4/9/9/9/4K4 b P 1"  # Black Pawn at (0,0) (SFEN: 1a)
        )
        game_nifu = ShogiGame.from_sfen(sfen_nifu_scenario)
        assert game_nifu.hands[Color.BLACK.value].get(PieceType.PAWN, 0) == 1

        # Attempt to drop another pawn on file 0 (column 0)
        # e.g., to (1,0) which was legal before, should now be illegal due to nifu.
        nifu_drop_attempt: MoveTuple = (
            None,
            None,
            1,
            0,
            PieceType.PAWN,
        )  # Drop to (1,0)

        nifu_legal_moves = game_nifu.get_legal_moves()
        nifu_mask = mapper.get_legal_mask(nifu_legal_moves, device=DEVICE)

        idx_nifu_drop = mapper.shogi_move_to_policy_index(nifu_drop_attempt)
        assert (
            nifu_mask[idx_nifu_drop].item() is False
        ), "Illegal pawn drop (Nifu) should not be in mask"

        # Check count for nifu scenario
        # King moves: 8
        # Pawn drops:
        # File 0 is now blocked for pawn drops due to existing pawn P at (0,0).
        # So, 8 files * (9 ranks - 1 last rank) = 8 * 8 = 64 potential drop squares.
        # Squares occupied by kings: (4,4) and (8,4). These are not on file 0.
        # Square (0,0) is occupied by a pawn.
        # Valid pawn drop squares = (Total empty squares - empty squares in row 0) - squares on file 0 (excluding row 0)
        # Total empty squares = 81 - 3 = 78.
        # Empty squares in row 0 (cannot drop): (0,1) to (0,8) -> 8 squares.
        # Valid drop squares if no nifu = 78 - 8 = 70.
        # Now, file 0 is disallowed.
        # Number of empty squares on file 0 not in row 0: (1,0) to (8,0) -> 8 squares.
        # King at (4,4) is not on file 0. King at (8,4) is not on file 0.
        # So, 70 - 8 (disallowed drops on file 0) = 62 pawn drops.
        # Total moves = 5 (king) + 62 (pawn drops) = 67.
        expected_nifu_moves = 67  # Corrected: Black King (K at 8,4) has 5 moves. Pawn drops are 62. 5 + 62 = 67.
        actual_nifu_moves = len(nifu_legal_moves)
        assert (
            actual_nifu_moves == expected_nifu_moves
        ), f"Nifu scenario: Expected {expected_nifu_moves} moves, got {actual_nifu_moves}. Moves: {nifu_legal_moves}"
        assert (
            nifu_mask.sum().item() == expected_nifu_moves
        ), f"Nifu scenario: Expected mask sum {expected_nifu_moves}, got {nifu_mask.sum().item()}"

    def test_promotion_mask(
        self,
    ):  # pylint: disable=too-many-statements, too-many-locals
        """Test that promotion and non-promotion are correctly represented in the mask."""
        # Position where a pawn can promote
        # Black pawn at (4,2) (e7), can move to (3,2) (e6) and promote or not.
        # Rows 0,1,2 are promotion zone for black.
        sfen_promote_option = "4k4/9/4p4/9/9/9/9/9/4K4 b - 1"  # Black Pawn at e7 (board (2,4) if SFEN e7 is 3rd rank from top)
        # My board: (0,0) is 9a. SFEN e7 is 3rd rank, 5th file.
        # SFEN ranks are 1-9 top to bottom. My rows 0-8 top to bottom.
        # SFEN files are a-i right to left. My cols 0-8 left to right (9-1).
        # '4p4' means rank 3 (my row 2) has a pawn at 'e' file (my col 4).
        # So, black pawn at (2,4) moving to (1,4).
        sfen_promote_option = (
            "4k4/9/4P4/9/9/9/9/9/4K4 b - 1"  # Black Pawn at (2,4) (SFEN e7)
        )
        game_promote = ShogiGame.from_sfen(sfen_promote_option)

        mapper = PolicyOutputMapper()
        legal_moves = game_promote.get_legal_moves()
        # print(f"DEBUG test_promotion_mask: Legal moves for {sfen_promote_option}: {legal_moves}")
        legal_mask = mapper.get_legal_mask(legal_moves, device=DEVICE)

        # Pawn at (2,4) can move to (1,4)
        move_no_promote: MoveTuple = (
            2,
            4,
            1,
            4,
            False,
        )  # Corrected: Pawn at (2,4) moves to (1,4)
        move_promote: MoveTuple = (
            2,
            4,
            1,
            4,
            True,
        )  # Corrected: Pawn at (2,4) moves to (1,4)

        idx_no_promote = mapper.shogi_move_to_policy_index(move_no_promote)
        idx_promote = mapper.shogi_move_to_policy_index(move_promote)

        assert (
            legal_mask[idx_no_promote].item() is True
        ), f"Non-promoting pawn move (2,4)->(1,4) not in mask. Moves: {legal_moves}"
        assert (
            legal_mask[idx_promote].item() is True
        ), f"Promoting pawn move (2,4)->(1,4) not in mask. Moves: {legal_moves}"

        # Ensure that an illegal promotion (e.g. king promoting) is not in the mask
        # King at (8,4) cannot promote by moving to (7,4)
        sfen_king_move = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        game_king = ShogiGame.from_sfen(sfen_king_move)
        king_legal_moves = game_king.get_legal_moves()
        king_mask = mapper.get_legal_mask(king_legal_moves, device=DEVICE)

        king_move_attempt_promote: MoveTuple = (
            8,
            4,
            7,
            4,
            True,
        )  # King e1-e2, attempt promote
        idx_king_promote = mapper.shogi_move_to_policy_index(king_move_attempt_promote)
        assert (
            king_mask[idx_king_promote].item() is False
        ), "Illegal king promotion found in mask"

    def test_specific_board_moves_are_legal(
        self,
    ):  # pylint: disable=too-many-statements
        """Test a few specific known-legal board moves."""
        game = ShogiGame()  # Initial position
        mapper = PolicyOutputMapper()
        legal_moves = game.get_legal_moves()
        legal_mask = mapper.get_legal_mask(legal_moves, device=DEVICE)

        # Pawn moves (e.g., 7g7f or (6,6) to (5,6))
        pawn_move: MoveTuple = (
            6,
            6,
            5,
            6,
            False,
        )  # Corrected: Pawn at (6,6) moves to (5,6)
        idx_pawn = mapper.shogi_move_to_policy_index(pawn_move)
        assert (
            legal_mask[idx_pawn].item() is True
        ), "Legal pawn move 7g7f (6,6)->(5,6) not in mask"

        # Rook move (e.g., 2h3h or (7,7) to (7,6)) - Standard Black Rook is at (7,7) (h8)
        # Moving from (7,7) to (7,6) is one step left.
        rook_move: MoveTuple = (7, 7, 7, 6, False)
        idx_rook = mapper.shogi_move_to_policy_index(rook_move)
        assert (
            legal_mask[idx_rook].item() is True
        ), "Legal rook move H8-G8 (7,7)->(7,6) not in mask"

        # Knight move (e.g., 2i3g or (8,1) to (6,2)) - Standard Black Knight is at (8,1) (b9)
        # Moving from (8,1) to (6,2) is illegal as (6,2) is occupied by a friendly pawn.
        knight_move: MoveTuple = (8, 1, 6, 2, False)  # N-8i to 7g (my (8,1) to (6,2))
        idx_knight = mapper.shogi_move_to_policy_index(knight_move)
        assert (
            legal_mask[idx_knight].item() is False
        ), "Illegal knight move N8i-7g (8,1)->(6,2) (blocked by own pawn) should not be in mask"

    def test_no_legal_moves_for_opponent_in_checkmate(
        self,
    ):  # pylint: disable=too-many-statements
        """Test that if Black checkmates White, White has no legal moves."""
        # SFEN: White King at a9 (my (0,8)), Black Rook at a1 (my (8,8)). White to move.
        # This is checkmate for White.
        sfen_white_checkmated = "K8/9/9/9/9/9/9/9/r8 w - 1"

        game = ShogiGame.from_sfen(sfen_white_checkmated)
        mapper = PolicyOutputMapper()

        assert game.current_player == Color.WHITE
        assert (
            game.is_in_check(Color.WHITE) is True
        ), "White should be in check in this position."
        assert game.game_over is True, "Game should be over due to checkmate for White."
        assert game.winner == Color.BLACK, "Black should be the winner."

        white_legal_moves = game.get_legal_moves()
        assert (
            not white_legal_moves
        ), f"White should have no legal moves in checkmate, got {white_legal_moves}"

        legal_mask = mapper.get_legal_mask(white_legal_moves, device=DEVICE)
        assert (
            legal_mask.sum().item() == 0
        ), "Mask sum for White should be 0 in checkmate."
        assert (
            not legal_mask.any()
        ), "No element in the legal mask should be true for White in checkmate."

    def test_stalemate_mask(self):  # pylint: disable=too-many-statements
        """Test that no moves are legal in a stalemate position (詰み)."""
        # SFEN for a stalemate position (Black to move, Black has no legal moves but is not in check)
        # This specific SFEN represents a position where it's Black's turn,
        # Black is not in check, but has no legal moves.
        # King vs King, Black to move. Black king on e9 (0,4), White king on e1 (8,4)
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        game = ShogiGame.from_sfen(sfen)
        mapper = PolicyOutputMapper()  # Moved mapper instantiation here
        legal_moves = game.get_legal_moves()
        legal_mask = mapper.get_legal_mask(legal_moves, device=DEVICE)

        # In this King vs King position, black king 'k' at (0,4) has 5 moves:
        # (0,3), (0,5), (1,3), (1,4), (1,5)
        assert (
            game.is_in_check(Color.BLACK) is False
        ), "King should not be in check in this K vs K setup."
        assert (
            game.game_over is False
        )  # Should not be game over by stalemate yet by ShogiGame rules for this position
        assert (
            len(legal_moves) == 5
        ), f"Expected 5 moves, got {len(legal_moves)}. Moves: {legal_moves}"
        # Use legal_mask.sum().item() which returns a Python number
        assert (
            legal_mask.sum().item() == 5
        ), f"Expected mask sum 5, got {legal_mask.sum().item()}"  # Black king can move to 5 squares

    def test_nifu_scenario_with_explicit_game_instance(
        self,
    ):  # pylint: disable=too-many-statements
        """Test nifu (two pawns on a file) specifically with direct game manipulation."""
        # Initial position
        game = ShogiGame()
        mapper = PolicyOutputMapper()

        # 1. Black moves P-7f (pawn from (6,2) to (5,2))
        game.make_move((6, 2, 5, 2, False))  # P-7f
        # 2. White moves P-3d (pawn from (2,6) to (3,6))
        game.make_move((2, 6, 3, 6, False))  # P-3d
        # 3. Black drops a pawn P*7d (drop pawn at (3,2))
        # First, ensure Black has a pawn to drop (needs to capture one)
        # Let's set up a simpler Nifu:
        # Start with empty board, black king, white king. Black has 2 pawns in hand.
        # Place one black pawn on the board. Then try to drop another on the same file.
        game_nifu_direct = ShogiGame.from_sfen(
            "4k4/9/9/9/9/9/9/9/4K4 b PP 1"
        )  # Black has two pawns
        assert game_nifu_direct.hands[Color.BLACK.value][PieceType.PAWN] == 2

        # Drop first pawn P*5e (at (4,4), assuming kings are elsewhere or this is fine)
        # Let's use a clear file, e.g. file 0 (column 0). Kings at e1, e9.
        # SFEN: 4k4/9/9/9/9/9/9/9/4K4 b PP 1. Kings at (0,4) and (8,4).
        # Drop P*9e (at (4,0))
        move1_drop_pawn: MoveTuple = (None, None, 4, 0, PieceType.PAWN)
        game_nifu_direct.make_move(move1_drop_pawn)
        assert game_nifu_direct.board[4][0] is not None
        assert (
            game_nifu_direct.board[4][0].type == PieceType.PAWN
        )  # type, not piece_type
        assert game_nifu_direct.board[4][0].color == Color.BLACK
        assert game_nifu_direct.hands[Color.BLACK.value][PieceType.PAWN] == 1
        assert game_nifu_direct.current_player == Color.WHITE  # Turn changes

        # White makes a pass move (or any simple move)
        # For simplicity in testing drops, let's assume white passes or we manually switch turn
        game_nifu_direct.current_player = (
            Color.BLACK
        )  # Manually switch back for testing black's drop

        # Now Black tries to drop another pawn on file 0 (column 0), e.g., P*9d (at (3,0))
        move2_nifu_drop: MoveTuple = (None, None, 3, 0, PieceType.PAWN)

        # Get legal moves for Black
        legal_moves_black_nifu = game_nifu_direct.get_legal_moves()
        legal_mask_black_nifu = mapper.get_legal_mask(
            legal_moves_black_nifu, device=DEVICE
        )

        idx_nifu_drop_attempt = mapper.shogi_move_to_policy_index(move2_nifu_drop)

        # Assert that this nifu drop is NOT in the legal mask
        assert (
            legal_mask_black_nifu[idx_nifu_drop_attempt].item() is False
        ), f"Nifu drop {move2_nifu_drop} should be illegal but was found in mask."

        # Also check that the move is not in legal_moves_black_nifu
        assert (
            move2_nifu_drop not in legal_moves_black_nifu
        ), f"Nifu drop {move2_nifu_drop} should be illegal but was found in get_legal_moves()."
