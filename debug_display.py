#!/usr/bin/env python3
"""Debug the display rendering to see what's happening with the pieces."""

import sys
import os

# Add the project root to the Python path
project_root = "/home/john/keisei"
sys.path.insert(0, project_root)

from keisei.shogi.shogi_game import ShogiGame
from keisei.training.display_components import ShogiBoard

def debug_display_rendering():
    """Debug what's happening in the display rendering."""
    game = ShogiGame()
    game.reset()
    
    print("=== Raw Board State ===")
    for r_idx in range(9):
        row_pieces = []
        for c_idx in range(9):
            piece = game.board[r_idx][c_idx]
            if piece:
                row_pieces.append(f"{piece.type.name[0]}{piece.color.name[0]}")
            else:
                row_pieces.append("--")
        print(f"Row {r_idx}: {' '.join(row_pieces)}")
    
    print("\n=== Testing Display Component ===")
    board_display = ShogiBoard(use_unicode=True)
    
    # Test the _piece_to_symbol method directly
    print("\nTesting _piece_to_symbol for some pieces:")
    test_piece = game.board[0][0]  # Should be White Lance
    if test_piece:
        symbol = board_display._piece_to_symbol(test_piece)
        print(f"White Lance symbol: '{symbol}'")
        
    test_piece = game.board[8][0]  # Should be Black Lance  
    if test_piece:
        symbol = board_display._piece_to_symbol(test_piece)
        print(f"Black Lance symbol: '{symbol}'")
    
    # Test what happens in _create_cell_panel
    print("\n=== Testing cell panel creation ===")
    for r_idx in range(3):  # Just test first 3 rows
        print(f"\nRow {r_idx}:")
        row = game.board[r_idx]
        for c_idx, piece in enumerate(reversed(row)):
            symbol = board_display._piece_to_symbol(piece)
            if piece:
                print(f"  Display Col {c_idx}: {piece.type.name} -> '{symbol}'")
            else:
                print(f"  Display Col {c_idx}: EMPTY -> '{symbol}'")

if __name__ == "__main__":
    debug_display_rendering()
