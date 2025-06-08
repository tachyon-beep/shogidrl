#!/usr/bin/env python3
"""Debug the reversed() operation in the display."""

import sys
import os

# Add the project root to the Python path
project_root = "/home/john/keisei"
sys.path.insert(0, project_root)

from keisei.shogi.shogi_game import ShogiGame

def test_reverse_operation():
    """Test the reversed() operation to understand what's happening."""
    game = ShogiGame()
    game.reset()
    
    print("=== Testing reversed() operation ===")
    
    # Test with row 0 (White's back rank)
    row_0 = game.board[0]
    
    print("\nOriginal row 0:")
    for i, piece in enumerate(row_0):
        piece_str = f"{piece.type.name}({piece.color.name})" if piece else "EMPTY"
        print(f"  Index {i}: {piece_str}")
    
    print("\nAfter reversed(row_0):")
    for i, piece in enumerate(reversed(row_0)):
        piece_str = f"{piece.type.name}({piece.color.name})" if piece else "EMPTY"
        print(f"  Index {i}: {piece_str}")
    
    print("\nExpected Shogi board layout (from Black's perspective):")
    print("  File:  9 8 7 6 5 4 3 2 1")
    print("  Index: 0 1 2 3 4 5 6 7 8")
    print("  White should appear as: L N S G K G S N L")
    print("  Black should appear as: L N S G K G S N L")
    
    print("\nLet's check file mapping:")
    files = list(range(9, 0, -1))  # [9, 8, 7, 6, 5, 4, 3, 2, 1]
    print(f"  Files: {files}")
    print("  This means:")
    print("    game.board[0][0] should display at file 9 (leftmost)")
    print("    game.board[0][8] should display at file 1 (rightmost)")
    print("  But with reversed(), we're mapping:")
    print("    game.board[0][8] to display position 0 (leftmost)")
    print("    game.board[0][0] to display position 8 (rightmost)")

if __name__ == "__main__":
    try:
        test_reverse_operation()
        print("\n✅ Reverse operation test completed!")
    except Exception as e:
        print(f"\n❌ Reverse operation test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
