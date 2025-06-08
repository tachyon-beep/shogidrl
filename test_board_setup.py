#!/usr/bin/env python3
"""Test script to verify board setup and piece positions."""

import sys
import os

# Add the project root to the Python path
project_root = "/home/john/keisei"
sys.path.insert(0, project_root)

from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_core_definitions import Color, PieceType

def test_board_setup():
    """Test that the board is set up correctly with all pieces."""
    game = ShogiGame()
    game.reset()
    
    print("=== Testing Board Setup ===")
    
    # Check White's back rank (row 0)
    print("\nWhite's back rank (row 0):")
    for col in range(9):
        piece = game.get_piece(0, col)
        if piece:
            print(f"  Col {col}: {piece.type.name} ({piece.color.name})")
        else:
            print(f"  Col {col}: EMPTY")
    
    # Check White's second rank (row 1)
    print("\nWhite's second rank (row 1):")
    for col in range(9):
        piece = game.get_piece(1, col)
        if piece:
            print(f"  Col {col}: {piece.type.name} ({piece.color.name})")
        else:
            print(f"  Col {col}: EMPTY")
    
    # Check White's pawn rank (row 2)
    print("\nWhite's pawn rank (row 2):")
    for col in range(9):
        piece = game.get_piece(2, col)
        if piece:
            print(f"  Col {col}: {piece.type.name} ({piece.color.name})")
        else:
            print(f"  Col {col}: EMPTY")
    
    # Check Black's pawn rank (row 6)
    print("\nBlack's pawn rank (row 6):")
    for col in range(9):
        piece = game.get_piece(6, col)
        if piece:
            print(f"  Col {col}: {piece.type.name} ({piece.color.name})")
        else:
            print(f"  Col {col}: EMPTY")
    
    # Check Black's back rank (row 8)
    print("\nBlack's back rank (row 8):")
    for col in range(9):
        piece = game.get_piece(8, col)
        if piece:
            print(f"  Col {col}: {piece.type.name} ({piece.color.name})")
        else:
            print(f"  Col {col}: EMPTY")
    
    # Print total piece count
    total_pieces = 0
    for r in range(9):
        for c in range(9):
            if game.get_piece(r, c):
                total_pieces += 1
    
    print(f"\nTotal pieces on board: {total_pieces} (should be 40)")
    
    return game

if __name__ == "__main__":
    try:
        game = test_board_setup()
        print("\n✅ Board setup test completed!")
    except Exception as e:
        print(f"\n❌ Board setup test failed: {e}")
        sys.exit(1)
