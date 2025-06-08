#!/usr/bin/env python3
"""Debug script to check Black piece placement."""

import sys
import os

# Add the project root to the Python path
project_root = "/home/john/keisei"
sys.path.insert(0, project_root)

from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_core_definitions import Color, PieceType

def debug_black_pieces():
    """Debug Black piece placement."""
    game = ShogiGame()
    game.reset()
    
    print("=== Checking Black Pieces ===")
    
    # Check Black's pawn rank (row 6 = rank 3)
    print("\nBlack's pawn rank (row 6 = rank 3):")
    for col in range(9):
        piece = game.get_piece(6, col)
        if piece:
            print(f"  Col {col}: {piece.type.name} ({piece.color.name})")
        else:
            print(f"  Col {col}: EMPTY")
    
    # Check Black's second rank (row 7 = rank 2) - should have rook and bishop
    print("\nBlack's second rank (row 7 = rank 2) - should have rook and bishop:")
    for col in range(9):
        piece = game.get_piece(7, col)
        if piece:
            print(f"  Col {col}: {piece.type.name} ({piece.color.name})")
        else:
            print(f"  Col {col}: EMPTY")
    
    # Check Black's back rank (row 8 = rank 1) - should have all major pieces
    print("\nBlack's back rank (row 8 = rank 1) - should have all major pieces:")
    for col in range(9):
        piece = game.get_piece(8, col)
        if piece:
            print(f"  Col {col}: {piece.type.name} ({piece.color.name})")
        else:
            print(f"  Col {col}: EMPTY")
    
    # Count Black pieces
    black_piece_count = 0
    for r in range(9):
        for c in range(9):
            piece = game.get_piece(r, c)
            if piece and piece.color == Color.BLACK:
                black_piece_count += 1
    
    print(f"\nTotal BLACK pieces on board: {black_piece_count} (should be 20)")
    
    # Count White pieces  
    white_piece_count = 0
    for r in range(9):
        for c in range(9):
            piece = game.get_piece(r, c)
            if piece and piece.color == Color.WHITE:
                white_piece_count += 1
    
    print(f"Total WHITE pieces on board: {white_piece_count} (should be 20)")

if __name__ == "__main__":
    debug_black_pieces()
