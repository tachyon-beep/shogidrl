#!/usr/bin/env python3
"""Test script to verify coordinate mapping in the display."""

import sys
import os

# Add the project root to the Python path
project_root = "/home/john/keisei"
sys.path.insert(0, project_root)

from keisei.shogi.shogi_game import ShogiGame
from keisei.training.display_components import ShogiBoard

def test_coordinate_mapping():
    """Test coordinate mapping between game board and display."""
    game = ShogiGame()
    game.reset()
    
    print("=== Testing Coordinate Mapping ===")
    
    # Check what's at specific known positions
    print("\nActual game board positions:")
    print(f"(0,0): {game.get_piece(0,0)}")  # Should be White Lance
    print(f"(0,4): {game.get_piece(0,4)}")  # Should be White King  
    print(f"(0,8): {game.get_piece(0,8)}")  # Should be White Lance
    print(f"(8,0): {game.get_piece(8,0)}")  # Should be Black Lance
    print(f"(8,4): {game.get_piece(8,4)}")  # Should be Black King
    print(f"(8,8): {game.get_piece(8,8)}")  # Should be Black Lance
    
    # Test the rendering
    board_display = ShogiBoard(use_unicode=True)
    
    print("\nTesting board display rendering:")
    print("Row 0 (White's back rank) as it appears in game.board[0]:")
    for c in range(9):
        piece = game.board[0][c]
        symbol = board_display._piece_to_symbol(piece)
        print(f"  Col {c}: {symbol} ({piece})")
    
    print("\nRow 0 reversed (as it would appear in display):")
    for c, piece in enumerate(reversed(game.board[0])):
        symbol = board_display._piece_to_symbol(piece)
        print(f"  Display Col {c}: {symbol} ({piece})")
    
    print("\nRow 8 (Black's back rank) as it appears in game.board[8]:")
    for c in range(9):
        piece = game.board[8][c]
        symbol = board_display._piece_to_symbol(piece)
        print(f"  Col {c}: {symbol} ({piece})")
    
    print("\nRow 8 reversed (as it would appear in display):")
    for c, piece in enumerate(reversed(game.board[8])):
        symbol = board_display._piece_to_symbol(piece)
        print(f"  Display Col {c}: {symbol} ({piece})")
    
    return game

if __name__ == "__main__":
    try:
        game = test_coordinate_mapping()
        print("\n✅ Coordinate mapping test completed!")
    except Exception as e:
        print(f"\n❌ Coordinate mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
