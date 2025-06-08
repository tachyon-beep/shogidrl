#!/usr/bin/env python3
"""Test script to verify board rendering with the color fix."""

import sys
import os

# Add the project root to the Python path
project_root = "/home/john/keisei"
sys.path.insert(0, project_root)

from rich.console import Console
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.display_components import ShogiBoard

def test_board_rendering():
    """Test the board rendering with the new color scheme."""
    console = Console()
    
    # Create a new game
    game = ShogiGame()
    game.reset()
    
    # Create the board display component
    board_display = ShogiBoard(use_unicode=True)
    
    # Render the board
    board_panel = board_display.render(game)
    
    # Print the board
    console.print("\n=== Testing Board Rendering with Color Fix ===")
    console.print(board_panel)
    console.print("\n=== Test Notes ===")
    console.print("- Color.BLACK pieces (red team) should now appear as [bold bright_white]bright white[/bold bright_white]")
    console.print("- Color.WHITE pieces (blue team) should appear as [bold bright_blue]bright blue[/bold bright_blue]")
    console.print("- Both colors should have good contrast against the brown board backgrounds")
    
    return True

if __name__ == "__main__":
    try:
        test_board_rendering()
        print("\n✅ Board rendering test completed successfully!")
    except Exception as e:
        print(f"\n❌ Board rendering test failed: {e}")
        sys.exit(1)
