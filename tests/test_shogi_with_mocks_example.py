"""
Example test that demonstrates how to use mock utilities 
to test Shogi modules without PyTorch dependencies.
"""

import sys
import types
from unittest.mock import patch
import pytest
import numpy as np

# Import our mock classes
from mock_utilities import MockTensor, MockModule, MockPolicyOutputMapper

# Create a mock test that demonstrates the approach
def test_shogi_with_mocks():
    """
    This is a demonstration of how to test Shogi components
    by mocking the PyTorch dependencies.
    """
    # Step 1: Create a mock torch module
    mock_torch = types.ModuleType('torch')
    mock_torch.Tensor = MockTensor
    mock_torch.nn = types.ModuleType('torch.nn')
    mock_torch.nn.Module = MockModule
    
    # Step 2: Apply our patches
    with patch.dict('sys.modules', {'torch': mock_torch}):
        with patch('keisei.utils.PolicyOutputMapper', MockPolicyOutputMapper):
            # Step 3: Now we can safely import the modules that depend on torch
            # This import would normally cause the PyTorch error
            from keisei.shogi.shogi_core_definitions import (
                OBS_CURR_PLAYER_UNPROMOTED_START,
                Color, Piece, PieceType
            )
            from keisei.shogi.shogi_game import ShogiGame
            
            # Step 4: Run our actual test code
            game = ShogiGame()
            
            # A simple assertion to make sure the game was initialized
            assert game.current_player == Color.BLACK
            
            # If we need to test observation generation:
            from keisei.shogi.shogi_game_io import generate_neural_network_observation
            
            obs = generate_neural_network_observation(game)
            assert isinstance(obs, np.ndarray)
            assert obs.shape[0] >= 42  # At least the core channels
            
            # Test a basic game operation
            black_pawn_pos = (6, 4)  # Position of a black pawn in initial setup
            target_pos = (5, 4)      # Move one square forward
            assert game.get_piece(*black_pawn_pos).type == PieceType.PAWN
            
            # Make the move
            game.make_move((black_pawn_pos[0], black_pawn_pos[1], 
                           target_pos[0], target_pos[1], False))
            
            # Verify the move was made
            assert game.get_piece(*black_pawn_pos) is None
            assert game.get_piece(*target_pos).type == PieceType.PAWN
            assert game.current_player == Color.WHITE
            
            # Test works as expected
            print("Test completed successfully!")


# This won't run automatically when the file is imported
# but can be run manually or as part of a test suite
if __name__ == "__main__":
    test_shogi_with_mocks()
    print("All tests passed!")
