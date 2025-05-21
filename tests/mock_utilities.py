"""
Mock utilities for testing Shogi modules without PyTorch dependencies.

This module provides mock implementations of the PyTorch-dependent classes
used in the Shogi implementation, allowing for testing the game logic
without requiring the full PyTorch library.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class MockTensor:
    """Mock implementation of torch.Tensor for testing."""
    
    def __init__(self, data):
        """Initialize with numpy array data."""
        self.data = np.array(data)
        self.shape = self.data.shape
        self.device = "cpu"
    
    def numpy(self):
        """Return the underlying numpy array."""
        return self.data
    
    def to(self, device):
        """Mock device transfer, returns self."""
        self.device = device
        return self
    
    def __getitem__(self, idx):
        """Support indexing."""
        return MockTensor(self.data[idx])
    
    def squeeze(self, dim=None):
        """Mock squeeze operation."""
        if dim is None:
            return MockTensor(np.squeeze(self.data))
        return MockTensor(np.squeeze(self.data, axis=dim))
    
    def detach(self):
        """Mock detach operation."""
        return self
    
    def __repr__(self):
        """String representation."""
        return f"MockTensor(shape={self.shape}, device={self.device})"


class MockModule:
    """Mock implementation of torch.nn.Module for testing."""
    
    def __init__(self):
        """Initialize module."""
        self.training = True
    
    def __call__(self, *args, **kwargs):
        """Mock forward call."""
        raise NotImplementedError("Subclasses must implement __call__")
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
        return self
    
    def train(self, mode=True):
        """Set to training mode."""
        self.training = mode
        return self
    
    def to(self, device):
        """Mock device transfer."""
        return self


class MockPolicyValueNetwork(MockModule):
    """Mock implementation of PolicyValueNetwork for testing."""
    
    def __init__(self, input_planes=46, board_size=9):
        """Initialize with same signature as the real network."""
        super().__init__()
        self.input_planes = input_planes
        self.board_size = board_size
    
    def __call__(self, x, valid_moves_mask=None):
        """Mock forward pass returning policy and value."""
        # Mock a valid policy distribution over moves
        batch_size = x.shape[0] if len(x.shape) > 3 else 1
        
        # Policy has shape [batch_size, action_space_size]
        # For Shogi, action_space_size is typically around 2187
        policy = MockTensor(np.ones((batch_size, 2187)) / 2187)
        
        # Apply mask if provided
        if valid_moves_mask is not None:
            valid_moves = valid_moves_mask.data
            masked_policy_data = policy.data * valid_moves
            # Renormalize
            sums = np.sum(masked_policy_data, axis=1, keepdims=True)
            sums = np.where(sums == 0, 1.0, sums)  # Avoid division by zero
            masked_policy_data = masked_policy_data / sums
            policy = MockTensor(masked_policy_data)
        
        # Value has shape [batch_size, 1]
        value = MockTensor(np.zeros((batch_size, 1)))
        
        return policy, value


class MockPolicyOutputMapper:
    """Mock implementation of PolicyOutputMapper for testing."""
    
    def __init__(self, board_size=9):
        """Initialize with same parameters as real mapper."""
        self.board_size = board_size
        self.action_space_size = 2187  # Typical for 9x9 Shogi
    
    def get_valid_moves_mask(self, legal_moves):
        """Create a mock mask tensor from legal moves."""
        mask = np.zeros(self.action_space_size)
        # In real implementation, this would map legal moves to indices
        # For mock, we'll just set some random indices to 1
        for _ in range(min(len(legal_moves), 20)):
            idx = np.random.randint(0, self.action_space_size)
            mask[idx] = 1.0
        return MockTensor(mask)
    
    def get_move_from_policy_index(self, idx):
        """Convert policy index to a move tuple."""
        # Mock implementation returns a generic move
        return (4, 4, 3, 3, False)


# Usage example in tests:
"""
# Import the real functions but mock their dependencies
import sys
from unittest.mock import patch
import importlib

# First, mock the torch module
mock_torch = types.ModuleType('torch')
mock_torch.Tensor = MockTensor
mock_torch.nn = types.ModuleType('torch.nn')
mock_torch.nn.Module = MockModule
sys.modules['torch'] = mock_torch

# Then, patch the PolicyOutputMapper
with patch('keisei.utils.PolicyOutputMapper', MockPolicyOutputMapper):
    # Now import the modules that depend on torch
    from keisei.shogi.shogi_game_io import generate_neural_network_observation
    # And run tests as normal
"""
