"""
Mock utilities for testing Shogi modules without PyTorch dependencies.

This module provides mock implementations of the PyTorch-dependent classes
used in the Shogi implementation, allowing for testing the game logic
without requiring the full PyTorch library.

It also provides a patching mechanism to handle the PyTorch docstring conflict
that causes the error: `RuntimeError: function '_has_torch_function' already has a docstring`
"""

import sys
import types
import importlib
from unittest.mock import patch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


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
    
    def get_move_from_policy_index(self, index):
        """Convert policy index to a move tuple."""
        # Mock implementation returns a generic move
        return (4, 4, 3, 3, False)


# Create a patched version of _add_docstr that handles re-initialization gracefully
def patched_add_docstr(obj, docstr, warn_on_existing=True):
    """A patched version of torch.overrides._add_docstr that doesn't error on repeated calls."""
    if hasattr(obj, "__doc__") and obj.__doc__ is not None:
        # Already has a docstring, just return without changing anything
        return obj
    obj.__doc__ = docstr
    return obj


def setup_pytorch_mock_environment():
    """
    Sets up a mocked PyTorch environment to prevent import errors.
    
    This function:
    1. Creates a mock torch module
    2. Sets up all necessary submodules and classes
    3. Adds patches for problematic functions like _add_docstr
    4. Installs the mock into sys.modules
    
    Returns:
        A context manager that can be used in a with statement
    """
    # Create mock torch module and submodules
    mock_torch = types.ModuleType('torch')
    mock_torch.Tensor = MockTensor
    mock_torch.nn = types.ModuleType('torch.nn')
    mock_torch.nn.Module = MockModule
    
    # Create overrides submodule with patched _add_docstr
    mock_torch.overrides = types.ModuleType('torch.overrides')
    mock_torch.overrides._add_docstr = patched_add_docstr
    
    # Create other necessary submodules
    mock_torch.functional = types.ModuleType('torch.functional')
    mock_torch.nn.functional = types.ModuleType('torch.nn.functional')
    
    # Create patches dictionary for all relevant modules
    patches = {
        'torch': mock_torch,
        'torch.nn': mock_torch.nn,
        'torch.overrides': mock_torch.overrides,
        'torch.functional': mock_torch.functional,
        'torch.nn.functional': mock_torch.nn.functional,
    }
    
    # Create a context manager for the patches
    sys_modules_patch = patch.dict('sys.modules', patches)
    
    # Add additional patch for PolicyOutputMapper
    policy_mapper_patch = patch('keisei.utils.PolicyOutputMapper', MockPolicyOutputMapper)
    
    # Import contextlib for nested context managers
    import contextlib
    
    # Return a nested context manager to handle both patches
    @contextlib.contextmanager
    def combined_context():
        with contextlib.ExitStack() as stack:
            stack.enter_context(sys_modules_patch)
            stack.enter_context(policy_mapper_patch)
            yield
    
    return combined_context()


# Example usage:
"""
# Use the setup function to create a patched environment
with setup_pytorch_mock_environment():
    # Now you can safely import modules that depend on PyTorch
    from keisei.shogi.shogi_game_io import generate_neural_network_observation
    from keisei.shogi.shogi_game import ShogiGame
    
    # And use them normally in your tests
    game = ShogiGame()
    obs = generate_neural_network_observation(game)
"""
