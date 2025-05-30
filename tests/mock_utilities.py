"""
Mock utilities for testing Shogi modules without PyTorch dependencies.

This module provides mock implementations of the PyTorch-dependent classes
used in the Shogi implementation, allowing for testing the game logic
without requiring the full PyTorch library.

It also provides a patching mechanism to handle the PyTorch docstring conflict
that causes the error: `RuntimeError: function '_has_torch_function' already has a docstring`
"""

import contextlib  # Ensure contextlib is imported
import types
from typing import (  # MODIFIED: Added ContextManager, Iterator
    Any,
    ContextManager,
    Iterator,
    List,
)
from unittest.mock import patch

import numpy as np


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

    def to(self, _device):  # MODIFIED: _device to indicate unused argument
        """Mock device transfer."""
        # self.device = _device # Optionally, make MockModule track device too
        return self


class MockPolicyValueNetwork(MockModule):
    """Mock implementation of PolicyValueNetwork for testing."""

    def __init__(self, input_planes=2, board_size=9):  # MODIFIED: Local constants
        """Initialize with same signature as the real network."""
        super().__init__()
        self.input_planes = input_planes
        self.board_size = board_size

    def __call__(self, x, valid_moves_mask=None):
        """Mock forward pass returning policy and value."""
        batch_size = x.shape[0] if len(x.shape) > 3 else 1
        policy = MockTensor(np.ones((batch_size, 2187)) / 2187)
        if valid_moves_mask is not None:
            valid_moves = valid_moves_mask.data
            masked_policy_data = policy.data * valid_moves
            sums = np.sum(masked_policy_data, axis=1, keepdims=True)
            sums = np.where(sums == 0, 1.0, sums)
            masked_policy_data = masked_policy_data / sums
            policy = MockTensor(masked_policy_data)
        value = MockTensor(np.zeros((batch_size, 1)))
        return policy, value


class MockPolicyOutputMapper:
    """Mock implementation of PolicyOutputMapper for testing."""

    def __init__(self, board_size=9):
        """Initialize with same parameters as real mapper."""
        self.board_size = board_size
        self.action_space_size = 13527

    def get_total_actions(self) -> int:
        """Return the total number of actions, matching real PolicyOutputMapper interface."""
        return self.action_space_size

    def get_valid_moves_mask(self, legal_moves: List[Any]):
        """Create a mock mask tensor from legal moves."""
        mask = np.zeros(self.action_space_size)
        num_moves_to_set = min(len(legal_moves), 20)
        if self.action_space_size > 0 and num_moves_to_set > 0:
            rng = np.random.default_rng(seed=42)  # Fixed seed for deterministic tests
            for _ in range(num_moves_to_set):
                idx = rng.integers(0, self.action_space_size)
                mask[idx] = 1.0
        return MockTensor(mask)

    def get_move_from_policy_index(
        self, _index: int
    ):  # MODIFIED: _index to indicate unused argument
        """Convert policy index to a move tuple."""
        # Consider making this mock more flexible if tests need varied outputs based on index.
        return (4, 4, 3, 3, False)


def patched_add_docstr(
    obj, docstr, _warn_on_existing=True
):  # MODIFIED: _warn_on_existing
    """A patched version of torch.overrides._add_docstr that doesn't error on repeated calls."""
    if hasattr(obj, "__doc__") and obj.__doc__ is not None:
        return obj
    obj.__doc__ = docstr
    return obj


def setup_pytorch_mock_environment() -> (
    ContextManager[None]
):  # MODIFIED: Added return type hint
    """
    Sets up a mocked PyTorch environment to prevent import errors.
    ...
    """
    mock_torch = types.ModuleType("torch")
    mock_torch.Tensor = MockTensor  # type: ignore[attr-defined]

    mock_torch_nn = types.ModuleType("torch.nn")
    mock_torch_nn.Module = MockModule  # type: ignore[attr-defined]

    mock_torch_nn_functional = types.ModuleType("torch.nn.functional")
    # Add mock functions to mock_torch_nn_functional if used (e.g., relu, softmax)
    mock_torch_nn.functional = mock_torch_nn_functional  # type: ignore[attr-defined]

    mock_torch.nn = mock_torch_nn  # type: ignore[attr-defined]

    mock_torch_overrides = types.ModuleType("torch.overrides")
    # pylint: disable=protected-access # MODIFIED: Disabled warning for this intentional access
    mock_torch_overrides._add_docstr = patched_add_docstr  # type: ignore[attr-defined]
    mock_torch.overrides = mock_torch_overrides  # type: ignore[attr-defined]

    mock_torch_functional = types.ModuleType("torch.functional")
    # Add mock functions to mock_torch_functional if used
    mock_torch.functional = mock_torch_functional  # type: ignore[attr-defined]

    patches = {
        "torch": mock_torch,
        "torch.nn": mock_torch.nn,  # pylint: disable=no-member
        "torch.overrides": mock_torch.overrides,  # pylint: disable=no-member
        "torch.functional": mock_torch.functional,  # pylint: disable=no-member
        "torch.nn.functional": mock_torch.nn.functional,  # pylint: disable=no-member
    }

    sys_modules_patch = patch.dict(
        "sys.modules", patches
    )  # Intentionally using "sys.modules" directly
    policy_mapper_patch = patch(
        "keisei.utils.PolicyOutputMapper", MockPolicyOutputMapper
    )

    @contextlib.contextmanager
    def combined_context() -> (
        Iterator[None]
    ):  # MODIFIED: Added return type hint for clarity
        with contextlib.ExitStack() as stack:
            stack.enter_context(sys_modules_patch)
            stack.enter_context(policy_mapper_patch)
            yield

    return combined_context()
