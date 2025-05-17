"""
conftest.py: Shared fixtures for all tests in the DRL Shogi Client project.
"""

import pytest


# Place all test scaffolding here, not in individual test files.


@pytest.fixture(scope="session")
def sample_board_state():
    """Return a minimal board state for testing (placeholder)."""
    return None  # Replace with actual board state as needed


# Add more fixtures as the codebase grows (e.g., mock agents, sample moves, etc.)
