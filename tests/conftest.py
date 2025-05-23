"""
conftest.py: Shared fixtures for all tests in the DRL Shogi Client project.
"""

import multiprocessing as mp
import sys  # Add this import

import pytest

# Try to set the start method as early as possible for pytest runs
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
        print(
            "Successfully set multiprocessing start method to 'spawn' in conftest.py",
            file=sys.stderr,
        )
except RuntimeError as e:
    print(
        f"Could not set multiprocessing start method in conftest.py: {e}",
        file=sys.stderr,
    )
except AttributeError:
    # Fallback for older Python versions that might not have get_start_method with allow_none
    # or if get_start_method itself fails when no method has been set yet.
    try:
        # Attempt to set it directly if get_start_method is problematic or indicates no method set.
        # This path is more speculative and depends on Python version specifics.
        mp.set_start_method("spawn", force=True)  # Still use force=True
        print(
            "Successfully set multiprocessing start method to 'spawn' in conftest.py (fallback/direct set)",
            file=sys.stderr,
        )
    except RuntimeError as e_inner:
        print(
            f"Could not set multiprocessing start method in conftest.py (fallback/direct set): {e_inner}",
            file=sys.stderr,
        )


# Place all test scaffolding here, not in individual test files.


@pytest.fixture(scope="session")
def sample_board_state():
    """Return a minimal board state for testing (placeholder)."""
    return None  # Replace with actual board state as needed


# Add more fixtures as the codebase grows (e.g., mock agents, sample moves, etc.)
