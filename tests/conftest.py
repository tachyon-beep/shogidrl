"""
conftest.py: Shared fixtures for all tests in the DRL Shogi Client project.
"""

import multiprocessing as mp
import sys  # Add this import
from unittest.mock import MagicMock, Mock, patch

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

# W&B Fixtures for consistent mocking across tests
@pytest.fixture
def mock_wandb_disabled():
    """Fixture that completely disables W&B for tests by mocking all common functions."""
    mock_run = MagicMock()
    mock_run.return_value = None  # wandb.run returns None when disabled
    
    with (
        patch("wandb.init") as mock_init,
        patch("wandb.log") as mock_log,
        patch("wandb.finish") as mock_finish,
        patch("wandb.watch") as mock_watch,
        patch("wandb.run", mock_run),  # Use a mock object instead of None
        patch("wandb.Artifact") as mock_artifact,
        patch("wandb.log_artifact") as mock_log_artifact,
    ):
        # Configure return values for common usage patterns
        mock_init.return_value = None
        mock_log.return_value = None
        mock_finish.return_value = None
        mock_watch.return_value = None
        mock_artifact.return_value = Mock()
        mock_log_artifact.return_value = None
        
        yield {
            "init": mock_init,
            "log": mock_log,
            "finish": mock_finish,
            "watch": mock_watch,
            "run": mock_run,  # Add run to the dictionary
            "artifact": mock_artifact,
            "log_artifact": mock_log_artifact,
        }


@pytest.fixture
def mock_wandb_active():
    """Fixture that mocks W&B as active with proper mock objects."""
    mock_run = MagicMock()
    mock_artifact = Mock()
    
    with (
        patch("wandb.init") as mock_init,
        patch("wandb.log") as mock_log,
        patch("wandb.finish") as mock_finish,
        patch("wandb.watch") as mock_watch,
        patch("wandb.run", mock_run),  # Set wandb.run to active mock
        patch("wandb.Artifact", return_value=mock_artifact) as mock_artifact_class,
        patch("wandb.log_artifact") as mock_log_artifact,
    ):
        # Configure return values for active W&B session
        mock_init.return_value = mock_run
        mock_log.return_value = None
        mock_finish.return_value = None
        mock_watch.return_value = None
        mock_log_artifact.return_value = None
        
        yield {
            "init": mock_init,
            "log": mock_log,
            "finish": mock_finish,
            "watch": mock_watch,
            "run": mock_run,
            "artifact_class": mock_artifact_class,
            "artifact": mock_artifact,
            "log_artifact": mock_log_artifact,
        }


@pytest.fixture
def mock_setup_wandb():
    """Fixture for mocking the setup_wandb utility function."""
    with patch("keisei.training.utils.setup_wandb") as mock_setup:
        mock_setup.return_value = False  # Default to disabled
        yield mock_setup

# Add more fixtures as the codebase grows (e.g., mock agents, sample moves, etc.)
