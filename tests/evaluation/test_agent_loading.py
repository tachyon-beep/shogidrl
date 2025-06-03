"""
Tests for agent loading functionality in evaluation system.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from keisei.utils.agent_loading import load_evaluation_agent

from .conftest import INPUT_CHANNELS


@patch(
    "keisei.core.ppo_agent.PPOAgent"  # Corrected patch target - patch at the definition site
)
def test_load_evaluation_agent_mocked(mock_ppo_agent_class, policy_mapper, tmp_path):
    """Test that load_evaluation_agent returns a PPOAgent instance when checkpoint exists."""
    # Mock the PPOAgent constructor to return a specific mock instance,
    # and then mock the load_model method on that instance.
    mock_created_agent_instance = MagicMock()  # No spec constraint
    mock_created_agent_instance.load_model.return_value = {}  # Mock load_model behavior
    mock_ppo_agent_class.return_value = (
        mock_created_agent_instance  # Ensure PPOAgent() returns our mock
    )

    # Create a dummy checkpoint file (must exist for pre-load checks)
    dummy_ckpt = tmp_path / "dummy_checkpoint.pth"
    dummy_ckpt.write_bytes(b"dummy_pytorch_model_data")

    agent = load_evaluation_agent(str(dummy_ckpt), "cpu", policy_mapper, INPUT_CHANNELS)
    assert agent == mock_created_agent_instance
    mock_created_agent_instance.load_model.assert_called_once_with(str(dummy_ckpt))


def test_load_evaluation_agent_missing_checkpoint(policy_mapper):
    """
    Test that load_evaluation_agent raises a FileNotFoundError for a missing checkpoint file.

    This ensures that the evaluation pipeline fails fast and clearly when a model checkpoint
    is missing, rather than failing later or with a cryptic error.
    """
    # Create a path that does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = os.path.join(tmpdir, "nonexistent_checkpoint.pth")
        with pytest.raises(FileNotFoundError):
            load_evaluation_agent(missing_path, "cpu", policy_mapper, INPUT_CHANNELS)
