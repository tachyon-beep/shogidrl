"""
Tests for agent loading functionality in evaluation system.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from keisei.config_schema import (  # Corrected: TrainingConfig is directly in config_schema
    AppConfig,
    EvaluationConfig,
    TrainingConfig,
)
from keisei.evaluation.core.model_manager import ModelWeightManager
from keisei.utils.agent_loading import load_evaluation_agent
from tests.evaluation.conftest import INPUT_CHANNELS


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


@patch("keisei.core.ppo_agent.PPOAgent")  # Ensure this patch target is correct
def test_load_evaluation_agent_updates_minibatch_size(
    mock_ppo_agent_class, policy_mapper, tmp_path
):
    """Test that the minibatch_size is updated in TrainingConfig when loading an evaluation agent."""
    # Mock the PPOAgent constructor and its methods
    mock_created_agent_instance = MagicMock()

    def ppo_agent_constructor_side_effect(*args, **kwargs):
        # Capture the config passed to PPOAgent constructor and set it on the mock instance
        mock_created_agent_instance.config = kwargs.get("config")
        # Ensure load_model is a mock method on this instance, as expected by the function
        if not hasattr(mock_created_agent_instance, "load_model") or not isinstance(
            mock_created_agent_instance.load_model, MagicMock
        ):
            mock_created_agent_instance.load_model = MagicMock(return_value={})
        return mock_created_agent_instance

    mock_ppo_agent_class.side_effect = ppo_agent_constructor_side_effect

    # Create a dummy checkpoint file
    dummy_ckpt = tmp_path / "dummy_checkpoint.pth"
    dummy_ckpt.write_bytes(b"dummy_pytorch_model_data")

    # Load the evaluation agent
    agent = load_evaluation_agent(str(dummy_ckpt), "cpu", policy_mapper, INPUT_CHANNELS)

    # Check that the agent's TrainingConfig has the updated minibatch_size
    assert isinstance(
        agent, mock_created_agent_instance.__class__
    )  # Ensure agent is of correct class
    # The agent.config will be an AppConfig instance, so we check its training part.
    assert (
        agent.config.training.minibatch_size == 2
    )  # Check that minibatch_size is set to 2

    # Construct a mock TrainingConfig for comparison
    mock_training_config_for_comparison = TrainingConfig(
        total_timesteps=1,  # Updated to match load_evaluation_agent
        steps_per_epoch=1,  # Updated to match load_evaluation_agent
        ppo_epochs=1,
        minibatch_size=2,  # Expected value
        learning_rate=1e-4,  # Updated to match load_evaluation_agent
        gamma=0.99,
        # Corrected field names based on keisei/config_schema.py
        lambda_gae=0.95,  # Corresponds to gae_lambda in some contexts, schema uses lambda_gae
        clip_epsilon=0.2,  # Corrected from clip_coef
        entropy_coef=0.01,  # Corrected from ent_coef
        value_loss_coeff=0.5,  # Corrected from vf_coef
        gradient_clip_max_norm=0.5,  # Corresponds to max_grad_norm
        normalize_advantages=True,
        # Fields from schema that might be missing or need defaults for a valid TrainingConfig
        input_features="core46",  # Added default from schema
        tower_depth=9,  # Added default from schema
        tower_width=256,  # Added default from schema
        se_ratio=0.25,  # Added default from schema
        model_type="resnet",  # Added default from schema
        mixed_precision=False,  # Added default from schema
        ddp=False,  # Added default from schema
        checkpoint_interval_timesteps=10000,  # Added default from schema
        evaluation_interval_timesteps=50000,  # Added default from schema
        weight_decay=0.0,  # Added default from schema
        enable_value_clipping=False,  # Added default from schema
        lr_schedule_type=None,  # Added default from schema
        lr_schedule_kwargs=None,  # Added default from schema
        lr_schedule_step_on="epoch",  # Added default from schema
        render_every_steps=1,
        refresh_per_second=4,
        enable_spinner=True,
    )
    # Compare the training part of the agent's AppConfig with the mock TrainingConfig
    assert agent.config.training == mock_training_config_for_comparison
