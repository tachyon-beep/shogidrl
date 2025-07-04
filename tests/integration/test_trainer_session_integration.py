"""
test_trainer_session_integration.py: Consolidated integration tests for Trainer and SessionManager.

This file combines two previous test files to verify that the SessionManager
is properly integrated into the Trainer and that session management
functionality works correctly end-to-end, removing redundancy.
"""

import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest

from keisei.training.session_manager import SessionManager
from keisei.training.trainer import Trainer


class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self, **kwargs):
        self.run_name = kwargs.get("run_name")
        self.resume = kwargs.get("resume")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_config(fast_app_config):
    """Create a mock configuration for testing using the centralized fixture."""
    return fast_app_config


@pytest.fixture
def mock_args():
    """Create mock command-line arguments."""
    return MockArgs()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.mark.integration
class TestTrainerSessionIntegration:
    """Test SessionManager integration in Trainer."""

    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.ppo_agent.PPOAgent")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    def test_trainer_initialization_and_properties(
        self,
        mock_model_factory,
        _mock_experience_buffer,
        mock_ppo_agent,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_session_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that Trainer initializes SessionManager and delegates properties."""
        # --- Setup Mocks ---
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model_factory.return_value = Mock()
        mock_ppo_agent.return_value = Mock(name="TestAgent")

        # Get a handle to the mock instance that will be created
        mock_session_instance = mock_session_manager_class.return_value

        # --- Test with CLI run_name ---
        args_with_name = MockArgs(run_name="cli_run_name")
        mock_session_instance.run_name = "cli_run_name"
        trainer_with_name = Trainer(mock_config, args_with_name)

        assert hasattr(trainer_with_name, "session_manager")
        assert trainer_with_name.run_name == "cli_run_name"

        # --- Test default initialization and property delegation ---
        args_without_name = MockArgs()
        mock_session_instance.run_name = "default_run_name"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        # FIX: The property on SessionManager is 'is_wandb_active'
        mock_session_instance.is_wandb_active = True

        trainer = Trainer(mock_config, args_without_name)

        assert trainer.session_manager is not None
        assert trainer.run_name == "default_run_name"
        assert trainer.run_artifact_dir == f"{temp_dir}/artifacts"
        assert trainer.model_dir == f"{temp_dir}/models"
        assert trainer.log_file_path == f"{temp_dir}/train.log"
        assert trainer.is_train_wandb_active is True

    @patch("keisei.training.utils.setup_seeding")
    @patch("keisei.training.utils.setup_directories")
    @patch("keisei.training.utils.setup_wandb")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.ppo_agent.PPOAgent")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    def test_session_manager_finalization(
        self,
        _mock_model_factory,
        _mock_experience_buffer,
        _mock_ppo_agent,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_setup_wandb,
        mock_setup_dirs,
        _mock_setup_seeding,
        mock_wandb_disabled,  # Use the correct fixture name
        mock_config,
        mock_args,
        temp_dir,
    ):
        """Test that session finalization is called correctly."""
        mock_setup_dirs.return_value = {
            "run_artifact_dir": f"{temp_dir}/artifacts",
            "model_dir": f"{temp_dir}/models",
            "log_file_path": f"{temp_dir}/train.log",
            "eval_log_file_path": f"{temp_dir}/eval.log",
        }
        mock_config.wandb.enabled = True
        mock_setup_wandb.return_value = True

        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        with (
            patch("builtins.open", mock_open()),
            patch("keisei.training.utils.serialize_config"),
        ):
            trainer = Trainer(mock_config, mock_args)

        trainer.session_manager.finalize_session()
        mock_wandb_disabled["finish"].assert_called_once()

    @patch("keisei.training.utils.setup_directories")
    @patch("keisei.training.models.model_factory")
    def test_session_manager_error_handling(
        self, _mock_model_factory, mock_setup_dirs, mock_config, mock_args
    ):
        """Test that SessionManager errors are handled during Trainer initialization."""
        mock_setup_dirs.side_effect = OSError("Permission denied")

        with pytest.raises(RuntimeError, match="Failed to setup directories"):
            Trainer(mock_config, mock_args)
