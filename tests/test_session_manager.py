"""
test_session_manager.py: Comprehensive unit tests for SessionManager class.

Tests cover initialization, directory setup, WandB configuration, logging,
error handling, and session lifecycle management.
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, mock_open

import pytest

from keisei.config_schema import AppConfig, EnvConfig, LoggingConfig, TrainingConfig, WandBConfig
from keisei.training.session_manager import SessionManager


class MockArgs:
    """Mock command-line arguments for testing."""
    def __init__(self, **kwargs):
        self.run_name = kwargs.get("run_name")
        self.resume = kwargs.get("resume")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=AppConfig)
    
    # Environment config
    env_config = Mock(spec=EnvConfig)
    env_config.seed = 42
    env_config.device = "cuda"
    config.env = env_config
    
    # Training config
    training_config = Mock(spec=TrainingConfig)
    training_config.total_timesteps = 1000000
    training_config.steps_per_epoch = 2048
    training_config.model_type = "resnet"
    training_config.input_features = "core46"
    config.training = training_config
     # Logging config
    logging_config = Mock(spec=LoggingConfig)
    logging_config.run_name = None
    config.logging = logging_config

    # WandB config
    wandb_config = Mock(spec=WandBConfig)
    wandb_config.run_name_prefix = "keisei"
    config.wandb = wandb_config

    return config


@pytest.fixture
def mock_args():
    """Create mock command-line arguments."""
    return MockArgs()


class TestSessionManagerInitialization:
    """Test SessionManager initialization logic."""
    
    def test_init_with_explicit_run_name(self, mock_config, mock_args):
        """Test initialization with explicit run_name parameter."""
        explicit_name = "explicit_test_run"
        manager = SessionManager(mock_config, mock_args, run_name=explicit_name)
        assert manager.run_name == explicit_name
    
    def test_init_with_args_run_name(self, mock_config):
        """Test initialization with run_name from CLI args."""
        args_name = "args_test_run"
        args = MockArgs(run_name=args_name)
        manager = SessionManager(mock_config, args)
        assert manager.run_name == args_name
    
    def test_init_with_config_run_name(self, mock_config, mock_args):
        """Test initialization with run_name from config."""
        config_name = "config_test_run"
        mock_config.logging.run_name = config_name
        manager = SessionManager(mock_config, mock_args)
        assert manager.run_name == config_name
    
    @patch('keisei.training.session_manager.generate_run_name')
    def test_init_with_auto_generated_name(self, mock_generate, mock_config, mock_args):
        """Test initialization with auto-generated run_name."""
        generated_name = "auto_generated_run"
        mock_generate.return_value = generated_name
        
        manager = SessionManager(mock_config, mock_args)
        assert manager.run_name == generated_name
        mock_generate.assert_called_once_with(mock_config, None)
    
    def test_init_precedence_explicit_over_args(self, mock_config):
        """Test that explicit run_name takes precedence over args."""
        explicit_name = "explicit_name"
        args_name = "args_name"
        args = MockArgs(run_name=args_name)
        
        manager = SessionManager(mock_config, args, run_name=explicit_name)
        assert manager.run_name == explicit_name
    
    def test_init_precedence_args_over_config(self, mock_config):
        """Test that args run_name takes precedence over config."""
        args_name = "args_name"
        config_name = "config_name"
        args = MockArgs(run_name=args_name)
        mock_config.logging.run_name = config_name
        
        manager = SessionManager(mock_config, args)
        assert manager.run_name == args_name
    
    def test_init_properties_not_accessible_before_setup(self, mock_config, mock_args):
        """Test that directory properties raise errors before setup."""
        manager = SessionManager(mock_config, mock_args, run_name="test")
        
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = manager.run_artifact_dir
            
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = manager.model_dir
            
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = manager.log_file_path
            
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = manager.eval_log_file_path
    
    def test_init_wandb_property_not_accessible_before_setup(self, mock_config, mock_args):
        """Test that WandB property raises error before setup."""
        manager = SessionManager(mock_config, mock_args, run_name="test")
        
        with pytest.raises(RuntimeError, match="WandB not yet initialized"):
            _ = manager.is_wandb_active


class TestSessionManagerDirectorySetup:
    """Test directory setup functionality."""
    
    @patch('keisei.training.utils.setup_directories')
    def test_setup_directories_success(self, mock_setup_dirs, mock_config, mock_args):
        """Test successful directory setup."""
        expected_dirs = {
            "run_artifact_dir": "/tmp/test_run",
            "model_dir": "/tmp/test_run/models",
            "log_file_path": "/tmp/test_run/training.log",
            "eval_log_file_path": "/tmp/test_run/eval.log"
        }
        mock_setup_dirs.return_value = expected_dirs
        
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        result = manager.setup_directories()
        
        assert result == expected_dirs
        assert manager.run_artifact_dir == expected_dirs["run_artifact_dir"]
        assert manager.model_dir == expected_dirs["model_dir"]
        assert manager.log_file_path == expected_dirs["log_file_path"]
        assert manager.eval_log_file_path == expected_dirs["eval_log_file_path"]
        
        mock_setup_dirs.assert_called_once_with(mock_config, "test_run")
    
    @patch('keisei.training.utils.setup_directories')
    def test_setup_directories_failure(self, mock_setup_dirs, mock_config, mock_args):
        """Test directory setup failure handling."""
        mock_setup_dirs.side_effect = OSError("Permission denied")
        
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        
        with pytest.raises(RuntimeError, match="Failed to setup directories"):
            manager.setup_directories()


class TestSessionManagerWandBSetup:
    """Test WandB setup functionality."""
    
    @patch('keisei.training.utils.setup_wandb')
    def test_setup_wandb_success(self, mock_setup_wandb, mock_config, mock_args):
        """Test successful WandB setup."""
        mock_setup_wandb.return_value = True
        
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        # Setup directories first
        manager._run_artifact_dir = "/tmp/test_run"
        
        result = manager.setup_wandb()
        
        assert result is True
        assert manager.is_wandb_active is True
        mock_setup_wandb.assert_called_once_with(mock_config, "test_run", "/tmp/test_run")
    
    @patch('keisei.training.utils.setup_wandb')
    def test_setup_wandb_failure(self, mock_setup_wandb, mock_config, mock_args):
        """Test WandB setup failure handling."""
        mock_setup_wandb.side_effect = Exception("WandB connection failed")
        
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._run_artifact_dir = "/tmp/test_run"
        
        with patch('sys.stderr'):
            result = manager.setup_wandb()
        
        assert result is False
        assert manager.is_wandb_active is False
    
    def test_setup_wandb_without_directories(self, mock_config, mock_args):
        """Test WandB setup fails without directory setup."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        
        with pytest.raises(RuntimeError, match="Directories must be set up before initializing WandB"):
            manager.setup_wandb()


class TestSessionManagerConfigSaving:
    """Test configuration saving functionality."""
    
    @patch('keisei.training.utils.serialize_config')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    def test_save_effective_config_success(self, mock_join, mock_file, mock_serialize, 
                                         mock_config, mock_args):
        """Test successful configuration saving."""
        mock_serialize.return_value = '{"test": "config"}'
        mock_join.return_value = "/tmp/test_run/effective_config.json"
        
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._run_artifact_dir = "/tmp/test_run"
        
        manager.save_effective_config()
        
        mock_serialize.assert_called_once_with(mock_config)
        mock_file.assert_called_once_with("/tmp/test_run/effective_config.json", "w", encoding="utf-8")
        mock_file().write.assert_called_once_with('{"test": "config"}')
    
    @patch('keisei.training.utils.serialize_config')
    def test_save_effective_config_serialization_error(self, mock_serialize, mock_config, mock_args):
        """Test configuration saving with serialization error."""
        mock_serialize.side_effect = TypeError("Cannot serialize")
        
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._run_artifact_dir = "/tmp/test_run"
        
        with pytest.raises(RuntimeError, match="Failed to save effective config"):
            manager.save_effective_config()
    
    def test_save_effective_config_without_directories(self, mock_config, mock_args):
        """Test configuration saving fails without directory setup."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        
        with pytest.raises(RuntimeError, match="Directories must be set up before saving config"):
            manager.save_effective_config()


class TestSessionManagerLogging:
    """Test session logging functionality."""
    
    def test_log_session_info_basic(self, mock_config, mock_args):
        """Test basic session info logging."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._run_artifact_dir = "/tmp/test_run"
        manager._is_wandb_active = False
        
        logged_messages = []
        def mock_logger(msg):
            logged_messages.append(msg)
        
        manager.log_session_info(mock_logger)
        
        assert any("Keisei Training Run: test_run" in msg for msg in logged_messages)
        assert any("Run directory: /tmp/test_run" in msg for msg in logged_messages)
        assert any("Random seed: 42" in msg for msg in logged_messages)
        assert any("Device: cuda" in msg for msg in logged_messages)
        assert any("Starting fresh training." in msg for msg in logged_messages)
    
    @patch('wandb.run')
    def test_log_session_info_with_wandb(self, mock_wandb_run, mock_config, mock_args):
        """Test session info logging with WandB active."""
        mock_wandb_run.url = "https://wandb.ai/test/run"
        
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._run_artifact_dir = "/tmp/test_run"
        manager._is_wandb_active = True
        
        logged_messages = []
        def mock_logger(msg):
            logged_messages.append(msg)
        
        manager.log_session_info(mock_logger)
        
        assert any("W&B: https://wandb.ai/test/run" in msg for msg in logged_messages)
    
    def test_log_session_info_with_resume(self, mock_config, mock_args):
        """Test session info logging with resume information."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._run_artifact_dir = "/tmp/test_run"
        manager._is_wandb_active = False
        
        logged_messages = []
        def mock_logger(msg):
            logged_messages.append(msg)
        
        manager.log_session_info(
            mock_logger,
            agent_info={"type": "PPO", "name": "TestAgent"},
            resumed_from_checkpoint="/path/to/checkpoint.pth",
            global_timestep=50000,
            total_episodes_completed=200
        )
        
        assert any("Agent: PPO (TestAgent)" in msg for msg in logged_messages)
        assert any("Resumed training from checkpoint" in msg for msg in logged_messages)
        assert any("Resuming from timestep 50000, 200 episodes completed" in msg for msg in logged_messages)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('keisei.training.session_manager.datetime')
    def test_log_session_start_success(self, mock_datetime, mock_file, mock_config, mock_args):
        """Test successful session start logging."""
        mock_datetime.now.return_value.strftime.return_value = "2025-05-28 10:30:00"
        
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._log_file_path = "/tmp/test_run/training.log"
        
        manager.log_session_start()
        
        mock_file.assert_called_once_with("/tmp/test_run/training.log", "a", encoding="utf-8")
        mock_file().write.assert_called_once_with(
            "[2025-05-28 10:30:00] --- SESSION START: test_run ---\n"
        )
    
    @patch('builtins.open', side_effect=IOError("File write error"))
    def test_log_session_start_failure(self, mock_file, mock_config, mock_args):
        """Test session start logging failure handling."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._log_file_path = "/tmp/test_run/training.log"
        
        with patch('sys.stderr'):
            manager.log_session_start()  # Should not raise, just print warning
    
    def test_log_session_start_without_directories(self, mock_config, mock_args):
        """Test session start logging fails without directory setup."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        
        with pytest.raises(RuntimeError, match="Directories must be set up before logging"):
            manager.log_session_start()


class TestSessionManagerFinalization:
    """Test session finalization functionality."""
    
    @patch('wandb.run')
    @patch('wandb.finish')
    def test_finalize_session_with_wandb(self, mock_wandb_finish, mock_wandb_run, 
                                       mock_config, mock_args):
        """Test session finalization with active WandB."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._is_wandb_active = True
        
        manager.finalize_session()
        
        mock_wandb_finish.assert_called_once()
    
    @patch('wandb.run', None)
    def test_finalize_session_without_wandb(self, mock_config, mock_args):
        """Test session finalization without WandB."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._is_wandb_active = False
        
        manager.finalize_session()  # Should not raise
    
    @patch('wandb.run')
    @patch('wandb.finish', side_effect=Exception("WandB error"))
    def test_finalize_session_wandb_error(self, mock_wandb_finish, mock_wandb_run,
                                        mock_config, mock_args):
        """Test session finalization with WandB error."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._is_wandb_active = True
        
        with patch('sys.stderr'):
            manager.finalize_session()  # Should not raise, just print warning


class TestSessionManagerUtilityMethods:
    """Test utility methods."""
    
    @patch('keisei.training.utils.setup_seeding')
    def test_setup_seeding(self, mock_setup_seeding, mock_config, mock_args):
        """Test seeding setup delegation."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        
        manager.setup_seeding()
        
        mock_setup_seeding.assert_called_once_with(mock_config)
    
    def test_get_session_summary(self, mock_config, mock_args):
        """Test session summary generation."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        manager._run_artifact_dir = "/tmp/test_run"
        manager._model_dir = "/tmp/test_run/models"
        manager._log_file_path = "/tmp/test_run/training.log"
        manager._is_wandb_active = True
        
        summary = manager.get_session_summary()
        
        expected_summary = {
            "run_name": "test_run",
            "run_artifact_dir": "/tmp/test_run",
            "model_dir": "/tmp/test_run/models",
            "log_file_path": "/tmp/test_run/training.log",
            "is_wandb_active": True,
            "seed": 42,
            "device": "cuda"
        }
        
        assert summary == expected_summary
    
    def test_get_session_summary_partial_setup(self, mock_config, mock_args):
        """Test session summary with partial setup."""
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        
        summary = manager.get_session_summary()
        
        assert summary["run_name"] == "test_run"
        assert summary["run_artifact_dir"] is None
        assert summary["is_wandb_active"] is None
        assert summary["seed"] == 42
        assert summary["device"] == "cuda"


class TestSessionManagerIntegration:
    """Integration tests for complete SessionManager workflows."""
    
    @patch('keisei.training.utils.setup_directories')
    @patch('keisei.training.utils.setup_wandb')
    @patch('keisei.training.utils.serialize_config')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    def test_complete_session_setup_workflow(self, mock_join, mock_file, mock_serialize,
                                           mock_setup_wandb, mock_setup_dirs,
                                           mock_config, mock_args):
        """Test complete session setup workflow."""
        # Setup mocks
        mock_setup_dirs.return_value = {
            "run_artifact_dir": "/tmp/test_run",
            "model_dir": "/tmp/test_run/models",
            "log_file_path": "/tmp/test_run/training.log",
            "eval_log_file_path": "/tmp/test_run/eval.log"
        }
        mock_setup_wandb.return_value = True
        mock_serialize.return_value = '{"test": "config"}'
        mock_join.return_value = "/tmp/test_run/effective_config.json"
        
        # Execute workflow
        manager = SessionManager(mock_config, mock_args, run_name="test_run")
        
        # Setup directories
        dirs = manager.setup_directories()
        assert dirs["run_artifact_dir"] == "/tmp/test_run"
        
        # Setup WandB
        wandb_active = manager.setup_wandb()
        assert wandb_active is True
        
        # Save config
        manager.save_effective_config()
        
        # Log session start
        with patch('keisei.training.session_manager.datetime'):
            manager.log_session_start()
        
        # Verify all components work together
        assert manager.run_name == "test_run"
        assert manager.run_artifact_dir == "/tmp/test_run"
        assert manager.is_wandb_active is True
        
        # Get summary
        summary = manager.get_session_summary()
        assert summary["run_name"] == "test_run"
        assert summary["is_wandb_active"] is True
