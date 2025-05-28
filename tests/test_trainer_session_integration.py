"""
test_trainer_session_integration.py: Integration tests for Trainer and SessionManager.

Tests that verify the SessionManager is properly integrated into the Trainer
and that session management functionality works correctly end-to-end.
"""

import tempfile
from unittest.mock import Mock, patch, MagicMock
import pytest

from keisei.config_schema import AppConfig, EnvConfig, LoggingConfig, TrainingConfig, WandBConfig
from keisei.training.trainer import Trainer


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
    env_config.device = "cpu"
    env_config.num_actions_total = 4096
    env_config.input_channels = 46
    config.env = env_config
    
    # Training config - include ALL required attributes
    training_config = Mock(spec=TrainingConfig)
    training_config.total_timesteps = 1000
    training_config.steps_per_epoch = 64
    training_config.model_type = "resnet"
    training_config.input_features = "core46"
    training_config.tower_depth = 5
    training_config.tower_width = 128
    training_config.se_ratio = 0.25
    training_config.mixed_precision = False
    training_config.checkpoint_interval_timesteps = 1000
    training_config.evaluation_interval_timesteps = 1000
    training_config.gamma = 0.99
    training_config.lambda_gae = 0.95
    training_config.learning_rate = 0.0003
    training_config.batch_size = 256
    training_config.epochs = 4
    training_config.clip_range = 0.2
    training_config.value_loss_coeff = 0.5
    training_config.entropy_loss_coeff = 0.01
    training_config.max_grad_norm = 0.5
    training_config.target_kl = 0.01
    config.training = training_config
    
    # Logging config
    logging_config = Mock(spec=LoggingConfig)
    logging_config.run_name = None
    logging_config.log_level = "INFO"
    logging_config.model_dir = None
    logging_config.savedir = "/tmp/test_logs"
    config.logging = logging_config
    
    # WandB config
    wandb_config = Mock(spec=WandBConfig)
    wandb_config.enabled = False
    wandb_config.run_name_prefix = "test"
    wandb_config.project = "test-project"
    config.wandb = wandb_config
    
    # Demo config
    demo_config = Mock()
    demo_config.enable_demo_mode = False
    demo_config.demo_mode_delay = 0
    config.demo = demo_config
    
    return config


@pytest.fixture
def mock_args():
    """Create mock command-line arguments."""
    return MockArgs(
        run_name=None,
        resume=None,
        input_features=None,
        model=None,
        tower_depth=None,
        tower_width=None,
        se_ratio=None
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestTrainerSessionIntegration:
    """Test integration between Trainer and SessionManager."""

    @patch('torch.device')
    @patch('keisei.training.trainer.Console')
    @patch('keisei.training.trainer.TrainingLogger')
    @patch('keisei.training.trainer.ShogiGame')
    @patch('keisei.training.trainer.PolicyOutputMapper')
    @patch('keisei.training.trainer.PPOAgent')
    @patch('keisei.training.trainer.ExperienceBuffer')
    @patch('keisei.training.models.model_factory')
    @patch('keisei.shogi.features.FEATURE_SPECS')
    @patch('keisei.training.trainer.display.TrainingDisplay')
    @patch('keisei.training.trainer.callbacks.CheckpointCallback')
    @patch('keisei.training.trainer.callbacks.EvaluationCallback')
    def test_trainer_initialization_with_session_manager(
        self,
        mock_eval_callback,
        mock_checkpoint_callback,
        mock_display,
        mock_feature_specs,
        mock_model_factory,
        mock_experience_buffer,
        mock_ppo_agent,
        mock_policy_mapper,
        mock_shogi_game,
        mock_training_logger,
        mock_console,
        mock_torch_device,
        mock_config,
        mock_args,
        temp_dir
    ):
        """Test that Trainer properly initializes with SessionManager."""
        # Setup feature specs mock
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock
        
        # Setup model factory mock
        mock_model = Mock()
        mock_model_factory.return_value = mock_model
        
        # Setup device mock
        mock_device_instance = Mock()
        mock_device_instance.type = "cpu"
        mock_torch_device.return_value = mock_device_instance
        
        # Setup game mock
        game_instance = Mock()
        game_instance.seed = Mock()
        mock_shogi_game.return_value = game_instance
        
        # Setup policy mapper mock
        policy_mapper_instance = Mock()
        policy_mapper_instance.get_total_actions.return_value = 4096
        mock_policy_mapper.return_value = policy_mapper_instance
        
        # Setup agent mock
        agent_instance = Mock()
        agent_instance.name = "test_agent"
        agent_instance.model = mock_model
        mock_ppo_agent.return_value = agent_instance
        
        # Setup buffer mock
        buffer_instance = Mock()
        mock_experience_buffer.return_value = buffer_instance
        
        # Setup logger mock
        logger_instance = Mock()
        mock_training_logger.return_value = logger_instance
        
        # Setup console mock
        console_instance = Mock()
        mock_console.return_value = console_instance
        
        # Mock evaluation config
        eval_config = Mock()
        eval_config.evaluation_interval_timesteps = 1000
        mock_config.evaluation = eval_config
        
        # Use temp directory for session manager
        with patch('keisei.training.trainer.SessionManager') as mock_session_manager_class, \
             patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
             patch('glob.glob', return_value=[]), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=True):
            
            # Create mock session manager instance
            mock_session_manager = Mock()
            mock_session_manager.run_name = "test_run_12345"
            mock_session_manager.run_artifact_dir = f"{temp_dir}/artifacts"
            mock_session_manager.model_dir = f"{temp_dir}/models"
            mock_session_manager.log_file_path = f"{temp_dir}/train.log"
            mock_session_manager.eval_log_file_path = f"{temp_dir}/eval.log"
            mock_session_manager.is_wandb_active = False
            
            # Mock setup methods
            mock_session_manager.setup_directories = Mock()
            mock_session_manager.setup_wandb = Mock()
            mock_session_manager.save_effective_config = Mock()  
            mock_session_manager.setup_seeding = Mock()
            mock_session_manager.log_session_info = Mock()
            mock_session_manager.finalize_session = Mock()
            
            mock_session_manager_class.return_value = mock_session_manager
            
            # Create Trainer instance
            trainer = Trainer(mock_config, mock_args)
            
            # Verify SessionManager was created and initialized
            assert hasattr(trainer, 'session_manager')
            assert trainer.session_manager is not None
            
            # Verify session setup methods were called
            mock_session_manager.setup_directories.assert_called_once()
            mock_session_manager.setup_wandb.assert_called_once()
            mock_session_manager.save_effective_config.assert_called_once()
            mock_session_manager.setup_seeding.assert_called_once()
            
            # Verify session properties were set on trainer
            assert hasattr(trainer, 'run_name')
            assert hasattr(trainer, 'model_dir')
            assert hasattr(trainer, 'log_file_path')
            
            # Verify other trainer components were initialized
            assert hasattr(trainer, 'agent')
            assert hasattr(trainer, 'experience_buffer')
            assert hasattr(trainer, 'game')
            assert hasattr(trainer, 'policy_output_mapper')

    @patch('torch.device')
    @patch('keisei.training.trainer.Console')
    @patch('keisei.training.trainer.TrainingLogger')
    @patch('keisei.training.trainer.ShogiGame')
    @patch('keisei.training.trainer.PolicyOutputMapper')
    @patch('keisei.training.trainer.PPOAgent')
    @patch('keisei.training.trainer.ExperienceBuffer')
    @patch('keisei.training.models.model_factory')
    @patch('keisei.shogi.features.FEATURE_SPECS')
    @patch('keisei.training.trainer.display.TrainingDisplay')
    @patch('keisei.training.trainer.callbacks.CheckpointCallback')
    @patch('keisei.training.trainer.callbacks.EvaluationCallback')
    def test_trainer_session_properties_delegation(
        self,
        mock_eval_callback,
        mock_checkpoint_callback,
        mock_display,
        mock_feature_specs,
        mock_model_factory,
        mock_experience_buffer,
        mock_ppo_agent,
        mock_policy_mapper,
        mock_shogi_game,
        mock_training_logger,
        mock_console,
        mock_torch_device,
        mock_config,
        mock_args,
        temp_dir
    ):
        """Test that Trainer properly delegates session properties to SessionManager."""
        # Setup mocks (similar to previous test)
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock
        
        mock_model = Mock()
        mock_model_factory.return_value = mock_model
        
        mock_device_instance = Mock()
        mock_device_instance.type = "cpu"
        mock_torch_device.return_value = mock_device_instance
        
        game_instance = Mock()
        game_instance.seed = Mock()
        mock_shogi_game.return_value = game_instance
        
        policy_mapper_instance = Mock()
        policy_mapper_instance.get_total_actions.return_value = 4096
        mock_policy_mapper.return_value = policy_mapper_instance
        
        agent_instance = Mock()
        agent_instance.name = "test_agent"
        agent_instance.model = mock_model
        mock_ppo_agent.return_value = agent_instance
        
        buffer_instance = Mock()
        mock_experience_buffer.return_value = buffer_instance
        
        logger_instance = Mock()
        mock_training_logger.return_value = logger_instance
        
        console_instance = Mock()
        mock_console.return_value = console_instance
        
        eval_config = Mock()
        eval_config.evaluation_interval_timesteps = 1000
        mock_config.evaluation = eval_config
        
        # Mock session manager with specific property values
        with patch('keisei.training.session_manager.SessionManager') as mock_session_manager_class:
            mock_session_manager = Mock()
            mock_session_manager.run_name = "test_run_session"
            mock_session_manager.run_artifact_dir = f"{temp_dir}/artifacts"
            mock_session_manager.model_dir = f"{temp_dir}/models"
            mock_session_manager.log_file_path = f"{temp_dir}/train.log"
            mock_session_manager.eval_log_file_path = f"{temp_dir}/eval.log"
            mock_session_manager.is_wandb_active = False
            
            mock_session_manager_class.return_value = mock_session_manager
            
            # Create Trainer instance
            trainer = Trainer(mock_config, mock_args)
            
            # Verify properties were copied from session manager
            assert trainer.run_name == "test_run_session"
            assert trainer.run_artifact_dir == f"{temp_dir}/artifacts"
            assert trainer.model_dir == f"{temp_dir}/models"
            assert trainer.log_file_path == f"{temp_dir}/train.log"
            assert trainer.eval_log_file_path == f"{temp_dir}/eval.log"
            assert trainer.is_train_wandb_active == False

    def test_session_manager_method_integration(self, mock_config, mock_args, temp_dir):
        """Test that session manager methods are properly integrated."""
        with patch('keisei.training.session_manager.SessionManager') as mock_session_manager_class:
            mock_session_manager = Mock()
            mock_session_manager.log_session_info = Mock()
            mock_session_manager.finalize_session = Mock()
            
            mock_session_manager_class.return_value = mock_session_manager
            
            # Mock all trainer dependencies to avoid initialization issues
            with patch.multiple(
                'keisei.training.trainer',
                Console=Mock(),
                TrainingLogger=Mock(),
                ShogiGame=Mock(),
                PolicyOutputMapper=Mock(),
                PPOAgent=Mock(),
                ExperienceBuffer=Mock(),
                model_factory=Mock(),
                display=Mock(),
                callbacks=Mock()
            ), \
            patch('torch.device'), \
            patch('keisei.shogi.features.FEATURE_SPECS', {'core46': Mock(num_planes=46)}):
                
                trainer = Trainer(mock_config, mock_args)
                
                # Test session info logging delegation
                mock_log_both = Mock()
                trainer._log_run_info(mock_log_both)
                
                # Verify session manager's log_session_info was called
                mock_session_manager.log_session_info.assert_called_once()
                
                # Test session finalization (would be called at end of training)
                # This would typically be called in training loop completion
                trainer.session_manager.finalize_session()
                mock_session_manager.finalize_session.assert_called_once()
