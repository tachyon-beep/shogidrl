"""
test_trainer_training_loop_integration_fixed.py: Integration tests for Trainer.run_training_loop() with proper mocking.

Tests verify end-to-end training loop functionality including:
- Complete training loop execution with mocked components
- Resume state logging integration
- Error handling during training loop
- Finalization behavior
"""

import tempfile
from unittest.mock import Mock, patch

import pytest

from keisei.config_schema import (
    AppConfig,
    DemoConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.training.trainer import Trainer


class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self, **kwargs):
        self.resume = kwargs.get("resume")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return AppConfig(
        env=EnvConfig(
            device="cpu", num_actions_total=13527, input_channels=46, seed=42
        ),
        training=TrainingConfig(
            total_timesteps=100,  # Small for testing
            steps_per_epoch=64,
            ppo_epochs=10,
            minibatch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            lambda_gae=0.95,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            clip_epsilon=0.2,
            gradient_clip_max_norm=0.5,
            checkpoint_interval_timesteps=100,
            input_features="core46",
            # Required parameters for testing
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            evaluation_interval_timesteps=1000,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(
            num_games=20, opponent_type="random", evaluation_interval_timesteps=1000
        ),
        logging=LoggingConfig(
            log_file="test.log", model_dir="/tmp/test_models", run_name=None
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test-project",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestTrainerTrainingLoopIntegration:
    """Test end-to-end training loop integration with mocked components."""

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.core.ppo_agent.PPOAgent")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.training.trainer.TrainingLoopManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    def test_run_training_loop_with_checkpoint_resume_logging(
        self,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        _mock_training_loop_manager_class,
        mock_session_manager_class,
        mock_ppo_agent_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that run_training_loop() logs checkpoint resume information correctly."""
        # Setup comprehensive mocks like in resume state tests
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock PPOAgent
        mock_agent = Mock()
        mock_ppo_agent_class.return_value = mock_agent

        # Mock ModelManager with checkpoint data side effect
        mock_model_manager = Mock()

        def setup_checkpoint_data(**_kwargs):
            mock_model_manager.checkpoint_data = {
                "global_timestep": 1500,
                "total_episodes_completed": 100,
                "black_wins": 40,
                "white_wins": 35,
                "draws": 25,
            }
            mock_model_manager.resumed_from_checkpoint = (
                "/path/to/resume_test_checkpoint.pth"
            )

        mock_model_manager.handle_checkpoint_resume.side_effect = setup_checkpoint_data
        mock_model_manager.save_final_model.return_value = None
        mock_model_manager.save_final_checkpoint.return_value = None
        mock_model_manager_class.return_value = mock_model_manager

        # Mock EnvManager
        mock_env_manager = Mock()
        mock_env_manager.setup_environment.return_value = (Mock(), Mock())
        mock_env_manager_class.return_value = mock_env_manager

        # Create trainer with resume
        checkpoint_path = "/path/to/resume_test_checkpoint.pth"
        args = MockArgs(resume=checkpoint_path)

        # Create trainer
        trainer = Trainer(mock_config, args)

        # Mock TrainingLogger to capture log_both calls
        mock_logger = Mock()
        with patch(
            "keisei.training.trainer.TrainingLogger"
        ) as mock_training_logger_class:
            mock_training_logger_class.return_value.__enter__.return_value = mock_logger
            mock_training_logger_class.return_value.__exit__.return_value = None

            # Mock display context manager
            trainer.display = Mock()
            trainer.display.start.return_value.__enter__ = Mock()
            trainer.display.start.return_value.__exit__ = Mock()

            # Mock episode state initialization
            with patch.object(
                trainer, "_initialize_game_state"
            ) as mock_init_game_state:
                mock_episode_state = Mock()
                mock_init_game_state.return_value = mock_episode_state

                # Run the training loop
                trainer.run_training_loop()

                # Verify resume logging was called via the logger
                mock_logger.log.assert_any_call(
                    f"Resumed training from checkpoint: {checkpoint_path}"
                )

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.core.ppo_agent.PPOAgent")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.training.trainer.TrainingLoopManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    def test_run_training_loop_fresh_start_no_resume_logging(
        self,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        _mock_training_loop_manager_class,
        mock_session_manager_class,
        mock_ppo_agent_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that run_training_loop() does not log resume message when starting fresh."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock PPOAgent
        mock_agent = Mock()
        mock_ppo_agent_class.return_value = mock_agent

        # Mock ModelManager with no checkpoint data
        mock_model_manager = Mock()
        mock_model_manager.checkpoint_data = None
        mock_model_manager.resumed_from_checkpoint = None
        mock_model_manager.save_final_model.return_value = None
        mock_model_manager.save_final_checkpoint.return_value = None
        mock_model_manager_class.return_value = mock_model_manager

        # Mock EnvManager
        mock_env_manager = Mock()
        mock_env_manager.setup_environment.return_value = (Mock(), Mock())
        mock_env_manager_class.return_value = mock_env_manager

        # Create trainer without resume
        args = MockArgs()  # No resume

        # Create trainer
        trainer = Trainer(mock_config, args)

        # Mock TrainingLogger to capture log_both calls
        mock_logger = Mock()
        with patch(
            "keisei.training.trainer.TrainingLogger"
        ) as mock_training_logger_class:
            mock_training_logger_class.return_value.__enter__.return_value = mock_logger
            mock_training_logger_class.return_value.__exit__.return_value = None

            # Mock display context manager
            trainer.display = Mock()
            trainer.display.start.return_value.__enter__ = Mock()
            trainer.display.start.return_value.__exit__ = Mock()

            # Mock episode state initialization
            with patch.object(
                trainer, "_initialize_game_state"
            ) as mock_init_game_state:
                mock_episode_state = Mock()
                mock_init_game_state.return_value = mock_episode_state

                # Run the training loop
                trainer.run_training_loop()

                # Verify no resume logging was called (check all calls don't contain resume message)
                all_calls = [str(call) for call in mock_logger.log.call_args_list]
                resume_calls = [
                    call
                    for call in all_calls
                    if "Resumed training from checkpoint" in call
                ]
                assert (
                    len(resume_calls) == 0
                ), f"Unexpected resume logging: {resume_calls}"

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.core.ppo_agent.PPOAgent")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.training.trainer.TrainingLoopManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    def test_run_training_loop_keyboard_interrupt_handling(
        self,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_training_loop_manager_class,
        mock_session_manager_class,
        mock_ppo_agent_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that run_training_loop() handles KeyboardInterrupt gracefully and calls finalization."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock PPOAgent
        mock_agent = Mock()
        mock_ppo_agent_class.return_value = mock_agent

        # Mock ModelManager
        mock_model_manager = Mock()
        mock_model_manager.checkpoint_data = None
        mock_model_manager.resumed_from_checkpoint = None
        mock_model_manager.save_final_model.return_value = None
        mock_model_manager.save_final_checkpoint.return_value = None
        mock_model_manager_class.return_value = mock_model_manager

        # Mock EnvManager
        mock_env_manager = Mock()
        mock_env_manager.setup_environment.return_value = (Mock(), Mock())
        mock_env_manager_class.return_value = mock_env_manager

        # Mock TrainingLoopManager to raise KeyboardInterrupt
        mock_training_loop_instance = mock_training_loop_manager_class.return_value
        mock_training_loop_instance.run.side_effect = KeyboardInterrupt(
            "User interrupted"
        )

        # Create trainer
        args = MockArgs()

        # Create trainer
        trainer = Trainer(mock_config, args)

        # Mock TrainingLogger to capture log_both calls
        mock_logger = Mock()
        with patch(
            "keisei.training.trainer.TrainingLogger"
        ) as mock_training_logger_class:
            mock_training_logger_class.return_value.__enter__.return_value = mock_logger
            mock_training_logger_class.return_value.__exit__.return_value = None

            # Mock display context manager
            trainer.display = Mock()
            trainer.display.start.return_value.__enter__ = Mock()
            trainer.display.start.return_value.__exit__ = Mock()

            # Mock finalization method
            with patch.object(trainer, "_finalize_training") as mock_finalize:
                # Mock episode state initialization
                with patch.object(
                    trainer, "_initialize_game_state"
                ) as mock_init_game_state:
                    mock_episode_state = Mock()
                    mock_init_game_state.return_value = mock_episode_state

                    # Run the training loop (should handle KeyboardInterrupt)
                    trainer.run_training_loop()

                    # Verify KeyboardInterrupt was logged
                    mock_logger.log.assert_any_call(
                        "Trainer caught KeyboardInterrupt from TrainingLoopManager. Finalizing."
                    )

                    # Verify finalization was called
                    mock_finalize.assert_called_once()

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.core.ppo_agent.PPOAgent")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.training.trainer.TrainingLoopManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    def test_run_training_loop_general_exception_handling(
        self,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_training_loop_manager_class,
        mock_session_manager_class,
        mock_ppo_agent_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that run_training_loop() handles general exceptions and calls finalization."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock PPOAgent
        mock_agent = Mock()
        mock_ppo_agent_class.return_value = mock_agent

        # Mock ModelManager
        mock_model_manager = Mock()
        mock_model_manager.checkpoint_data = None
        mock_model_manager.resumed_from_checkpoint = None
        mock_model_manager.save_final_model.return_value = None
        mock_model_manager.save_final_checkpoint.return_value = None
        mock_model_manager_class.return_value = mock_model_manager

        # Mock EnvManager
        mock_env_manager = Mock()
        mock_env_manager.setup_environment.return_value = (Mock(), Mock())
        mock_env_manager_class.return_value = mock_env_manager

        # Mock TrainingLoopManager to raise general exception
        mock_training_loop_instance = mock_training_loop_manager_class.return_value
        test_exception = RuntimeError("Test runtime error")
        mock_training_loop_instance.run.side_effect = test_exception

        # Create trainer
        args = MockArgs()

        # Create trainer
        trainer = Trainer(mock_config, args)

        # Mock TrainingLogger to capture log_both calls
        mock_logger = Mock()
        with patch(
            "keisei.training.trainer.TrainingLogger"
        ) as mock_training_logger_class:
            mock_training_logger_class.return_value.__enter__.return_value = mock_logger
            mock_training_logger_class.return_value.__exit__.return_value = None

            # Mock display context manager
            trainer.display = Mock()
            trainer.display.start.return_value.__enter__ = Mock()
            trainer.display.start.return_value.__exit__ = Mock()

            # Mock finalization method
            with patch.object(trainer, "_finalize_training") as mock_finalize:
                # Mock episode state initialization
                with patch.object(
                    trainer, "_initialize_game_state"
                ) as mock_init_game_state:
                    mock_episode_state = Mock()
                    mock_init_game_state.return_value = mock_episode_state

                    # Run the training loop (should handle general exception)
                    trainer.run_training_loop()

                    # Verify exception was logged
                    mock_logger.log.assert_any_call(
                        f"Trainer caught unhandled exception from TrainingLoopManager: {test_exception}. Finalizing."
                    )

                    # Verify finalization was called
                    mock_finalize.assert_called_once()

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.core.ppo_agent.PPOAgent")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.training.trainer.TrainingLoopManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    def test_training_loop_state_consistency_throughout_execution(
        self,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        _mock_training_loop_manager_class,
        mock_session_manager_class,
        mock_ppo_agent_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that training state remains consistent throughout run_training_loop() execution."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock PPOAgent
        mock_agent = Mock()
        mock_ppo_agent_class.return_value = mock_agent

        # Mock ModelManager with checkpoint data side effect
        mock_model_manager = Mock()

        def setup_checkpoint_data(**_kwargs):
            mock_model_manager.checkpoint_data = {
                "global_timestep": 2000,
                "total_episodes_completed": 150,
                "black_wins": 60,
                "white_wins": 55,
                "draws": 35,
            }
            mock_model_manager.resumed_from_checkpoint = (
                "/path/to/state_consistency_checkpoint.pth"
            )

        mock_model_manager.handle_checkpoint_resume.side_effect = setup_checkpoint_data
        mock_model_manager.save_final_model.return_value = None
        mock_model_manager.save_final_checkpoint.return_value = None
        mock_model_manager_class.return_value = mock_model_manager

        # Mock EnvManager
        mock_env_manager = Mock()
        mock_env_manager.setup_environment.return_value = (Mock(), Mock())
        mock_env_manager_class.return_value = mock_env_manager

        # Create trainer with resume to test state preservation
        checkpoint_path = "/path/to/state_consistency_checkpoint.pth"
        args = MockArgs(resume=checkpoint_path)

        # Create trainer
        trainer = Trainer(mock_config, args)

        # Mock TrainingLogger to capture log_both calls
        mock_logger = Mock()
        with patch(
            "keisei.training.trainer.TrainingLogger"
        ) as mock_training_logger_class:
            mock_training_logger_class.return_value.__enter__.return_value = mock_logger
            mock_training_logger_class.return_value.__exit__.return_value = None

            # Mock display context manager
            trainer.display = Mock()
            trainer.display.start.return_value.__enter__ = Mock()
            trainer.display.start.return_value.__exit__ = Mock()

            # Mock finalization method to capture state at finalization time
            finalization_state = {}

            def capture_finalization_state(_log_func):
                finalization_state["global_timestep"] = trainer.global_timestep
                finalization_state["total_episodes_completed"] = (
                    trainer.total_episodes_completed
                )
                finalization_state["black_wins"] = trainer.black_wins
                finalization_state["white_wins"] = trainer.white_wins
                finalization_state["draws"] = trainer.draws

            with patch.object(
                trainer, "_finalize_training", side_effect=capture_finalization_state
            ):
                # Mock episode state initialization
                with patch.object(
                    trainer, "_initialize_game_state"
                ) as mock_init_game_state:
                    mock_episode_state = Mock()
                    mock_init_game_state.return_value = mock_episode_state

                    # Mock session logger context
                    with patch.object(trainer, "logger") as mock_logger:
                        mock_logger.get_context.return_value.__enter__ = Mock(
                            return_value=trainer
                        )
                        mock_logger.get_context.return_value.__exit__ = Mock()

                        # Verify state after checkpoint resume but before running training loop
                        assert trainer.global_timestep == 2000
                        assert trainer.total_episodes_completed == 150
                        assert trainer.black_wins == 60
                        assert trainer.white_wins == 55
                        assert trainer.draws == 35

                        # Run the training loop
                        trainer.run_training_loop()

                        # Verify state consistency was maintained through execution
                        assert finalization_state["global_timestep"] == 2000
                        assert finalization_state["total_episodes_completed"] == 150
                        assert finalization_state["black_wins"] == 60
                        assert finalization_state["white_wins"] == 55
                        assert finalization_state["draws"] == 35
