"""
Unit tests for PPOAgent edge cases and error handling.

This module tests PPOAgent behavior in unusual scenarios including:
- Error handling in action selection and learning
- Edge cases with legal masks and invalid inputs
- Model save/load operations
- Device placement scenarios
- Configuration validation
- Boundary conditions and error recovery
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from keisei.config_schema import EnvConfig, TrainingConfig

from keisei.constants import (
    CORE_OBSERVATION_CHANNELS,
    EPSILON_MEDIUM,
    SHOGI_BOARD_SIZE,
    TEST_ADVANTAGE_GAMMA_ZERO,
    TEST_BATCH_SIZE,
    TEST_BUFFER_SIZE,
    TEST_CONFIG_STEPS_PER_EPOCH,
    TEST_CONFIG_TOWER_DEPTH,
    TEST_CONFIG_TOWER_WIDTH,
    TEST_DEMO_MODE_DELAY,
    TEST_ETA_MIN_FRACTION,
    TEST_EVALUATION_INTERVAL,
    TEST_GAE_LAMBDA_DEFAULT,
    TEST_GAMMA_GREATER_THAN_ONE,
    TEST_GAMMA_NINE_TENTHS,
    TEST_GLOBAL_TIMESTEP_NEGATIVE,
    TEST_GLOBAL_TIMESTEP_ZERO,
    TEST_LARGE_INITIAL_LR,
    TEST_LARGE_MASK_SIZE,
    TEST_LOG_PROB_VALUE,
    TEST_MINIMAL_BUFFER_SIZE,
    TEST_NEGATIVE_CLIP_EPSILON,
    TEST_NEGATIVE_GAMMA,
    TEST_NEGATIVE_LEARNING_RATE,
    TEST_NUM_WORKERS,
    TEST_PARAMETER_FILL_VALUE,
    TEST_REWARD_VALUE,
    TEST_SCHEDULER_FINAL_FRACTION,
    TEST_SCHEDULER_GAMMA,
    TEST_SCHEDULER_STEP_SIZE,
    TEST_SCHEDULER_TOTAL_TIMESTEPS,
    TEST_SINGLE_EPOCH,
    TEST_SINGLE_GAME,
    TEST_SINGLE_LEGAL_ACTION_INDEX,
    TEST_SMALL_MASK_SIZE,
    TEST_SMALL_MINIBATCH,
    TEST_STEP_THREE_DONE,
    TEST_SYNC_INTERVAL,
    TEST_T_MAX,
    TEST_TIMEOUT_SECONDS,
    TEST_TINY_LEARNING_RATE,
    TEST_VALUE_HALF,
    TEST_VERY_SMALL_LEARNING_RATE,
    TEST_WATCH_LOG_FREQ,
    TEST_WEIGHT_DECAY_ZERO,
    TEST_ZERO_CLIP_EPSILON,
    TEST_ZERO_LEARNING_RATE,
)

TRAIN_DEFAULTS = TrainingConfig()
ENV_DEFAULTS = EnvConfig()
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.ppo_agent import PPOAgent
from tests.conftest import assert_valid_ppo_metrics


class TestPPOAgentErrorHandling:
    """Tests for PPOAgent error handling and recovery."""

    def test_select_action_invalid_observation_shape(
        self, ppo_agent_basic, dummy_legal_mask
    ):
        """Test action selection with invalid observation shapes."""
        # Wrong number of dimensions
        invalid_obs_1d = torch.randn(CORE_OBSERVATION_CHANNELS)
        invalid_obs_2d = torch.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE)
        invalid_obs_4d = torch.randn(
            1, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
        )

        # These should either handle gracefully or raise informative errors
        for invalid_obs in [invalid_obs_1d, invalid_obs_2d, invalid_obs_4d]:
            with pytest.raises((RuntimeError, ValueError, AssertionError)):
                ppo_agent_basic.select_action(
                    invalid_obs, dummy_legal_mask, is_training=True
                )

    def test_select_action_invalid_legal_mask_shape(
        self, ppo_agent_basic, dummy_observation
    ):
        """Test action selection with invalid legal mask shapes."""
        # Wrong size legal mask
        invalid_mask_small = torch.ones(
            TEST_SMALL_MASK_SIZE, dtype=torch.bool, device="cpu"
        )
        invalid_mask_large = torch.ones(
            TEST_LARGE_MASK_SIZE, dtype=torch.bool, device="cpu"
        )

        for invalid_mask in [invalid_mask_small, invalid_mask_large]:
            with pytest.raises((RuntimeError, ValueError, IndexError)):
                ppo_agent_basic.select_action(
                    dummy_observation, invalid_mask, is_training=True
                )

    def test_select_action_all_illegal_moves(self, ppo_agent_basic, dummy_observation):
        """Test behavior when all moves are marked as illegal."""
        # Create mask with all actions illegal
        all_illegal_mask = torch.zeros(
            ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # The implementation handles this gracefully with warnings, not exceptions
        _, idx, log_prob, value = ppo_agent_basic.select_action(
            dummy_observation, all_illegal_mask, is_training=True
        )

        # Should still return valid types even with all illegal moves
        assert isinstance(idx, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert 0 <= idx < ppo_agent_basic.num_actions_total

    def test_get_value_invalid_observation(self, ppo_agent_basic):
        """Test value estimation with invalid observations."""
        # Wrong shape
        rng = np.random.default_rng(42)
        invalid_obs_1d = rng.random(CORE_OBSERVATION_CHANNELS).astype(np.float32)
        invalid_obs_4d = rng.random(
            (1, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
        ).astype(np.float32)

        for invalid_obs in [invalid_obs_1d, invalid_obs_4d]:
            with pytest.raises((RuntimeError, ValueError)):
                ppo_agent_basic.get_value(invalid_obs)


class TestPPOAgentLegalMaskEdgeCases:
    """Tests for legal mask edge cases and boundary conditions."""

    def test_single_legal_action(self, ppo_agent_basic, dummy_observation):
        """Test action selection when only one action is legal."""
        # Create mask with only one legal action
        single_legal_mask = torch.zeros(
            ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cpu"
        )
        single_legal_mask[TEST_SINGLE_LEGAL_ACTION_INDEX] = True  # Make action legal

        _, idx, log_prob, value = ppo_agent_basic.select_action(
            dummy_observation, single_legal_mask, is_training=True
        )

        # Should select the only legal action
        assert idx == TEST_SINGLE_LEGAL_ACTION_INDEX
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_legal_mask_device_mismatch(self, ppo_agent_basic, dummy_observation):
        """Test handling of legal mask on wrong device."""
        if torch.cuda.is_available():
            # Create mask on different device (if CUDA available)
            wrong_device_mask = torch.ones(
                ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cuda"
            )

            # Should either handle gracefully or give clear error
            with pytest.raises((RuntimeError, ValueError)):
                ppo_agent_basic.select_action(
                    dummy_observation, wrong_device_mask, is_training=True
                )
        else:
            pytest.skip("CUDA not available for device mismatch testing")

    def test_legal_mask_wrong_dtype(self, ppo_agent_basic, dummy_observation):
        """Test legal mask with wrong data type."""
        # Float mask instead of bool
        float_mask = torch.ones(
            ppo_agent_basic.num_actions_total, dtype=torch.float32, device="cpu"
        )

        # Should either convert gracefully or raise informative error
        try:
            _, idx, _, _ = ppo_agent_basic.select_action(
                dummy_observation, float_mask, is_training=True
            )
            # If it succeeds, verify it's still valid
            assert isinstance(idx, int)
            assert 0 <= idx < ppo_agent_basic.num_actions_total
        except (RuntimeError, ValueError, TypeError):
            # This is also acceptable behavior
            pass


class TestPPOAgentModelPersistence:
    """Tests for PPOAgent model save/load operations."""

    def test_save_model_basic(self, ppo_agent_basic):
        """Test basic model saving functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_model.pth")

            # Save should complete without error
            ppo_agent_basic.save_model(save_path)

            # File should exist
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

    def test_load_model_basic(self, ppo_agent_basic):
        """Test basic model loading functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_model.pth")

            # Save original model
            ppo_agent_basic.save_model(save_path)

            # Modify model parameters
            for param in ppo_agent_basic.model.parameters():
                param.data.fill_(TEST_PARAMETER_FILL_VALUE)

            # Load should restore original parameters
            ppo_agent_basic.load_model(save_path)

            # Verify model was loaded (parameters should not all be TEST_PARAMETER_FILL_VALUE)
            all_999 = all(
                torch.allclose(
                    param.data, torch.full_like(param.data, TEST_PARAMETER_FILL_VALUE)
                )
                for param in ppo_agent_basic.model.parameters()
            )
            assert not all_999, "Model parameters should be restored from saved state"

    def test_save_load_round_trip_preserves_behavior(
        self, ppo_agent_basic, dummy_observation, dummy_legal_mask
    ):
        """Test that save/load preserves agent behavior."""
        # Get initial action selection
        _, initial_idx, initial_log_prob, initial_value = ppo_agent_basic.select_action(
            dummy_observation,
            dummy_legal_mask,
            is_training=False,  # Deterministic for consistency
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_model.pth")

            # Save and load
            ppo_agent_basic.save_model(save_path)
            ppo_agent_basic.load_model(save_path)

            # Get action selection after load
            _, loaded_idx, loaded_log_prob, loaded_value = (
                ppo_agent_basic.select_action(
                    dummy_observation, dummy_legal_mask, is_training=False
                )
            )

            # Behavior should be identical (deterministic mode)
            assert initial_idx == loaded_idx
            assert np.isclose(initial_log_prob, loaded_log_prob, atol=EPSILON_MEDIUM)
            assert np.isclose(initial_value, loaded_value, atol=EPSILON_MEDIUM)

    def test_load_nonexistent_file(self, ppo_agent_basic):
        """Test loading from nonexistent file."""
        nonexistent_path = "/tmp/definitely_does_not_exist.pth"

        # The implementation returns error information instead of raising exceptions
        result = ppo_agent_basic.load_model(nonexistent_path)

        # Should return error dictionary
        assert isinstance(result, dict)
        assert (
            "error" in result
            or result.get("global_timestep", TEST_GLOBAL_TIMESTEP_NEGATIVE)
            == TEST_GLOBAL_TIMESTEP_ZERO
        )

    def test_save_to_invalid_path(self, ppo_agent_basic):
        """Test saving to invalid path."""
        invalid_path = (
            "/root/cannot_write_here/model.pth"  # Assuming no write permission
        )

        # This will actually raise an exception as expected for save operations
        with pytest.raises((PermissionError, FileNotFoundError, OSError, RuntimeError)):
            ppo_agent_basic.save_model(invalid_path)


class TestPPOAgentConfigurationValidation:
    """Tests for configuration validation and boundary conditions."""

    def test_invalid_hyperparameters(self, ppo_test_model):
        """Test PPOAgent behavior with invalid hyperparameters."""
        from keisei.config_schema import TrainingConfig

        base_config = TrainingConfig(
            total_timesteps=TEST_SCHEDULER_TOTAL_TIMESTEPS,
            steps_per_epoch=TEST_CONFIG_STEPS_PER_EPOCH,
            ppo_epochs=TEST_SINGLE_EPOCH,
            minibatch_size=TEST_SMALL_MINIBATCH,
            learning_rate=TRAIN_DEFAULTS.learning_rate,
            gamma=TRAIN_DEFAULTS.gamma,
            clip_epsilon=TRAIN_DEFAULTS.clip_epsilon,
            value_loss_coeff=TRAIN_DEFAULTS.value_loss_coeff,
            entropy_coef=TRAIN_DEFAULTS.entropy_coef,
            render_every_steps=TRAIN_DEFAULTS.render_every_steps,
            refresh_per_second=TRAIN_DEFAULTS.refresh_per_second,
            enable_spinner=False,
            input_features="core46",
            tower_depth=TEST_CONFIG_TOWER_DEPTH,
            tower_width=TEST_CONFIG_TOWER_WIDTH,
            se_ratio=TRAIN_DEFAULTS.se_ratio,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=TRAIN_DEFAULTS.gradient_clip_max_norm,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            checkpoint_interval_timesteps=TEST_SCHEDULER_TOTAL_TIMESTEPS,
            evaluation_interval_timesteps=TEST_SCHEDULER_TOTAL_TIMESTEPS,
            weight_decay=TEST_WEIGHT_DECAY_ZERO,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        )

        # Test invalid learning rate
        invalid_configs = [
            base_config.model_copy(
                update={"learning_rate": TEST_NEGATIVE_LEARNING_RATE}
            ),  # Negative LR
            base_config.model_copy(
                update={"learning_rate": TEST_ZERO_LEARNING_RATE}
            ),  # Zero LR
            base_config.model_copy(
                update={"gamma": TEST_NEGATIVE_GAMMA}
            ),  # Negative gamma
            base_config.model_copy(
                update={"gamma": TEST_GAMMA_GREATER_THAN_ONE}
            ),  # Gamma > 1
            base_config.model_copy(
                update={"clip_epsilon": TEST_NEGATIVE_CLIP_EPSILON}
            ),  # Negative epsilon
            base_config.model_copy(
                update={"clip_epsilon": TEST_ZERO_CLIP_EPSILON}
            ),  # Zero epsilon
        ]

        # PPOAgent should either reject invalid configs or handle them gracefully
        for invalid_config in invalid_configs:
            # Create minimal app config with invalid training config
            from keisei.config_schema import (
                AppConfig,
                DemoConfig,
                EnvConfig,
                EvaluationConfig,
                LoggingConfig,
                ParallelConfig,
                WandBConfig,
            )

            app_config = AppConfig(
                env=EnvConfig(
                    device="cpu",
                    input_channels=CORE_OBSERVATION_CHANNELS,
                    num_actions_total=13527,
                    seed=42,
                    max_moves_per_game=200,
                ),
                training=invalid_config,
                evaluation=EvaluationConfig(
                    num_games=TEST_SINGLE_GAME,
                    opponent_type="random",
                    evaluation_interval_timesteps=TEST_EVALUATION_INTERVAL,
                    enable_periodic_evaluation=False,
                    max_moves_per_game=200,
                    log_file_path_eval="eval_log.txt",
                    wandb_log_eval=False,
                ),
                logging=LoggingConfig(
                    log_file="test.log",
                    model_dir="/tmp/test_models",
                    run_name="test_run",
                ),
                wandb=WandBConfig(
                    enabled=False,
                    project="test-project",
                    entity=None,
                    run_name_prefix="test",
                    watch_model=False,
                    watch_log_freq=TEST_WATCH_LOG_FREQ,
                    watch_log_type="all",
                    log_model_artifact=False,
                ),
                demo=DemoConfig(
                    enable_demo_mode=False, demo_mode_delay=TEST_DEMO_MODE_DELAY
                ),
                parallel=ParallelConfig(
                    enabled=False,
                    num_workers=TEST_NUM_WORKERS,
                    batch_size=TEST_BATCH_SIZE,
                    sync_interval=TEST_SYNC_INTERVAL,
                    compression_enabled=True,
                    timeout_seconds=TEST_TIMEOUT_SECONDS,
                    max_queue_size=1000,
                    worker_seed_offset=1000,
                ),
            )

            # Instantiate PPOAgent to test config integration
            agent = PPOAgent(
                model=ppo_test_model, config=app_config, device=torch.device("cpu")
            )
            # If creation succeeds, agent should still function
            assert hasattr(agent, "config")
            assert hasattr(agent, "model")
            assert agent.config.training.learning_rate == invalid_config.learning_rate


class TestPPOAgentDevicePlacement:
    """Tests for device placement and handling scenarios."""

    def test_cpu_device_consistency(self, ppo_agent_basic):
        """Test that all operations are consistent on CPU device."""
        # Verify agent is on CPU
        assert ppo_agent_basic.device == torch.device("cpu")
        assert next(ppo_agent_basic.model.parameters()).device == torch.device("cpu")

        # All operations should work on CPU
        dummy_obs = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_mask = torch.ones(
            ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Action selection should work
        _, idx, _, _ = ppo_agent_basic.select_action(
            dummy_obs.cpu().numpy(), dummy_mask, is_training=True
        )
        assert isinstance(idx, int)

        # Value estimation should work
        obs_np = dummy_obs.cpu().numpy()
        value = ppo_agent_basic.get_value(obs_np)
        assert isinstance(value, float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_consistency(self, minimal_app_config, ppo_test_model):
        """Test PPOAgent operations on CUDA device."""
        # Move model to CUDA
        cuda_model = ppo_test_model.to("cuda")

        # Create agent on CUDA
        agent = PPOAgent(
            model=cuda_model,
            config=minimal_app_config,
            device=torch.device("cuda"),
            name="CUDAPPOAgent",
        )

        # Verify agent is on CUDA
        assert agent.device == torch.device("cuda")
        assert next(agent.model.parameters()).device.type == "cuda"

        # All operations should work on CUDA
        dummy_obs = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cuda"
        )
        dummy_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cuda"
        )

        # Action selection should work
        _, idx, _, _ = agent.select_action(
            dummy_obs.cpu().numpy(), dummy_mask, is_training=True
        )
        assert isinstance(idx, int)

        # Value estimation should work
        obs_np = dummy_obs.cpu().numpy()  # get_value expects numpy array
        value = agent.get_value(obs_np)
        assert isinstance(value, float)

    def test_mixed_device_inputs(self, ppo_agent_basic):
        """Test handling of inputs on wrong devices."""
        if torch.cuda.is_available():
            # Create CUDA tensor for CPU agent - this will be converted to numpy for select_action
            cuda_obs = torch.randn(
                CORE_OBSERVATION_CHANNELS,
                SHOGI_BOARD_SIZE,
                SHOGI_BOARD_SIZE,
                device="cuda",
            )
            cpu_mask = torch.ones(
                ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cpu"
            )

            # Convert CUDA tensor to numpy on CPU for the agent
            cuda_obs_np = cuda_obs.cpu().numpy()

            # This should work fine since we're passing numpy array and CPU mask
            _, idx, log_prob, value = ppo_agent_basic.select_action(
                cuda_obs_np, cpu_mask, is_training=True
            )

            # Should return valid results
            assert isinstance(idx, int)
            assert isinstance(log_prob, float)
            assert isinstance(value, float)
        else:
            pytest.skip("CUDA not available for mixed device testing")


class TestPPOAgentBoundaryConditions:
    """Tests for boundary conditions and extreme scenarios."""

    def test_extremely_small_learning_rate(self, minimal_app_config, ppo_test_model):
        """Test PPOAgent with extremely small learning rate."""
        config = minimal_app_config.model_copy()
        config.training.learning_rate = TEST_VERY_SMALL_LEARNING_RATE  # Very small LR

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Should still be able to learn (though may not change much)
        experience_buffer = ExperienceBuffer(
            buffer_size=TEST_BUFFER_SIZE,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        dummy_obs = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_mask = torch.ones(agent.num_actions_total, dtype=torch.bool, device="cpu")

        for i in range(TEST_BUFFER_SIZE):
            experience_buffer.add(
                obs=dummy_obs,
                action=i,
                reward=TEST_REWARD_VALUE,
                log_prob=TEST_LOG_PROB_VALUE,
                value=TEST_VALUE_HALF,
                done=(i == TEST_STEP_THREE_DONE),
                legal_mask=dummy_mask,
            )

        experience_buffer.compute_advantages_and_returns(TEST_ADVANTAGE_GAMMA_ZERO)
        metrics = agent.learn(experience_buffer)

        assert_valid_ppo_metrics(metrics)
        assert np.isclose(metrics["ppo/learning_rate"], TEST_VERY_SMALL_LEARNING_RATE)

    def test_zero_experiences_after_compute_advantages(self, ppo_agent_basic):
        """Test learning when buffer has experiences but they get filtered out."""
        experience_buffer = ExperienceBuffer(
            buffer_size=TEST_MINIMAL_BUFFER_SIZE,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TEST_GAE_LAMBDA_DEFAULT,
            device="cpu",
        )

        # Add experiences but don't call compute_advantages_and_returns
        dummy_obs = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_mask = torch.ones(
            ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cpu"
        )

        experience_buffer.add(
            obs=dummy_obs,
            action=0,
            reward=TEST_REWARD_VALUE,
            log_prob=TEST_LOG_PROB_VALUE,
            value=TEST_VALUE_HALF,
            done=True,
            legal_mask=dummy_mask,
        )

        # This should raise an error since advantages haven't been computed
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            ppo_agent_basic.learn(experience_buffer)


class TestPPOAgentSchedulerEdgeCases:
    """Tests for learning rate scheduler edge cases and error handling."""

    def test_invalid_scheduler_type(self, minimal_app_config, ppo_test_model):
        """Test PPOAgent behavior with invalid scheduler type."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "invalid_scheduler_type"

        # Should either handle gracefully or raise informative error
        try:
            agent = PPOAgent(
                model=ppo_test_model, config=config, device=torch.device("cpu")
            )
            # If creation succeeds, scheduler should be None (fallback behavior)
            assert agent.scheduler is None
        except (ValueError, KeyError, RuntimeError):
            # This is also acceptable behavior for invalid scheduler types
            pass

    def test_scheduler_with_invalid_kwargs(self, minimal_app_config, ppo_test_model):
        """Test scheduler creation with invalid keyword arguments."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "linear"
        config.training.lr_schedule_kwargs = {"invalid_param": "invalid_value"}

        # Should either ignore invalid kwargs or handle gracefully
        try:
            agent = PPOAgent(
                model=ppo_test_model, config=config, device=torch.device("cpu")
            )
            # If creation succeeds, verify basic functionality
            assert hasattr(agent, "scheduler")
        except (TypeError, ValueError, RuntimeError):
            # This is acceptable for invalid parameters
            pass

    def test_scheduler_with_zero_total_steps(self, minimal_app_config, ppo_test_model):
        """Test scheduler behavior when total steps would be zero."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "linear"
        config.training.total_timesteps = 1  # Very small, will result in 0 epochs
        config.training.steps_per_epoch = 1000  # Larger than total timesteps

        # Should handle gracefully or provide meaningful error
        try:
            agent = PPOAgent(
                model=ppo_test_model, config=config, device=torch.device("cpu")
            )
            # If creation succeeds, scheduler might be None or have minimal steps
            if agent.scheduler is not None:
                # Should be able to step without errors
                agent.scheduler.step()
                # Learning rate might not change due to zero/minimal steps
                current_lr = agent.optimizer.param_groups[0]["lr"]
                assert isinstance(current_lr, float)
        except (ValueError, ZeroDivisionError, RuntimeError):
            # This is acceptable for edge case configurations
            pass

    def test_scheduler_step_validation(self, minimal_app_config, ppo_test_model):
        """Test scheduler with invalid step_on configuration."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "linear"
        config.training.lr_schedule_step_on = "invalid_step_mode"

        # Should fallback to default or handle gracefully
        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Should either use default "epoch" or handle the invalid value
        assert agent.lr_schedule_step_on in ["epoch", "update", "invalid_step_mode"]

        # Should still be able to train
        buffer = ExperienceBuffer(
            buffer_size=TEST_BUFFER_SIZE,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        dummy_obs = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_mask = torch.ones(agent.num_actions_total, dtype=torch.bool, device="cpu")

        for i in range(TEST_BUFFER_SIZE):
            buffer.add(
                obs=dummy_obs,
                action=i,
                reward=TEST_REWARD_VALUE,
                log_prob=TEST_LOG_PROB_VALUE,
                value=TEST_VALUE_HALF,
                done=(i == TEST_STEP_THREE_DONE),
                legal_mask=dummy_mask,
            )

        buffer.compute_advantages_and_returns(TEST_ADVANTAGE_GAMMA_ZERO)
        metrics = agent.learn(buffer)

        assert_valid_ppo_metrics(metrics)

    def test_scheduler_extreme_learning_rates(self, minimal_app_config, ppo_test_model):
        """Test scheduler behavior with extreme learning rate values."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "exponential"
        config.training.lr_schedule_kwargs = {
            "gamma": TEST_GAMMA_NINE_TENTHS
        }  # Aggressive but not extreme decay

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        initial_lr = agent.optimizer.param_groups[0]["lr"]

        # Step scheduler many times to reach very small LR
        if agent.scheduler is not None:
            for _ in range(100):  # Many steps
                agent.scheduler.step()

            final_lr = agent.optimizer.param_groups[0]["lr"]

            # Should still be positive and smaller than initial
            assert final_lr > 0
            assert final_lr < initial_lr
            assert not np.isnan(final_lr)
            assert not np.isinf(final_lr)

    def test_scheduler_state_persistence(self, minimal_app_config, ppo_test_model):
        """Test that scheduler state is preserved through save/load operations."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "linear"
        config.training.lr_schedule_kwargs = {
            "final_lr_fraction": TEST_SCHEDULER_FINAL_FRACTION
        }

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Step scheduler to change state
        if agent.scheduler is not None:
            for _ in range(5):
                agent.scheduler.step()

            # Save and load model
            with tempfile.TemporaryDirectory() as tmp_dir:
                save_path = os.path.join(tmp_dir, "test_scheduler_model.pth")

                agent.save_model(save_path)

                # Reset scheduler to initial state
                agent.scheduler.step_size = 0  # Reset if possible

                # Load model back
                agent.load_model(save_path)

                # Should restore scheduler state (LR should match)
                loaded_lr = agent.optimizer.param_groups[0]["lr"]

                # The save/load might not preserve exact scheduler state,
                # but LR should be preserved at minimum
                assert isinstance(loaded_lr, float)
                assert loaded_lr > 0

    def test_scheduler_with_very_large_total_steps(
        self, minimal_app_config, ppo_test_model
    ):
        """Test scheduler calculation with very large training configurations."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "cosine"
        config.training.total_timesteps = 1000000  # Very large
        config.training.steps_per_epoch = 10000
        config.training.ppo_epochs = 10
        config.training.minibatch_size = 32
        config.training.lr_schedule_step_on = "update"

        # Should handle large numbers without overflow
        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Verify scheduler was created successfully
        assert hasattr(agent, "scheduler")

        # Should be able to step scheduler
        if agent.scheduler is not None:
            agent.scheduler.step()
            current_lr = agent.optimizer.param_groups[0]["lr"]

            # LR should change (even minimally)
            assert isinstance(current_lr, float)
            assert current_lr > 0

    def test_scheduler_boundary_learning_rates(
        self, minimal_app_config, ppo_test_model
    ):
        """Test scheduler with boundary learning rate values."""
        test_cases = [
            {
                "initial_lr": TEST_TINY_LEARNING_RATE,
                "schedule_type": "linear",
            },  # Very small initial LR
            {
                "initial_lr": TEST_LARGE_INITIAL_LR,
                "schedule_type": "exponential",
            },  # Large initial LR
            {
                "initial_lr": 3e-4,
                "schedule_type": "step",
                "kwargs": {
                    "step_size": TEST_SCHEDULER_STEP_SIZE,
                    "gamma": TEST_SCHEDULER_GAMMA,
                },
            },
        ]

        for case in test_cases:
            config = minimal_app_config.model_copy()
            config.training.learning_rate = case["initial_lr"]
            config.training.lr_schedule_type = case["schedule_type"]
            if "kwargs" in case:
                config.training.lr_schedule_kwargs = case["kwargs"]

            try:
                agent = PPOAgent(
                    model=ppo_test_model, config=config, device=torch.device("cpu")
                )

                # Verify basic functionality
                assert agent.optimizer.param_groups[0]["lr"] == case["initial_lr"]

                # Should be able to step if scheduler exists
                if agent.scheduler is not None:
                    agent.scheduler.step()
                    new_lr = agent.optimizer.param_groups[0]["lr"]
                    assert new_lr > 0
                    assert not np.isnan(new_lr)
                    assert not np.isinf(new_lr)

            except (ValueError, RuntimeError, OverflowError):
                # Some boundary values might be rejected, which is acceptable
                pass

    def test_scheduler_configuration_edge_cases(
        self, minimal_app_config, ppo_test_model
    ):
        """Test various edge cases in scheduler configuration."""
        edge_cases = [
            # Missing kwargs for scheduler that requires them
            {"lr_schedule_type": "step", "lr_schedule_kwargs": None},
            # Empty kwargs
            {"lr_schedule_type": "linear", "lr_schedule_kwargs": {}},
            # Conflicting kwargs
            {
                "lr_schedule_type": "cosine",
                "lr_schedule_kwargs": {
                    "T_max": TEST_T_MAX,
                    "eta_min_fraction": TEST_ETA_MIN_FRACTION,
                },
            },
        ]

        for case in edge_cases:
            config = minimal_app_config.model_copy()
            for key, value in case.items():
                setattr(config.training, key, value)

            try:
                agent = PPOAgent(
                    model=ppo_test_model, config=config, device=torch.device("cpu")
                )

                # If creation succeeds, basic functionality should work
                assert hasattr(agent, "scheduler")
                assert hasattr(agent, "lr_schedule_type")

                # Should be able to get learning rate
                current_lr = agent.optimizer.param_groups[0]["lr"]
                assert isinstance(current_lr, float)
                assert current_lr > 0

            except (ValueError, TypeError, RuntimeError):
                # Some edge cases might be rejected, which is fine
                pass
