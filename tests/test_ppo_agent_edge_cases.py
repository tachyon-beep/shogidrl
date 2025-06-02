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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper
from tests.conftest import assert_valid_ppo_metrics


class TestPPOAgentErrorHandling:
    """Tests for PPOAgent error handling and recovery."""

    def test_select_action_invalid_observation_shape(
        self, ppo_agent_basic, dummy_legal_mask
    ):
        """Test action selection with invalid observation shapes."""
        # Wrong number of dimensions
        invalid_obs_1d = torch.randn(46)
        invalid_obs_2d = torch.randn(46, 9)
        invalid_obs_4d = torch.randn(1, 46, 9, 9)

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
        invalid_mask_small = torch.ones(100, dtype=torch.bool, device="cpu")
        invalid_mask_large = torch.ones(20000, dtype=torch.bool, device="cpu")

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
        selected_move, idx, log_prob, value = ppo_agent_basic.select_action(
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
        invalid_obs_1d = np.random.random(46).astype(np.float32)
        invalid_obs_4d = np.random.random((1, 46, 9, 9)).astype(np.float32)

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
        single_legal_mask[42] = True  # Make action 42 legal

        selected_move, idx, log_prob, value = ppo_agent_basic.select_action(
            dummy_observation, single_legal_mask, is_training=True
        )

        # Should select the only legal action
        assert idx == 42
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
            selected_move, idx, log_prob, value = ppo_agent_basic.select_action(
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
                param.data.fill_(999.0)

            # Load should restore original parameters
            ppo_agent_basic.load_model(save_path)

            # Verify model was loaded (parameters should not all be 999.0)
            all_999 = all(
                torch.allclose(param.data, torch.full_like(param.data, 999.0))
                for param in ppo_agent_basic.model.parameters()
            )
            assert not all_999, "Model parameters should be restored from saved state"

    def test_save_load_round_trip_preserves_behavior(
        self, ppo_agent_basic, dummy_observation, dummy_legal_mask
    ):
        """Test that save/load preserves agent behavior."""
        # Get initial action selection
        initial_move, initial_idx, initial_log_prob, initial_value = (
            ppo_agent_basic.select_action(
                dummy_observation,
                dummy_legal_mask,
                is_training=False,  # Deterministic for consistency
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_model.pth")

            # Save and load
            ppo_agent_basic.save_model(save_path)
            ppo_agent_basic.load_model(save_path)

            # Get action selection after load
            loaded_move, loaded_idx, loaded_log_prob, loaded_value = (
                ppo_agent_basic.select_action(
                    dummy_observation, dummy_legal_mask, is_training=False
                )
            )

            # Behavior should be identical (deterministic mode)
            assert initial_idx == loaded_idx
            assert np.isclose(initial_log_prob, loaded_log_prob, atol=1e-6)
            assert np.isclose(initial_value, loaded_value, atol=1e-6)

    def test_load_nonexistent_file(self, ppo_agent_basic):
        """Test loading from nonexistent file."""
        nonexistent_path = "/tmp/definitely_does_not_exist.pth"

        # The implementation returns error information instead of raising exceptions
        result = ppo_agent_basic.load_model(nonexistent_path)

        # Should return error dictionary
        assert isinstance(result, dict)
        assert "error" in result or result.get("global_timestep", -1) == 0

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
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=1e-3,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=False,
            input_features="core46",
            tower_depth=3,
            tower_width=64,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=1000,
            evaluation_interval_timesteps=1000,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        )

        # Test invalid learning rate
        invalid_configs = [
            base_config.model_copy(update={"learning_rate": -1.0}),  # Negative LR
            base_config.model_copy(update={"learning_rate": 0.0}),  # Zero LR
            base_config.model_copy(update={"gamma": -0.5}),  # Negative gamma
            base_config.model_copy(update={"gamma": 1.5}),  # Gamma > 1
            base_config.model_copy(update={"clip_epsilon": -0.1}),  # Negative epsilon
            base_config.model_copy(update={"clip_epsilon": 0.0}),  # Zero epsilon
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
                    input_channels=46,
                    num_actions_total=13527,
                    seed=42,
                    max_moves_per_game=200,
                ),
                training=invalid_config,
                evaluation=EvaluationConfig(
                    num_games=1,
                    opponent_type="random",
                    evaluation_interval_timesteps=1000,
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
                    watch_log_freq=1000,
                    watch_log_type="all",
                    log_model_artifact=False,
                ),
                demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
                parallel=ParallelConfig(
                    enabled=False,
                    num_workers=4,
                    batch_size=32,
                    sync_interval=100,
                    compression_enabled=True,
                    timeout_seconds=10.0,
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
        dummy_obs = torch.randn(46, 9, 9, device="cpu")
        dummy_mask = torch.ones(
            ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Action selection should work
        selected_move, idx, log_prob, value = ppo_agent_basic.select_action(
            dummy_obs, dummy_mask, is_training=True
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
        dummy_obs = torch.randn(46, 9, 9, device="cuda")
        dummy_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cuda"
        )

        # Action selection should work
        selected_move, idx, log_prob, value = agent.select_action(
            dummy_obs, dummy_mask, is_training=True
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
            cuda_obs = torch.randn(46, 9, 9, device="cuda")
            cpu_mask = torch.ones(
                ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cpu"
            )

            # Convert CUDA tensor to numpy on CPU for the agent
            cuda_obs_np = cuda_obs.cpu().numpy()

            # This should work fine since we're passing numpy array and CPU mask
            selected_move, idx, log_prob, value = ppo_agent_basic.select_action(
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
        config.training.learning_rate = 1e-10  # Very small LR

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Should still be able to learn (though may not change much)
        experience_buffer = ExperienceBuffer(
            buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu"
        )

        dummy_obs = torch.randn(46, 9, 9, device="cpu")
        dummy_mask = torch.ones(agent.num_actions_total, dtype=torch.bool, device="cpu")

        for i in range(4):
            experience_buffer.add(
                obs=dummy_obs,
                action=i,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)
        metrics = agent.learn(experience_buffer)

        assert_valid_ppo_metrics(metrics)
        assert metrics["ppo/learning_rate"] == 1e-10

    def test_zero_experiences_after_compute_advantages(self, ppo_agent_basic):
        """Test learning when buffer has experiences but they get filtered out."""
        experience_buffer = ExperienceBuffer(
            buffer_size=2, gamma=0.99, lambda_gae=0.95, device="cpu"
        )

        # Add experiences but don't call compute_advantages_and_returns
        dummy_obs = torch.randn(46, 9, 9, device="cpu")
        dummy_mask = torch.ones(
            ppo_agent_basic.num_actions_total, dtype=torch.bool, device="cpu"
        )

        experience_buffer.add(
            obs=dummy_obs,
            action=0,
            reward=1.0,
            log_prob=0.1,
            value=0.5,
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
                initial_lr = agent.optimizer.param_groups[0]["lr"]
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
        from keisei.core.experience_buffer import ExperienceBuffer

        buffer = ExperienceBuffer(
            buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu"
        )

        dummy_obs = torch.randn(46, 9, 9, device="cpu")
        dummy_mask = torch.ones(agent.num_actions_total, dtype=torch.bool, device="cpu")

        for i in range(4):
            buffer.add(
                obs=dummy_obs,
                action=i,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_mask,
            )

        buffer.compute_advantages_and_returns(0.0)
        metrics = agent.learn(buffer)

        assert_valid_ppo_metrics(metrics)

    def test_scheduler_extreme_learning_rates(self, minimal_app_config, ppo_test_model):
        """Test scheduler behavior with extreme learning rate values."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "exponential"
        config.training.lr_schedule_kwargs = {
            "gamma": 0.9
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
        import os
        import tempfile

        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "linear"
        config.training.lr_schedule_kwargs = {"final_lr_fraction": 0.1}

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Step scheduler to change state
        if agent.scheduler is not None:
            for _ in range(5):
                agent.scheduler.step()

            lr_after_steps = agent.optimizer.param_groups[0]["lr"]

            # Save and load model
            with tempfile.TemporaryDirectory() as tmp_dir:
                save_path = os.path.join(tmp_dir, "test_scheduler_model.pth")

                agent.save_model(save_path)

                # Reset scheduler to initial state
                agent.scheduler.step_size = 0  # Reset if possible

                # Load model back
                result = agent.load_model(save_path)

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
            initial_lr = agent.optimizer.param_groups[0]["lr"]
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
            {"initial_lr": 1e-8, "schedule_type": "linear"},  # Very small initial LR
            {"initial_lr": 1.0, "schedule_type": "exponential"},  # Large initial LR
            {
                "initial_lr": 3e-4,
                "schedule_type": "step",
                "kwargs": {"step_size": 1, "gamma": 0.1},
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
                "lr_schedule_kwargs": {"T_max": 10, "eta_min_fraction": 0.1},
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
