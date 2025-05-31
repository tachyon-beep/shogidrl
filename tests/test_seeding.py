"""
Test suite for environment seeding functionality.

Tests the enhanced seeding implementation in ShogiGame and EnvManager.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from keisei.config_schema import AppConfig
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.env_manager import EnvManager


class TestShogiGameSeeding:
    """Test ShogiGame seeding functionality."""

    def test_seed_method_exists(self):
        """Test that the seed method exists and is callable."""
        game = ShogiGame()
        assert hasattr(game, "seed")
        assert callable(game.seed)

    def test_seed_accepts_integer(self):
        """Test that seed method accepts integer values."""
        game = ShogiGame()

        # Should not raise any exceptions
        game.seed(42)
        game.seed(0)
        game.seed(999999)

    def test_seed_accepts_none(self):
        """Test that seed method accepts None value."""
        game = ShogiGame()

        # Should not raise any exceptions
        game.seed(None)

    def test_seed_value_storage(self):
        """Test that seed values are properly stored."""
        game = ShogiGame()

        # Test with integer seed
        game.seed(42)
        assert hasattr(game, "_seed_value")
        assert game._seed_value == 42

        # Test with different seed
        game.seed(123)
        assert game._seed_value == 123

        # Test with None
        game.seed(None)
        assert game._seed_value is None

    def test_seed_logging(self):
        """Test that seeding operations are properly logged."""
        game = ShogiGame()

        with patch("keisei.shogi.shogi_game.logger") as mock_logger:
            game.seed(42)
            mock_logger.debug.assert_called_once()

            # Check that the log message contains the seed value
            log_call = mock_logger.debug.call_args[0][0]
            assert "42" in str(log_call)

    def test_seed_returns_self(self):
        """Test that seed method returns self for method chaining."""
        game = ShogiGame()
        result = game.seed(42)
        assert result is game

    def test_multiple_seeding_operations(self):
        """Test multiple seeding operations on the same game instance."""
        game = ShogiGame()

        # Test sequence of seeding operations
        seeds = [42, 123, 999, None, 0]
        for seed in seeds:
            game.seed(seed)
            assert game._seed_value == seed

    def test_seed_debugging_hook(self):
        """Test that seeding provides debugging hooks."""
        game = ShogiGame()

        # The seeding should be accessible for debugging
        game.seed(42)
        assert game._seed_value == 42

        # Should be able to retrieve the current seed
        current_seed = getattr(game, "_seed_value", None)
        assert current_seed == 42


class TestEnvManagerSeeding:
    """Test EnvManager seeding integration."""

    def setup_method(self):
        """Setup test environment."""
        from keisei.config_schema import (
            DemoConfig,
            EnvConfig,
            EvaluationConfig,
            LoggingConfig,
            TrainingConfig,
            WandBConfig,
        )

        self.config = AppConfig(
            env=EnvConfig(
                device="cpu", input_channels=46, num_actions_total=13527, seed=42
            ),
            training=TrainingConfig(
                total_timesteps=1000,
                steps_per_epoch=64,
                ppo_epochs=2,
                minibatch_size=32,
                learning_rate=3e-4,
                gamma=0.99,
                clip_epsilon=0.2,
                value_loss_coeff=0.5,
                entropy_coef=0.01,
                render_every_steps=1,
                refresh_per_second=4,
                enable_spinner=False,
                input_features="core46",
                tower_depth=5,
                tower_width=64,
                se_ratio=0.0,
                model_type="resnet",
                mixed_precision=False,
                ddp=False,
                gradient_clip_max_norm=0.5,
                lambda_gae=0.95,
                checkpoint_interval_timesteps=1000,
                evaluation_interval_timesteps=1000,
                weight_decay=0.0,
            ),
            evaluation=EvaluationConfig(
                num_games=1, opponent_type="random", evaluation_interval_timesteps=1000
            ),
            logging=LoggingConfig(
                log_file="test.log", model_dir="/tmp/test_models", run_name="test_run"
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

    def test_env_manager_seeding_integration(self):
        """Test that EnvManager properly integrates with seeding."""
        env_manager = EnvManager(config=self.config)

        # Setup the environment to initialize the game
        game, _ = env_manager.setup_environment()

        # Should be able to access the underlying game
        assert hasattr(env_manager, "game")
        assert isinstance(env_manager.game, ShogiGame)
        assert env_manager.game is game

    def test_env_manager_seed_propagation(self):
        """Test that seeding propagates through EnvManager."""
        env_manager = EnvManager(config=self.config)

        # Setup the environment to initialize the game
        game, _ = env_manager.setup_environment()

        # Seed through the game instance
        game.seed(42)

        # Verify the seed was set
        assert game._seed_value == 42

    def test_env_manager_seeding_logging(self):
        """Test that EnvManager seeding operations are logged."""
        # Create a mock logger function to track calls
        mock_logger = Mock()
        env_manager = EnvManager(config=self.config, logger_func=mock_logger)

        # Setup the environment to initialize the game
        _, _ = env_manager.setup_environment()

        # The EnvManager initialization should have logged something
        # (This test verifies the logging integration exists)
        assert (
            mock_logger.called
        ), "Logger function should have been called during environment setup"


class TestSeedingReproducibility:
    """Test seeding reproducibility contract."""

    def test_seeding_contract_consistency(self):
        """Test that seeding provides consistent interface."""
        game1 = ShogiGame()
        game2 = ShogiGame()

        # Both games should support the same seeding interface
        game1.seed(42)
        game2.seed(42)

        assert game1._seed_value == game2._seed_value

    def test_seeding_state_independence(self):
        """Test that seeding doesn't interfere with game state."""
        game = ShogiGame()

        # Get initial state
        initial_state = game.get_observation()

        # Seed the game
        game.seed(42)

        # State should be unchanged (seeding is just for future stochastic operations)
        current_state = game.get_observation()
        np.testing.assert_array_equal(initial_state, current_state)

    def test_seeding_future_extensibility(self):
        """Test that seeding infrastructure supports future extensions."""
        game = ShogiGame()

        # The seeding should support various data types for future use
        test_seeds = [42, "string_seed", {"complex": "seed"}, [1, 2, 3]]

        for seed in test_seeds:
            # Should not raise exceptions (flexibility for future use)
            try:
                game.seed(seed)
                assert game._seed_value == seed
            except Exception as e:
                # If there are type restrictions, they should be documented
                assert False, f"Seeding failed unexpectedly for {type(seed)}: {e}"


class TestSeedingEdgeCases:
    """Test edge cases and error conditions for seeding."""

    def test_seed_with_negative_values(self):
        """Test seeding with negative values."""
        game = ShogiGame()

        game.seed(-1)
        assert game._seed_value == -1

        game.seed(-999999)
        assert game._seed_value == -999999

    def test_seed_with_large_values(self):
        """Test seeding with very large values."""
        game = ShogiGame()

        large_seed = 2**32 - 1
        game.seed(large_seed)
        assert game._seed_value == large_seed

    def test_seed_type_handling(self):
        """Test how seeding handles different types."""
        game = ShogiGame()

        # Test with float (should be accepted or converted)
        game.seed(42.0)
        # Use isinstance to handle type flexibility
        assert isinstance(game._seed_value, (int, float))

        # Test with string representation of number
        game.seed("123")
        assert game._seed_value == "123"  # Should store as-is for flexibility

    def test_seed_memory_efficiency(self):
        """Test that seeding doesn't cause memory leaks."""
        game = ShogiGame()

        # Repeated seeding should not accumulate memory
        for i in range(1000):
            game.seed(i)

        # Only the last seed should be stored
        assert game._seed_value == 999

        # Should not have accumulated a history of seeds
        assert (
            not hasattr(game, "_seed_history")
            or len(getattr(game, "_seed_history", [])) <= 1
        )


@pytest.mark.integration
class TestSeedingIntegration:
    """Integration tests for seeding across the system."""

    def test_full_system_seeding_workflow(self):
        """Test complete seeding workflow from config to game."""
        from keisei.config_schema import (
            DemoConfig,
            EnvConfig,
            EvaluationConfig,
            LoggingConfig,
            TrainingConfig,
            WandBConfig,
        )

        config = AppConfig(
            env=EnvConfig(
                device="cpu", input_channels=46, num_actions_total=13527, seed=42
            ),
            training=TrainingConfig(
                total_timesteps=1000,
                steps_per_epoch=64,
                ppo_epochs=2,
                minibatch_size=32,
                learning_rate=3e-4,
                gamma=0.99,
                clip_epsilon=0.2,
                value_loss_coeff=0.5,
                entropy_coef=0.01,
                render_every_steps=1,
                refresh_per_second=4,
                enable_spinner=False,
                input_features="core46",
                tower_depth=5,
                tower_width=64,
                se_ratio=0.0,
                model_type="resnet",
                mixed_precision=False,
                ddp=False,
                gradient_clip_max_norm=0.5,
                lambda_gae=0.95,
                checkpoint_interval_timesteps=1000,
                evaluation_interval_timesteps=1000,
                weight_decay=0.0,
            ),
            evaluation=EvaluationConfig(
                num_games=1, opponent_type="random", evaluation_interval_timesteps=1000
            ),
            logging=LoggingConfig(
                log_file="test.log", model_dir="/tmp/test_models", run_name="test_run"
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
        env_manager = EnvManager(config=config)

        # Setup the environment to initialize the game
        game, _ = env_manager.setup_environment()

        # Seed the environment
        game.seed(42)

        # Verify seeding propagated correctly
        assert game._seed_value == 42

        # The system should remain functional
        state = game.get_observation()
        assert state is not None

    def test_seeding_with_real_training_components(self):
        """Test seeding works with actual training components."""
        from keisei.config_schema import (
            DemoConfig,
            EnvConfig,
            EvaluationConfig,
            LoggingConfig,
            TrainingConfig,
            WandBConfig,
        )

        config = AppConfig(
            env=EnvConfig(
                device="cpu", input_channels=46, num_actions_total=13527, seed=42
            ),
            training=TrainingConfig(
                total_timesteps=1000,
                steps_per_epoch=64,
                ppo_epochs=2,
                minibatch_size=32,
                learning_rate=3e-4,
                gamma=0.99,
                clip_epsilon=0.2,
                value_loss_coeff=0.5,
                entropy_coef=0.01,
                render_every_steps=1,
                refresh_per_second=4,
                enable_spinner=False,
                input_features="core46",
                tower_depth=5,
                tower_width=64,
                se_ratio=0.0,
                model_type="resnet",
                mixed_precision=False,
                ddp=False,
                gradient_clip_max_norm=0.5,
                lambda_gae=0.95,
                checkpoint_interval_timesteps=1000,
                evaluation_interval_timesteps=1000,
                weight_decay=0.0,
            ),
            evaluation=EvaluationConfig(
                num_games=1, opponent_type="random", evaluation_interval_timesteps=1000
            ),
            logging=LoggingConfig(
                log_file="test.log", model_dir="/tmp/test_models", run_name="test_run"
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
        env_manager = EnvManager(config=config)

        # Setup the environment to initialize the game
        game, _ = env_manager.setup_environment()

        # This should work without errors
        game.seed(42)

        # Basic game operations should still work
        assert game.get_observation() is not None
        assert hasattr(game, "_seed_value")

    def test_seeding_persistence_across_operations(self):
        """Test that seed values persist across game operations."""
        game = ShogiGame()
        game.seed(42)

        # Perform some game operations
        initial_state = game.get_observation()

        # Seed should still be accessible
        assert game._seed_value == 42

        # Game should still be functional
        current_state = game.get_observation()
        np.testing.assert_array_equal(initial_state, current_state)


if __name__ == "__main__":
    pytest.main([__file__])
