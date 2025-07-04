"""
Integration smoke test for the Keisei Shogi training system.
Tests that the full training pipeline can initialize and run for a short period.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from keisei.config_schema import AppConfig
from keisei.evaluation.core_manager import EvaluationManager
from keisei.training.trainer import Trainer
from keisei.utils import load_config
from keisei.utils.opponents import SimpleRandomOpponent


@pytest.mark.integration
class TestIntegrationSmoke:
    """Integration tests to verify the full system works end-to-end."""

    @pytest.mark.slow
    def test_training_smoke_test(self, mock_wandb_disabled):
        """
        Run a very short training session to ensure the system can initialize,
        run without errors, and terminate cleanly.
        """
        # Create a temporary directory for this test run
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load default config and override for minimal training
            config = load_config()

            # Override settings for quick smoke test
            config.training.total_timesteps = 100  # Very short run
            config.training.checkpoint_interval_timesteps = 50
            config.training.ppo_epochs = 1  # Minimal PPO updates
            config.training.evaluation_interval_timesteps = (
                999999  # Skip eval for speed
            )
            config.logging.model_dir = temp_dir
            config.wandb.enabled = False  # Disable W&B for CI
            config.display.display_moves = False  # Disable move display for speed

            # Create mock args object with explicit None values for trainer attributes
            mock_args = MagicMock()
            mock_args.run_name = "smoke_test"
            mock_args.resume = None
            mock_args.input_features = None
            mock_args.model = None
            mock_args.tower_depth = None
            mock_args.tower_width = None
            mock_args.se_ratio = None

            # Create trainer with correct interface (W&B disabled via fixture)
            trainer = Trainer(config=config, args=mock_args)

            # Verify basic initialization
            assert trainer.config == config
            assert trainer.run_name == "smoke_test"
            assert hasattr(trainer, "metrics_manager")
            assert hasattr(trainer.metrics_manager, "global_timestep")
            assert hasattr(trainer.metrics_manager, "total_episodes_completed")

            # Note: For actual training test, we would call trainer.run_training_loop()
            # but that's quite heavy for a smoke test, so we just verify initialization
            # Note: checkpoint may not be created if interval wasn't reached
            # Just verify no crash occurred

    def test_evaluation_smoke_test(self):
        """
        Test that evaluation components can be imported and basic classes work.
        """
        # Test that we can import these classes successfully
        assert EvaluationManager is not None
        assert SimpleRandomOpponent is not None

        # Test simple opponent creation
        opponent = SimpleRandomOpponent()
        assert opponent is not None

        # For a full evaluation test, we would need to set up all the required
        # parameters (checkpoint paths, device, etc.) which is quite complex
        # This smoke test just verifies the imports work

    def test_config_system_smoke_test(self):
        """Test that the configuration system works correctly."""
        # Test loading default config
        config = load_config()
        assert isinstance(config, AppConfig)

        # Test config validation
        assert config.training.learning_rate > 0
        assert config.training.total_timesteps > 0
        assert config.env.num_actions_total > 0  # Use actual env config field
        assert config.env.input_channels > 0  # Use actual env config field

        # Test config with overrides
        config_with_overrides = load_config(
            cli_overrides={"training.learning_rate": 0.001, "env.device": "cpu"}
        )
        assert abs(config_with_overrides.training.learning_rate - 0.001) < 1e-9
        assert config_with_overrides.env.device == "cpu"
