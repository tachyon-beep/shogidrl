"""
W&B Sweep-enabled training script for Keisei Shogi RL agent.
This script can be used with both regular training and W&B hyperparameter sweeps.
"""

import argparse
import multiprocessing
import sys

from keisei.config_schema import AppConfig
from keisei.training.trainer import Trainer
from keisei.training.utils import apply_wandb_sweep_config, build_cli_overrides
from keisei.utils import load_config
from keisei.utils.unified_logger import log_error_to_stderr


def main():
    """Main entry point for the W&B sweep-enabled training script."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for Shogi with W&B Sweep support."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default_config.yaml",
        help="Path to a YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from, or 'latest' to auto-detect.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total timesteps to train for. Overrides config value.",
    )

    args = parser.parse_args()

    # Get W&B sweep overrides if running in a sweep
    sweep_overrides = apply_wandb_sweep_config()

    # Build CLI overrides using shared utility
    cli_overrides = build_cli_overrides(args)

    # Merge sweep overrides with CLI overrides (CLI takes precedence)
    final_overrides = {**sweep_overrides, **cli_overrides}

    # Load config with overrides
    config: AppConfig = load_config(args.config, final_overrides)

    # Initialize and run the trainer
    trainer = Trainer(config=config, args=args)
    trainer.run_training_loop()


if __name__ == "__main__":
    # Set multiprocessing start method for safety, especially with CUDA
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        log_error_to_stderr("TrainWandbSweep", f"Could not set multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method(allow_none=True)}")
    except OSError as e:
        log_error_to_stderr("TrainWandbSweep", f"Error setting multiprocessing start_method: {e}")

    main()
