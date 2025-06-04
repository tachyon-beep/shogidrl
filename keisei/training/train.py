"""
Main training script for Keisei Shogi RL agent.
Refactored to use the Trainer class for better modularity.
"""

import argparse
import multiprocessing
import sys

from keisei.config_schema import AppConfig

# Import config module and related components
from keisei.utils import load_config
from keisei.utils.unified_logger import log_info_to_stderr, log_error_to_stderr

from .trainer import Trainer
from .utils import apply_wandb_sweep_config, build_cli_overrides


def main():
    """Main entry point for the training script (Pydantic config version)."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for Shogi with Rich TUI (Pydantic config)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
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
    # --- Model/feature CLI flags ---
    parser.add_argument(
        "--model", type=str, default=None, help="Model type (e.g. 'resnet')."
    )
    parser.add_argument(
        "--input_features",
        type=str,
        default=None,
        help="Feature set for observation builder.",
    )
    parser.add_argument(
        "--tower_depth", type=int, default=None, help="ResNet tower depth."
    )
    parser.add_argument(
        "--tower_width", type=int, default=None, help="ResNet tower width."
    )
    parser.add_argument(
        "--se_ratio", type=float, default=None, help="SE block squeeze ratio."
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed-precision training.",
    )
    parser.add_argument(
        "--ddp", action="store_true", help="Enable DistributedDataParallel training."
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
    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Directory to save models and logs. Overrides config MODEL_DIR.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values via KEY.SUBKEY=VALUE format.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=None,
        help="Update display every N steps to reduce flicker. Overrides config value.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this run (overrides config and auto-generated name).",
    )
    # --- W&B Sweep support flags ---
    parser.add_argument(
        "--wandb-enabled",
        action="store_true",
        help="Force enable W&B logging (useful for sweeps).",
    )
    args = parser.parse_args()

    # Get W&B sweep overrides if running in a sweep
    sweep_overrides = apply_wandb_sweep_config()

    # Build CLI overrides dict (dot notation)
    cli_overrides = {}
    for override in args.override:
        if "=" in override:
            k, v = override.split("=", 1)
            cli_overrides[k] = v

    # Add direct CLI args as overrides using shared utility
    direct_cli_overrides = build_cli_overrides(args)
    cli_overrides.update(direct_cli_overrides)
    # Do NOT generate run_name here; let Trainer handle it for correct CLI/config/auto priority

    # Merge sweep overrides with CLI overrides (CLI takes precedence)
    final_overrides = {**sweep_overrides, **cli_overrides}

    # Load config (YAML/JSON + CLI overrides + sweep overrides)
    config: AppConfig = load_config(args.config, final_overrides)

    # Initialize and run the trainer (Trainer will determine run_name)
    trainer = Trainer(config=config, args=args)
    trainer.run_training_loop()


if __name__ == "__main__":
    # Fix B6: Add multiprocessing.freeze_support() for Windows compatibility
    multiprocessing.freeze_support()
    
    # Set multiprocessing start method for safety, especially with CUDA
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        log_error_to_stderr("Train", f"Could not set multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method(allow_none=True)}")
    except Exception as e:
        log_error_to_stderr("Train", f"Error setting multiprocessing start_method: {e}")

    main()
