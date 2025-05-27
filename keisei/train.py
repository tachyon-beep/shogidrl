"""
Main training script for Keisei Shogi RL agent.
Refactored to use the Trainer class for better modularity.
"""
import os
import sys
import json
import argparse
import multiprocessing
from types import SimpleNamespace
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import config module and related components
from keisei.utils import load_config
from keisei.config_schema import AppConfig
from .trainer import Trainer


def main():
    """Main entry point for the training script (Pydantic config version)."""
    parser = argparse.ArgumentParser(description="Train PPO agent for Shogi with Rich TUI (Pydantic config).")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML or JSON configuration file.")
    parser.add_argument("--run_name", type=str, default=f"ppo_shogi_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Name of the run for logging and checkpointing.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume training from, or 'latest' to auto-detect.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (e.g., 'cpu', 'cuda').")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total timesteps to train for. Overrides config value.")
    parser.add_argument("--savedir", type=str, default=None, help="Directory to save models and logs. Overrides config MODEL_DIR.")
    parser.add_argument("--override", action="append", default=[], help="Override config values via KEY.SUBKEY=VALUE format.")
    args = parser.parse_args()

    # Build CLI overrides dict (dot notation)
    cli_overrides = {}
    for override in args.override:
        if '=' in override:
            k, v = override.split('=', 1)
            cli_overrides[k] = v

    # Add direct CLI args as overrides if set
    if args.seed is not None:
        cli_overrides['env.seed'] = args.seed
    if args.device is not None:
        cli_overrides['env.device'] = args.device
    if args.total_timesteps is not None:
        cli_overrides['training.total_timesteps'] = args.total_timesteps
    if args.savedir is not None:
        cli_overrides['logging.model_dir'] = args.savedir

    # Load config (YAML/JSON + CLI overrides)
    config: AppConfig = load_config(args.config, cli_overrides)

    # Pass run_name and resume directly to Trainer
    trainer = Trainer(config, args)
    trainer.run_training_loop()


if __name__ == "__main__":
    # Set multiprocessing start method for safety, especially with CUDA
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(
            f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method(allow_none=True)}.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Error setting multiprocessing start_method: {e}", file=sys.stderr)

    main()
