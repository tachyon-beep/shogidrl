"""
W&B Sweep-enabled training script for Keisei Shogi RL agent.
This script can be used with both regular training and W&B hyperparameter sweeps.
"""

import argparse
import multiprocessing
import sys

import wandb

from keisei.config_schema import AppConfig
from keisei.utils import load_config
from keisei.training.trainer import Trainer


def apply_wandb_sweep_config():
    """Apply W&B sweep configuration to override parameters."""
    if wandb.run is None:
        return {}
    
    sweep_config = wandb.config
    print(f"Running W&B sweep with config: {dict(sweep_config)}")
    
    # Map W&B sweep parameters to config paths
    sweep_param_mapping = {
        'learning_rate': 'training.learning_rate',
        'gamma': 'training.gamma', 
        'clip_epsilon': 'training.clip_epsilon',
        'ppo_epochs': 'training.ppo_epochs',
        'minibatch_size': 'training.minibatch_size',
        'value_loss_coeff': 'training.value_loss_coeff',
        'entropy_coef': 'training.entropy_coef',
        'tower_depth': 'training.tower_depth',
        'tower_width': 'training.tower_width',
        'se_ratio': 'training.se_ratio',
        'steps_per_epoch': 'training.steps_per_epoch',
        'gradient_clip_max_norm': 'training.gradient_clip_max_norm',
        'lambda_gae': 'training.lambda_gae',
    }
    
    # Apply sweep parameters as overrides
    sweep_overrides = {'wandb.enabled': True}  # Force enable W&B for sweeps
    for sweep_key, config_path in sweep_param_mapping.items():
        if hasattr(sweep_config, sweep_key):
            sweep_overrides[config_path] = getattr(sweep_config, sweep_key)
    
    return sweep_overrides


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

    # Build CLI overrides dict (dot notation)
    cli_overrides = {}
    if args.seed is not None:
        cli_overrides["env.seed"] = args.seed
    if args.device is not None:
        cli_overrides["env.device"] = args.device
    if args.total_timesteps is not None:
        cli_overrides["training.total_timesteps"] = args.total_timesteps

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
        print(
            f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method(allow_none=True)}.",
            file=sys.stderr,
        )
    except OSError as e:
        print(f"Error setting multiprocessing start_method: {e}", file=sys.stderr)

    main()
