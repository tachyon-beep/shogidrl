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
import config as config_module
from .trainer import Trainer


def setup_config() -> SimpleNamespace:
    """Creates a configuration object from config_module constants."""
    cfg = SimpleNamespace()

    # Copy all uppercase constants from config_module to cfg
    for key in dir(config_module):
        if key.isupper():  # Standard convention for constants
            setattr(cfg, key, getattr(config_module, key))

    # Ensure some essential defaults if not in config_module
    cfg.DEVICE = getattr(cfg, "DEVICE", "cpu")
    cfg.TOTAL_TIMESTEPS = getattr(cfg, "TOTAL_TIMESTEPS", 500000)
    cfg.STEPS_PER_EPOCH = getattr(cfg, "STEPS_PER_EPOCH", 2048)  # PPO buffer size
    cfg.LEARNING_RATE = getattr(cfg, "LEARNING_RATE", 3e-4)
    cfg.GAMMA = getattr(cfg, "GAMMA", 0.99)
    cfg.LAMBDA_GAE = getattr(cfg, "LAMBDA_GAE", 0.95)
    cfg.CLIP_EPSILON = getattr(cfg, "CLIP_EPSILON", 0.2)
    cfg.PPO_EPOCHS = getattr(cfg, "PPO_EPOCHS", 10)
    cfg.MINIBATCH_SIZE = getattr(cfg, "MINIBATCH_SIZE", 64)
    cfg.VALUE_LOSS_COEFF = getattr(cfg, "VALUE_LOSS_COEFF", 0.5)
    cfg.ENTROPY_COEFF = getattr(cfg, "ENTROPY_COEFF", 0.01)

    cfg.TRAIN_OUTPUT_DIR = getattr(cfg, "TRAIN_OUTPUT_DIR", "training_output")
    cfg.CHECKPOINT_INTERVAL_TIMESTEPS = getattr(cfg, "CHECKPOINT_INTERVAL_TIMESTEPS", 50000)

    # W&B Defaults
    cfg.WANDB_PROJECT_TRAIN = getattr(cfg, "WANDB_PROJECT_TRAIN", "keisei-shogi-drl")
    cfg.WANDB_ENTITY_TRAIN = getattr(cfg, "WANDB_ENTITY_TRAIN", None)
    cfg.WANDB_MODE = getattr(cfg, "WANDB_MODE", "online")

    # Setup nested EVALUATION_CONFIG
    cfg.EVALUATION_CONFIG = SimpleNamespace()
    cfg.EVALUATION_CONFIG.ENABLE_PERIODIC_EVALUATION = getattr(cfg, "EVAL_DURING_TRAINING", True)
    cfg.EVALUATION_CONFIG.EVALUATION_INTERVAL_TIMESTEPS = getattr(
        cfg, "EVAL_INTERVAL_TIMESTEPS", cfg.CHECKPOINT_INTERVAL_TIMESTEPS
    )
    cfg.EVALUATION_CONFIG.NUM_GAMES = getattr(cfg, "EVAL_NUM_GAMES", 10)
    cfg.EVALUATION_CONFIG.OPPONENT_TYPE = getattr(cfg, "EVAL_OPPONENT_TYPE", "random")
    cfg.EVALUATION_CONFIG.MAX_MOVES_PER_GAME = getattr(cfg, "EVAL_MAX_MOVES_PER_GAME", 200)
    cfg.EVALUATION_CONFIG.DEVICE = getattr(cfg, "EVAL_DEVICE", cfg.DEVICE)
    cfg.EVALUATION_CONFIG.OPPONENT_CHECKPOINT_PATH = getattr(cfg, "EVAL_OPPONENT_CHECKPOINT_PATH", None)
    cfg.EVALUATION_CONFIG.WANDB_LOG_EVAL = getattr(cfg, "WANDB_LOG_EVAL", True)
    cfg.EVALUATION_CONFIG.WANDB_PROJECT_EVAL = getattr(
        cfg, "WANDB_PROJECT_EVAL", cfg.WANDB_PROJECT_TRAIN
    )
    cfg.EVALUATION_CONFIG.WANDB_ENTITY_EVAL = getattr(cfg, "WANDB_ENTITY_EVAL", cfg.WANDB_ENTITY_TRAIN)
    cfg.EVALUATION_CONFIG.WANDB_RUN_NAME_PREFIX = getattr(
        cfg, "WANDB_RUN_NAME_PREFIX_EVAL", "eval_"
    )

    # Board dimensions (standard Shogi)
    cfg.BOARD_ROWS = getattr(cfg, "BOARD_ROWS", 9)
    cfg.BOARD_COLS = getattr(cfg, "BOARD_COLS", 9)
    cfg.INPUT_CHANNELS = getattr(cfg, "INPUT_CHANNELS", 46)  # Default channel count for Shogi observation

    # Rich TUI config
    cfg.RICH_REFRESH_RATE = getattr(cfg, "RICH_REFRESH_RATE", 4)
    cfg.tui_max_log_messages = getattr(cfg, "TUI_MAX_LOG_MESSAGES", 20)

    # Log file path
    cfg.LOG_FILE = getattr(cfg, "LOG_FILE", "training_log.txt")
    cfg.MODEL_DIR = getattr(cfg, "MODEL_DIR", "models")

    return cfg


def update_config_from_dict(config_obj: SimpleNamespace, data_dict: Dict[str, Any]):
    """Update config object from a dictionary of overrides."""
    for key, value in data_dict.items():
        if not hasattr(config_obj, key):
            setattr(config_obj, key, value)
            continue

        target_attr = getattr(config_obj, key)
        if isinstance(target_attr, SimpleNamespace) and isinstance(value, dict):
            update_config_from_dict(target_attr, value)  # Recurse for nested SimpleNamespace
        else:
            try:
                setattr(config_obj, key, value)
            except AttributeError:
                print(f"Warning: Could not set attribute {key} in config.", file=sys.stderr)


def apply_config_overrides(config_obj: Any, overrides_list: List[str]) -> Any:
    """Apply command-line overrides to the config object."""
    if not overrides_list:
        return config_obj
        
    print(f"[INFO] Config overrides specified: {overrides_list}", file=sys.stderr)
    
    for override in overrides_list:
        try:
            key_path, value_str = override.split('=', 1)
            keys = key_path.split('.')
            
            # Parse the value string to appropriate type
            if value_str.lower() == 'true':
                value = True
            elif value_str.lower() == 'false':
                value = False
            elif value_str.lower() == 'none':
                value = None
            else:
                try:
                    # Try parsing as number
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    # Keep as string if not a number
                    value = value_str
            
            # Navigate to the correct nesting level
            current_obj = config_obj
            for key_part in keys[:-1]:
                if not hasattr(current_obj, key_part):
                    setattr(current_obj, key_part, SimpleNamespace())
                current_obj = getattr(current_obj, key_part)
            
            # Set the final attribute
            setattr(current_obj, keys[-1], value)
            print(f"  - Set {key_path} = {value}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error applying override '{override}': {e}", file=sys.stderr)
    
    return config_obj


def parse_args(cfg_defaults: SimpleNamespace):
    """Parse command-line arguments with defaults from config."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for Shogi with Rich TUI."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON configuration file.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"ppo_shogi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name of the run for logging and checkpointing.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from, or 'latest' to auto-detect.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=cfg_defaults.WANDB_PROJECT_TRAIN,
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=cfg_defaults.WANDB_ENTITY_TRAIN,
        help="W&B entity.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=cfg_defaults.WANDB_MODE,
        choices=["online", "offline", "disabled"],
        help="W&B mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg_defaults.DEVICE,
        help="Device to use for training (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help=f"Total timesteps to train for. Overrides config value ({cfg_defaults.TOTAL_TIMESTEPS}).",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help=f"Directory to save models and logs. Overrides config MODEL_DIR ({cfg_defaults.MODEL_DIR}).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values via KEY.SUBKEY=VALUE format.",
    )

    return parser.parse_args()


def main():
    """Main entry point for the training script."""
    # Set up the configuration
    cfg = setup_config()
    
    # Parse command-line arguments
    args = parse_args(cfg)
    
    # Apply configuration overrides from file
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                file_overrides = json.load(f)
            update_config_from_dict(cfg, file_overrides)
        except Exception as e:
            print(
                f"Error loading config file {args.config}: {e}. Using defaults/CLI overrides only.",
                file=sys.stderr,
            )
    
    # Apply command-line overrides
    cfg = apply_config_overrides(cfg, args.override)
    
    # Update cfg with direct argument overrides
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.device is not None:
        cfg.DEVICE = args.device
    if args.total_timesteps is not None:
        cfg.TOTAL_TIMESTEPS = args.total_timesteps
    
    # Set W&B configurations
    cfg.WANDB_PROJECT_TRAIN = args.wandb_project
    cfg.WANDB_ENTITY_TRAIN = args.wandb_entity
    cfg.WANDB_MODE = args.wandb_mode
    
    # Create and run the trainer
    trainer = Trainer(cfg, args)
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
