"""
Main training script for Keisei Shogi RL agent using Rich for TUI.
"""

import os
import json
import glob
import re
import sys
import argparse
import time  # MODIFIED: Added time import
from datetime import datetime
from types import SimpleNamespace  # For creating cfg object
import random  # Added import

import numpy as np
import torch
import wandb

# Rich imports
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)  # MODIFIED: Removed SpeedColumn
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout  # For more complex layouts if needed

from typing import List, Optional, Dict, Any, Tuple

# Keisei imports
from keisei.ppo_agent import PPOAgent
from keisei.experience_buffer import ExperienceBuffer
from keisei.utils import PolicyOutputMapper, TrainingLogger

# ShogiGame and Color are expected to be in keisei.shogi
from keisei.shogi import (
    ShogiGame,
    Color,
)  # Or from keisei.shogi.shogi_game import ShogiGame
from keisei.evaluate import execute_full_evaluation_run


# --- STUB IMPLEMENTATIONS for missing utility functions ---
# These would ideally be in keisei.utils.py or a similar shared module.


def find_latest_checkpoint(model_dir_path: str) -> Optional[str]:
    """Basic stub to find the latest Pytorch checkpoint file based on modification time."""
    try:
        # Common checkpoint naming patterns
        patterns = [
            "checkpoint_ts*.pth",
            "checkpoint_ep*.pth",
            "model_step_*.pt",
            "*.ckpt",
        ]
        checkpoints = []
        for pattern in patterns:
            checkpoints.extend(glob.glob(os.path.join(model_dir_path, pattern)))

        if not checkpoints:
            # Try to list all .pth files if specific patterns fail
            checkpoints = glob.glob(os.path.join(model_dir_path, "*.pth"))
            if not checkpoints:
                checkpoints = glob.glob(
                    os.path.join(model_dir_path, "*.pt")
                )  # Try .pt as well
            if not checkpoints:
                return None

        # Sort by modification time, newest first
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        # print(f"Found checkpoints: {checkpoints}, latest: {checkpoints[0]}", file=sys.stderr)
        return checkpoints[0]
    except (OSError, FileNotFoundError) as e:
        print(f"Error in find_latest_checkpoint: {e}", file=sys.stderr)
        return None


def serialize_config(config_obj: Any) -> str:
    """Basic stub to serialize a config object (SimpleNamespace or class instance) to a JSON string."""
    conf_dict = {}
    if hasattr(config_obj, "__dict__"):
        # For SimpleNamespace or class instances
        source_dict = config_obj.__dict__
    elif isinstance(config_obj, dict):
        source_dict = config_obj
    else:
        # Fallback for module-like objects (less common for dynamic cfg)
        source_dict = {
            key: getattr(config_obj, key)
            for key in dir(config_obj)
            if not key.startswith("__") and not callable(getattr(config_obj, key))
        }

    # Filter out non-JSON serializable types or convert them
    for k, v in source_dict.items():
        if isinstance(v, (int, float, str, bool, list, dict, tuple)) or v is None:
            conf_dict[k] = v
        elif isinstance(v, SimpleNamespace):  # Recursively serialize SimpleNamespace
            conf_dict[k] = json.loads(serialize_config(v))  # Store as dict
        # Add other type conversions if necessary (e.g., torch.device to str)
        elif hasattr(
            v, "__str__"
        ):  # Fallback to string representation for unknown types
            pass  # print(f"Warning: Skipping non-serializable config key '{k}' of type {type(v)}", file=sys.stderr)
            # conf_dict[k] = str(v) # Option: convert to string, but might not be what W&B expects

    try:
        return json.dumps(conf_dict, indent=4, sort_keys=True)
    except TypeError as e:
        print(
            f"Error serializing config: {e}. Partial dict: {conf_dict}", file=sys.stderr
        )
        return "{}"


def apply_config_overrides(config_obj: Any, overrides_list: List[str]) -> Any:
    """Basic stub for applying command-line overrides. Currently a no-op."""
    if overrides_list:
        print(f"[INFO] Config overrides specified: {overrides_list}", file=sys.stderr)
        # A real implementation would parse "KEY.SUBKEY=VALUE" and update config_obj
        # For example:
        # for override in overrides_list:
        #   key_path, value_str = override.split('=', 1)
        #   keys = key_path.split('.')
        #   current_level = config_obj
        #   for key_part in keys[:-1]:
        #       if hasattr(current_level, key_part):
        #           current_level = getattr(current_level, key_part)
        #       else: # Or create if SimpleNamespace
        #           print(f"Warning: Path {key_path} not found in config for override.", file=sys.stderr)
        #           break
        #   else:
        #       # Attempt to cast value_str to appropriate type (e.g., int, float, bool)
        #       # This is complex; for a stub, we might just assign as string or skip.
        #       # setattr(current_level, keys[-1], cast_value(value_str))
        #       print(f"Stub: Would attempt to set {keys[-1]} in {key_path}", file=sys.stderr)
        print(
            "Warning: 'apply_config_overrides' is a stub and does not actually apply overrides yet.",
            file=sys.stderr,
        )
    return config_obj


# --- End of STUB IMPLEMENTATIONS ---


def update_config_from_dict(config_obj: SimpleNamespace, data_dict: Dict[str, Any]):
    for key, value in data_dict.items():
        if not hasattr(config_obj, key):
            setattr(config_obj, key, value)
            continue

        target_attr = getattr(config_obj, key)
        if isinstance(target_attr, SimpleNamespace) and isinstance(value, dict):
            update_config_from_dict(
                target_attr, value
            )  # Recurse for nested SimpleNamespace
        else:
            try:
                setattr(config_obj, key, value)
            except AttributeError:
                print(
                    f"Warning: Could not set attribute {key} in config.",
                    file=sys.stderr,
                )


def parse_args(cfg_defaults: SimpleNamespace):
    parser = argparse.ArgumentParser(
        description="Train PPO agent for Shogi with Rich TUI."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to a YAML/JSON configuration file (currently not deeply integrated).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"ppo_shogi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name of the run for logging and checkpointing.",
    )
    parser.add_argument(
        "--resume",
        type=str,  # Can be path or 'latest'
        default=None,
        help="Path to a checkpoint file to resume training from, or 'latest' to auto-detect.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    # W&B arguments, using defaults from cfg
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

    # Arguments that were previously causing "unrecognized arguments" errors
    parser.add_argument(
        "--device",
        type=str,
        default=cfg_defaults.DEVICE,
        help="Device to use for training (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Overrides the save directory construction. If set, this path is used as run_artifact_dir directly.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=cfg_defaults.TOTAL_TIMESTEPS,
        help="Total number of timesteps to train for.",
    )

    parser.add_argument(
        "--override",
        nargs="+",
        default=[],
        help="Override config values (e.g., TOTAL_TIMESTEPS=10000 LEARNING_RATE=0.001). Stub implementation.",
    )

    return parser.parse_args()


def setup_config() -> SimpleNamespace:
    """Creates a configuration object from config_module constants."""
    cfg = SimpleNamespace()

    # --- Hardcoded defaults and schema values ---
    # These replace the removed config_module imports

    # General training settings
    cfg.DEVICE = "cpu"  # Default device
    cfg.TOTAL_TIMESTEPS = 500000  # Total timesteps for training
    cfg.STEPS_PER_EPOCH = 2048  # PPO buffer size
    cfg.LEARNING_RATE = 3e-4  # Learning rate for the optimizer
    cfg.GAMMA = 0.99  # Discount factor for rewards
    cfg.LAMBDA_GAE = 0.95  # GAE lambda for advantage computation
    cfg.CLIP_EPSILON = 0.2  # PPO clip epsilon
    cfg.PPO_EPOCHS = 10  # Number of PPO epochs per update
    cfg.MINIBATCH_SIZE = 64  # Minibatch size for PPO
    cfg.VALUE_LOSS_COEFF = 0.5  # Coefficient for value loss
    cfg.ENTROPY_COEFF = 0.01  # Coefficient for entropy term

    # Directories and file settings
    cfg.TRAIN_OUTPUT_DIR = "training_output"  # Default output directory
    cfg.CHECKPOINT_INTERVAL_TIMESTEPS = 50000  # Checkpointing interval

    # W&B settings
    cfg.WANDB_PROJECT_TRAIN = "keisei-shogi-drl"
    cfg.WANDB_ENTITY_TRAIN = None  # None uses W&B default
    cfg.WANDB_MODE = "online"

    # Evaluation config
    cfg.EVALUATION_CONFIG = SimpleNamespace()
    cfg.EVALUATION_CONFIG.ENABLE_PERIODIC_EVALUATION = True
    cfg.EVALUATION_CONFIG.EVALUATION_INTERVAL_TIMESTEPS = cfg.CHECKPOINT_INTERVAL_TIMESTEPS
    cfg.EVALUATION_CONFIG.NUM_GAMES = 10
    cfg.EVALUATION_CONFIG.OPPONENT_TYPE = "random"  # 'random', 'heuristic', 'self'
    cfg.EVALUATION_CONFIG.MAX_MOVES_PER_GAME = 256
    cfg.EVALUATION_CONFIG.DEVICE = "cpu"
    cfg.EVALUATION_CONFIG.OPPONENT_CHECKPOINT_PATH = None

    cfg.EVALUATION_CONFIG.WANDB_LOG_EVAL = True
    cfg.EVALUATION_CONFIG.WANDB_PROJECT_EVAL = "keisei-shogi-eval"
    cfg.EVALUATION_CONFIG.WANDB_ENTITY_EVAL = cfg.WANDB_ENTITY_TRAIN  # Default to train entity
    cfg.EVALUATION_CONFIG.WANDB_RUN_NAME_PREFIX = "eval_"

    # Board dimensions (standard Shogi) - PPOAgent might need this if not inferred
    cfg.BOARD_ROWS = 9
    cfg.BOARD_COLS = 9
    cfg.INPUT_CHANNELS = 46  # From config.py

    # Rich TUI settings
    cfg.RICH_MAX_LOG_MESSAGES = 100
    cfg.RICH_REFRESH_RATE = 4  # Hz
    cfg.tui_max_log_messages = 100  # MODIFIED: Added separate setting for TUI log message limit

    return cfg


def main():
    cfg = setup_config()
    args = parse_args(cfg)  # Pass cfg to parse_args for defaults

    # Load from --config_file (JSON) first
    if args.config_file:
        try:
            with open(args.config_file, "r", encoding="utf-8") as f:
                file_overrides = json.load(f)
            update_config_from_dict(cfg, file_overrides)  # Apply these overrides
            # print(f"Loaded config overrides from {args.config_file}", file=sys.stderr) # For debugging
        except (OSError, FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(
                f"Error loading config file {args.config_file}: {e}. Using defaults/CLI overrides only.",
                file=sys.stderr,
            )

    # Apply command-line overrides to cfg (using stub function)
    cfg = apply_config_overrides(cfg, args.override)
    # Update cfg with any args that directly map to cfg attributes if not handled by apply_config_overrides
    if args.seed is not None:
        cfg.SEED = args.seed  # Allow --seed to override config
    if args.device is not None:
        cfg.DEVICE = args.device  # Allow --device to override config
    if args.total_timesteps is not None:
        cfg.TOTAL_TIMESTEPS = (
            args.total_timesteps
        )  # Allow --total-timesteps to override config

    cfg.WANDB_PROJECT_TRAIN = args.wandb_project
    cfg.WANDB_ENTITY_TRAIN = args.wandb_entity
    cfg.WANDB_MODE = args.wandb_mode

    run_name = args.run_name

    # Determine the parent directory for the run artifacts
    if args.savedir:
        parent_artifact_dir = args.savedir
    else:
        parent_artifact_dir = cfg.MODEL_DIR.strip("/")

    # Define the main directory for this run's artifacts.
    # This will be parent_artifact_dir / run_name
    run_artifact_dir = os.path.join(parent_artifact_dir, run_name)

    # Model checkpoints will be saved directly within the run_artifact_dir.
    model_dir = run_artifact_dir

    # Log file will be named based on cfg.LOG_FILE's basename, inside run_artifact_dir.
    log_file_path = os.path.join(run_artifact_dir, os.path.basename(cfg.LOG_FILE))

    # Evaluation log path, also inside run_artifact_dir.
    cfg.EVALUATION_CONFIG.LOG_FILE_PATH_EVAL = os.path.join(
        run_artifact_dir, "rich_periodic_eval_log.txt"
    )

    # Create the main run artifact directory.
    # This single call covers model_dir and the parent directory for log files.
    os.makedirs(run_artifact_dir, exist_ok=True)

    # Save effective config to run_artifact_dir
    try:
        effective_config_str = serialize_config(cfg)
        # with open(os.path.join(run_artifact_dir, "effective_config_rich.json"), "w", encoding="utf-8") # MODIFIED
        with open(
            os.path.join(run_artifact_dir, "effective_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(effective_config_str)
    except Exception as e:  # Already correct
        print(f"Error saving effective_config.json: {e}", file=sys.stderr)

    # Seed everything
    if hasattr(cfg, "SEED") and cfg.SEED is not None:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)  # Standard library random
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.SEED)
        # Consider game.seed(cfg.SEED) if the game env supports it

    # --- Rich TUI Setup ---
    rich_console = Console(
        file=sys.stderr, record=True
    )  # Record for saving log, output to stderr
    rich_log_messages: List[Text] = []  # Stores Text objects for the log panel

    # --- WandB Setup ---
    is_train_wandb_active = cfg.WANDB_MODE != "disabled"
    if is_train_wandb_active:
        try:
            # Convert cfg SimpleNamespace to dict for wandb.init
            config_dict_for_wandb = (
                json.loads(serialize_config(cfg)) if serialize_config(cfg) else {}
            )

            wandb.init(
                project=cfg.WANDB_PROJECT_TRAIN,
                entity=cfg.WANDB_ENTITY_TRAIN,
                name=run_name,
                config=config_dict_for_wandb,
                mode=cfg.WANDB_MODE,
                dir=run_artifact_dir,  # MODIFIED: Use run_artifact_dir for W&B output
                resume="allow",
                id=run_name,  # Use run_name for resumability
            )
        except (OSError, RuntimeError, ValueError) as e:  # Already correct
            rich_console.print(
                f"[bold red]Error initializing W&B: {e}. W&B logging disabled.[/bold red]"
            )
            is_train_wandb_active = False

    if not is_train_wandb_active:
        rich_console.print(
            "[yellow]Weights & Biases logging is disabled or failed to initialize.[/yellow]"
        )

    # Initialize game environment
    try:
        game = ShogiGame()  # Assuming default constructor
        # To get obs_space_shape, we might need to reset and check, or use config
        # temp_obs, _ = game.reset()
        # obs_space_shape = temp_obs.shape # (Channels, Height, Width)
        # For now, rely on cfg.INPUT_CHANNELS and standard board dims
        obs_space_shape = (cfg.INPUT_CHANNELS, cfg.BOARD_ROWS, cfg.BOARD_COLS)
    except Exception as e:  # Corrected
        rich_console.print(
            f"[bold red]Error initializing ShogiGame: {e}. Aborting.[/bold red]"
        )
        return

    policy_output_mapper = PolicyOutputMapper()  # No args
    action_space_size = policy_output_mapper.get_total_actions()

    # Initialize agent
    try:
        agent = PPOAgent(
            input_channels=cfg.INPUT_CHANNELS,  # From cfg
            policy_output_mapper=policy_output_mapper,
            learning_rate=cfg.LEARNING_RATE,
            gamma=cfg.GAMMA,
            clip_epsilon=cfg.CLIP_EPSILON,
            ppo_epochs=cfg.PPO_EPOCHS,
            minibatch_size=cfg.MINIBATCH_SIZE,
            value_loss_coeff=cfg.VALUE_LOSS_COEFF,
            entropy_coef=cfg.ENTROPY_COEFF,
            device=cfg.DEVICE,
            name=run_name,  # Agent name
        )
    except Exception as e:  # Corrected
        rich_console.print(
            f"[bold red]Error initializing PPOAgent: {e}. Aborting.[/bold red]"
        )
        if is_train_wandb_active and wandb.run:
            wandb.finish(exit_code=1)
        return

    # Experience buffer
    experience_buffer = ExperienceBuffer(
        buffer_size=cfg.STEPS_PER_EPOCH,
        gamma=cfg.GAMMA,
        lambda_gae=cfg.LAMBDA_GAE,
        device=cfg.DEVICE,
    )

    global_timestep = 0
    total_episodes_completed = 0

    # --- Checkpoint resume logic ---
    potential_resume_path = None
    attempt_resume = False

    if args.resume:
        attempt_resume = True
        if args.resume == "latest":
            potential_resume_path = find_latest_checkpoint(model_dir)
            if potential_resume_path:
                rich_console.print(
                    f"Found latest checkpoint via --resume latest: {potential_resume_path}"
                )
                print(
                    f"Found latest checkpoint via --resume latest: {potential_resume_path}",
                    file=sys.stderr,
                )
            else:
                rich_console.print(
                    f"[yellow]'--resume latest' specified, but no checkpoint found in {model_dir}. Starting fresh.[/yellow]"
                )
                print(
                    f"'--resume latest' specified, but no checkpoint found in {model_dir}. Starting fresh.",
                    file=sys.stderr,
                )
        else:  # Specific path given
            potential_resume_path = args.resume
            if not os.path.exists(potential_resume_path):
                rich_console.print(
                    f"[bold red]Specified resume checkpoint {potential_resume_path} not found. Starting fresh.[/bold red]"
                )
                print(
                    f"Specified resume checkpoint {potential_resume_path} not found. Starting fresh.",
                    file=sys.stderr,
                )
                potential_resume_path = None  # Ensure it's None if path doesn't exist
    else:  # args.resume is None, try auto-detection
        attempt_resume = True  # We will attempt to find one
        # print(f"No --resume flag. Checking for existing checkpoints in {model_dir} for auto-resume...", file=sys.stderr)
        potential_resume_path = find_latest_checkpoint(model_dir)
        if potential_resume_path:
            rich_console.print(
                f"Auto-detected checkpoint: {potential_resume_path}. Attempting to resume."
            )
            print(
                f"Auto-detected checkpoint: {potential_resume_path}. Attempting to resume.",
                file=sys.stderr,
            )
        else:
            # rich_console.print(f"No checkpoint found in {model_dir} for auto-resume. Starting fresh.")
            # print(f"No checkpoint found in {model_dir} for auto-resume. Starting fresh.", file=sys.stderr)
            attempt_resume = False  # No checkpoint found to attempt resume

    if (
        attempt_resume
        and potential_resume_path
        and os.path.exists(potential_resume_path)
    ):
        try:
            checkpoint_data = agent.load_model(potential_resume_path)
            global_timestep = checkpoint_data.get("global_timestep", 0)
            total_episodes_completed = checkpoint_data.get(
                "total_episodes_completed", 0
            )
            # Load win/loss/draw counts if they exist in the checkpoint
            black_wins = checkpoint_data.get("black_wins", 0)
            white_wins = checkpoint_data.get("white_wins", 0)
            draws = checkpoint_data.get("draws", 0)
            rich_console.print(
                f"[green]Resumed training from checkpoint: {potential_resume_path} at timestep {global_timestep}[/green]"
            )
            print(
                f"Resuming from checkpoint: {potential_resume_path} at timestep {global_timestep}",
                file=sys.stderr,
            )
        except (OSError, RuntimeError, ValueError) as e:
            rich_console.print(
                f"[bold red]Error loading checkpoint {potential_resume_path}: {e}. Starting fresh.[/bold red]"
            )
            print(
                f"Error loading checkpoint {potential_resume_path}: {e}. Starting fresh.",
                file=sys.stderr,
            )
            global_timestep = 0
            total_episodes_completed = 0
            black_wins = 0
            white_wins = 0
            draws = 0
    elif (
        args.resume
        and args.resume != "latest"
        and not (potential_resume_path and os.path.exists(potential_resume_path))
    ):
        if args.resume and args.resume != "latest":  # Re-check condition
            rich_console.print(
                f"[bold red]Specified resume checkpoint {args.resume} not found. Starting fresh.[/bold red]"
            )
            print(
                f"Specified resume checkpoint {args.resume} not found. Starting fresh.",
                file=sys.stderr,
            )  # ADDED
        black_wins = 0
        white_wins = 0
        draws = 0
    else:  # No resume or auto-detection failed, or fresh start
        global_timestep = 0
        total_episodes_completed = 0
        black_wins = 0
        white_wins = 0
        draws = 0

    # TrainingLogger context manager
    with TrainingLogger(
        log_file_path, rich_console=rich_console, rich_log_panel=rich_log_messages
    ) as logger:

        def log_both(
            message: str,
            level: str = "info",
            also_to_wandb: bool = False,
            wandb_data: Optional[Dict] = None,
        ):
            # TrainingLogger's log method now handles Rich panel appending
            logger.log(message)  # Logs to file and appends to rich_log_messages
            if is_train_wandb_active and also_to_wandb and wandb.run:
                log_payload = {"train_message": message}
                if wandb_data:
                    log_payload.update(wandb_data)
                wandb.log(log_payload, step=global_timestep)

        # Log session start
        session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_both(f"--- SESSION START: {run_name} at {session_start_time} ---")

        # MODIFIED: Changed "Rich TUI" to "Keisei" and added placeholder for W&B link
        run_title = f"Keisei Training Run: {run_name}"
        if is_train_wandb_active and wandb.run and hasattr(wandb.run, "url"):
            run_title += f" (W&B: {wandb.run.url})"
        log_both(run_title)
        log_both(f"Run directory: {run_artifact_dir}")
        # log_both(f"Effective config saved to: {os.path.join(run_artifact_dir, 'effective_config_rich.json')}") # MODIFIED
        log_both(
            f"Effective config saved to: {os.path.join(run_artifact_dir, 'effective_config.json')}"
        )
        if hasattr(cfg, "SEED") and cfg.SEED is not None:
            log_both(f"Random seed: {cfg.SEED}")
        log_both(f"Device: {cfg.DEVICE}")
        log_both(f"Agent: {type(agent).__name__} ({agent.name})")
        log_both(
            f"Total timesteps: {cfg.TOTAL_TIMESTEPS}, Steps per PPO epoch: {cfg.STEPS_PER_EPOCH}"
        )
        if global_timestep > 0:
            log_both(
                f"Resuming from timestep {global_timestep}, {total_episodes_completed} episodes completed."
            )
        else:
            log_both("Starting fresh training.")

        log_both(
            f"Model Structure:\n{agent.model}", also_to_wandb=False
        )  # Keep model structure log local

        # --- RICH Progress Bar and Layout ---
        progress_bar = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),  # Percentage
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn(
                "• Steps: {task.completed}/{task.total} ({task.fields[speed]:.1f} it/s)"
            ),  # MODIFIED: Integrated speed here
            TextColumn("• {task.fields[ep_metrics]}", style="bright_cyan"),
            TextColumn("• {task.fields[ppo_metrics]}", style="bright_yellow"),
            TextColumn(
                "• Wins B:{task.fields[black_wins_cum]} W:{task.fields[white_wins_cum]} D:{task.fields[draws_cum]}",
                style="bright_green",
            ),
            TextColumn(
                "• Rates B:{task.fields[black_win_rate]:.1%} W:{task.fields[white_win_rate]:.1%} D:{task.fields[draw_rate]:.1%}",
                style="bright_blue",
            ),
            console=rich_console,
            transient=False,
        )

        initial_black_win_rate = (
            black_wins / total_episodes_completed
            if total_episodes_completed > 0
            else 0.0
        )
        initial_white_win_rate = (
            white_wins / total_episodes_completed
            if total_episodes_completed > 0
            else 0.0
        )
        initial_draw_rate = (
            draws / total_episodes_completed if total_episodes_completed > 0 else 0.0
        )

        training_task = progress_bar.add_task(
            "Training",
            total=cfg.TOTAL_TIMESTEPS,
            completed=global_timestep,
            ep_metrics="Ep L:0 R:0.0",
            ppo_metrics="",
            black_wins_cum=black_wins,
            white_wins_cum=white_wins,
            draws_cum=draws,
            black_win_rate=initial_black_win_rate,
            white_win_rate=initial_white_win_rate,
            draw_rate=initial_draw_rate,
            speed=0.0,  # MODIFIED: Add speed field
            start=(global_timestep < cfg.TOTAL_TIMESTEPS),
        )

        log_panel = Panel(
            Text(""),  # Initial content, updated in the loop
            title="[b]Live Training Log[/b]",
            border_style="bright_green",
            expand=True,
        )

        # Simple layout: Log panel above, progress bar below
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="main_log", ratio=1),  # Log panel takes most space
            Layout(
                name="progress_display", size=4
            ),  # Progress bar fixed size, changed from progress_bar.height + 1
        )
        layout["main_log"].update(log_panel)
        layout["progress_display"].update(progress_bar)

        # --- MAIN TRAINING LOOP ---
        try:
            # ShogiGame.reset() now returns np.ndarray directly
            reset_result = game.reset()
            if not isinstance(reset_result, np.ndarray) or reset_result.ndim == 0:
                log_both(
                    f"CRITICAL: Initial game.reset() did not return valid observation. Got: {type(reset_result)}. Aborting.",
                    level="error",
                    also_to_wandb=True,
                )
                if is_train_wandb_active and wandb.run:
                    wandb.finish(exit_code=1)
                return
            current_obs_np: np.ndarray = reset_result
        except Exception as e:
            log_both(
                f"CRITICAL: Error during initial game.reset(): {e}. Aborting.",
                level="error",
                also_to_wandb=True,
            )
            if is_train_wandb_active and wandb.run:
                wandb.finish(exit_code=1)
            return

        current_episode_reward = 0.0
        current_episode_length = 0

        current_obs_tensor = torch.tensor(
            current_obs_np, dtype=torch.float32, device=cfg.DEVICE
        ).unsqueeze(0)

        last_time = time.time()  # For speed calculation
        steps_since_last_time = 0  # For speed calculation

        with Live(
            layout,
            console=rich_console,
            refresh_per_second=cfg.RICH_REFRESH_RATE,
            transient=False,
            vertical_overflow="visible",
        ) as live:
            while global_timestep < cfg.TOTAL_TIMESTEPS:
                legal_shogi_moves = game.get_legal_moves()
                if not legal_shogi_moves:
                    log_both(
                        f"Warning: No legal moves available at timestep {global_timestep}.",
                        level="warning",
                    )
                    legal_mask_tensor = torch.zeros(
                        action_space_size, dtype=torch.bool, device=cfg.DEVICE
                    )
                else:
                    legal_mask_tensor = policy_output_mapper.get_legal_mask(
                        legal_shogi_moves, device=cfg.DEVICE
                    )

                if not legal_mask_tensor.any():
                    log_both(
                        f"Warning: No legal actions in mask at timestep {global_timestep}. This might be game end or mapping issue.",
                        level="warning",
                    )
                    pass

                selected_shogi_move, policy_index, log_prob, value_pred = (
                    agent.select_action(
                        current_obs_np,
                        legal_shogi_moves,
                        legal_mask_tensor,
                        is_training=True,
                    )
                )

                if selected_shogi_move is None:
                    log_both(
                        f"CRITICAL: Agent failed to select a move at timestep {global_timestep}. Resetting episode.",
                        level="error",
                        also_to_wandb=True,
                    )
                    reset_result_agent_fail = game.reset()
                    if not isinstance(reset_result_agent_fail, np.ndarray):
                        log_both(
                            f"CRITICAL: game.reset() after agent failure did not return ndarray. Got {type(reset_result_agent_fail)}. Aborting.",
                            "error",
                            True,
                        )
                        if is_train_wandb_active and wandb.run:
                            wandb.finish(exit_code=1)
                        return
                    current_obs_np = reset_result_agent_fail
                    current_obs_tensor = torch.tensor(
                        current_obs_np, dtype=torch.float32, device=cfg.DEVICE
                    ).unsqueeze(0)
                    current_episode_reward = 0.0
                    current_episode_length = 0
                    continue

                # Environment step
                next_obs_np_typed: np.ndarray
                reward_typed: float
                done_typed: bool
                info_typed: Dict[str, Any]

                try:
                    # Get piece information BEFORE making the move for demo logging
                    piece_info_for_demo = None
                    if (
                        getattr(cfg, "ENABLE_DEMO_MODE", False)
                        and selected_shogi_move is not None
                    ):
                        if (
                            len(selected_shogi_move) == 5
                            and selected_shogi_move[0] is not None
                            and selected_shogi_move[1] is not None
                        ):
                            # Board move - get piece at source position
                            from_r, from_c = (
                                selected_shogi_move[0],
                                selected_shogi_move[1],
                            )
                            try:
                                piece_info_for_demo = game.get_piece(from_r, from_c)
                            except:
                                piece_info_for_demo = None

                    # We expect a 4-tuple here as is_simulation is False by default in ShogiGame.make_move
                    # and we are not passing it.
                    move_result = game.make_move(selected_shogi_move)
                    if not (
                        isinstance(move_result, tuple)
                        and len(move_result) == 4
                        and isinstance(move_result[0], np.ndarray)
                        and isinstance(move_result[1], float)
                        and isinstance(move_result[2], bool)
                        and isinstance(move_result[3], dict)
                    ):
                        log_both(
                            f"CRITICAL: game.make_move did not return expected 4-tuple. Got: {type(move_result)}. Aborting episode.",
                            "error",
                            True,
                        )
                        done_typed = True
                        obs_space_shape = (
                            cfg.INPUT_CHANNELS,
                            cfg.BOARD_ROWS,
                            cfg.BOARD_COLS,
                        )
                        next_obs_np_typed = np.zeros(obs_space_shape, dtype=np.float32)
                        reward_typed = -1.0
                        info_typed = {
                            "error_in_make_move_result_type": str(type(move_result))
                        }
                    else:
                        next_obs_np_typed, reward_typed, done_typed, info_typed = (
                            move_result
                        )

                        # Demo mode per-move logging and delay
                        if getattr(cfg, "ENABLE_DEMO_MODE", False):
                            current_player_name = (
                                getattr(
                                    game.current_player,
                                    "name",
                                    str(game.current_player),
                                )
                                if hasattr(game, "current_player")
                                else "Unknown"
                            )
                            move_str = format_move_with_description_enhanced(
                                selected_shogi_move,
                                policy_output_mapper,
                                piece_info_for_demo,
                            )
                            log_both(
                                f"Move {current_episode_length + 1}: {current_player_name} played {move_str}"
                            )

                            # Add delay for easier observation
                            demo_delay = getattr(cfg, "DEMO_MODE_DELAY", 0.5)
                            if demo_delay > 0:
                                time.sleep(demo_delay)
                except (RuntimeError, ValueError, OSError) as e:
                    log_both(
                        f"CRITICAL: Error during game.make_move(): {e}. Move: {selected_shogi_move}. Aborting episode.",
                        level="error",
                        also_to_wandb=True,
                    )
                    done_typed = True
                    obs_space_shape = (
                        cfg.INPUT_CHANNELS,
                        cfg.BOARD_ROWS,
                        cfg.BOARD_COLS,
                    )
                    next_obs_np_typed = np.copy(
                        current_obs_np
                    )  # Fallback to current obs
                    reward_typed = -1.0
                    info_typed = {"error_in_make_move_exception": str(e)}

                current_episode_reward += reward_typed
                current_episode_length += 1

                experience_buffer.add(
                    current_obs_tensor.squeeze(0),
                    policy_index,
                    reward_typed,
                    log_prob,
                    value_pred,
                    done_typed,
                    legal_mask_tensor,
                )

                current_obs_np = next_obs_np_typed
                current_obs_tensor = torch.tensor(
                    current_obs_np, dtype=torch.float32, device=cfg.DEVICE
                ).unsqueeze(0)

                if done_typed:
                    total_episodes_completed += 1
                    ep_metrics_str = (
                        f"Ep L:{current_episode_length} R:{current_episode_reward:.2f}"
                    )

                    game_outcome_message = "Game outcome: Unknown"
                    winner_color = None

                    if "winner" in info_typed:
                        winner = info_typed["winner"]
                        if winner is not None:
                            game_outcome_message = f"Game outcome: {winner.name} won."
                            winner_color = winner
                        else:
                            game_outcome_message = "Game outcome: Draw."
                    elif game.winner is not None:
                        winner = game.winner
                        game_outcome_message = f"Game outcome: {winner.name} won."
                        winner_color = winner
                    elif game.game_over and game.winner is None:
                        game_outcome_message = (
                            "Game outcome: Draw (max moves or stalemate)."
                        )

                    if winner_color == Color.BLACK:
                        black_wins += 1
                    elif winner_color == Color.WHITE:
                        white_wins += 1
                    else:
                        draws += 1

                    current_black_win_rate = (
                        black_wins / total_episodes_completed
                        if total_episodes_completed > 0
                        else 0.0
                    )
                    current_white_win_rate = (
                        white_wins / total_episodes_completed
                        if total_episodes_completed > 0
                        else 0.0
                    )
                    current_draw_rate = (
                        draws / total_episodes_completed
                        if total_episodes_completed > 0
                        else 0.0
                    )

                    progress_bar.update(
                        training_task,
                        advance=0,
                        ep_metrics=ep_metrics_str,
                        black_wins_cum=black_wins,
                        white_wins_cum=white_wins,
                        draws_cum=draws,
                        black_win_rate=current_black_win_rate,
                        white_win_rate=current_white_win_rate,
                        draw_rate=current_draw_rate,
                    )

                    log_both(
                        f"Episode {total_episodes_completed} finished. Length: {current_episode_length}, Reward: {current_episode_reward:.2f}. {game_outcome_message}",
                        also_to_wandb=True,
                        wandb_data={
                            "episode_reward": current_episode_reward,
                            "episode_length": current_episode_length,
                            "total_episodes": total_episodes_completed,
                            "black_wins_cumulative": black_wins,
                            "white_wins_cumulative": white_wins,
                            "draws_cumulative": draws,  # MODIFIED: More descriptive keys for W&B
                            "black_win_rate": current_black_win_rate,
                            "white_win_rate": current_white_win_rate,
                            "draw_rate": current_draw_rate,
                        },
                    )  # MODIFIED: Log rates to W&B

                    reset_result_done = game.reset()
                    if not isinstance(reset_result_done, np.ndarray):
                        log_both(
                            f"CRITICAL: game.reset() after episode done did not return ndarray. Got {type(reset_result_done)}. Aborting.",
                            "error",
                            True,
                        )
                        if is_train_wandb_active and wandb.run:
                            wandb.finish(exit_code=1)
                        return
                    current_obs_np = reset_result_done
                    current_obs_tensor = torch.tensor(
                        current_obs_np, dtype=torch.float32, device=cfg.DEVICE
                    ).unsqueeze(0)
                    current_episode_reward = 0.0
                    current_episode_length = 0

                # PPO Update
                if (
                    (global_timestep + 1) % cfg.STEPS_PER_EPOCH == 0
                    and experience_buffer.ptr == cfg.STEPS_PER_EPOCH
                ):
                    with torch.no_grad():
                        last_value_pred_for_gae = agent.get_value(current_obs_np)

                    experience_buffer.compute_advantages_and_returns(
                        last_value_pred_for_gae
                    )

                    learn_metrics = agent.learn(experience_buffer)
                    experience_buffer.clear()

                    ppo_metrics_str_parts = []
                    # MODIFIED: Changed formatting from .2f to .4f for PPO metrics
                    if "ppo/kl_divergence_approx" in learn_metrics:
                        ppo_metrics_str_parts.append(
                            f"KL:{learn_metrics['ppo/kl_divergence_approx']:.4f}"
                        )
                    if "ppo/policy_loss" in learn_metrics:
                        ppo_metrics_str_parts.append(
                            f"PolL:{learn_metrics['ppo/policy_loss']:.4f}"
                        )
                    if "ppo/value_loss" in learn_metrics:
                        ppo_metrics_str_parts.append(
                            f"ValL:{learn_metrics['ppo/value_loss']:.4f}"
                        )
                    if "ppo/entropy" in learn_metrics:
                        ppo_metrics_str_parts.append(
                            f"Ent:{learn_metrics['ppo/entropy']:.4f}"
                        )

                    ppo_metrics_display = " ".join(ppo_metrics_str_parts)
                    progress_bar.update(
                        training_task, advance=0, ppo_metrics=ppo_metrics_display
                    )

                    log_both(
                        f"PPO Update @ ts {global_timestep+1}. Metrics: {json.dumps({k: f'{v:.4f}' for k,v in learn_metrics.items()})}",
                        also_to_wandb=True,
                        wandb_data=learn_metrics,
                    )

                # Checkpointing
                if (global_timestep + 1) % cfg.CHECKPOINT_INTERVAL_TIMESTEPS == 0:
                    ckpt_save_path = os.path.join(
                        model_dir, f"checkpoint_ts{global_timestep+1}.pth"
                    )
                    try:
                        agent.save_model(
                            ckpt_save_path,
                            global_timestep + 1,
                            total_episodes_completed,
                            # Pass the new stats to save_model
                            stats_to_save={
                                "black_wins": black_wins,
                                "white_wins": white_wins,
                                "draws": draws,
                            },
                        )
                        log_both(
                            f"Checkpoint saved to {ckpt_save_path}", also_to_wandb=True
                        )
                    except Exception as e:
                        log_both(
                            f"Error saving checkpoint {ckpt_save_path}: {e}",
                            level="error",
                            also_to_wandb=True,
                        )

                # Periodic Evaluation
                if (
                    cfg.EVALUATION_CONFIG.ENABLE_PERIODIC_EVALUATION
                    and (global_timestep + 1)
                    % cfg.EVALUATION_CONFIG.EVALUATION_INTERVAL_TIMESTEPS
                    == 0
                ):

                    # Save a temporary checkpoint for evaluation if different from main checkpointing
                    eval_ckpt_path = os.path.join(
                        model_dir, f"eval_checkpoint_ts{global_timestep+1}.pth"
                    )
                    agent.save_model(
                        eval_ckpt_path, global_timestep + 1, total_episodes_completed
                    )
                    log_both(
                        f"Starting periodic evaluation at timestep {global_timestep + 1}...",
                        also_to_wandb=True,
                    )

                    agent.model.eval()  # Set agent to evaluation mode

                    eval_results = execute_full_evaluation_run(
                        agent_checkpoint_path=eval_ckpt_path,
                        opponent_type=cfg.EVALUATION_CONFIG.OPPONENT_TYPE,
                        opponent_checkpoint_path=cfg.EVALUATION_CONFIG.OPPONENT_CHECKPOINT_PATH,
                        num_games=cfg.EVALUATION_CONFIG.NUM_GAMES,
                        max_moves_per_game=cfg.EVALUATION_CONFIG.MAX_MOVES_PER_GAME,
                        device_str=cfg.EVALUATION_CONFIG.DEVICE,  # Device for eval
                        log_file_path_eval=cfg.EVALUATION_CONFIG.LOG_FILE_PATH_EVAL,
                        policy_mapper=policy_output_mapper,
                        seed=(
                            cfg.SEED if hasattr(cfg, "SEED") else None
                        ),  # Use training seed for consistency
                        wandb_log_eval=cfg.EVALUATION_CONFIG.WANDB_LOG_EVAL,
                        wandb_project_eval=cfg.EVALUATION_CONFIG.WANDB_PROJECT_EVAL,
                        wandb_entity_eval=cfg.EVALUATION_CONFIG.WANDB_ENTITY_EVAL,
                        wandb_run_name_eval=f"{cfg.EVALUATION_CONFIG.WANDB_RUN_NAME_PREFIX}{run_name}_ts{global_timestep+1}",
                        wandb_group=run_name,  # Group eval runs under the main training run in W&B
                        wandb_reinit=True,  # Important if W&B is used in the same process
                        logger_also_stdout=False,  # Let Rich handle TUI output
                    )
                    agent.model.train()  # Set agent back to training mode
                    log_both(
                        f"Periodic evaluation finished. Results: {eval_results}",
                        also_to_wandb=True,
                        wandb_data=(
                            eval_results
                            if isinstance(eval_results, dict)
                            else {"eval_summary": str(eval_results)}
                        ),
                    )

                # MODIFIED: Increment global_timestep and update progress bar for step counting
                global_timestep += 1
                steps_since_last_time += 1

                current_time = time.time()
                time_delta = current_time - last_time
                current_speed = 0.0
                if time_delta > 1:  # Update speed roughly every second
                    current_speed = steps_since_last_time / time_delta
                    last_time = current_time
                    steps_since_last_time = 0

                if training_task is not None:
                    progress_bar.update(
                        training_task,
                        completed=global_timestep,
                        speed=(
                            current_speed
                            if current_speed > 0
                            else progress_bar.tasks[training_task].fields.get(
                                "speed", 0.0
                            )
                        ),
                    )

                # Update Rich log panel display with latest messages
                if rich_log_messages:
                    display_messages = rich_log_messages[
                        -cfg.tui_max_log_messages :
                    ]  # Limit displayed messages
                    log_panel.renderable = (
                        Group(*display_messages) if display_messages else Text("")
                    )

            # End of training loop
            if (
                training_task is not None
                and not progress_bar.tasks[training_task].finished
            ):
                progress_bar.update(
                    training_task, completed=cfg.TOTAL_TIMESTEPS
                )  # Ensure it shows 100% if loop finishes early due to other conditions

        # After Live context exits
        # Save final log
        log_both(
            f"Training loop finished at timestep {global_timestep}. Total episodes: {total_episodes_completed}.",
            also_to_wandb=True,
        )

        if global_timestep >= cfg.TOTAL_TIMESTEPS:
            log_both(
                "Training successfully completed all timesteps.", also_to_wandb=True
            )
            final_model_path = os.path.join(model_dir, "final_model.pth")
            try:
                agent.save_model(
                    final_model_path, global_timestep, total_episodes_completed
                )
                log_both(f"Final model saved to {final_model_path}", also_to_wandb=True)
                if is_train_wandb_active and wandb.run:
                    model_artifact = wandb.Artifact(f"{run_name}-model", type="model")
                    model_artifact.add_file(final_model_path)
                    wandb.log_artifact(model_artifact)
                    log_both("Final model logged as W&B artifact.")
            except Exception as e:  # Corrected
                log_both(
                    f"Error saving final model {final_model_path}: {e}",
                    level="error",
                    also_to_wandb=True,
                )
        else:
            log_both(
                f"Training interrupted at timestep {global_timestep} (before {cfg.TOTAL_TIMESTEPS} total).",
                level="warning",
                also_to_wandb=True,
            )

        if is_train_wandb_active and wandb.run:
            wandb.finish()
            log_both("Weights & Biases run finished.")

        # Save the full console log from Rich
        console_log_path = os.path.join(
            run_artifact_dir, "full_console_output_rich.html"
        )  # Or .txt # MODIFIED: Use run_artifact_dir
        try:
            rich_console.save_html(console_log_path)  # Or save_text
            print(
                f"Full Rich console output saved to {console_log_path}", file=sys.stderr
            )
        except Exception as e:  # Corrected
            print(f"Error saving Rich console log: {e}", file=sys.stderr)

    # --- End of TrainingLogger context ---
    # Final messages to actual stderr after Rich's Live context is done.
    rich_console.rule("[bold green]Run Finished[/bold green]")
    rich_console.print(
        f"[bold green]Run '{run_name}' processing finished.[/bold green]"
    )
    rich_console.print(
        f"Output and logs are in: {run_artifact_dir}"
    )  # MODIFIED: Use run_artifact_dir


if __name__ == "__main__":
    # Set multiprocessing start method. 'spawn' is often safer, especially with CUDA.
    # This should be done outside main() if main() can be imported and called multiple times,
    # or if other parts of the module use multiprocessing.
    # For a script that's the main entry point, here is fine.
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method(
                "spawn", force=True
            )  # force=True if confident
    except RuntimeError as e:  # Already correct
        # This can happen if the context has already been set or used.
        print(
            f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method(allow_none=True)}.",
            file=sys.stderr,
        )
    except Exception as e:  # Corrected
        print(f"Error setting multiprocessing start_method: {e}", file=sys.stderr)

    main()

# --- End of STUB IMPLEMENTATIONS ---


def format_move_with_description(selected_shogi_move, policy_output_mapper, game=None):
    """
    Formats a shogi move with USI notation and English description.

    Args:
        selected_shogi_move: The MoveTuple (either BoardMoveTuple or DropMoveTuple)
        policy_output_mapper: PolicyOutputMapper instance for USI conversion
        game: Optional ShogiGame instance for getting piece information

    Returns:
        str: Formatted string like "7g7f (pawn move to 7f)" or "P*5e (pawn drop to 5e)"
    """
    if selected_shogi_move is None:
        return "None"

    try:
        # Get USI notation
        usi_notation = policy_output_mapper.shogi_move_to_usi(selected_shogi_move)

        # Determine if it's a drop or board move and create description
        if len(selected_shogi_move) == 5 and selected_shogi_move[0] is None:
            # Drop move: (None, None, to_r, to_c, piece_type)
            _, _, to_r, to_c, piece_type = selected_shogi_move
            piece_name = _get_piece_name(piece_type, False)
            to_square = _coords_to_square_name(to_r, to_c)
            description = f"{piece_name} drop to {to_square}"
        else:
            # Board move: (from_r, from_c, to_r, to_c, promote_flag)
            from_r, from_c, to_r, to_c, promote_flag = selected_shogi_move
            from_square = _coords_to_square_name(from_r, from_c)
            to_square = _coords_to_square_name(to_r, to_c)

            # Try to get piece information from game if available
            piece_name = "piece"
            if game is not None:
                try:
                    piece = game.get_piece(from_r, from_c)
                    if piece is not None:
                        piece_name = _get_piece_name(piece.type, promote_flag)
                except:
                    pass  # Fall back to generic "piece"
            else:
                # Without game context, assume it's a piece that can promote if promote_flag is True
                if promote_flag:
                    piece_name = "piece promoting"

            description = f"{piece_name} moving from {from_square} to {to_square}"

        return f"{usi_notation} - {description}."

    except Exception as e:
        # Fallback to string representation if formatting fails
        return f"{str(selected_shogi_move)} (format error: {e})"


def _get_piece_name(piece_type, is_promoting=False):
    """Convert PieceType enum to Japanese name with English translation."""
    from keisei.shogi.shogi_core_definitions import PieceType

    piece_names = {
        PieceType.PAWN: "Fuhyō (Pawn)",
        PieceType.LANCE: "Kyōsha (Lance)",
        PieceType.KNIGHT: "Keima (Knight)",
        PieceType.SILVER: "Ginsho (Silver General)",
        PieceType.GOLD: "Kinshō (Gold General)",
        PieceType.BISHOP: "Kakugyō (Bishop)",
        PieceType.ROOK: "Hisha (Rook)",
        PieceType.KING: "Ōshō (King)",
        PieceType.PROMOTED_PAWN: "Tokin (Promoted Pawn)",
        PieceType.PROMOTED_LANCE: "Narikyo (Promoted Lance)",
        PieceType.PROMOTED_KNIGHT: "Narikei (Promoted Knight)",
        PieceType.PROMOTED_SILVER: "Narigin (Promoted Silver)",
        PieceType.PROMOTED_BISHOP: "Ryūma (Dragon Horse)",
        PieceType.PROMOTED_ROOK: "Ryūō (Dragon King)",
    }

    # If promoting during this move, show the transformation
    base_names = {
        PieceType.PAWN: "Fuhyō (Pawn) → Tokin (Promoted Pawn)",
        PieceType.LANCE: "Kyōsha (Lance) → Narikyo (Promoted Lance)",
        PieceType.KNIGHT: "Keima (Knight) → Narikei (Promoted Knight)",
        PieceType.SILVER: "Ginsho (Silver General) → Narigin (Promoted Silver)",
        PieceType.BISHOP: "Kakugyō (Bishop) → Ryūma (Dragon Horse)",
        PieceType.ROOK: "Hisha (Rook) → Ryūō (Dragon King)",
    }
    return base_names.get(piece_type, piece_names.get(piece_type, str(piece_type)))

    return piece_names.get(piece_type, str(piece_type))


def _coords_to_square_name(row, col):
    """Convert 0-indexed coordinates to square name like '7f'."""
    file = str(
        9 - col
    )  # Convert column to file (9-col because shogi files go 9-1 from left to right)
    rank = chr(ord("a") + row)  # Convert row to rank (a-i)
    return f"{file}{rank}"


# --- End of helper functions ---


def format_move_with_description_enhanced(
    selected_shogi_move, policy_output_mapper, piece_info=None
):
    """
    Enhanced move formatting that takes piece info as parameter for better demo logging.

    Args:
        selected_shogi_move: The MoveTuple (either BoardMoveTuple or DropMoveTuple)
        policy_output_mapper: PolicyOutputMapper instance for USI conversion
        piece_info: Piece object from game.get_piece() call made before the move

    Returns:
        str: Formatted string like "7g7f - Fuhyō (Pawn) moving from 7g to 7f."
    """
    if selected_shogi_move is None:
        return "None"

    try:
        # Get USI notation
        usi_notation = policy_output_mapper.shogi_move_to_usi(selected_shogi_move)

        # Determine if it's a drop or board move and create description
        if len(selected_shogi_move) == 5 and selected_shogi_move[0] is None:
            # Drop move: (None, None, to_r, to_c, piece_type)
            _, _, to_r, to_c, piece_type = selected_shogi_move
            piece_name = _get_piece_name(piece_type, False)
            to_square = _coords_to_square_name(to_r, to_c)
            description = f"{piece_name} drop to {to_square}"
        else:
            # Board move: (from_r, from_c, to_r, to_c, promote_flag)
            from_r, from_c, to_r, to_c, promote_flag = selected_shogi_move
            from_square = _coords_to_square_name(from_r, from_c)
            to_square = _coords_to_square_name(to_r, to_c)

            # Use the piece info passed as parameter if available
            piece_name = "piece"
            if piece_info is not None:
                try:
                    piece_name = _get_piece_name(piece_info.type, promote_flag)
                except:
                    piece_name = "piece"  # Fall back to generic "piece"

            description = f"{piece_name} moving from {from_square} to {to_square}"

        return f"{usi_notation} - {description}."

    except Exception as e:
        # Fallback to string representation if formatting fails
        return f"{str(selected_shogi_move)} (format error: {e})"
