"""
trainer.py: Contains the Trainer class for managing the Shogi RL training loop.
"""
import os
import sys
import json
import glob
import time
import random
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional  # Removed Tuple

import numpy as np
import torch
import wandb

# Rich imports for TUI
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
)
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout

# Keisei imports
from .ppo_agent import PPOAgent
from .experience_buffer import ExperienceBuffer
from .utils import TrainingLogger, PolicyOutputMapper
from .shogi import ShogiGame, Color
from .evaluate import execute_full_evaluation_run
from keisei.config_schema import AppConfig


# --- Helper Functions ---

def find_latest_checkpoint(model_dir_path):
    try:
        checkpoints = glob.glob(os.path.join(model_dir_path, "*.pth"))
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(model_dir_path, "*.pt"))
        if not checkpoints:
            return None
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    except (OSError, FileNotFoundError) as e:
        print(f"Error in find_latest_checkpoint: {e}", file=sys.stderr)
        return None


def serialize_config(config_obj: Any) -> str:
    """Serialize a config object (SimpleNamespace or class instance) to a JSON string."""
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
        # Skip non-serializable types

    try:
        return json.dumps(conf_dict, indent=4, sort_keys=True)
    except TypeError as e:
        print(f"Error serializing config: {e}", file=sys.stderr)
        return "{}"

# Move formatting utilities are now in keisei.move_formatting
from .move_formatting import format_move_with_description_enhanced


class Trainer:
    """
    Manages the training process for the PPO Shogi agent.
    This class encapsulates the setup, training loop, evaluation, and logging.
    """

    def __init__(self, config: AppConfig, args: Any):
        """
        Initializes the Trainer with configuration and command-line arguments.

        Args:
            config: An AppConfig Pydantic object containing all configuration.
            args: Parsed command-line arguments.
        """
        self.config = config
        self.args = args
        
        # Initialize core attributes
        self.run_name = args.run_name
        self.global_timestep = 0
        self.total_episodes_completed = 0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0
        self.resumed_from_checkpoint: Optional[str] = None
        
        # Setup directories
        self._setup_directories()
        
        # Save effective config
        self._save_effective_config()
        
        # Setup seeding
        self._setup_seeding()
        
        # Initialize Rich TUI
        self.rich_console = Console(file=sys.stderr, record=True)
        self.rich_log_messages: List[Text] = []
        
        # Initialize WandB
        self.is_train_wandb_active = self._setup_wandb()
        
        # Initialize game and components
        self._setup_game_components()
        
        # Setup training components
        self._setup_training_components()
        
        # Handle checkpoint resuming
        self._handle_checkpoint_resume()

    def _setup_directories(self):
        """Setup run directories for artifacts and logging."""
        # Use new config structure
        model_dir = self.config.logging.model_dir
        log_file = self.config.logging.log_file
        self.run_artifact_dir = os.path.join(model_dir.strip("/"), self.run_name)
        self.model_dir = self.run_artifact_dir
        self.log_file_path = os.path.join(self.run_artifact_dir, os.path.basename(log_file))
        # Setup evaluation log path
        self.eval_log_file_path = os.path.join(self.run_artifact_dir, "rich_periodic_eval_log.txt")
        os.makedirs(self.run_artifact_dir, exist_ok=True)

    def _save_effective_config(self):
        """Save the effective configuration to the run directory."""
        try:
            effective_config_str = serialize_config(self.config)
            config_path = os.path.join(self.run_artifact_dir, "effective_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(effective_config_str)
        except (OSError, TypeError) as e:
            print(f"Error saving effective_config.json: {e}", file=sys.stderr)

    def _setup_seeding(self):
        """Setup random seeds for reproducibility."""
        seed = self.config.env.seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _setup_wandb(self) -> bool:
        """Setup Weights & Biases logging."""
        wandb_cfg = self.config.wandb
        is_active = wandb_cfg.enabled
        if is_active:
            try:
                config_dict_for_wandb = json.loads(serialize_config(self.config)) if serialize_config(self.config) else {}
                wandb.init(
                    project=wandb_cfg.project,
                    entity=wandb_cfg.entity,
                    name=self.run_name,
                    config=config_dict_for_wandb,
                    mode="online" if wandb_cfg.enabled else "disabled",
                    dir=self.run_artifact_dir,
                    resume="allow",
                    id=self.run_name,
                )
            except (TypeError, ValueError, OSError) as e:
                self.rich_console.print(
                    f"[bold red]Error initializing W&B: {e}. W&B logging disabled.[/bold red]"
                )
                is_active = False
        if not is_active:
            self.rich_console.print(
                "[yellow]Weights & Biases logging is disabled or failed to initialize.[/yellow]"
            )
        return is_active

    def _setup_game_components(self):
        """Initialize game environment and policy mapper."""
        try:
            self.game = ShogiGame()
            if hasattr(self.game, "seed") and self.config.env.seed is not None:
                self.game.seed(self.config.env.seed)
            self.obs_space_shape = (self.config.env.input_channels, 9, 9)
        except (RuntimeError, ValueError, OSError) as e:
            self.rich_console.print(
                f"[bold red]Error initializing ShogiGame: {e}. Aborting.[/bold red]"
            )
            raise RuntimeError(f"Failed to initialize ShogiGame: {e}") from e
        self.policy_output_mapper = PolicyOutputMapper()
        self.action_space_size = self.policy_output_mapper.get_total_actions()

    def _setup_training_components(self):
        """Initialize PPO agent and experience buffer."""
        # Use nested config structure
        self.agent = PPOAgent(
            input_channels=self.config.env.input_channels,
            policy_output_mapper=self.policy_output_mapper,
            learning_rate=self.config.training.learning_rate,
            gamma=self.config.training.gamma,
            clip_epsilon=self.config.training.clip_epsilon,
            ppo_epochs=self.config.training.ppo_epochs,
            minibatch_size=self.config.training.minibatch_size,
            value_loss_coeff=self.config.training.value_loss_coeff,
            entropy_coef=self.config.training.entropy_coef,
            device=self.config.env.device,
        )
        self.experience_buffer = ExperienceBuffer(
            buffer_size=self.config.training.steps_per_epoch,
            gamma=self.config.training.gamma,
            lambda_gae=getattr(self.config.training, 'lambda_gae', 0.95),
            device=self.config.env.device,
        )

    def _handle_checkpoint_resume(self):
        """Handle checkpoint resuming logic."""
        potential_resume_path = None
        attempt_resume = False
        
        if self.args.resume:
            attempt_resume = True
            if self.args.resume == "latest":
                potential_resume_path = find_latest_checkpoint(self.model_dir)
                if potential_resume_path:
                    self.rich_console.print(
                        f"Found latest checkpoint via --resume latest: {potential_resume_path}"
                    )
                else:
                    self.rich_console.print(
                        f"[yellow]'--resume latest' specified, but no checkpoint found in {self.model_dir}. Starting fresh.[/yellow]"
                    )
            else:
                potential_resume_path = self.args.resume
                if not (potential_resume_path and os.path.exists(potential_resume_path)):
                    self.rich_console.print(
                        f"[bold red]Specified resume checkpoint {potential_resume_path} not found. Starting fresh.[/bold red]"
                    )
                    potential_resume_path = None
        else:
            attempt_resume = True
            potential_resume_path = find_latest_checkpoint(self.model_dir)
            if potential_resume_path:
                self.rich_console.print(
                    f"Auto-detected checkpoint: {potential_resume_path}. Attempting to resume."
                )
            else:
                attempt_resume = False
        if attempt_resume and potential_resume_path and os.path.exists(potential_resume_path):
            try:
                checkpoint_data = self.agent.load_model(potential_resume_path)
                self.global_timestep = checkpoint_data.get("global_timestep", 0)
                self.total_episodes_completed = checkpoint_data.get("total_episodes_completed", 0)
                self.black_wins = checkpoint_data.get("black_wins", 0)
                self.white_wins = checkpoint_data.get("white_wins", 0)
                self.draws = checkpoint_data.get("draws", 0)
                self.resumed_from_checkpoint = potential_resume_path
                self.rich_console.print(
                    f"[green]Resumed training from checkpoint: {potential_resume_path} at timestep {self.global_timestep}[/green]"
                )
            except (OSError, RuntimeError, ValueError) as e:
                self.rich_console.print(
                    f"[bold red]Error loading checkpoint {potential_resume_path}: {e}. Starting fresh.[/bold red]"
                )
                self.global_timestep = 0
                self.total_episodes_completed = 0
                self.black_wins = 0
                self.white_wins = 0
                self.draws = 0

    def run_training_loop(self):
        """Executes the main training loop."""
        # TrainingLogger context manager
        with TrainingLogger(
            self.log_file_path, rich_console=self.rich_console, rich_log_panel=self.rich_log_messages
        ) as logger:
            
            def log_both(
                message: str,
                also_to_wandb: bool = False,
                wandb_data: Optional[Dict] = None,
            ):
                logger.log(message)
                if self.is_train_wandb_active and also_to_wandb and wandb.run:
                    log_payload = {"train_message": message}
                    if wandb_data:
                        log_payload.update(wandb_data)
                    wandb.log(log_payload, step=self.global_timestep)
            
            # Log session start
            session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_both(f"--- SESSION START: {self.run_name} at {session_start_time} ---")
            
            # Setup run information logging
            self._log_run_info(log_both)
            
            # Setup Rich progress bar and layout
            progress_bar, training_task, layout, log_panel = self._setup_rich_progress_display()
            
            # Main training loop
            try:
                current_obs_np = self._initialize_game_state(log_both)
                current_episode_reward = 0.0
                current_episode_length = 0
                
                current_obs_tensor = torch.tensor(
                    current_obs_np, dtype=torch.float32, device=torch.device(self.config.env.device)
                ).unsqueeze(0)
                last_time = time.time()
                steps_since_last_time = 0
                with Live(
                    layout,
                    console=self.rich_console,
                    refresh_per_second=10,  # Default refresh rate
                    transient=False,
                    vertical_overflow="visible",
                ) as _:
                    while self.global_timestep < self.config.training.total_timesteps:
                        # Training step
                        current_obs_np, current_obs_tensor, current_episode_reward, current_episode_length = (
                            self._execute_training_step(
                                current_obs_np, current_obs_tensor, current_episode_reward, 
                                current_episode_length, log_both, progress_bar, training_task, _
                            )
                        )
                        
                        # Update progress and display
                        self._update_progress_display(progress_bar, training_task, log_panel)
                        
                        # Update step counters
                        self.global_timestep += 1
                        steps_since_last_time += 1
                        
                        # Update speed calculation
                        current_time = time.time()
                        time_delta = current_time - last_time
                        if time_delta > 1:
                            current_speed = steps_since_last_time / time_delta
                            last_time = current_time
                            steps_since_last_time = 0
                            if training_task is not None:
                                progress_bar.update(training_task, speed=current_speed)
                
                # End of training loop
                self._finalize_training(log_both, progress_bar, training_task)
                
            except RuntimeError as e:
                log_both(f"CRITICAL: Error in training loop: {e}", also_to_wandb=True)
                raise

    def _log_run_info(self, log_both):
        """Log run information at the start of training."""
        run_title = f"Keisei Training Run: {self.run_name}"
        if self.is_train_wandb_active and wandb.run and hasattr(wandb.run, "url"):
            run_title += f" (W&B: {wandb.run.url})"
        
        log_both(run_title)
        log_both(f"Run directory: {self.run_artifact_dir}")
        log_both(f"Effective config saved to: {os.path.join(self.run_artifact_dir, 'effective_config.json')}")
        
        if self.config.env.seed is not None:
            log_both(f"Random seed: {self.config.env.seed}")
        
        log_both(f"Device: {self.config.env.device}")
        log_both(f"Agent: {type(self.agent).__name__} ({self.agent.name})")
        log_both(f"Total timesteps: {self.config.training.total_timesteps}, Steps per PPO epoch: {self.config.training.steps_per_epoch}")
        
        if self.global_timestep > 0:
            if self.resumed_from_checkpoint:
                log_both(f"[green]Resumed training from checkpoint: {self.resumed_from_checkpoint}[/green]")
            log_both(f"Resuming from timestep {self.global_timestep}, {self.total_episodes_completed} episodes completed.")
        else:
            log_both("Starting fresh training.")
        
        log_both(f"Model Structure:\n{self.agent.model}", also_to_wandb=False)

    def _setup_rich_progress_display(self):
        """Setup Rich progress bar and layout."""
        progress_bar = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("• Steps: {task.completed}/{task.total} ({task.fields[speed]:.1f} it/s)"),
            TextColumn("• {task.fields[ep_metrics]}", style="bright_cyan"),
            TextColumn("• {task.fields[ppo_metrics]}", style="bright_yellow"),
            TextColumn("• Wins B:{task.fields[black_wins_cum]} W:{task.fields[white_wins_cum]} D:{task.fields[draws_cum]}", style="bright_green"),
            TextColumn("• Rates B:{task.fields[black_win_rate]:.1%} W:{task.fields[white_win_rate]:.1%} D:{task.fields[draw_rate]:.1%}", style="bright_blue"),
            console=self.rich_console,
            transient=False,
        )
        
        initial_black_win_rate = self.black_wins / self.total_episodes_completed if self.total_episodes_completed > 0 else 0.0
        initial_white_win_rate = self.white_wins / self.total_episodes_completed if self.total_episodes_completed > 0 else 0.0
        initial_draw_rate = self.draws / self.total_episodes_completed if self.total_episodes_completed > 0 else 0.0
        
        training_task = progress_bar.add_task(
            "Training",
            total=self.config.training.total_timesteps,
            completed=self.global_timestep,
            ep_metrics="Ep L:0 R:0.0",
            ppo_metrics="",
            black_wins_cum=self.black_wins,
            white_wins_cum=self.white_wins,
            draws_cum=self.draws,
            black_win_rate=initial_black_win_rate,
            white_win_rate=initial_white_win_rate,
            draw_rate=initial_draw_rate,
            speed=0.0,
            start=(self.global_timestep < self.config.training.total_timesteps),
        )
        
        log_panel = Panel(
            Text(""),
            title="[b]Live Training Log[/b]",
            border_style="bright_green",
            expand=True,
        )
        
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="main_log", ratio=1),
            Layout(name="progress_display", size=4),
        )
        layout["main_log"].update(log_panel)
        layout["progress_display"].update(progress_bar)
        
        return progress_bar, training_task, layout, log_panel

    def _initialize_game_state(self, log_both):
        """Initialize the game state for training."""
        try:
            reset_result = self.game.reset()
            if not isinstance(reset_result, np.ndarray):
                if self.is_train_wandb_active and wandb.run:
                    wandb.finish(exit_code=1)
                raise RuntimeError("Game reset failed")
            return reset_result
        except (RuntimeError, ValueError, OSError) as e:
            log_both(
                f"CRITICAL: Error during initial game.reset(): {e}. Aborting.",
                also_to_wandb=True,
            )
            if self.is_train_wandb_active and wandb.run:
                wandb.finish(exit_code=1)
            raise RuntimeError(f"Game initialization error: {e}") from e

    def _execute_training_step(self, current_obs_np, current_obs_tensor, current_episode_reward, 
                             current_episode_length, log_both, progress_bar, training_task, _):
        """Execute a single training step."""
        # Get legal moves
        legal_shogi_moves = self.game.get_legal_moves()
        legal_mask_tensor = self.policy_output_mapper.get_legal_mask(legal_shogi_moves, device=torch.device(self.config.env.device))
        
        # For demo mode - capture piece info before the move
        piece_info_for_demo = None
        if self.config.demo.enable_demo_mode and len(legal_shogi_moves) > 0 and legal_shogi_moves[0] is not None:
            try:
                sample_move = legal_shogi_moves[0]
                if len(sample_move) == 5 and sample_move[0] is not None and sample_move[1] is not None:
                    from_r, from_c = sample_move[0], sample_move[1]
                    piece_info_for_demo = self.game.get_piece(from_r, from_c)
            except (AttributeError, IndexError, ValueError):
                pass  # Silently ignore errors in demo mode preparation
        
        # Agent action selection
        selected_shogi_move, policy_index, log_prob, value_pred = self.agent.select_action(
            current_obs_np, legal_shogi_moves, legal_mask_tensor, is_training=True
        )
        
        if selected_shogi_move is None:
            log_both(
                f"CRITICAL: Agent failed to select a move at timestep {self.global_timestep}. Resetting episode.",
                also_to_wandb=True,
            )
            current_obs_np = self.game.reset()
            current_obs_tensor = torch.tensor(current_obs_np, dtype=torch.float32, device=torch.device(self.config.env.device)).unsqueeze(0)
            return current_obs_np, current_obs_tensor, 0.0, 0
        
        # Demo mode per-move logging and delay
        if self.config.demo.enable_demo_mode:
            current_player_name = (
                getattr(
                    self.game.current_player,
                    "name",
                    str(self.game.current_player),
                )
                if hasattr(self.game, "current_player")
                else "Unknown"
            )
            move_str = format_move_with_description_enhanced(
                selected_shogi_move,
                self.policy_output_mapper,
                piece_info_for_demo,
            )
            log_both(
                f"Move {current_episode_length + 1}: {current_player_name} played {move_str}"
            )

            # Add delay for easier observation
            demo_delay = self.config.demo.demo_mode_delay
            if demo_delay > 0:
                import time
                time.sleep(demo_delay)
        
        # Environment step
        try:
            move_result = self.game.make_move(selected_shogi_move)
            if not (isinstance(move_result, tuple) and len(move_result) == 4):
                raise ValueError(f"Invalid move result: {type(move_result)}")
            next_obs_np, reward, done, info = move_result
            current_episode_reward += reward
            current_episode_length += 1
            
            # Add experience to buffer
            self.experience_buffer.add(
                current_obs_tensor.squeeze(0),
                policy_index,
                reward,
                log_prob,
                value_pred,
                done,
                legal_mask_tensor,
            )
            
            # Update observations
            current_obs_np = next_obs_np
            current_obs_tensor = torch.tensor(current_obs_np, dtype=torch.float32, device=torch.device(self.config.env.device)).unsqueeze(0)
            
            if done:
                current_obs_np, current_obs_tensor, current_episode_reward, current_episode_length = (
                    self._handle_episode_end(
                        current_episode_reward, current_episode_length, 
                        info, log_both, progress_bar, training_task
                    )
                )
            
            # PPO Update
            if ((self.global_timestep + 1) % self.config.training.steps_per_epoch == 0 and 
                self.experience_buffer.ptr == self.config.training.steps_per_epoch):
                self._perform_ppo_update(current_obs_np, log_both, progress_bar, training_task)
            
            # Checkpointing (add to config_schema if not present)
            checkpoint_interval = getattr(self.config.training, "checkpoint_interval_timesteps", 10000)
            if (self.global_timestep + 1) % checkpoint_interval == 0:
                self._save_checkpoint(log_both)
            
            # Periodic Evaluation (add to config_schema if not present)
            eval_cfg = getattr(self.config, "evaluation", None)
            enable_periodic_eval = getattr(eval_cfg, "enable_periodic_evaluation", False)
            eval_interval = getattr(eval_cfg, "evaluation_interval_timesteps", 50000)
            if enable_periodic_eval and (self.global_timestep + 1) % eval_interval == 0:
                self._run_evaluation(log_both)
            
        except ValueError as e:
            log_both(
                f"CRITICAL: Error during training step: {e}. Resetting episode.",
                also_to_wandb=True,
            )
            current_obs_np = self.game.reset()
            current_obs_tensor = torch.tensor(current_obs_np, dtype=torch.float32, device=torch.device(self.config.env.device)).unsqueeze(0)
            current_episode_reward = 0.0
            current_episode_length = 0
        
        return current_obs_np, current_obs_tensor, current_episode_reward, current_episode_length

    def _handle_episode_end(self, current_episode_reward, current_episode_length, 
                          info, log_both, progress_bar, training_task):
        """Handle the end of an episode."""
        self.total_episodes_completed += 1
        ep_metrics_str = f"Ep L:{current_episode_length} R:{current_episode_reward:.2f}"
        
        # Determine game outcome
        game_outcome_message = "Game outcome: Unknown"
        winner_color = None
        
        if "winner" in info:
            winner = info["winner"]
            if winner is not None:
                game_outcome_message = f"Game outcome: {winner.name} won."
                winner_color = winner
            else:
                game_outcome_message = "Game outcome: Draw."
        elif self.game.winner is not None:
            winner = self.game.winner
            game_outcome_message = f"Game outcome: {winner.name} won."
            winner_color = winner
        elif self.game.game_over and self.game.winner is None:
            game_outcome_message = "Game outcome: Draw (max moves or stalemate)."
        
        # Update win/loss/draw counts
        if winner_color == Color.BLACK:
            self.black_wins += 1
        elif winner_color == Color.WHITE:
            self.white_wins += 1
        else:
            self.draws += 1
        
        # Calculate rates
        current_black_win_rate = self.black_wins / self.total_episodes_completed if self.total_episodes_completed > 0 else 0.0
        current_white_win_rate = self.white_wins / self.total_episodes_completed if self.total_episodes_completed > 0 else 0.0
        current_draw_rate = self.draws / self.total_episodes_completed if self.total_episodes_completed > 0 else 0.0
        
        # Update progress bar
        progress_bar.update(
            training_task,
            advance=0,
            ep_metrics=ep_metrics_str,
            black_wins_cum=self.black_wins,
            white_wins_cum=self.white_wins,
            draws_cum=self.draws,
            black_win_rate=current_black_win_rate,
            white_win_rate=current_white_win_rate,
            draw_rate=current_draw_rate,
        )
        
        # Log episode completion
        log_both(
            f"Episode {self.total_episodes_completed} finished. Length: {current_episode_length}, Reward: {current_episode_reward:.2f}. {game_outcome_message}",
            also_to_wandb=True,
            wandb_data={
                "episode_reward": current_episode_reward,
                "episode_length": current_episode_length,
                "total_episodes": self.total_episodes_completed,
                "black_wins_cumulative": self.black_wins,
                "white_wins_cumulative": self.white_wins,
                "draws_cumulative": self.draws,
                "black_win_rate": current_black_win_rate,
                "white_win_rate": current_white_win_rate,
                "draw_rate": current_draw_rate,
            },
        )
        
        # Reset game
        reset_result = self.game.reset()
        if not isinstance(reset_result, np.ndarray):
            log_both(
                f"CRITICAL: game.reset() after episode done did not return ndarray. Got {type(reset_result)}. Aborting.",
                also_to_wandb=True,
            )
            if self.is_train_wandb_active and wandb.run:
                wandb.finish(exit_code=1)
            raise RuntimeError("Game reset failed after episode end")
        
        current_obs_np = reset_result
        current_obs_tensor = torch.tensor(current_obs_np, dtype=torch.float32, device=torch.device(self.config.env.device)).unsqueeze(0)
        
        return current_obs_np, current_obs_tensor, 0.0, 0

    def _perform_ppo_update(self, current_obs_np, log_both, progress_bar, training_task):
        """Perform a PPO update."""
        with torch.no_grad():
            last_value_pred_for_gae = self.agent.get_value(current_obs_np)
        
        self.experience_buffer.compute_advantages_and_returns(last_value_pred_for_gae)
        learn_metrics = self.agent.learn(self.experience_buffer)
        self.experience_buffer.clear()
        
        # Format PPO metrics for display
        ppo_metrics_str_parts = []
        if "ppo/kl_divergence_approx" in learn_metrics:
            ppo_metrics_str_parts.append(f"KL:{learn_metrics['ppo/kl_divergence_approx']:.4f}")
        if "ppo/policy_loss" in learn_metrics:
            ppo_metrics_str_parts.append(f"PolL:{learn_metrics['ppo/policy_loss']:.4f}")
        if "ppo/value_loss" in learn_metrics:
            ppo_metrics_str_parts.append(f"ValL:{learn_metrics['ppo/value_loss']:.4f}")
        if "ppo/entropy" in learn_metrics:
            ppo_metrics_str_parts.append(f"Ent:{learn_metrics['ppo/entropy']:.4f}")
        
        ppo_metrics_display = " ".join(ppo_metrics_str_parts)
        progress_bar.update(training_task, advance=0, ppo_metrics=ppo_metrics_display)
        
        log_both(
            f"PPO Update @ ts {self.global_timestep+1}. Metrics: {json.dumps({k: f'{v:.4f}' for k,v in learn_metrics.items()})}",
            also_to_wandb=True,
            wandb_data=learn_metrics,
        )

    def _update_progress_display(self, progress_bar, training_task, log_panel):
        """Update the Rich progress display."""
        if training_task is not None:
            progress_bar.update(training_task, completed=self.global_timestep)
        
        # Use a default for max log messages
        max_log_messages = 50
        if self.rich_log_messages:
            display_messages = self.rich_log_messages[-max_log_messages:]
            updated_panel = Group(*display_messages) if display_messages else Text("")
            log_panel.renderable = updated_panel

    def _save_checkpoint(self, log_both):
        """Save a training checkpoint."""
        ckpt_save_path = os.path.join(self.model_dir, f"checkpoint_ts{self.global_timestep+1}.pth")
        try:
            self.agent.save_model(
                ckpt_save_path,
                self.global_timestep + 1,
                self.total_episodes_completed,
                stats_to_save={
                    "black_wins": self.black_wins,
                    "white_wins": self.white_wins,
                    "draws": self.draws,
                },
            )
            log_both(f"Checkpoint saved to {ckpt_save_path}", also_to_wandb=True)
        except (OSError, RuntimeError) as e:
            log_both(
                f"Error saving checkpoint {ckpt_save_path}: {e}",
                log_level="error",
                also_to_wandb=True,
            )

    def _run_evaluation(self, log_both):
        """Run periodic evaluation."""
        eval_cfg = getattr(self.config, "evaluation", None)
        eval_ckpt_path = os.path.join(self.model_dir, f"eval_checkpoint_ts{self.global_timestep+1}.pth")
        self.agent.save_model(eval_ckpt_path, self.global_timestep + 1, self.total_episodes_completed)
        
        log_both(f"Starting periodic evaluation at timestep {self.global_timestep + 1}...", also_to_wandb=True)
        
        self.agent.model.eval()
        
        log_file_path_eval = getattr(eval_cfg, "log_file_path_eval", "")
        eval_results = execute_full_evaluation_run(
            agent_checkpoint_path=eval_ckpt_path,
            opponent_type=getattr(eval_cfg, "opponent_type", "random"),
            opponent_checkpoint_path=getattr(eval_cfg, "opponent_checkpoint_path", None),
            num_games=getattr(eval_cfg, "num_games", 20),
            max_moves_per_game=getattr(eval_cfg, "max_moves_per_game", 256),
            device_str=self.config.env.device,
            log_file_path_eval=log_file_path_eval,
            policy_mapper=self.policy_output_mapper,
            seed=self.config.env.seed,
            wandb_log_eval=getattr(eval_cfg, "wandb_log_eval", False),
            wandb_project_eval=getattr(eval_cfg, "wandb_project_eval", None),
            wandb_entity_eval=getattr(eval_cfg, "wandb_entity_eval", None),
            wandb_run_name_eval=f"periodic_eval_{self.run_name}_ts{self.global_timestep+1}",
            wandb_group=self.run_name,
            wandb_reinit=True,
            logger_also_stdout=False,
        )
        
        self.agent.model.train()
        
        log_both(
            f"Periodic evaluation finished. Results: {eval_results}",
            also_to_wandb=True,
            wandb_data=(
                eval_results if isinstance(eval_results, dict) 
                else {"eval_summary": str(eval_results)}
            ),
        )

    def _finalize_training(self, log_both, progress_bar, training_task):
        """Finalize training and save final model."""
        if training_task is not None:
            progress_bar.update(training_task, completed=self.config.training.total_timesteps)

        log_both(
            f"Training loop finished at timestep {self.global_timestep}. Total episodes: {self.total_episodes_completed}.",
            also_to_wandb=True,
        )

        if self.global_timestep >= self.config.training.total_timesteps:
            log_both("Training successfully completed all timesteps.", also_to_wandb=True)
            final_model_path = os.path.join(self.model_dir, "final_model.pth")
            try:
                self.agent.save_model(final_model_path, self.global_timestep, self.total_episodes_completed)
                log_both(f"Final model saved to {final_model_path}", also_to_wandb=True)

                if self.is_train_wandb_active and wandb.run:
                    wandb.finish()
            except (OSError, RuntimeError) as e:
                log_both(
                    f"Error saving final model {final_model_path}: {e}",
                    log_level="error",
                    also_to_wandb=True,
                )
        else:
            log_both(
                f"Training interrupted at timestep {self.global_timestep} (before {self.config.training.total_timesteps} total).",
                log_level="warning",
                also_to_wandb=True,
            )

        if self.is_train_wandb_active and wandb.run:
            wandb.finish()
            log_both("Weights & Biases run finished.")

        # Save the full console log from Rich
        console_log_path = os.path.join(self.run_artifact_dir, "full_console_output_rich.html")
        try:
            self.rich_console.save_html(console_log_path)
            print(f"Full Rich console output saved to {console_log_path}", file=sys.stderr)
        except OSError as e:
            print(f"Error saving Rich console log: {e}", file=sys.stderr)

        # Final messages
        self.rich_console.rule("[bold green]Run Finished[/bold green]")
        self.rich_console.print(f"[bold green]Run '{self.run_name}' processing finished.[/bold green]")
        self.rich_console.print(f"Output and logs are in: {self.run_artifact_dir}")
