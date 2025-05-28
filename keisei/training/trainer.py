"""
trainer.py: Contains the Trainer class for managing the Shogi RL training loop (refactored).
"""

import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import (  # pylint: disable=unused-import
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch  # Add torch import
from rich.console import Console, Text
from torch.cuda.amp import GradScaler  # For mixed precision

import wandb
from keisei.config_schema import AppConfig
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.evaluate import execute_full_evaluation_run
from keisei.shogi import ShogiGame
from keisei.utils import (
    PolicyOutputMapper,
    TrainingLogger,
)

from . import callbacks, display, utils
from .session_manager import SessionManager
from .step_manager import EpisodeState, StepManager


class Trainer:
    """
    Manages the training process for the PPO Shogi agent.
    This class orchestrates setup, training loop, evaluation, and logging.
    """

    # Mixed-precision training
    use_mixed_precision: bool = False
    scaler: Optional[torch.cuda.amp.GradScaler] = None

    # Statistics for tracking and logging
    global_timestep: int = 0
    total_episodes_completed: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0

    def __init__(self, config: AppConfig, args: Any):
        """
        Initializes the Trainer with configuration and command-line arguments.

        Args:
            config: An AppConfig Pydantic object containing all configuration.
            args: Parsed command-line arguments.
        """
        self.config = config
        self.args = args

        # Initialize attributes that will be set later (to avoid pylint errors)
        self.log_both: Optional[Callable] = None
        self.execute_full_evaluation_run: Optional[Callable] = None

        # Initialize session manager for session-level concerns
        self.session_manager = SessionManager(config, args)

        # Setup session infrastructure
        self.session_manager.setup_directories()
        self.session_manager.setup_wandb()
        self.session_manager.save_effective_config()
        self.session_manager.setup_seeding()

        # Access session properties through manager
        self.run_name = self.session_manager.run_name
        self.run_artifact_dir = self.session_manager.run_artifact_dir
        self.model_dir = self.session_manager.model_dir
        self.log_file_path = self.session_manager.log_file_path
        self.eval_log_file_path = self.session_manager.eval_log_file_path
        self.is_train_wandb_active = self.session_manager.is_wandb_active

        # Initialize statistics
        self.global_timestep = 0
        self.total_episodes_completed = 0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0

        self.device = torch.device(config.env.device)
        self.console = Console()
        self.logger = TrainingLogger(self.log_file_path, self.console)

        # Mixed Precision Setup
        self.use_mixed_precision = (
            self.config.training.mixed_precision and self.device.type == "cuda"
        )
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.logger.log(
                str(
                    Text("Mixed precision training enabled (CUDA).", style="green")
                )  # Convert Text to str
            )
        elif self.config.training.mixed_precision and self.device.type != "cuda":
            self.logger.log(
                str(
                    Text(  # Convert Text to str
                        "Mixed precision training requested but CUDA is not available/selected. Proceeding without mixed precision.",
                        style="yellow",
                    )
                )
            )
            self.use_mixed_precision = False

        # --- Model/feature config integration ---
        self.input_features = (
            getattr(args, "input_features", None) or config.training.input_features
        )
        self.model_type = getattr(args, "model", None) or config.training.model_type
        self.tower_depth = (
            getattr(args, "tower_depth", None) or config.training.tower_depth
        )
        self.tower_width = (
            getattr(args, "tower_width", None) or config.training.tower_width
        )
        self.se_ratio = getattr(args, "se_ratio", None) or config.training.se_ratio
        # Feature builder
        from keisei.shogi import features

        self.feature_spec = features.FEATURE_SPECS[self.input_features]
        self.obs_shape = (self.feature_spec.num_planes, 9, 9)
        # Model factory
        from keisei.training.models import model_factory  # Corrected import

        # from keisei.training.models.resnet_tower import ActorCriticResTower # Old direct import
        # if self.model_type == "resnet": # Old direct instantiation
        self.model = model_factory(
            model_type=self.model_type,
            obs_shape=self.obs_shape,
            num_actions=config.env.num_actions_total,  # Added num_actions
            tower_depth=self.tower_depth,
            tower_width=self.tower_width,
            se_ratio=self.se_ratio if self.se_ratio > 0 else None,
            # Add any other kwargs your model_factory or models might need, e.g.:
            # num_actions_total=config.env.num_actions_total # Already passed as num_actions
        )
        # else:
        #     raise ValueError(f"Unknown model_type: {self.model_type}")

        # Initialize Rich TUI
        self.rich_console = Console(file=sys.stderr, record=True)
        self.rich_log_messages: List[Text] = []

        # WP-2: Store pending progress bar updates to consolidate them
        self.pending_progress_updates: Dict[str, Any] = {}

        # Initialize game and components
        self._setup_game_components()

        # Setup training components
        self._setup_training_components()

        # Handle checkpoint resuming
        self._handle_checkpoint_resume()

        # Display and callbacks
        self.display = display.TrainingDisplay(self.config, self, self.rich_console)
        eval_cfg = getattr(self.config, "evaluation", None)
        checkpoint_interval = (
            self.config.training.checkpoint_interval_timesteps
        )  # Use config value
        eval_interval = (
            eval_cfg.evaluation_interval_timesteps
            if eval_cfg
            else self.config.training.evaluation_interval_timesteps  # Use config value
        )
        self.callbacks = [
            callbacks.CheckpointCallback(checkpoint_interval, self.model_dir),
            callbacks.EvaluationCallback(eval_cfg, eval_interval),
        ]

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
        self.agent = PPOAgent(
            config=self.config,
            device=torch.device(self.config.env.device),
        )
        self.experience_buffer = ExperienceBuffer(
            buffer_size=self.config.training.steps_per_epoch,
            gamma=self.config.training.gamma,
            lambda_gae=self.config.training.lambda_gae,  # Use config value
            device=self.config.env.device,
        )

        # Initialize StepManager for step execution and episode management
        self.step_manager = StepManager(
            config=self.config,
            game=self.game,
            agent=self.agent,
            policy_mapper=self.policy_output_mapper,
            experience_buffer=self.experience_buffer,
        )

    def _log_event(self, message: str):
        """Log important events to the main training log file."""
        # Always log to the main log file
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except (OSError, IOError) as e:
            print(f"[Trainer] Failed to log event: {e}", file=sys.stderr)
        # No longer print to stderr for test compatibility

    def _handle_checkpoint_resume(self):
        """Handle resuming from checkpoint if specified or auto-detected."""
        resume_path = self.args.resume

        def find_ckpt_in_dir(directory):
            return utils.find_latest_checkpoint(directory)

        if resume_path == "latest" or resume_path is None:
            # Try to find latest checkpoint in the run's model_dir
            latest_ckpt = find_ckpt_in_dir(self.model_dir)
            # If not found, try the parent directory (savedir)
            if not latest_ckpt:
                parent_dir = os.path.dirname(self.model_dir.rstrip(os.sep))
                parent_ckpt = find_ckpt_in_dir(parent_dir)
                if parent_ckpt:
                    # Copy the checkpoint into the run's model_dir for consistency
                    dest_ckpt = os.path.join(
                        self.model_dir, os.path.basename(parent_ckpt)
                    )
                    shutil.copy2(parent_ckpt, dest_ckpt)
                    latest_ckpt = dest_ckpt
            if latest_ckpt:
                self.agent.load_model(latest_ckpt)
                self.resumed_from_checkpoint = latest_ckpt
                msg = f"Resumed training from checkpoint: {latest_ckpt}"
                self._log_event(msg)
                if hasattr(self, "rich_console"):
                    self.rich_console.print(f"[yellow]{msg}[/yellow]")
            else:
                self.resumed_from_checkpoint = None
        elif resume_path:
            self.agent.load_model(resume_path)
            self.resumed_from_checkpoint = resume_path
            msg = f"Resumed training from checkpoint: {resume_path}"
            self._log_event(msg)
            if hasattr(self, "rich_console"):
                self.rich_console.print(f"[yellow]{msg}[/yellow]")
        else:
            self.resumed_from_checkpoint = None

    def _log_run_info(self, log_both):
        """Log run information at the start of training."""
        # Delegate session info logging to SessionManager
        agent_info = {"type": type(self.agent).__name__, "name": self.agent.name}

        def log_wrapper(msg):
            log_both(msg)

        self.session_manager.log_session_info(
            logger_func=log_wrapper,
            agent_info=agent_info,
            resumed_from_checkpoint=getattr(self, "resumed_from_checkpoint", None),
            global_timestep=self.global_timestep,
            total_episodes_completed=self.total_episodes_completed,
        )

        # Log model structure (trainer-specific)
        log_both(f"Model Structure:\n{self.agent.model}", also_to_wandb=False)
        self._log_event(f"Model Structure:\n{self.agent.model}")

    def _initialize_game_state(self, log_both) -> EpisodeState:
        """Initialize the game state for training."""
        try:
            episode_state = self.step_manager.reset_episode()
            return episode_state
        except (RuntimeError, ValueError, OSError) as e:
            log_both(
                f"CRITICAL: Error during initial game.reset(): {e}. Aborting.",
                also_to_wandb=True,
            )
            if self.is_train_wandb_active and wandb.run:
                wandb.finish(exit_code=1)
            raise RuntimeError(f"Game initialization error: {e}") from e

    def _execute_training_step(
        self,
        episode_state: EpisodeState,
        log_both,
    ) -> EpisodeState:
        """Execute a single training step using StepManager."""
        # Execute the step using StepManager
        step_result = self.step_manager.execute_step(
            episode_state=episode_state,
            global_timestep=self.global_timestep,
            logger_func=log_both,
        )

        # Handle step failure by returning reset episode state
        if not step_result.success:
            return self.step_manager.reset_episode()

        # Update episode state with step results
        updated_episode_state = self.step_manager.update_episode_state(
            episode_state, step_result
        )

        # Handle episode end
        if step_result.done:
            # Update game statistics based on outcome
            if step_result.info and "winner" in step_result.info:
                winner = step_result.info["winner"]
                if winner == "black":
                    self.black_wins += 1
                elif winner == "white":
                    self.white_wins += 1
                else:
                    self.draws += 1
            else:
                self.draws += 1

            # Create game stats dict for StepManager
            game_stats = {
                "black_wins": self.black_wins,
                "white_wins": self.white_wins,
                "draws": self.draws,
            }

            # Handle episode end using StepManager
            new_episode_state = self.step_manager.handle_episode_end(
                updated_episode_state,
                step_result,
                game_stats,
                self.total_episodes_completed,
                log_both,
            )

            # Update trainer statistics
            self.total_episodes_completed += 1

            # Store pending progress updates for display
            ep_metrics_str = f"L:{updated_episode_state.episode_length} R:{updated_episode_state.episode_reward:.2f}"

            # Calculate win rates for display
            total_games = self.black_wins + self.white_wins + self.draws
            current_black_win_rate = (
                self.black_wins / total_games if total_games > 0 else 0.0
            )
            current_white_win_rate = (
                self.white_wins / total_games if total_games > 0 else 0.0
            )
            current_draw_rate = self.draws / total_games if total_games > 0 else 0.0

            # Store episode metrics for next throttled update (WP-2)
            self.pending_progress_updates.update(
                {
                    "ep_metrics": ep_metrics_str,
                    "black_wins_cum": self.black_wins,
                    "white_wins_cum": self.white_wins,
                    "draws_cum": self.draws,
                    "black_win_rate": current_black_win_rate,
                    "white_win_rate": current_white_win_rate,
                    "draw_rate": current_draw_rate,
                }
            )

            return new_episode_state

        # PPO Update check
        if (
            (self.global_timestep + 1) % self.config.training.steps_per_epoch == 0
            and self.experience_buffer.ptr == self.config.training.steps_per_epoch
        ):
            self._perform_ppo_update(updated_episode_state.current_obs, log_both)

        return updated_episode_state

    def _perform_ppo_update(self, current_obs_np, log_both):
        """Perform a PPO update."""
        with torch.no_grad():
            last_value_pred_for_gae = self.agent.get_value(current_obs_np)

        self.experience_buffer.compute_advantages_and_returns(last_value_pred_for_gae)
        learn_metrics = self.agent.learn(self.experience_buffer)
        self.experience_buffer.clear()

        # Format PPO metrics for display
        ppo_metrics_str_parts = []
        if "ppo/kl_divergence_approx" in learn_metrics:
            ppo_metrics_str_parts.append(
                f"KL:{learn_metrics['ppo/kl_divergence_approx']:.4f}"
            )
        if "ppo/policy_loss" in learn_metrics:
            ppo_metrics_str_parts.append(f"PolL:{learn_metrics['ppo/policy_loss']:.4f}")
        if "ppo/value_loss" in learn_metrics:
            ppo_metrics_str_parts.append(f"ValL:{learn_metrics['ppo/value_loss']:.4f}")
        if "ppo/entropy" in learn_metrics:
            ppo_metrics_str_parts.append(f"Ent:{learn_metrics['ppo/entropy']:.4f}")

        ppo_metrics_display = " ".join(ppo_metrics_str_parts)
        # Store PPO metrics for next throttled update (WP-2)
        self.pending_progress_updates["ppo_metrics"] = ppo_metrics_display

        log_both(
            f"PPO Update @ ts {self.global_timestep+1}. Metrics: {json.dumps({k: f'{v:.4f}' for k,v in learn_metrics.items()})}",
            also_to_wandb=True,
            wandb_data=learn_metrics,
        )

    def _create_model_artifact(
        self,
        model_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        log_both=None,
    ) -> bool:
        """
        Create and upload a W&B artifact for a model checkpoint.

        Args:
            model_path: Path to the model file to upload
            artifact_name: Name for the artifact (without run prefix)
            artifact_type: Type of artifact (default: "model")
            description: Optional description for the artifact
            metadata: Optional metadata dict to attach to the artifact
            aliases: Optional list of aliases (e.g., ["latest", "best"])
            log_both: Logging function to use

        Returns:
            bool: True if artifact was created successfully, False otherwise
        """
        if not (self.is_train_wandb_active and wandb.run):
            return False

        if not os.path.exists(model_path):
            if log_both:
                log_both(
                    f"Warning: Model file {model_path} does not exist, skipping artifact creation."
                )
            return False

        try:
            # Create artifact with run name prefix for uniqueness
            full_artifact_name = f"{self.run_name}-{artifact_name}"
            artifact = wandb.Artifact(
                name=full_artifact_name,
                type=artifact_type,
                description=description or f"Model checkpoint from run {self.run_name}",
                metadata=metadata or {},
            )

            # Add the model file
            artifact.add_file(model_path)

            # Log the artifact with optional aliases
            wandb.log_artifact(artifact, aliases=aliases)

            if log_both:
                aliases_str = f" with aliases {aliases}" if aliases else ""
                log_both(
                    f"Model artifact '{full_artifact_name}' created and uploaded{aliases_str}"
                )

            return True

        except (OSError, RuntimeError, TypeError, ValueError) as e:
            if log_both:
                log_both(
                    f"Error creating W&B artifact for {model_path}: {e}",
                    log_level="error",
                )
            return False

    def _finalize_training(self, log_both):
        """Finalize training and save final model."""
        log_both(
            f"Training loop finished at timestep {self.global_timestep}. Total episodes: {self.total_episodes_completed}.",
            also_to_wandb=True,
        )

        if self.global_timestep >= self.config.training.total_timesteps:
            log_both(
                "Training successfully completed all timesteps.", also_to_wandb=True
            )
            final_model_path = os.path.join(self.model_dir, "final_model.pth")
            try:
                self.agent.save_model(
                    final_model_path,
                    self.global_timestep,
                    self.total_episodes_completed,
                )
                log_both(f"Final model saved to {final_model_path}", also_to_wandb=True)

                # Create W&B artifact for final model
                final_metadata = {
                    "training_timesteps": self.global_timestep,
                    "total_episodes": self.total_episodes_completed,
                    "black_wins": self.black_wins,
                    "white_wins": self.white_wins,
                    "draws": self.draws,
                    "training_completed": True,
                    "model_type": getattr(self.config.training, "model_type", "resnet"),
                    "feature_set": getattr(self.config.env, "feature_set", "core"),
                }
                self._create_model_artifact(
                    model_path=final_model_path,
                    artifact_name="final-model",
                    description=f"Final trained model after {self.global_timestep} timesteps",
                    metadata=final_metadata,
                    aliases=["latest", "final"],
                    log_both=log_both,
                )

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

        # Always save a final checkpoint if one was not just saved at the last step
        last_ckpt_filename = os.path.join(
            self.model_dir, f"checkpoint_ts{self.global_timestep}.pth"
        )
        if self.global_timestep > 0 and not os.path.exists(last_ckpt_filename):
            self.agent.save_model(
                last_ckpt_filename,
                self.global_timestep,
                self.total_episodes_completed,
                stats_to_save={
                    "black_wins": self.black_wins,
                    "white_wins": self.white_wins,
                    "draws": self.draws,
                },
            )
            log_both(
                f"Final checkpoint saved to {last_ckpt_filename}", also_to_wandb=True
            )

            # Create W&B artifact for final checkpoint
            checkpoint_metadata = {
                "training_timesteps": self.global_timestep,
                "total_episodes": self.total_episodes_completed,
                "black_wins": self.black_wins,
                "white_wins": self.white_wins,
                "draws": self.draws,
                "checkpoint_type": "final",
                "model_type": getattr(self.config.training, "model_type", "resnet"),
                "feature_set": getattr(self.config.env, "feature_set", "core"),
            }
            self._create_model_artifact(
                model_path=last_ckpt_filename,
                artifact_name="final-checkpoint",
                description=f"Final checkpoint at timestep {self.global_timestep}",
                metadata=checkpoint_metadata,
                aliases=["latest-checkpoint"],
                log_both=log_both,
            )

        if self.is_train_wandb_active and wandb.run:
            self.session_manager.finalize_session()
            log_both("Weights & Biases run finished.")

        # Save the full console log from Rich
        console_log_path = os.path.join(
            self.run_artifact_dir, "full_console_output_rich.html"
        )
        try:
            self.rich_console.save_html(console_log_path)
            print(
                f"Full Rich console output saved to {console_log_path}", file=sys.stderr
            )
        except OSError as e:
            print(f"Error saving Rich console log: {e}", file=sys.stderr)

        # Final messages
        self.rich_console.rule("[bold green]Run Finished[/bold green]")
        self.rich_console.print(
            f"[bold green]Run '{self.run_name}' processing finished.[/bold green]"
        )
        self.rich_console.print(f"Output and logs are in: {self.run_artifact_dir}")

    def run_training_loop(self):
        """Executes the main training loop."""
        # Log session start using SessionManager
        self.session_manager.log_session_start()

        # TrainingLogger context manager
        with TrainingLogger(
            self.log_file_path,
            rich_console=self.rich_console,
            rich_log_panel=self.rich_log_messages,
        ) as logger:

            def log_both(
                message: str,
                also_to_wandb: bool = False,
                wandb_data: Optional[Dict] = None,
                log_level: str = "info",  # pylint: disable=unused-argument
            ):
                # Note: log_level parameter is available for future use if needed
                logger.log(message)
                if self.is_train_wandb_active and also_to_wandb and wandb.run:
                    log_payload = {"train_message": message}
                    if wandb_data:
                        log_payload.update(wandb_data)
                    wandb.log(log_payload, step=self.global_timestep)

            self.log_both = log_both  # Expose for callbacks
            self.execute_full_evaluation_run = (
                execute_full_evaluation_run  # Expose for callbacks
            )

            # Log session start
            session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_both(f"--- SESSION START: {self.run_name} at {session_start_time} ---")

            # Setup run information logging
            self._log_run_info(log_both)

            last_time = time.time()
            steps_since_last_time = 0
            episode_state = self._initialize_game_state(log_both)

            with self.display.start() as _:
                while self.global_timestep < self.config.training.total_timesteps:
                    episode_state = self._execute_training_step(
                        episode_state,
                        log_both,
                    )

                    # Update step counters
                    self.global_timestep += 1
                    steps_since_last_time += 1

                    # Display updates
                    if (
                        self.global_timestep % self.config.training.render_every_steps
                    ) == 0:
                        self.display.update_log_panel(self)

                    current_time = time.time()
                    time_delta = current_time - last_time
                    current_speed = (
                        steps_since_last_time / time_delta if time_delta > 0 else 0.0
                    )

                    if time_delta > 0.1:  # Update speed roughly every 100ms
                        last_time = current_time
                        steps_since_last_time = 0

                    self.display.update_progress(
                        self, current_speed, self.pending_progress_updates
                    )
                    self.pending_progress_updates.clear()

                    # Callbacks
                    for callback in self.callbacks:
                        callback.on_step_end(self)

                # End of training loop
                self._finalize_training(log_both)
