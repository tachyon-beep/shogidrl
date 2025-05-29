"""
trainer.py: Contains the Trainer class for managing the Shogi RL training loop (refactored).
"""

import json
import os
import sys
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

import wandb
from keisei.config_schema import AppConfig
from keisei.core.actor_critic_protocol import (  # Import ActorCriticProtocol
    ActorCriticProtocol,
)
from keisei.core.experience_buffer import ExperienceBuffer

# Backwards compatibility imports for tests (these classes are now used in managers)
from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.evaluate import execute_full_evaluation_run
from keisei.utils import (
    TrainingLogger,
)

from . import callbacks, display
from .env_manager import EnvManager
from .model_manager import ModelManager
from .session_manager import SessionManager
from .step_manager import EpisodeState, StepManager
from .training_loop_manager import TrainingLoopManager  # Added import


class Trainer:
    """
    Manages the training process for the PPO Shogi agent.
    This class orchestrates setup, training loop, evaluation, and logging.
    """

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

        # Declare instance attributes that will be set up
        self.model: Optional[ActorCriticProtocol] = None  # Model instance
        self.agent: Optional[PPOAgent] = None  # Agent instance
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

        # Initialize managers
        self.model_manager = ModelManager(config, args, self.device, self.logger.log)
        self.env_manager = EnvManager(config, self.logger.log)

        # Initialize Rich TUI
        self.rich_console = Console(file=sys.stderr, record=True)
        self.rich_log_messages: List[Text] = []

        # WP-2: Store pending progress bar updates to consolidate them
        self.pending_progress_updates: Dict[str, Any] = {}

        # Initialize game and components using EnvManager
        self._setup_game_components()

        # Setup training components
        self._setup_training_components()

        # Handle checkpoint resuming using ModelManager
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

        # Initialize TrainingLoopManager
        self.training_loop_manager = TrainingLoopManager(trainer=self)

    def _setup_game_components(self):
        """Initialize game environment and policy mapper using EnvManager."""
        try:
            # Call EnvManager's setup_environment to get game and mapper
            self.game, self.policy_output_mapper = self.env_manager.setup_environment()

            # Retrieve other info if needed, or ensure EnvManager sets them internally
            # For now, let's assume action_space_size and obs_space_shape are set within EnvManager
            # by its setup_environment method, and we can access them via properties or get_environment_info
            # if that method is still useful for other details.
            # Based on EnvManager changes, these are set as attributes during its setup_environment.
            self.action_space_size = self.env_manager.action_space_size
            self.obs_space_shape = self.env_manager.obs_space_shape

            if self.game is None or self.policy_output_mapper is None:
                raise RuntimeError(
                    "EnvManager.setup_environment() failed to return valid game or policy_output_mapper."
                )

        except (RuntimeError, ValueError, OSError) as e:
            self.rich_console.print(
                f"[bold red]Error initializing game components: {e}. Aborting.[/bold red]"
            )
            raise RuntimeError(f"Failed to initialize game components: {e}") from e

    def _setup_training_components(self):
        """Initialize PPO agent and experience buffer."""
        # Create model using ModelManager
        self.model = self.model_manager.create_model()  # Get the model instance

        # Initialize PPOAgent and assign the model
        self.agent = PPOAgent(
            config=self.config,
            device=self.device,
        )
        if self.model is None:
            # This should ideally not happen if model_manager.create_model() raises an error on failure
            raise RuntimeError(
                "Model was not created successfully before agent initialization."
            )
        self.agent.model = (
            self.model
        )  # self.model is now confirmed to be ActorCriticProtocol

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
            agent=self.agent,  # Agent is now guaranteed to be initialized
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
        """Handle resuming from checkpoint using ModelManager."""
        if not self.agent:
            self.logger.log(
                "[ERROR] Agent not initialized before handling checkpoint resume. This should not happen."
            )
            # Or raise an error, as this indicates a logic flaw
            raise RuntimeError("Agent not initialized before _handle_checkpoint_resume")

        # self.agent is confirmed to be not None by the check above.
        self.model_manager.handle_checkpoint_resume(
            agent=self.agent,
            model_dir=self.model_dir,
            # resume_path_override can be passed here if needed, e.g., from self.args
        )
        self.resumed_from_checkpoint = self.model_manager.resumed_from_checkpoint

        # Restore training state from checkpoint data
        if self.model_manager.checkpoint_data:
            checkpoint_data = self.model_manager.checkpoint_data

            # Restore global timestep and episode count
            self.global_timestep = checkpoint_data.get("global_timestep", 0)
            self.total_episodes_completed = checkpoint_data.get(
                "total_episodes_completed", 0
            )

            # Restore game statistics
            self.black_wins = checkpoint_data.get("black_wins", 0)
            self.white_wins = checkpoint_data.get("white_wins", 0)
            self.draws = checkpoint_data.get("draws", 0)

    def _log_run_info(self, log_both):
        """Log run information at the start of training."""
        # Delegate session info logging to SessionManager
        agent_name = "N/A"
        agent_type_name = "N/A"
        if self.agent:  # Check if agent is initialized
            agent_name = getattr(
                self.agent, "name", "N/A"
            )  # PPOAgent might not have a 'name' attribute
            agent_type_name = type(self.agent).__name__

        agent_info = {"type": agent_type_name, "name": agent_name}

        def log_wrapper(msg):
            log_both(msg)

        self.session_manager.log_session_info(
            logger_func=log_wrapper,
            agent_info=agent_info,
            resumed_from_checkpoint=getattr(self, "resumed_from_checkpoint", None),
            global_timestep=self.global_timestep,
            total_episodes_completed=self.total_episodes_completed,
        )

        # Log model structure (trainer-specific) using ModelManager
        model_info = self.model_manager.get_model_info()
        log_both(f"Model Structure:\n{model_info}", also_to_wandb=False)
        self._log_event(f"Model Structure:\n{model_info}")

    def _initialize_game_state(self, log_both) -> EpisodeState:
        """Initialize the game state for training using EnvManager."""
        try:
            # Reset game using EnvManager
            self.env_manager.reset_game()
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

    # _execute_training_step is removed as its logic is now in TrainingLoopManager._run_epoch
    # def _execute_training_step(...)

    def _perform_ppo_update(self, current_obs_np, log_both):
        """Perform a PPO update."""
        if not self.agent:
            log_both(
                "[ERROR] PPO update called but agent is not initialized.",
                also_to_wandb=True,
            )
            return

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

    def _finalize_training(self, log_both):
        """Finalize training and save final model and checkpoint via ModelManager."""
        log_both(
            f"Training loop finished at timestep {self.global_timestep}. Total episodes: {self.total_episodes_completed}.",
            also_to_wandb=True,
        )

        if not self.agent:
            log_both(
                "[ERROR] Finalize training: Agent not initialized. Cannot save model or checkpoint.",
                also_to_wandb=True,
            )
            if (
                self.is_train_wandb_active and wandb.run
            ):  # Ensure WandB is finalized if active
                self.session_manager.finalize_session()
                log_both("Weights & Biases run finished due to error.")
            return

        game_stats = {
            "black_wins": self.black_wins,
            "white_wins": self.white_wins,
            "draws": self.draws,
        }

        if self.global_timestep >= self.config.training.total_timesteps:
            log_both(
                "Training successfully completed all timesteps. Saving final model.",
                also_to_wandb=True,
            )
            success, final_model_path = self.model_manager.save_final_model(
                agent=self.agent,
                model_dir=self.model_dir,
                global_timestep=self.global_timestep,
                total_episodes_completed=self.total_episodes_completed,
                game_stats=game_stats,
                run_name=self.run_name,
                is_wandb_active=self.is_train_wandb_active,
            )
            if success and final_model_path:
                log_both(
                    f"Final model processing (save & artifact) successful: {final_model_path}",
                    also_to_wandb=True,
                )
            else:
                log_both(
                    f"[ERROR] Failed to save/artifact final model for timestep {self.global_timestep}.",
                    also_to_wandb=True,
                )

            # WandB finishing is handled by SessionManager or after all save attempts
        else:
            log_both(
                f"[WARNING] Training interrupted at timestep {self.global_timestep} (before {self.config.training.total_timesteps} total).",
                also_to_wandb=True,
            )

        # Always attempt to save a final checkpoint
        log_both(
            f"Attempting to save final checkpoint at timestep {self.global_timestep}.",
            also_to_wandb=False,
        )
        ckpt_success, final_ckpt_path = self.model_manager.save_final_checkpoint(
            agent=self.agent,
            model_dir=self.model_dir,
            global_timestep=self.global_timestep,
            total_episodes_completed=self.total_episodes_completed,
            game_stats=game_stats,
            run_name=self.run_name,
            is_wandb_active=self.is_train_wandb_active,
        )
        if ckpt_success and final_ckpt_path:
            log_both(
                f"Final checkpoint processing (save & artifact) successful: {final_ckpt_path}",
                also_to_wandb=True,
            )
        elif self.global_timestep > 0:  # Only log error if a checkpoint was expected
            log_both(
                f"[ERROR] Failed to save/artifact final checkpoint for timestep {self.global_timestep}.",
                also_to_wandb=True,
            )

        if self.is_train_wandb_active and wandb.run:
            self.session_manager.finalize_session()  # Finalize session after all save attempts
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
        """Executes the main training loop by delegating to TrainingLoopManager."""
        self.session_manager.log_session_start()

        with TrainingLogger(
            self.log_file_path,
            rich_console=self.rich_console,
            rich_log_panel=self.rich_log_messages,
        ) as logger:

            def log_both_impl(
                message: str,
                also_to_wandb: bool = False,
                wandb_data: Optional[Dict] = None,
                log_level: str = "info",  # Parameter for log level (unused by current TrainingLogger.log)
            ):
                logger.log(message)  # Current TrainingLogger.log does not take level
                if self.is_train_wandb_active and also_to_wandb and wandb.run:
                    log_payload = {"train_message": message}
                    if wandb_data:
                        log_payload.update(wandb_data)
                    wandb.log(log_payload, step=self.global_timestep)

            self.log_both = log_both_impl
            self.execute_full_evaluation_run = execute_full_evaluation_run

            session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_both(
                f"--- SESSION START: {self.run_name} at {session_start_time} ---"
            )

            # Setup run information logging
            self._log_run_info(self.log_both)

            # Log checkpoint resume status
            if self.resumed_from_checkpoint:
                self.log_both(
                    f"Resumed training from checkpoint: {self.resumed_from_checkpoint}"
                )

            initial_episode_state = self._initialize_game_state(self.log_both)
            self.training_loop_manager.set_initial_episode_state(initial_episode_state)

            # The display.start() context manager should wrap the loop execution
            # It's currently managed by the TrainingDisplay class itself if it uses a Live display.
            # If direct management is needed here, it would be: `with self.display.start() as ...:`
            # For now, assuming display manages its own lifecycle based on its methods.
            # If display.start() is a context manager that needs to wrap the loop:
            # with self.display.start() as _:
            #    self.training_loop_manager.run()
            # else, if display.start() just initializes and run() handles updates:
            # self.display.start() # Or similar initialization if needed

            # The TrainingDisplay.start() method in the current `display.py` (not shown here but assumed)
            # likely sets up the Rich Live display. The TrainingLoopManager will then call
            # display.update_progress and display.update_log_panel.

            try:
                self.training_loop_manager.run()  # Delegate the loop execution
            except KeyboardInterrupt:
                # This is already logged by TrainingLoopManager, but we ensure finalization.
                self.log_both(
                    "Trainer caught KeyboardInterrupt from TrainingLoopManager. Finalizing.",
                    also_to_wandb=True,
                )
            except Exception as e:
                # This is already logged by TrainingLoopManager.
                self.log_both(
                    f"Trainer caught unhandled exception from TrainingLoopManager: {e}. Finalizing.",
                    also_to_wandb=True,
                )
                # Optionally, re-raise if higher-level handling is needed: raise
            finally:
                # Finalization is critical and should always run.
                self._finalize_training(self.log_both)

    # The @property for model was removed to allow direct assignment to self.model.
    # The instance attribute self.model (Optional[ActorCriticProtocol]) should be used directly.
    # Other properties like feature_spec, obs_shape etc., might need adjustment
    # if they previously relied on a self.model property that accessed self.agent.model,
    # or if they should now access self.model directly (after checking it's not None).
    # For now, only the conflicting 'model' property is fully removed.

    @property
    def feature_spec(self):
        """Access the feature spec through ModelManager."""
        return (
            self.model_manager.feature_spec
            if hasattr(self.model_manager, "feature_spec")
            else None
        )

    @property
    def obs_shape(self):
        """Access the observation shape through ModelManager."""
        return (
            self.model_manager.obs_shape
            if hasattr(self.model_manager, "obs_shape")
            else None
        )

    @property
    def tower_depth(self):
        """Access the tower depth through ModelManager."""
        return (
            self.model_manager.tower_depth
            if hasattr(self.model_manager, "tower_depth")
            else None
        )

    @property
    def tower_width(self):
        """Access the tower width through ModelManager."""
        return (
            self.model_manager.tower_width
            if hasattr(self.model_manager, "tower_width")
            else None
        )

    @property
    def se_ratio(self):
        """Access the SE ratio through ModelManager."""
        return (
            self.model_manager.se_ratio
            if hasattr(self.model_manager, "se_ratio")
            else None
        )

    # Backward compatibility delegation methods
    def _create_model_artifact(
        self,
        model_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: str = "model",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        log_both: Optional[Callable] = None,
    ) -> bool:
        """Backward compatibility method - delegates to ModelManager."""
        # Use default artifact name if not provided
        if artifact_name is None:
            artifact_name = os.path.basename(model_path)

        # Use default description if not provided
        if description is None:
            # Use trainer's run_name for backward compatibility (tests set this manually)
            run_name = getattr(self, "run_name", self.session_manager.run_name)
            description = f"Model checkpoint from run {run_name}"

        # Use trainer's run_name for artifact naming (for backward compatibility with tests)
        run_name = getattr(self, "run_name", self.session_manager.run_name)

        # Store original logger function and temporarily replace if log_both provided
        original_logger = self.model_manager.logger_func
        if log_both:
            # Create a wrapper that detects error messages and adds log_level="error"
            def logger_wrapper(message):
                if "Error creating W&B artifact" in message:
                    return log_both(message, log_level="error")
                else:
                    return log_both(message)

            self.model_manager.logger_func = logger_wrapper

        try:
            result = self.model_manager.create_model_artifact(
                model_path=model_path,
                artifact_name=artifact_name,
                run_name=run_name,
                is_wandb_active=self.session_manager.is_wandb_active,
                artifact_type=artifact_type,
                description=description,
                metadata=metadata,
                aliases=aliases,
            )
        finally:
            # Restore original logger function
            self.model_manager.logger_func = original_logger

        return result
