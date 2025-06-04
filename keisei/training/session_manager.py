"""
training/session_manager.py: Session lifecycle management for Shogi RL training.

This module handles session-level concerns including:
- Run name generation and validation
- Directory structure creation and management
- Configuration serialization and persistence
- WandB initialization and configuration
- Session logging and reporting
"""

import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import wandb
from keisei.config_schema import AppConfig
from keisei.utils.unified_logger import log_error_to_stderr, log_warning_to_stderr
from keisei.utils.utils import generate_run_name

from . import utils


class SessionManager:
    """Manages session-level lifecycle for training runs."""

    def __init__(self, config: AppConfig, args: Any, run_name: Optional[str] = None):
        """
        Initialize the SessionManager.

        Args:
            config: Application configuration
            args: Command-line arguments
            run_name: Optional explicit run name (overrides config and auto-generation)
        """
        self.config = config
        self.args = args

        # Determine run_name: explicit > CLI > config > auto-generate
        if run_name:
            self._run_name = run_name
        elif hasattr(args, "run_name") and args.run_name:
            self._run_name = args.run_name
        elif (
            hasattr(config, "logging")
            and hasattr(config.logging, "run_name")
            and config.logging.run_name
        ):
            self._run_name = config.logging.run_name
        else:
            self._run_name = generate_run_name(config, None)

        # Initialize paths - will be set by setup_directories()
        self._run_artifact_dir: Optional[str] = None
        self._model_dir: Optional[str] = None
        self._log_file_path: Optional[str] = None
        self._eval_log_file_path: Optional[str] = None

        # WandB state
        self._is_wandb_active: Optional[bool] = None

    @property
    def run_name(self) -> str:
        """Get the run name."""
        return self._run_name

    @property
    def run_artifact_dir(self) -> str:
        """Get the run artifact directory path."""
        if self._run_artifact_dir is None:
            raise RuntimeError(
                "Directories not yet set up. Call setup_directories() first."
            )
        return self._run_artifact_dir

    @property
    def model_dir(self) -> str:
        """Get the model directory path."""
        if self._model_dir is None:
            raise RuntimeError(
                "Directories not yet set up. Call setup_directories() first."
            )
        return self._model_dir

    @property
    def log_file_path(self) -> str:
        """Get the log file path."""
        if self._log_file_path is None:
            raise RuntimeError(
                "Directories not yet set up. Call setup_directories() first."
            )
        return self._log_file_path

    @property
    def eval_log_file_path(self) -> str:
        """Get the evaluation log file path."""
        if self._eval_log_file_path is None:
            raise RuntimeError(
                "Directories not yet set up. Call setup_directories() first."
            )
        return self._eval_log_file_path

    @property
    def is_wandb_active(self) -> bool:
        """Check if WandB is active."""
        if self._is_wandb_active is None:
            raise RuntimeError("WandB not yet initialized. Call setup_wandb() first.")
        return bool(self._is_wandb_active)

    def setup_directories(self) -> Dict[str, str]:
        """
        Set up directory structure for the training run.

        Returns:
            Dictionary containing directory paths
        """
        try:
            dirs = utils.setup_directories(self.config, self._run_name)
            self._run_artifact_dir = dirs["run_artifact_dir"]
            self._model_dir = dirs["model_dir"]
            self._log_file_path = dirs["log_file_path"]
            self._eval_log_file_path = dirs["eval_log_file_path"]
            return dirs
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to setup directories: {e}") from e

    def setup_wandb(self) -> bool:
        """
        Initialize Weights & Biases logging.

        Returns:
            True if WandB is active, False otherwise
        """
        if self._run_artifact_dir is None:
            raise RuntimeError("Directories must be set up before initializing WandB.")

        try:
            self._is_wandb_active = utils.setup_wandb(
                self.config, self._run_name, self._run_artifact_dir
            )
            return bool(self._is_wandb_active)
        except Exception as e:  # Catch all exceptions for WandB setup
            log_warning_to_stderr("SessionManager", f"WandB setup failed: {e}")
            self._is_wandb_active = False
            return False

    def save_effective_config(self) -> None:
        """Save the effective configuration to a JSON file."""
        if self._run_artifact_dir is None:
            raise RuntimeError("Directories must be set up before saving config.")

        try:
            # Ensure the directory exists
            os.makedirs(self._run_artifact_dir, exist_ok=True)

            effective_config_str = utils.serialize_config(self.config)
            config_path = os.path.join(self._run_artifact_dir, "effective_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(effective_config_str)
        except (OSError, TypeError) as e:
            log_error_to_stderr(
                "SessionManager", f"Error saving effective_config.json: {e}"
            )
            raise RuntimeError(f"Failed to save effective config: {e}") from e

    def log_session_info(
        self,
        logger_func: Callable[[str], None],
        agent_info: Optional[Dict[str, Any]] = None,
        resumed_from_checkpoint: Optional[str] = None,
        global_timestep: int = 0,
        total_episodes_completed: int = 0,
    ) -> None:
        """
        Log comprehensive session information.

        Args:
            logger_func: Function to call for logging messages
            agent_info: Optional agent information (name, type)
            resumed_from_checkpoint: Optional checkpoint path if resumed
            global_timestep: Current global timestep
            total_episodes_completed: Total episodes completed so far
        """
        # Session title with optional WandB URL
        run_title = f"Keisei Training Run: {self._run_name}"
        if self._is_wandb_active and wandb.run and hasattr(wandb.run, "url"):
            run_title += f" (W&B: {wandb.run.url})"

        logger_func(run_title)
        logger_func(f"Run directory: {self._run_artifact_dir}")

        # Ensure directory exists before constructing paths
        if self._run_artifact_dir:
            config_path = os.path.join(self._run_artifact_dir, "effective_config.json")
            logger_func(f"Effective config saved to: {config_path}")
        else:
            logger_func("Warning: Run artifact directory not set")

        # Configuration information
        if self.config.env.seed is not None:
            logger_func(f"Random seed: {self.config.env.seed}")

        logger_func(f"Device: {self.config.env.device}")

        # Agent information
        if agent_info:
            logger_func(
                f"Agent: {agent_info.get('type', 'Unknown')} ({agent_info.get('name', 'Unknown')})"
            )

        logger_func(
            f"Total timesteps: {self.config.training.total_timesteps}, "
            f"Steps per PPO epoch: {self.config.training.steps_per_epoch}"
        )

        # Resume information
        if global_timestep > 0:
            if resumed_from_checkpoint:
                logger_func(
                    f"Resumed training from checkpoint: {resumed_from_checkpoint}"
                )
            logger_func(
                f"Resuming from timestep {global_timestep}, {total_episodes_completed} episodes completed."
            )
        else:
            logger_func("Starting fresh training.")

    def log_session_start(self) -> None:
        """Log session start event to file."""
        if self._log_file_path is None:
            raise RuntimeError(
                "Directories must be set up before logging session start."
            )

        try:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] --- SESSION START: {self._run_name} ---\n")
        except (OSError, IOError) as e:
            log_error_to_stderr("SessionManager", f"Failed to log session start: {e}")

    def finalize_session(self) -> None:
        """Finalize the training session."""
        if self._is_wandb_active and wandb.run:
            try:
                # Fix B5: Use cross-platform threading.Timer instead of POSIX signal
                import threading

                timeout_occurred = threading.Event()

                def timeout_handler():
                    timeout_occurred.set()

                # Set up timeout (10 seconds) - cross-platform compatible
                timer = threading.Timer(10.0, timeout_handler)
                timer.start()

                try:
                    # Check periodically if timeout occurred
                    import time

                    start_time = time.time()
                    while (
                        not timeout_occurred.is_set()
                        and (time.time() - start_time) < 10
                    ):
                        try:
                            wandb.finish()
                            break  # Success, exit loop
                        except Exception:
                            time.sleep(0.1)  # Brief wait before retry

                    if timeout_occurred.is_set():
                        raise TimeoutError("WandB finalization timed out")

                finally:
                    timer.cancel()  # Cancel the timer

            except (KeyboardInterrupt, TimeoutError):
                log_warning_to_stderr(
                    "SessionManager", "WandB finalization interrupted or timed out"
                )
                try:
                    # Force finish without waiting
                    wandb.finish(exit_code=1)
                except Exception:
                    pass
            except Exception as e:  # Catch all exceptions for WandB finalization
                log_warning_to_stderr(
                    "SessionManager", f"WandB finalization failed: {e}"
                )

    def setup_seeding(self) -> None:
        """Setup random seeding based on configuration."""
        utils.setup_seeding(self.config)

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the session configuration.

        Returns:
            Dictionary containing session summary information
        """
        return {
            "run_name": self._run_name,
            "run_artifact_dir": self._run_artifact_dir,
            "model_dir": self._model_dir,
            "log_file_path": self._log_file_path,
            "is_wandb_active": self._is_wandb_active,
            "seed": self.config.env.seed if hasattr(self.config.env, "seed") else None,
            "device": (
                self.config.env.device if hasattr(self.config.env, "device") else None
            ),
        }
