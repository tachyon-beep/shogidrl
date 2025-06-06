"""
training/callback_manager.py: Manages training callbacks and their execution.
"""

from typing import TYPE_CHECKING, Any, List

from . import callbacks

if TYPE_CHECKING:
    from .trainer import Trainer


class CallbackManager:
    """Manages the setup and execution of training callbacks."""

    def __init__(self, config: Any, model_dir: str):
        """
        Initialize the CallbackManager.

        Args:
            config: Training configuration
            model_dir: Directory for saving model checkpoints
        """
        self.config = config
        self.model_dir = model_dir
        self.callbacks: List[callbacks.Callback] = []

    def setup_default_callbacks(self) -> List[callbacks.Callback]:
        """
        Setup default callbacks for training.

        Returns:
            List of configured callbacks
        """
        callback_list: List[callbacks.Callback] = []

        # Align checkpoint interval with steps_per_epoch to ensure callbacks fire
        checkpoint_interval = self.config.training.checkpoint_interval_timesteps
        steps_per_epoch = self.config.training.steps_per_epoch
        if checkpoint_interval % steps_per_epoch != 0:
            aligned = ((checkpoint_interval // steps_per_epoch) + 1) * steps_per_epoch
            print(
                f"[INFO] Adjusting checkpoint_interval_timesteps from {checkpoint_interval} to {aligned} to align with steps_per_epoch {steps_per_epoch}."
            )
            checkpoint_interval = aligned

        callback_list.append(
            callbacks.CheckpointCallback(checkpoint_interval, self.model_dir)
        )

        # Evaluation callback
        eval_cfg = getattr(self.config, "evaluation", None)
        eval_interval = (
            eval_cfg.evaluation_interval_timesteps
            if eval_cfg and hasattr(eval_cfg, "evaluation_interval_timesteps")
            else getattr(self.config.training, "evaluation_interval_timesteps", 1000)
        )

        if eval_interval % steps_per_epoch != 0:
            aligned = ((eval_interval // steps_per_epoch) + 1) * steps_per_epoch
            print(
                f"[INFO] Adjusting evaluation_interval_timesteps from {eval_interval} to {aligned} to align with steps_per_epoch {steps_per_epoch}."
            )
            eval_interval = aligned

        callback_list.append(callbacks.EvaluationCallback(eval_cfg, eval_interval))

        self.callbacks = callback_list
        return callback_list

    def execute_step_callbacks(self, trainer: "Trainer") -> None:
        """
        Execute on_step_end callbacks for all registered callbacks.

        Args:
            trainer: The trainer instance
        """
        for callback in self.callbacks:
            try:
                callback.on_step_end(trainer)
            except Exception as e:
                # Log error but don't stop training
                if hasattr(trainer, "log_both") and trainer.log_both:
                    trainer.log_both(
                        f"[ERROR] Callback {type(callback).__name__} failed: {e}",
                        also_to_wandb=False,
                    )

    def add_callback(self, callback: callbacks.Callback) -> None:
        """
        Add a custom callback to the manager.

        Args:
            callback: The callback to add
        """
        self.callbacks.append(callback)

    def remove_callback(self, callback_type: type) -> bool:
        """
        Remove callbacks of a specific type.

        Args:
            callback_type: The type of callback to remove

        Returns:
            True if any callbacks were removed, False otherwise
        """
        original_count = len(self.callbacks)
        self.callbacks = [
            cb for cb in self.callbacks if not isinstance(cb, callback_type)
        ]
        return len(self.callbacks) < original_count

    def get_callbacks(self) -> List[callbacks.Callback]:
        """
        Get the list of registered callbacks.

        Returns:
            List of callbacks
        """
        return self.callbacks.copy()

    def clear_callbacks(self) -> None:
        """Clear all registered callbacks."""
        self.callbacks.clear()
