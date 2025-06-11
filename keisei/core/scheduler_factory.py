"""
Learning rate scheduler factory for PPO training.
"""

from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR, StepLR


class SchedulerFactory:
    """Factory for creating PyTorch learning rate schedulers."""

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        schedule_type: Optional[str],
        total_steps: int,
        schedule_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Create a learning rate scheduler based on configuration.

        Args:
            optimizer: PyTorch optimizer
            schedule_type: Type of scheduler ('linear', 'cosine', 'exponential', 'step')
            total_steps: Total number of training steps for scheduling
            schedule_kwargs: Additional arguments for the scheduler

        Returns:
            Configured scheduler or None if schedule_type is None
        """
        if schedule_type is None:
            return None
        schedule_kwargs = schedule_kwargs or {}

        if schedule_type == "linear":
            return SchedulerFactory._create_linear_scheduler(
                optimizer, total_steps, schedule_kwargs
            )
        elif schedule_type == "cosine":
            return SchedulerFactory._create_cosine_scheduler(
                optimizer, total_steps, schedule_kwargs
            )
        elif schedule_type == "exponential":
            return SchedulerFactory._create_exponential_scheduler(
                optimizer, schedule_kwargs
            )
        elif schedule_type == "step":
            return SchedulerFactory._create_step_scheduler(optimizer, schedule_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {schedule_type}")

    @staticmethod
    def _create_linear_scheduler(
        optimizer: torch.optim.Optimizer, total_steps: int, kwargs: Dict[str, Any]
    ) -> LambdaLR:
        """Create linear decay scheduler."""
        if total_steps <= 0:
            raise ValueError(
                "total_steps must be a positive integer for linear scheduler"
            )
        # Provide default if not present, to support tests relying on defaults
        final_lr_fraction = kwargs.get("final_lr_fraction", 0.1)

        def linear_decay(step: int) -> float:
            # step is 0-indexed by LambdaLR for the *current* epoch/step.
            # It represents the number of times scheduler.step() has been called.
            if step > total_steps:
                current_step = total_steps
            else:
                current_step = step
            progress = current_step / total_steps
            return (1.0 - progress) * (1.0 - final_lr_fraction) + final_lr_fraction

        return LambdaLR(optimizer, lr_lambda=linear_decay)

    @staticmethod
    def _create_cosine_scheduler(
        optimizer: torch.optim.Optimizer, total_steps: int, kwargs: Dict[str, Any]
    ) -> CosineAnnealingLR:
        """Create cosine annealing scheduler."""
        if total_steps <= 0:
            raise ValueError(
                "total_steps must be a positive integer for cosine scheduler"
            )
        eta_min_fraction = kwargs.get("eta_min_fraction", 0.0)
        initial_lr = optimizer.param_groups[0]["lr"]
        eta_min = initial_lr * eta_min_fraction

        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    @staticmethod
    def _create_exponential_scheduler(
        optimizer: torch.optim.Optimizer, kwargs: Dict[str, Any]
    ) -> ExponentialLR:
        """Create exponential decay scheduler."""
        # Provide default if not present
        gamma = kwargs.get("gamma", 0.995)
        return ExponentialLR(optimizer, gamma=gamma)

    @staticmethod
    def _create_step_scheduler(
        optimizer: torch.optim.Optimizer, kwargs: Dict[str, Any]
    ) -> StepLR:
        """Create step decay scheduler."""
        # Provide defaults if not present
        step_size = kwargs.get("step_size", 1000)
        gamma = kwargs.get("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
