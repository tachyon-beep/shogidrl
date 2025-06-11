"""
Unit tests for SchedulerFactory and learning rate scheduling functionality.

This module tests:
- SchedulerFactory creation of different scheduler types
- Scheduler configuration validation
- Learning rate progression for each scheduler type
- Error handling for invalid configurations
"""

import pytest
import torch
from torch.optim import Adam

from keisei.core.scheduler_factory import SchedulerFactory


class TestSchedulerFactory:
    """Tests for SchedulerFactory scheduler creation."""

    @pytest.fixture
    def dummy_optimizer(self):
        """Create a dummy optimizer for testing."""
        # Create a simple parameter to optimize
        param = torch.nn.Parameter(torch.randn(2, 2))
        return Adam([param], lr=1e-3, weight_decay=0.0)  # ADDED weight_decay

    def test_create_scheduler_none_type(self, dummy_optimizer):
        """Test that None schedule_type returns None."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer, schedule_type=None, total_steps=1000
        )
        assert scheduler is None

    def test_create_linear_scheduler(self, dummy_optimizer):
        """Test linear decay scheduler creation."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            schedule_type="linear",
            total_steps=100,
            schedule_kwargs={"final_lr_fraction": 0.1},
        )

        assert scheduler is not None
        assert hasattr(scheduler, "step")

        # Test learning rate progression
        initial_lr = dummy_optimizer.param_groups[0]["lr"]

        # After 0 steps, LR should be unchanged
        assert dummy_optimizer.param_groups[0]["lr"] == initial_lr

        # After 50 steps (halfway), LR should be 50% of initial (linear decay from 100% to 10%)
        for _ in range(50):
            if scheduler:  # Check if scheduler exists
                dummy_optimizer.step()
                scheduler.step()
        mid_lr = dummy_optimizer.param_groups[0]["lr"]
        # Original test expectation was 0.5 * initial_lr.
        # For a linear decay from initial_lr to final_lr_fraction * initial_lr over total_steps:
        # lr(step) = initial_lr - (initial_lr - final_lr_fraction * initial_lr) * (step / total_steps)
        # lr(50) = initial_lr - (initial_lr - 0.1 * initial_lr) * (50 / 100)
        # lr(50) = initial_lr - (0.9 * initial_lr) * 0.5
        # lr(50) = initial_lr - 0.45 * initial_lr = 0.55 * initial_lr
        expected_mid_lr = initial_lr * 0.55
        assert abs(mid_lr - expected_mid_lr) < 1e-6

        # After 100 steps, LR should be 10% of initial
        for _ in range(50):  # remaining 50 steps
            if scheduler:
                dummy_optimizer.step()
                scheduler.step()
        final_lr = dummy_optimizer.param_groups[0]["lr"]
        expected_final_lr = initial_lr * 0.1
        assert abs(final_lr - expected_final_lr) < 1e-6

    def test_create_cosine_scheduler(self, dummy_optimizer):
        """Test cosine annealing scheduler creation."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            schedule_type="cosine",
            total_steps=100,
            schedule_kwargs={"eta_min_fraction": 0.0},
        )

        assert scheduler is not None
        assert hasattr(scheduler, "step")

        initial_lr = dummy_optimizer.param_groups[0]["lr"]

        # After 100 steps, LR should approach eta_min (0.0)
        for _ in range(100):
            if scheduler:
                dummy_optimizer.step()
                scheduler.step()
        final_lr = dummy_optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr * 0.1  # Should be very small

    def test_create_exponential_scheduler(self, dummy_optimizer):
        """Test exponential decay scheduler creation."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            schedule_type="exponential",
            total_steps=100,  # Not used for exponential
            schedule_kwargs={"gamma": 0.9},
        )

        assert scheduler is not None
        assert hasattr(scheduler, "step")

        initial_lr = dummy_optimizer.param_groups[0]["lr"]

        # After 1 step, LR should be gamma * initial_lr
        if scheduler:
            dummy_optimizer.step()
            scheduler.step()
        new_lr = dummy_optimizer.param_groups[0]["lr"]
        expected_lr = initial_lr * 0.9
        assert abs(new_lr - expected_lr) < 1e-8

    def test_create_step_scheduler(self, dummy_optimizer):
        """Test step decay scheduler creation."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            schedule_type="step",
            total_steps=100,  # Not used for step
            schedule_kwargs={"step_size": 10, "gamma": 0.5},
        )

        assert scheduler is not None
        assert hasattr(scheduler, "step")

        initial_lr = dummy_optimizer.param_groups[0]["lr"]

        # After 9 steps, LR should be unchanged
        for _ in range(9):
            if scheduler:
                dummy_optimizer.step()
                scheduler.step()
        assert dummy_optimizer.param_groups[0]["lr"] == initial_lr

        # After 10 steps, LR should be halved
        if scheduler:
            dummy_optimizer.step()
            scheduler.step()
        new_lr = dummy_optimizer.param_groups[0]["lr"]
        expected_lr = initial_lr * 0.5
        assert abs(new_lr - expected_lr) < 1e-8

    def test_invalid_scheduler_type(self, dummy_optimizer):
        """Test that invalid scheduler type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            SchedulerFactory.create_scheduler(
                optimizer=dummy_optimizer, schedule_type="invalid_type", total_steps=100
            )

    def test_scheduler_with_default_kwargs(self, dummy_optimizer):
        """Test scheduler creation with default kwargs (None)."""
        # Linear scheduler with default kwargs
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            schedule_type="linear",
            total_steps=100,
            schedule_kwargs=None,
        )
        assert scheduler is not None

        # Exponential scheduler with default kwargs
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            schedule_type="exponential",
            total_steps=100,
            schedule_kwargs=None,
        )
        assert scheduler is not None

    def test_linear_scheduler_edge_cases(self, dummy_optimizer):
        """Test linear scheduler edge cases."""
        # Test with final_lr_fraction = 0.0
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            schedule_type="linear",
            total_steps=10,
            schedule_kwargs={"final_lr_fraction": 0.0},
        )

        initial_lr = dummy_optimizer.param_groups[0]["lr"]

        # After all steps, LR should approach 0
        for _ in range(10):
            if scheduler:
                dummy_optimizer.step()
                scheduler.step()
        final_lr = dummy_optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr * 0.1

    def test_cosine_scheduler_with_eta_min(self, dummy_optimizer):
        """Test cosine scheduler with non-zero eta_min_fraction."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            schedule_type="cosine",
            total_steps=100,
            schedule_kwargs={"eta_min_fraction": 0.1},
        )

        initial_lr = dummy_optimizer.param_groups[0]["lr"]

        # After all steps, LR should be at least 10% of initial
        for _ in range(100):
            if scheduler:
                dummy_optimizer.step()
                scheduler.step()
        final_lr = dummy_optimizer.param_groups[0]["lr"]
        expected_min_lr = initial_lr * 0.1
        assert final_lr >= expected_min_lr * 0.9  # Allow small numerical errors


class TestSchedulerFactoryIntegration:
    """Integration tests for SchedulerFactory with realistic scenarios."""

    def test_scheduler_type_coverage(self):
        """Test that all documented scheduler types are supported."""
        dummy_param = torch.nn.Parameter(torch.randn(1))
        optimizer = Adam(
            [dummy_param], lr=1e-3, weight_decay=0.0
        )  # Ensure weight_decay is present

        supported_types = ["linear", "cosine", "exponential", "step"]

        for schedule_type in supported_types:
            scheduler = SchedulerFactory.create_scheduler(
                optimizer=optimizer, schedule_type=schedule_type, total_steps=100
            )
            assert scheduler is not None, f"Failed to create {schedule_type} scheduler"

    def test_multiple_schedulers_same_optimizer(self):
        """Test that multiple schedulers can be created for the same optimizer type."""
        dummy_param1 = torch.nn.Parameter(torch.randn(1))
        dummy_param2 = torch.nn.Parameter(torch.randn(1))

        optimizer1 = Adam([dummy_param1], lr=1e-3)
        optimizer2 = Adam([dummy_param2], lr=1e-3)

        scheduler1 = SchedulerFactory.create_scheduler(
            optimizer=optimizer1, schedule_type="linear", total_steps=100
        )

        scheduler2 = SchedulerFactory.create_scheduler(
            optimizer=optimizer2, schedule_type="cosine", total_steps=100
        )

        assert scheduler1 is not None
        assert scheduler2 is not None
        assert scheduler1 != scheduler2

    def test_scheduler_kwargs_validation(self):
        """Test that schedulers are created with default kwargs if specific ones are missing."""
        dummy_param = torch.nn.Parameter(torch.randn(1))
        optimizer = Adam([dummy_param], lr=1e-3, weight_decay=0.0)

        # Test linear scheduler: uses default final_lr_fraction if not provided
        # schedule_kwargs is omitted, factory uses default for final_lr_fraction
        scheduler_linear = SchedulerFactory.create_scheduler(
            optimizer=optimizer, schedule_type="linear", total_steps=100
        )
        assert (
            scheduler_linear is not None
        ), "Linear scheduler should be created with default kwargs"

        # Test exponential scheduler: uses default gamma if not provided
        # Empty dict for schedule_kwargs, factory uses default for gamma
        scheduler_exp = SchedulerFactory.create_scheduler(
            optimizer=optimizer,
            schedule_type="exponential",
            total_steps=100,  # total_steps is not strictly used by exponential's core logic but factory requires it
            schedule_kwargs={},
        )
        assert (
            scheduler_exp is not None
        ), "Exponential scheduler should be created with default kwargs"

        # Test step scheduler: uses default step_size and gamma if not provided
        # Empty dict for schedule_kwargs, factory uses defaults for step_size and gamma
        scheduler_step = SchedulerFactory.create_scheduler(
            optimizer=optimizer,
            schedule_type="step",
            total_steps=100,  # total_steps is not strictly used by step's core logic but factory requires it
            schedule_kwargs={},
        )
        assert (
            scheduler_step is not None
        ), "Step scheduler should be created with default kwargs"

    def test_scheduler_total_steps_validation(self):
        """Test validation of total_steps for relevant schedulers."""
        dummy_param1 = torch.nn.Parameter(torch.randn(1))
        optimizer1 = Adam(
            [dummy_param1], lr=1e-3, weight_decay=0.0
        )  # Ensure weight_decay is present

        # Linear scheduler requires total_steps > 0
        with pytest.raises(ValueError, match="total_steps must be a positive integer"):
            SchedulerFactory.create_scheduler(
                optimizer=optimizer1,
                schedule_type="linear",
                total_steps=0,
                schedule_kwargs={"final_lr_fraction": 0.1},
            )

        with pytest.raises(ValueError, match="total_steps must be a positive integer"):
            SchedulerFactory.create_scheduler(
                optimizer=optimizer1,
                schedule_type="linear",
                total_steps=-10,
                schedule_kwargs={"final_lr_fraction": 0.1},
            )

        # Cosine scheduler requires total_steps > 0
        with pytest.raises(
            ValueError,
            match="total_steps must be a positive integer for cosine scheduler",
        ):
            SchedulerFactory.create_scheduler(
                optimizer=optimizer1,
                schedule_type="cosine",
                total_steps=0,
                schedule_kwargs={"eta_min_fraction": 0.1},
            )

        with pytest.raises(
            ValueError,
            match="total_steps must be a positive integer for cosine scheduler",
        ):
            SchedulerFactory.create_scheduler(
                optimizer=optimizer1,
                schedule_type="cosine",
                total_steps=-10,
                schedule_kwargs={"eta_min_fraction": 0.1},
            )
