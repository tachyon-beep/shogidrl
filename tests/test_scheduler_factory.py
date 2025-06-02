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
        return Adam([param], lr=1e-3)

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
            scheduler.step()
        mid_lr = dummy_optimizer.param_groups[0]["lr"]
        expected_mid_lr = initial_lr * 0.5  # max(0.1, 1.0 - 50/100) = 0.5
        assert abs(mid_lr - expected_mid_lr) < 1e-6

        # After 100 steps, LR should be 10% of initial
        for _ in range(50):
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
            scheduler.step()
        assert dummy_optimizer.param_groups[0]["lr"] == initial_lr

        # After 10 steps, LR should be halved
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
            scheduler.step()
        final_lr = dummy_optimizer.param_groups[0]["lr"]
        expected_min_lr = initial_lr * 0.1
        assert final_lr >= expected_min_lr * 0.9  # Allow small numerical errors


class TestSchedulerFactoryIntegration:
    """Integration tests for SchedulerFactory with realistic scenarios."""

    def test_scheduler_type_coverage(self):
        """Test that all documented scheduler types are supported."""
        dummy_param = torch.nn.Parameter(torch.randn(1))
        optimizer = Adam([dummy_param], lr=1e-3)

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
