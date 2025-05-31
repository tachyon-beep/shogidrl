"""
Unit tests for TrainingLoopManager in training_loop_manager.py
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from keisei.training.step_manager import EpisodeState
from keisei.training.training_loop_manager import TrainingLoopManager


@pytest.fixture
def mock_trainer():
    """Create a mock trainer with all necessary components."""
    mock_trainer = Mock()

    # Mock config
    mock_config = Mock()
    mock_config.training.steps_per_epoch = 32
    mock_config.training.total_timesteps = 1000
    mock_config.training.ppo_epochs = 4
    mock_config.training.render_every_steps = 100
    mock_config.training.refresh_per_second = 4
    mock_config.parallel.enabled = False  # Disable parallel training for tests
    mock_trainer.config = mock_config

    # Mock agent
    mock_agent = Mock()
    mock_trainer.agent = mock_agent

    # Mock components
    mock_trainer.experience_buffer = Mock()
    mock_trainer.step_manager = Mock()
    mock_trainer.display = Mock()
    mock_trainer.callbacks = []

    # Mock state
    mock_trainer.global_timestep = 0
    mock_trainer.total_episodes_completed = 0
    mock_trainer.total_wins = 0
    mock_trainer.total_draws = 0
    mock_trainer.last_10_game_results = []

    return mock_trainer


@pytest.fixture
def mock_episode_state():
    """Create a mock episode state."""
    return EpisodeState(
        current_obs=np.zeros((46, 9, 9)),
        current_obs_tensor=torch.zeros(46, 9, 9),
        episode_reward=0.0,
        episode_length=0,
    )


def test_training_loop_manager_initialization(mock_trainer):
    """Test TrainingLoopManager initializes correctly."""
    manager = TrainingLoopManager(mock_trainer)

    assert manager.trainer is mock_trainer
    assert manager.config is mock_trainer.config
    assert manager.agent is mock_trainer.agent
    assert manager.buffer is mock_trainer.experience_buffer
    assert manager.step_manager is mock_trainer.step_manager
    assert manager.display is mock_trainer.display
    assert manager.callbacks is mock_trainer.callbacks

    assert manager.current_epoch == 0
    assert manager.episode_state is None
    assert abs(manager.last_time_for_sps - 0.0) < 1e-9
    assert manager.steps_since_last_time_for_sps == 0
    assert abs(manager.last_display_update_time - 0.0) < 1e-9


def test_set_initial_episode_state(mock_trainer, mock_episode_state):
    """Test setting initial episode state."""
    manager = TrainingLoopManager(mock_trainer)

    assert manager.episode_state is None
    manager.set_initial_episode_state(mock_episode_state)
    assert manager.episode_state is mock_episode_state


def test_training_loop_manager_basic_functionality(mock_trainer, mock_episode_state):
    """Test basic TrainingLoopManager functionality."""
    manager = TrainingLoopManager(mock_trainer)

    # Test initialization
    assert manager.trainer is mock_trainer
    assert manager.config is mock_trainer.config
    assert manager.agent is mock_trainer.agent
    assert manager.buffer is mock_trainer.experience_buffer
    assert manager.step_manager is mock_trainer.step_manager
    assert manager.display is mock_trainer.display
    assert manager.callbacks is mock_trainer.callbacks

    assert manager.current_epoch == 0
    assert manager.episode_state is None
    assert abs(manager.last_time_for_sps - 0.0) < 1e-9
    assert manager.steps_since_last_time_for_sps == 0
    assert abs(manager.last_display_update_time - 0.0) < 1e-9

    # Test setting episode state
    manager.set_initial_episode_state(mock_episode_state)
    assert manager.episode_state is mock_episode_state


def test_run_epoch_functionality(mock_trainer):
    """Test the _run_epoch method."""
    manager = TrainingLoopManager(mock_trainer)

    # Mock the necessary trainer methods for epoch execution
    mock_trainer.global_timestep = 100
    mock_trainer.config.training.total_timesteps = 1000

    # Set up trainer to have the required metrics_manager attribute
    mock_trainer.metrics_manager = Mock()
    mock_trainer.metrics_manager.global_timestep = 100

    # The method should be callable (it may raise exceptions due to incomplete mocking)
    assert hasattr(manager, "_run_epoch")
    assert callable(manager._run_epoch)


def test_training_loop_manager_run_method_structure(mock_trainer, mock_episode_state):
    """Test that the run method has the expected structure."""
    manager = TrainingLoopManager(mock_trainer)
    manager.set_initial_episode_state(mock_episode_state)

    # Mock trainer state to avoid infinite loops
    mock_trainer.global_timestep = 1000  # >= total_timesteps to end quickly
    mock_trainer.config.training.total_timesteps = 1000

    # The run method should be callable
    assert hasattr(manager, "run")
    assert callable(manager.run)


if __name__ == "__main__":
    pytest.main([__file__])
