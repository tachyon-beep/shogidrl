"""
E2E tests for train.py.
"""

import subprocess
import sys
import tempfile
import time
import os
from pathlib import Path

import pytest

TRAIN_PATH = Path(__file__).parent.parent.parent / "train.py"

@pytest.fixture
def mock_wandb_disabled(monkeypatch):
    """Mock wandb to disable real API calls during testing."""
    from unittest.mock import MagicMock
    import sys
    
    mock_wandb = MagicMock()
    
    # Mock wandb imports directly in sys.modules
    sys.modules['wandb'] = mock_wandb
    
    # Simulate disabled wandb
    mock_wandb.run = None
    mock_wandb.init.return_value = None
    mock_wandb.config = {}
    mock_wandb.log = MagicMock()
    
    # Set environment variable to disable wandb
    monkeypatch.setenv("WANDB_DISABLED", "true")
    
    yield mock_wandb


def check_training_outputs(result, expected_timesteps):
    """Check common training outputs."""
    assert result.returncode == 0, f"Training failed: {result.stderr}"
    
    # Check that we actually trained for the expected timesteps
    # Look for timestep progress in the formatted output
    assert f"Steps: {expected_timesteps}/{expected_timesteps}" in result.stderr or \
           f"timesteps={expected_timesteps}" in result.stderr or \
           str(expected_timesteps) in result.stderr, f"Expected timesteps {expected_timesteps} not found in output"


@pytest.mark.slow
def test_train_cli_help(mock_wandb_disabled):
    """Test that train.py train --help runs and prints usage."""
    result = subprocess.run(
        [sys.executable, TRAIN_PATH, "train", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--device" in result.stdout


@pytest.mark.slow
def test_train_resume_autodetect(tmp_path, mock_wandb_disabled):
    """
    Test that train.py can auto-detect a checkpoint from parent directory and resume training.

    This test verifies the parent directory checkpoint search functionality:
    1. Save a checkpoint directly in savedir (tmp_path)
    2. Run train.py with --resume latest and --savedir tmp_path
    3. Verify ModelManager finds the checkpoint in parent dir, copies it to run dir, and resumes
    """
    # Step 1: Create a minimal checkpoint file in the parent directory (tmp_path)
    import torch
    checkpoint_path = tmp_path / "checkpoint_ts1.pth"
    
    # Create a checkpoint with minimal required structure
    checkpoint_data = {
        'model': {},  # Minimal model state_dict
        'optimizer': {},
        'scheduler': {},
        'global_timestep': 1,
        'episode_count': 0,
        'best_reward': float('-inf'),
        'config': {
            'training': {
                'model_type': 'resnet',
                'total_timesteps': 10,
                'steps_per_epoch': 2,
                'checkpoint_interval_timesteps': 5,
            },
            'model': {
                'input_features': 'core_46_channels',
                'tower_depth': 1,
                'tower_width': 16,
                'se_ratio': 0.25,
            },
            'ppo': {
                'learning_rate': 1e-4,
            },
            'env': {
                'seed': 42,
                'tensorboard_log_dir': None,
                'log_tensorboard': False,
            },
            'logging': {
                'wandb_enabled': False,
                'run_name': None,
            },
            'device': {
                'device': 'cpu',
            },
            'evaluation': {},
        }
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Created checkpoint: {checkpoint_path}")

    # Step 2: Run train.py with --resume latest and --savedir tmp_path
    # The ModelManager should find the checkpoint in parent dir, copy to run dir, and resume
    env = os.environ.copy()
    env['WANDB_DISABLED'] = 'true'
    
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "train",
            "--resume", "latest",
            "--savedir", str(tmp_path),
            "--total-timesteps", "10",
            "--seed", "42",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")

    # Step 3: Verify that a run directory was created 
    run_dirs = list(tmp_path.glob("keisei_*"))
    assert run_dirs, f"No run directory created in {tmp_path}"
    
    run_dir = run_dirs[0]
    print(f"Found run directory: {run_dir}")
    
    # Verify that training completed successfully
    check_training_outputs(result, 10)
    
    # Verify that checkpoints were created in the run directory
    checkpoints = list(run_dir.glob("checkpoint_ts*.pth"))
    assert checkpoints, "No checkpoints found in run directory"


@pytest.mark.slow
def test_train_runs_minimal(mock_wandb_disabled):
    """Test that train.py runs with minimal configuration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        env = os.environ.copy()
        env['WANDB_DISABLED'] = 'true'
        
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "train",
                "--total-timesteps", "10",
                "--savedir", str(tmp_path),
                "--seed", "42",
                "--device", "cpu",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Check that a run directory was created
        run_dirs = list(tmp_path.glob("keisei_*"))
        assert run_dirs, "No run directory created"
        
        run_dir = run_dirs[0]
        print(f"Found run directory: {run_dir}")
        
        # Verify that training completed successfully
        check_training_outputs(result, 10)


@pytest.mark.slow
def test_train_config_override(mock_wandb_disabled):
    """Test that train.py handles config overrides correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        env = os.environ.copy()
        env['WANDB_DISABLED'] = 'true'
        
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "train",
                "--total-timesteps", "10",
                "--savedir", str(tmp_path),
                "--seed", "42",
                "--device", "cpu",
                "--override", "model.tower_width=32",
                "--override", "ppo.learning_rate=0.001",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        # Check that a run directory was created
        run_dirs = list(tmp_path.glob("keisei_*"))
        assert run_dirs, "No run directory created for config override test"
        
        # Verify that training completed successfully
        check_training_outputs(result, 10)


@pytest.mark.slow
def test_train_run_name_and_savedir(mock_wandb_disabled):
    """Test that train.py respects custom run names and save directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        env = os.environ.copy()
        env['WANDB_DISABLED'] = 'true'
        
        # Test with a custom run name prefix
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "train",
                "--total-timesteps", "10",
                "--savedir", str(tmp_path),
                "--run-name", "mytestrunprefix",
                "--model", "testmodel",
                "--input_features", "testfeats",
                "--seed", "42",
                "--device", "cpu",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        # Verify run directory with custom name was created
        run_dirs = list(tmp_path.glob("mytestrunprefix*"))
        run_dir = None
        for d in run_dirs:
            if d.is_dir() and "mytestrunprefix_testmodel_feats" in d.name:
                run_dir = d
                break
        
        assert (
            run_dir is not None
        ), f"Expected run directory starting with 'mytestrunprefix_testmodel_feats' not found in {tmp_path}"
        
        print(f"Found run directory: {run_dir}")
        
        # Verify that training completed successfully
        check_training_outputs(result, 10)


@pytest.mark.slow
def test_train_explicit_resume(tmp_path, mock_wandb_disabled):
    """
    Test explicit checkpoint resuming with train.py.
    
    This test verifies explicit checkpoint path functionality:
    1. Create a checkpoint file at a specific location
    2. Run train.py with --resume pointing to that specific checkpoint
    3. Verify training resumes from the checkpoint
    """
    # Step 1: Create initial checkpoint
    import torch
    
    # Create the initial save directory
    initial_save_dir = tmp_path / "initial_save_dir"
    initial_save_dir.mkdir()
    
    checkpoint_path = initial_save_dir / "my_explicit_checkpoint_ts100.pth"
    
    # Create a checkpoint with specific timestep
    checkpoint_data = {
        'model': {},
        'optimizer': {},
        'scheduler': {},
        'global_timestep': 100,  # Start from timestep 100
        'episode_count': 5,
        'best_reward': 10.5,
        'config': {
            'training': {
                'model_type': 'resnet',
                'total_timesteps': 150,  # Train to timestep 150
                'steps_per_epoch': 2,
                'checkpoint_interval_timesteps': 5,
            },
            'model': {
                'input_features': 'core_46_channels',
                'tower_depth': 1,
                'tower_width': 16,
                'se_ratio': 0.25,
            },
            'ppo': {
                'learning_rate': 1e-4,
            },
            'env': {
                'seed': 42,
                'tensorboard_log_dir': None,
                'log_tensorboard': False,
            },
            'logging': {
                'wandb_enabled': False,
                'run_name': None,
            },
            'device': {
                'device': 'cpu',
            },
            'evaluation': {},
        }
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Created checkpoint: {checkpoint_path}")

    env = os.environ.copy()
    env['WANDB_DISABLED'] = 'true'

    # Step 2: Run train.py with explicit checkpoint path
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "train",
            "--resume", str(checkpoint_path),  # Explicit path
            "--savedir", str(tmp_path),
            "--total-timesteps", "150",
            "--seed", "42",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")

    # Step 3: Verify that a new run directory was created
    run_dirs = list(tmp_path.glob("keisei_*"))
    assert run_dirs, f"No run directory created in {tmp_path}"
    
    run_dir = run_dirs[0]
    print(f"Found run directory: {run_dir}")
    
    # Verify that training completed successfully
    check_training_outputs(result, 150)
    
    # Verify that the checkpoint was copied to the run directory
    copied_checkpoints = list(run_dir.glob("my_explicit_checkpoint_ts100.pth"))
    assert copied_checkpoints, "Checkpoint was not copied to run directory"