"""
Unit tests for train2.py CLI and checkpoint logic (smoke test).
"""


import os
import sys
import subprocess

from keisei.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper
import config

TRAIN2_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train2.py'))


def test_train2_cli_help():
    """Test that train2.py --help runs and prints usage."""
    result = subprocess.run([sys.executable, TRAIN2_PATH, '--help'], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--device" in result.stdout


def test_train2_resume_autodetect(tmp_path):
    """Test that train2.py can auto-detect a checkpoint (mocked)."""
    # Create a fake model directory and a valid checkpoint using PPOAgent

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    fake_ckpt = model_dir / "ppo_shogi_ep1_ts1.pth"
    # Create a minimal valid PPOAgent and save its checkpoint
    policy_mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=config.INPUT_CHANNELS, policy_output_mapper=policy_mapper,
        learning_rate=config.LEARNING_RATE, gamma=config.GAMMA, clip_epsilon=config.CLIP_EPSILON,
        ppo_epochs=config.PPO_EPOCHS, minibatch_size=config.MINIBATCH_SIZE,
        value_loss_coeff=config.VALUE_LOSS_COEFF, entropy_coef=config.ENTROPY_COEFF, device=config.DEVICE,
    )
    agent.save_model(str(fake_ckpt))
    # Run train2.py with logdir set to tmp_path
    result = subprocess.run([
        sys.executable, TRAIN2_PATH,
        '--logdir', str(tmp_path),
        '--total-timesteps', '1',  # Only run for 1 step
    ], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "Resuming from checkpoint" in result.stdout or "Resuming from checkpoint" in result.stderr


def test_train2_runs_minimal(tmp_path):
    """Test that train2.py runs for 1 step and creates log/model files."""
    logdir = tmp_path / "run"
    result = subprocess.run([
        sys.executable, TRAIN2_PATH,
        '--logdir', str(logdir),
        '--total-timesteps', '1',
    ], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    # Check log file exists
    log_file = logdir / "training_log.txt"
    assert log_file.exists()
    # Check model dir exists
    model_dir = logdir / "models"
    assert model_dir.exists()
