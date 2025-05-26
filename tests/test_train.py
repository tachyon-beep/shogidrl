"""
Unit tests for train.py CLI and checkpoint logic (smoke test).
"""

import json as pyjson
import os
import subprocess
import sys

import config
from keisei.ppo_agent import PPOAgent  # Add back PPOAgent import
from keisei.utils import PolicyOutputMapper  # Add back PolicyOutputMapper import

TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train.py"))


def test_train_cli_help():
    """Test that train.py --help runs and prints usage."""
    result = subprocess.run(
        [sys.executable, TRAIN_PATH, "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--device" in result.stdout


def test_train_resume_autodetect(tmp_path):
    """Test that train.py can auto-detect a checkpoint (mocked)."""
    # Create a run directory and place checkpoint there
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake_ckpt = run_dir / "checkpoint_ts1.pth"  # Corrected filename pattern
    policy_mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=config.INPUT_CHANNELS,
        policy_output_mapper=policy_mapper,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        clip_epsilon=config.CLIP_EPSILON,
        ppo_epochs=config.PPO_EPOCHS,
        minibatch_size=config.MINIBATCH_SIZE,
        value_loss_coeff=config.VALUE_LOSS_COEFF,
        entropy_coef=config.ENTROPY_COEFF,
        device=config.DEVICE,
    )
    agent.save_model(str(fake_ckpt))
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "--savedir",
            str(tmp_path),
            "--run_name",
            "run",
            "--total-timesteps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    # The 'Resumed training from checkpoint' message is in stderr (Rich logs)
    assert "Resumed training from checkpoint" in result.stderr


def test_train_runs_minimal(tmp_path):
    """Test that train.py runs for 1 step and creates log/model files."""
    savedir = tmp_path / "run"
    # Set SAVE_FREQ_EPISODES=1 so a checkpoint is always saved
    config_override = {"CHECKPOINT_INTERVAL_TIMESTEPS": 1}
    config_path = tmp_path / "override.json"
    with open(config_path, "w", encoding="utf-8") as f:
        pyjson.dump(config_override, f)
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "--savedir",
            str(savedir),
            "--config",
            str(config_path),
            "--total-timesteps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    run_dirs = [d for d in savedir.iterdir() if d.is_dir()]
    assert run_dirs, "No run directory created"
    run_dir = run_dirs[0]
    log_file = run_dir / "training_log.txt"
    assert log_file.exists()
    ckpt_files = list(run_dir.glob("checkpoint_ts*.pth"))
    assert ckpt_files, "No checkpoint file found in run directory"


def test_train_config_override(tmp_path):
    """Test that --config JSON override works and is saved in effective_config.json."""
    config_override = {"TOTAL_TIMESTEPS": 2, "LEARNING_RATE": 0.12345}
    config_path = tmp_path / "override.json"
    with open(config_path, "w", encoding="utf-8") as f:
        pyjson.dump(config_override, f)
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "--savedir",
            str(tmp_path),
            "--run_name",
            "run",
            "--config",
            str(config_path),
            "--total-timesteps",
            "2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    eff_cfg = run_dir / "effective_config.json"
    assert eff_cfg.exists()
    with open(eff_cfg, encoding="utf-8") as f:
        eff = pyjson.load(f)
    assert eff["TOTAL_TIMESTEPS"] == 2
    assert abs(eff["LEARNING_RATE"] - 0.12345) < 1e-6


def test_train_run_name_and_savedir(tmp_path):
    """Test that --run_name and --savedir create the correct directory structure."""
    run_name = "mytestrun"
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "--savedir",
            str(tmp_path),
            "--run_name",
            run_name,
            "--total-timesteps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    run_dir = tmp_path / run_name
    assert run_dir.exists()
    assert (run_dir / "training_log.txt").exists()
    assert (run_dir / "effective_config.json").exists()


def test_train_explicit_resume(tmp_path):
    """Test that --resume overrides auto-detection and resumes from the specified checkpoint."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt_path = (
        run_dir / "ppo_shogi_agent_episode_10_ts_100.pth"
    )  # Corrected filename pattern
    # Create a minimal valid PPOAgent and save its checkpoint
    policy_mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=config.INPUT_CHANNELS,
        policy_output_mapper=policy_mapper,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        clip_epsilon=config.CLIP_EPSILON,
        ppo_epochs=config.PPO_EPOCHS,
        minibatch_size=config.MINIBATCH_SIZE,
        value_loss_coeff=config.VALUE_LOSS_COEFF,
        entropy_coef=config.ENTROPY_COEFF,
        device=config.DEVICE,
    )
    agent.save_model(str(ckpt_path), global_timestep=100, total_episodes_completed=10)
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "--savedir",
            str(tmp_path),
            "--run_name",
            "run",
            "--resume",
            str(ckpt_path),
            "--total-timesteps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    # The 'Resumed training from checkpoint' message is in stderr (Rich logs)
    assert "Resumed training from checkpoint" in result.stderr


# --- Tests for Periodic Evaluation ---

# Remove all tests that reference run_evaluation or monkey-patching in the root train.py
# All tests below this line are up to date and relevant to the new architecture.
