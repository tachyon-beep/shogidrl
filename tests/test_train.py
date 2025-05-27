"""
Unit tests for train.py CLI and checkpoint logic (smoke test).
"""

import json as pyjson
import os
import subprocess
import sys

import torch

from keisei.config_schema import (
    AppConfig,
    DemoConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper

# Local config constants for test compatibility with new config system
INPUT_CHANNELS = 46
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01
DEVICE = "cpu"

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
    config = AppConfig(
        env=EnvConfig(
            device=DEVICE,
            input_channels=INPUT_CHANNELS,
            num_actions_total=policy_mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            clip_epsilon=CLIP_EPSILON,
            value_loss_coeff=VALUE_LOSS_COEFF,
            entropy_coef=ENTROPY_COEFF,
        ),
        evaluation=EvaluationConfig(num_games=1, opponent_type="random", evaluation_interval_timesteps=1000), # Added evaluation_interval_timesteps
        logging=LoggingConfig(log_file="/tmp/test.log", model_dir="/tmp/"),
        wandb=WandBConfig(enabled=False, project="test", entity=None),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )
    agent = PPOAgent(config=config, device=torch.device(DEVICE))
    agent.save_model(str(fake_ckpt))
    try:
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
    except subprocess.CalledProcessError as e:
        print("STDERR from train.py:")
        print(e.stderr)
        raise
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
    # Check nested config structure
    assert eff["training"]["total_timesteps"] == 2
    assert abs(eff["training"]["learning_rate"] - 0.12345) < 1e-6


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
    config = AppConfig(
        env=EnvConfig(
            device=DEVICE,
            input_channels=INPUT_CHANNELS,
            num_actions_total=policy_mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            clip_epsilon=CLIP_EPSILON,
            value_loss_coeff=VALUE_LOSS_COEFF,
            entropy_coef=ENTROPY_COEFF,
        ),
        evaluation=EvaluationConfig(num_games=1, opponent_type="random", evaluation_interval_timesteps=1000), # Added evaluation_interval_timesteps
        logging=LoggingConfig(log_file="/tmp/test.log", model_dir="/tmp/"),
        wandb=WandBConfig(enabled=False, project="test", entity=None),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )
    agent = PPOAgent(config=config, device=torch.device(DEVICE))
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
