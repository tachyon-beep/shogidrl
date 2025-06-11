"""
Unit tests for train.py CLI and checkpoint logic (smoke test).
"""

import json as pyjson
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from keisei.config_schema import AppConfig, ParallelConfig
from keisei.core.neural_network import ActorCritic  # Added for dependency injection
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper


def _create_test_model(config):
    """Helper function to create ActorCritic model for PPOAgent testing."""
    mapper = PolicyOutputMapper()
    return ActorCritic(config.env.input_channels, mapper.get_total_actions())


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

ROOT_DIR = Path(__file__).parent.parent.parent  # Go up three levels for project root
TRAIN_PATH = str(ROOT_DIR / "train.py")  # Correct path to train.py


@pytest.mark.slow
def test_train_cli_help(mock_wandb_disabled):
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


@pytest.mark.slow
def test_train_resume_autodetect(tmp_path, mock_wandb_disabled):
    """
    Test that train.py can auto-detect a checkpoint from parent directory and resume training.

    This test verifies the parent directory checkpoint search functionality:
    1. Save a checkpoint directly in savedir (tmp_path)
    2. Run train.py with --resume latest and --savedir tmp_path
    3. Verify ModelManager finds the checkpoint in parent dir, copies it to run dir, and resumes
    """
    policy_mapper = PolicyOutputMapper()  # Used by PPOAgent and initial AppConfig

    # Config for the initial agent save, and for run name generation
    base_config_data = {
        "env": {
            "device": DEVICE,
            "input_channels": INPUT_CHANNELS,
            "num_actions_total": policy_mapper.get_total_actions(),
            "seed": 42,
        },
        "training": {
            "total_timesteps": 1000,
            "steps_per_epoch": 32,
            "ppo_epochs": 1,
            "minibatch_size": 2,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "clip_epsilon": CLIP_EPSILON,
            "value_loss_coeff": VALUE_LOSS_COEFF,
            "entropy_coef": ENTROPY_COEFF,
            "model_type": "dummy",
            "input_features": "dummyfeats",
            "checkpoint_interval_timesteps": 1000,  # Added default
            "evaluation_interval_timesteps": 1000,  # Added default
            "mixed_precision": False,
            "ddp": False,
            "gradient_clip_max_norm": 0.5,
            "lambda_gae": 0.95,  # Added defaults
            "render_every_steps": 1,
            "refresh_per_second": 4,
            "enable_spinner": True,  # Added defaults
        },
        "evaluation": {
            "num_games": 1,
            "opponent_type": "random",
            "evaluation_interval_timesteps": 1000,
        },
        "logging": {
            "log_file": "training.log",
            "model_dir": str(tmp_path),
        },  # model_dir is tmp_path for the run
        "wandb": {
            "enabled": False,
            "project": "test",
            "entity": None,
            "run_name_prefix": "autodetect",
            "watch_model": False,
            "watch_log_freq": 1000,
            "watch_log_type": "all",
        },  # Added defaults
        "demo": {"enable_demo_mode": False, "demo_mode_delay": 0.0},
        "parallel": {
            "enabled": False,
            "start_method": "fork",
            "num_envs": 1,
            "base_port": 50000,
        },
    }
    initial_agent_config = AppConfig.model_validate(base_config_data)

    # Create model for dependency injection
    model = _create_test_model(initial_agent_config)
    agent = PPOAgent(
        model=model, config=initial_agent_config, device=torch.device(DEVICE)
    )

    # Save the initial checkpoint in tmp_path (the savedir/parent directory)
    # This simulates a previous training run that saved a checkpoint in the savedir
    final_ckpt_path = tmp_path / "checkpoint_ts1.pth"
    agent.save_model(
        str(final_ckpt_path), global_timestep=1, total_episodes_completed=0
    )

    # Create the config file that the subprocess will use
    subprocess_config_path = tmp_path / "subprocess_config_autodetect.yaml"
    with open(subprocess_config_path, "w", encoding="utf-8") as f:
        pyjson.dump(base_config_data, f)  # Use the same base_config_data

    try:
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "--savedir",
                str(tmp_path),  # This is the parent for the run_dir
                "--config",
                str(
                    subprocess_config_path
                ),  # This config defines the run name and other params
                "--resume",
                "latest",  # Test the 'latest' feature
                "--total-timesteps",
                "2",  # Run a bit more
            ],
            capture_output=True,
            text=True,
            check=True,
            # W&B is mocked by the fixture, no need for environment variable
        )
    except subprocess.CalledProcessError as e:
        print("STDERR from train.py (test_train_resume_autodetect with 'latest'):")
        print(e.stderr)
        raise
    assert result.returncode == 0

    # Find the run directory (should be the only new directory under tmp_path)
    run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert run_dirs, f"No run directory created in {tmp_path}"
    run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)  # Most recently modified

    log_file = run_dir / "training.log"
    assert log_file.exists(), f"Log file {log_file} does not exist"
    with open(log_file, encoding="utf-8") as f:
        log_contents = f.read()

    # Verify that the checkpoint was copied to the run directory and we resumed from it
    copied_checkpoint = run_dir / "checkpoint_ts1.pth"
    assert (
        copied_checkpoint.exists()
    ), f"Expected checkpoint {copied_checkpoint} was not copied to run directory"

    # Check that the log file contains a resume message specifically for the copied checkpoint
    expected_resume_message = (
        f"Resumed training from checkpoint: {str(copied_checkpoint)}"
    )
    assert expected_resume_message in log_contents, (
        f"Expected resume message '{expected_resume_message}' not found in log. "
        f"Log contents:\n{log_contents}"
    )

    # Verify that the original checkpoint timestep was preserved (we saved at ts=1, should resume from ts=1)
    # And that training progressed beyond that point (we set total_timesteps=2, so should reach ts=2)
    final_checkpoint = run_dir / "checkpoint_ts2.pth"
    assert final_checkpoint.exists(), (
        f"Expected final checkpoint {final_checkpoint} was not created, "
        f"indicating training did not progress from resumed timestep"
    )


@pytest.mark.slow
def test_train_runs_minimal(tmp_path, mock_wandb_disabled):
    """Test that train.py runs for 1 step and creates log/model files."""
    savedir = tmp_path
    config_override = {
        "training.checkpoint_interval_timesteps": 1,
        "logging.model_dir": str(savedir),
    }
    config_path = tmp_path / "override.json"
    with open(config_path, "w", encoding="utf-8") as f:
        pyjson.dump(config_override, f)
    try:
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
    except subprocess.CalledProcessError as e:
        print("STDERR from train.py (test_train_runs_minimal):")
        print(e.stderr)
        raise

    assert result.returncode == 0
    run_dirs = [d for d in savedir.iterdir() if d.is_dir() and "keisei" in d.name]
    assert run_dirs, "No run directory created"
    run_dir = run_dirs[0]
    log_file = run_dir / "training_log.txt"
    assert log_file.exists()
    ckpt_files = list(run_dir.glob("checkpoint_ts*.pth"))
    assert ckpt_files, "No checkpoint file found in run directory"


@pytest.mark.slow
def test_train_config_override(tmp_path, mock_wandb_disabled):
    """Test that --config JSON override works and is saved in effective_config.json."""
    config_override = {
        "training.total_timesteps": 2,
        "training.learning_rate": 0.12345,
        "logging.model_dir": str(tmp_path),
    }
    config_path = tmp_path / "override.json"
    with open(config_path, "w", encoding="utf-8") as f:
        pyjson.dump(config_override, f)
    try:
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "--savedir",
                str(tmp_path),
                "--config",
                str(config_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("STDERR from train.py (test_train_config_override):")
        print(e.stderr)
        raise
    assert result.returncode == 0
    run_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and "keisei" in d.name]
    assert run_dirs, "No run directory created for config override test"
    run_dir = run_dirs[0]
    eff_cfg = run_dir / "effective_config.json"
    assert eff_cfg.exists()
    with open(eff_cfg, encoding="utf-8") as f:
        eff = pyjson.load(f)
    assert eff["training"]["total_timesteps"] == 2
    assert abs(eff["training"]["learning_rate"] - 0.12345) < 1e-6


@pytest.mark.slow
def test_train_run_name_and_savedir(tmp_path, mock_wandb_disabled):
    """Test that --savedir influences the base path of the generated run directory."""
    dummy_config_content = {
        "wandb": {
            "enabled": False,
            "run_name_prefix": "mytestrunprefix",
        },  # Explicitly disable W&B
        "training": {"model_type": "testmodel", "input_features": "testfeats"},
        "logging": {
            "model_dir": str(tmp_path)
        },  # This should be the parent for the run
    }
    dummy_config_path = tmp_path / "dummy_config_for_run_name.yaml"
    with open(dummy_config_path, "w", encoding="utf-8") as f:
        pyjson.dump(dummy_config_content, f)

    try:
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "--savedir",
                str(
                    tmp_path
                ),  # This is effectively the same as logging.model_dir in this case
                "--config",
                str(dummy_config_path),
                "--total-timesteps",
                "1",
            ],
            capture_output=True,
            text=True,
            check=True,
            # W&B is mocked by the fixture, no need for environment variable
        )
    except subprocess.CalledProcessError as e:
        print("STDERR from train.py (test_train_run_name_and_savedir):")
        print(e.stderr)
        raise

    assert result.returncode == 0
    found_run_dir = None
    for item in tmp_path.iterdir():
        # The run_name format is: prefix_modeltype_feats_inputfeatures_timestamp
        # So look for 'mytestrunprefix_testmodel_feats' in the name
        if item.is_dir() and item.name.startswith("mytestrunprefix_testmodel_feats"):
            found_run_dir = item
            break
    assert (
        found_run_dir is not None
    ), f"Expected run directory starting with 'mytestrunprefix_testmodel_feats' not found in {tmp_path}"
    assert (found_run_dir / "training_log.txt").exists()
    assert (found_run_dir / "effective_config.json").exists()


@pytest.mark.slow
def test_train_explicit_resume(tmp_path, mock_wandb_disabled):
    """Test that --resume overrides auto-detection and resumes from the specified checkpoint."""
    run_name_prefix = "explicitresume"
    model_type = "resumemodel"
    input_features = "resumefeats"
    policy_mapper = PolicyOutputMapper()

    # Config for the initial agent save
    initial_save_config_dict = {
        "env": {
            "device": DEVICE,
            "input_channels": INPUT_CHANNELS,
            "num_actions_total": policy_mapper.get_total_actions(),
            "seed": 42,
        },
        "training": {
            "total_timesteps": 1000,
            "steps_per_epoch": 32,
            "ppo_epochs": 1,
            "minibatch_size": 2,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "clip_epsilon": CLIP_EPSILON,
            "value_loss_coeff": VALUE_LOSS_COEFF,
            "entropy_coef": ENTROPY_COEFF,
            "model_type": model_type,
            "input_features": input_features,
            "checkpoint_interval_timesteps": 1000,
            "evaluation_interval_timesteps": 1000,  # Added defaults
            "mixed_precision": False,
            "ddp": False,
            "gradient_clip_max_norm": 0.5,
            "lambda_gae": 0.95,  # Added defaults
            "render_every_steps": 1,
            "refresh_per_second": 4,
            "enable_spinner": True,  # Added defaults
        },
        "evaluation": {
            "num_games": 1,
            "opponent_type": "random",
            "evaluation_interval_timesteps": 1000,
        },
        "logging": {
            "log_file": "/tmp/initial_save.log",
            "model_dir": str(tmp_path / "initial_save_dir"),
        },
        "wandb": {
            "enabled": False,
            "project": "test",
            "entity": None,
            "run_name_prefix": run_name_prefix,
            "watch_model": False,
            "watch_log_freq": 1000,
            "watch_log_type": "all",
        },  # Added defaults
        "demo": {"enable_demo_mode": False, "demo_mode_delay": 0.0},
        "parallel": {
            "enabled": False,
            "start_method": "fork",
            "num_envs": 1,
            "base_port": 50000,
        },
    }
    initial_save_config_obj = AppConfig.model_validate(initial_save_config_dict)

    checkpoint_save_dir = tmp_path / "initial_save_dir"  # As per logging.model_dir
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    # The checkpoint name itself doesn't need to match the generated run name for explicit resume
    ckpt_path = checkpoint_save_dir / "my_explicit_checkpoint_ts100.pth"

    # Create model for dependency injection
    model = _create_test_model(initial_save_config_obj)
    agent = PPOAgent(
        model=model, config=initial_save_config_obj, device=torch.device(DEVICE)
    )
    agent.save_model(str(ckpt_path), global_timestep=100, total_episodes_completed=10)

    # Config for the actual training run that will resume
    resume_run_config_dict: dict = (
        initial_save_config_dict.copy()
    )  # Start with a copy, explicitly type as dict
    # Mypy needs help understanding the nested structure when using .copy()
    # For logging, ensure it knows it's a dict
    if not isinstance(resume_run_config_dict.get("logging"), dict):
        resume_run_config_dict["logging"] = (
            {}
        )  # Initialize if not a dict (should not happen with .copy())
    resume_run_config_dict["logging"]["model_dir"] = str(
        tmp_path
    )  # New run will save to tmp_path as parent
    resume_run_config_dict["logging"]["log_file"] = "resumed_training.log"

    # For training, ensure it knows it's a dict
    if not isinstance(resume_run_config_dict.get("training"), dict):
        resume_run_config_dict["training"] = {}  # Initialize if not a dict
    # Ensure training params are suitable for resuming and running a bit more
    resume_run_config_dict["training"]["total_timesteps"] = 101

    resume_config_file_path = tmp_path / "resume_run_config.yaml"
    with open(resume_config_file_path, "w", encoding="utf-8") as f:
        pyjson.dump(resume_run_config_dict, f)

    try:
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "--savedir",
                str(tmp_path),  # Parent directory for the new run
                "--config",
                str(resume_config_file_path),  # Config for this run
                "--resume",
                str(ckpt_path),  # Explicit checkpoint to resume from
                # total_timesteps is now in the config file
            ],
            capture_output=True,
            text=True,
            check=True,
            # W&B is mocked by the fixture, no need for environment variable
        )
    except subprocess.CalledProcessError as e:
        print("STDERR from train.py (test_train_explicit_resume):")
        print(e.stderr)
        raise

    assert result.returncode == 0
    # Find the run directory (should be the only new directory under tmp_path not initial_save_dir)
    run_dirs = [
        d for d in tmp_path.iterdir() if d.is_dir() and d.name != "initial_save_dir"
    ]
    assert run_dirs, f"No run directory created in {tmp_path}"
    found_new_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    log_file = found_new_run_dir / "resumed_training.log"
    assert (
        found_new_run_dir.exists() and found_new_run_dir.is_dir()
    ), f"New run directory not found in {tmp_path}"
    assert log_file.exists(), f"Log file {log_file} does not exist"
    with open(log_file, encoding="utf-8") as f:
        log_contents = f.read()
    assert f"Resumed training from checkpoint: {str(ckpt_path)}" in log_contents

    # Verify that a new run directory was created under tmp_path
    # and that it contains a checkpoint reflecting the continued training.
    new_ckpt_found = False
    for ckpt_file_path_obj in found_new_run_dir.glob(
        "checkpoint_ts*.pth"
    ):  # Use a different var name
        try:
            ts_str = ckpt_file_path_obj.stem.split("_ts")[-1]
            if ts_str.isdigit() and int(ts_str) > 100:  # Should be 101 or more
                new_ckpt_found = True
                break
        except ValueError:
            continue
    assert (
        new_ckpt_found
    ), f"No new checkpoint found in {found_new_run_dir} with timestep > 100"


# --- Tests for Periodic Evaluation ---

# Remove all tests that reference run_evaluation or monkey-patching in the root train.py
# All tests below this line are up to date and relevant to the new architecture.
