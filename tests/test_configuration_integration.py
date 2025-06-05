"""
Test configuration system integration and log file usage.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from keisei.config_schema import EvaluationConfig
from keisei.training.utils import setup_directories
from keisei.utils import load_config


def test_config_schema_matches_yaml():
    """Test that all fields in default_config.yaml are defined in the schema."""
    # Load raw YAML
    config_path = Path(__file__).parent.parent / "default_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    # Load via Pydantic schema
    config = load_config(str(config_path))

    # Test that key fields are accessible
    assert hasattr(config.evaluation, "log_file_path_eval")
    assert hasattr(config.logging, "log_file")
    assert hasattr(config.wandb, "log_model_artifact")

    # Test that values match YAML
    assert (
        config.evaluation.log_file_path_eval
        == yaml_data["evaluation"]["log_file_path_eval"]
    )
    assert config.logging.log_file == yaml_data["logging"]["log_file"]
    assert config.wandb.log_model_artifact == yaml_data["wandb"]["log_model_artifact"]


def test_training_log_file_configuration():
    """Test that training log file configuration is properly processed."""
    config = load_config()
    run_name = "test_run"

    # Test directory setup
    dirs = setup_directories(config, run_name)

    # Verify log file path is constructed correctly
    assert "log_file_path" in dirs
    expected_filename = os.path.basename(config.logging.log_file)
    assert dirs["log_file_path"].endswith(expected_filename)
    assert run_name in dirs["log_file_path"]


def test_evaluation_log_file_configuration():
    """Test that evaluation log file configuration is accessible for the evaluator."""
    config = load_config()

    # Test that evaluation config has the log file field
    assert hasattr(config.evaluation, "log_file_path_eval")
    assert config.evaluation.log_file_path_eval == "eval_log.txt"

    # Test getattr usage pattern from callbacks.py
    log_file_path_eval = getattr(config.evaluation, "log_file_path_eval", "")
    assert log_file_path_eval == "eval_log.txt"


def test_config_validation_with_missing_fields():
    """Test that config validation works properly with schema."""
    # Test that evaluation config validates field values properly
    with pytest.raises(Exception):  # Pydantic validation error
        EvaluationConfig(
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=-1,  # Invalid negative value should fail
            num_games=20,
            opponent_type="random",
            max_moves_per_game=500,
            log_file_path_eval="eval_log.txt",
            wandb_log_eval=False,
        )

    with pytest.raises(Exception):  # Pydantic validation error
        EvaluationConfig(
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=1000,
            num_games=0,  # Invalid zero value should fail
            opponent_type="random",
            max_moves_per_game=500,
            log_file_path_eval="eval_log.txt",
            wandb_log_eval=False,
        )

    with pytest.raises(Exception):  # Pydantic validation error
        EvaluationConfig(
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=1000,
            num_games=20,
            opponent_type="random",
            max_moves_per_game=-10,  # Invalid negative value should fail
            log_file_path_eval="eval_log.txt",
            wandb_log_eval=False,
        )

    # Test that minimal valid config works
    eval_config = EvaluationConfig(
        enable_periodic_evaluation=True,
        evaluation_interval_timesteps=1000,
        num_games=5,
        opponent_type="random",
        max_moves_per_game=100,
        log_file_path_eval="test_eval.log",
        wandb_log_eval=False,
    )
    assert eval_config.log_file_path_eval == "test_eval.log"


def test_config_with_custom_yaml():
    """Test configuration loading with custom YAML content."""
    custom_config = {
        "env": {
            "seed": 42,
            "device": "cpu",
            "input_channels": 46,
            "max_moves_per_game": 500,
        },
        "training": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "ppo_epochs": 4,
            "minibatch_size": 64,
            "value_loss_coeff": 0.5,
            "entropy_coef": 0.01,
            "steps_per_epoch": 1024,
            "total_timesteps": 10000,
            "checkpoint_interval_timesteps": 1000,
        },
        "evaluation": {
            "enable_periodic_evaluation": True,
            "evaluation_interval_timesteps": 5000,
            "num_games": 10,
            "opponent_type": "random",
            "max_moves_per_game": 200,
            "log_file_path_eval": "custom_eval.log",
            "wandb_log_eval": True,
        },
        "logging": {"model_dir": "custom_models", "log_file": "custom_training.log"},
        "wandb": {"enabled": False, "entity": None, "log_model_artifact": True},
        "parallel": {
            "enabled": False,
            "num_workers": 2,
            "batch_size": 16,
            "sync_interval": 50,
            "compression_enabled": False,
            "timeout_seconds": 5.0,
            "max_queue_size": 500,
            "worker_seed_offset": 100,
        },
        "display": {"display_moves": True, "turn_tick": 0.1},
    }

    # Write to temporary file and load
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom_config, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)

        # Test custom values are loaded
        assert config.evaluation.log_file_path_eval == "custom_eval.log"
        assert config.logging.log_file == "custom_training.log"
        assert config.wandb.log_model_artifact is True
        assert config.evaluation.wandb_log_eval is True
        assert config.display.display_moves is True

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_config_schema_matches_yaml()
    test_training_log_file_configuration()
    test_evaluation_log_file_configuration()
    test_config_validation_with_missing_fields()
    test_config_with_custom_yaml()
    print("All configuration tests passed!")
