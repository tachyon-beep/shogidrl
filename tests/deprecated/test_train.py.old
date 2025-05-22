"""
Unit tests for train.py (smoke test for main loop).
"""
from datetime import datetime, timezone # For timestamping
import importlib
import os # Import os for environment variable patching
from unittest import mock # For patching environment variables
from wandb import init as original_wandb_init

import config  # Import config directly to modify it

# Store original environment variables for W&B that we might patch
ORIGINAL_WANDB_PROJECT = os.getenv("WANDB_PROJECT")
ORIGINAL_WANDB_ENTITY = os.getenv("WANDB_ENTITY")
ORIGINAL_WANDB_RUN_ID = os.getenv("WANDB_RUN_ID") # If you use fixed IDs for resuming

def _patch_wandb_init(patched_env):
    """Return a mock for wandb.init that applies test-specific naming and tags."""
    def mock_wandb_init(*args, **kwargs):
        if 'name' in kwargs:
            kwargs['name'] = f"test_smoke_{kwargs['name']}"
        elif len(args) > 2 and isinstance(args[2], dict) and 'name' in args[2]:
            args[2]['name'] = f"test_smoke_{args[2]['name']}"
        else:
            current_utc_time = datetime.now(timezone.utc)
            kwargs['name'] = f"test_smoke_unnamed_{current_utc_time.strftime('%y%m%d%H%M%S%Z')}"
        kwargs['project'] = patched_env.get("WANDB_PROJECT", kwargs.get('project'))
        kwargs['entity'] = patched_env.get("WANDB_ENTITY", kwargs.get('entity'))
        kwargs['notes'] = "Automated smoke test run."
        kwargs['tags'] = ["smoke-test"] + kwargs.get('tags', [])
        if os.environ.get("WANDB_MODE") == "disabled":
            print("W&B init call suppressed by WANDB_MODE=disabled in test.")
            return None
        print(f"Mock W&B Init called with modified args: name='{kwargs.get('name')}', project='{kwargs.get('project')}'")
        return original_wandb_init(*args, **kwargs)
    return mock_wandb_init

def _apply_test_config_overrides(config, original_minibatch_size):
    config.TOTAL_TIMESTEPS = 128
    config.STEPS_PER_EPOCH = 32
    config.EVAL_FREQ_EPISODES = 1
    config.EVAL_NUM_GAMES = 1
    config.SAVE_FREQ_EPISODES = 1
    config.MAX_MOVES_PER_GAME_EVAL = 20
    config.MINIBATCH_SIZE = min(original_minibatch_size, config.STEPS_PER_EPOCH)

def _restore_config(config, orig):
    config.TOTAL_TIMESTEPS = orig['TOTAL_TIMESTEPS']
    config.STEPS_PER_EPOCH = orig['STEPS_PER_EPOCH']
    config.EVAL_FREQ_EPISODES = orig['EVAL_FREQ_EPISODES']
    config.EVAL_NUM_GAMES = orig['EVAL_NUM_GAMES']
    config.SAVE_FREQ_EPISODES = orig['SAVE_FREQ_EPISODES']
    config.MAX_MOVES_PER_GAME_EVAL = orig['MAX_MOVES_PER_GAME_EVAL']
    config.MINIBATCH_SIZE = orig['MINIBATCH_SIZE']

def test_train_main_runs():
    """Smoke test: train.py main() runs without error with reduced timesteps and test W&B naming."""
    orig = {
        'TOTAL_TIMESTEPS': config.TOTAL_TIMESTEPS,
        'STEPS_PER_EPOCH': config.STEPS_PER_EPOCH,
        'EVAL_FREQ_EPISODES': config.EVAL_FREQ_EPISODES,
        'EVAL_NUM_GAMES': config.EVAL_NUM_GAMES,
        'SAVE_FREQ_EPISODES': config.SAVE_FREQ_EPISODES,
        'MAX_MOVES_PER_GAME_EVAL': config.MAX_MOVES_PER_GAME_EVAL,
        'MINIBATCH_SIZE': config.MINIBATCH_SIZE,
    }
    train_module_name = "train"
    train = importlib.import_module(train_module_name)
    patched_env = {
        "WANDB_PROJECT": "shogi-drl-tests",
        "WANDB_MODE": "disabled",
        "WANDB_RUN_ID": "smoke-test-fixed-id"
    }
    env_wandb_entity = os.getenv("WANDB_ENTITY")
    if env_wandb_entity is not None:
        patched_env["WANDB_ENTITY"] = env_wandb_entity
    try:
        with mock.patch.dict(os.environ, patched_env):
            importlib.reload(config)
            _apply_test_config_overrides(config, orig['MINIBATCH_SIZE'])
            importlib.reload(train)
            if patched_env.get("WANDB_MODE") != "disabled":
                mock_wandb_init = _patch_wandb_init(patched_env)
                with mock.patch(f"{train_module_name}.wandb.init", side_effect=mock_wandb_init):
                    train.main()
            else:
                train.main()
    finally:
        _restore_config(config, orig)
        importlib.reload(config)
        importlib.reload(train)
