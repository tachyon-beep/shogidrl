"""
Unit tests for train.py (smoke test for main loop).
"""
from datetime import datetime, timezone # For timestamping
import importlib
import os # Import os for environment variable patching
from unittest import mock # For patching environment variables

import wandb # Import wandb for patching
import config  # Import config directly to modify it

# Store original environment variables for W&B that we might patch
ORIGINAL_WANDB_PROJECT = os.getenv("WANDB_PROJECT")
ORIGINAL_WANDB_ENTITY = os.getenv("WANDB_ENTITY")
ORIGINAL_WANDB_RUN_ID = os.getenv("WANDB_RUN_ID") # If you use fixed IDs for resuming

def test_train_main_runs():
    """Smoke test: train.py main() runs without error with reduced timesteps and test W&B naming."""
    original_total_timesteps = config.TOTAL_TIMESTEPS
    original_steps_per_epoch = config.STEPS_PER_EPOCH
    original_eval_freq = config.EVAL_FREQ_EPISODES
    original_eval_episodes = config.EVAL_NUM_GAMES
    original_save_freq = config.SAVE_FREQ_EPISODES
    original_max_moves_eval = config.MAX_MOVES_PER_GAME_EVAL
    original_minibatch_size_from_config_file = config.MINIBATCH_SIZE
    #original_run_name_prefix = getattr(config, "RUN_NAME_PREFIX", "") # If you add this to config
    #original_wandb_notes = getattr(config, "WANDB_NOTES", None) # If you make notes configurable

    train_module_name = "train" # Assuming train.py is in the same directory or accessible via python path
    train = importlib.import_module(train_module_name)

    # --- Patch W&B related environment variables
    patched_env = {
        "WANDB_PROJECT": "shogi-drl-tests",  # Use a dedicated test project
        "WANDB_MODE": "disabled", # Completely disable W&B calls for smoke tests if preferred
                                 # OR "offline" to run offline without syncing
        "WANDB_RUN_ID": "smoke-test-fixed-id" # If you want a predictable ID for test runs
    }

    # Get potentially None environment variables
    env_wandb_entity = os.getenv("WANDB_ENTITY")

    # Add them to patched_env only if they are not None
    if env_wandb_entity is not None:
        patched_env["WANDB_ENTITY"] = env_wandb_entity

    try:
        # Apply patches
        with mock.patch.dict(os.environ, patched_env):
            # Reload config in case it also reads env vars (unlikely for these)
            importlib.reload(config)
            # --- Apply test-specific config overrides *after* reload ---
            config.TOTAL_TIMESTEPS = 128 # Minimal steps
            config.STEPS_PER_EPOCH = 32
            config.EVAL_FREQ_EPISODES = 1 # Eval once
            config.EVAL_NUM_GAMES = 1
            config.SAVE_FREQ_EPISODES = 1 # Save once (if episodes are hit)
            config.MAX_MOVES_PER_GAME_EVAL = 20
            config.MINIBATCH_SIZE = min(original_minibatch_size_from_config_file, config.STEPS_PER_EPOCH)
            # Optionally, add a prefix to config if train.py uses it for the run_name
            # config.RUN_NAME_PREFIX = "test_smoke_"
            # config.WANDB_NOTES = "This is an automated smoke test run."

            # Now reload train to pick up the patched config
            importlib.reload(train)

            # If you want to directly patch wandb.init to modify its arguments:
            original_wandb_init = wandb.init
            def mock_wandb_init(*args, **kwargs):
                # Modify run name if present
                if 'name' in kwargs:
                    kwargs['name'] = f"test_smoke_{kwargs['name']}"
                elif len(args) > 2 and isinstance(args[2], dict) and 'name' in args[2]: # config is often 3rd arg
                    args[2]['name'] = f"test_smoke_{args[2]['name']}"
                else: # Add a default test name if none was going to be set
                    current_utc_time = datetime.now(timezone.utc)
                    kwargs['name'] = f"test_smoke_unnamed_{current_utc_time.strftime('%y%m%d%H%M%S%Z')}"

                # Modify project/entity if not using WANDB_MODE=disabled and env vars weren't enough
                kwargs['project'] = patched_env.get("WANDB_PROJECT", kwargs.get('project'))
                kwargs['entity'] = patched_env.get("WANDB_ENTITY", kwargs.get('entity'))
                kwargs['notes'] = "Automated smoke test run."
                kwargs['tags'] = ["smoke-test"] + kwargs.get('tags', [])

                if os.environ.get("WANDB_MODE") == "disabled":
                    print("W&B init call suppressed by WANDB_MODE=disabled in test.")
                    return None # Simulate a disabled run or a mock object

                print(f"Mock W&B Init called with modified args: name='{kwargs.get('name')}', project='{kwargs.get('project')}'")
                return original_wandb_init(*args, **kwargs)

            # If WANDB_MODE is not 'disabled', apply the init patch
            if patched_env.get("WANDB_MODE") != "disabled":
                with mock.patch(f"{train_module_name}.wandb.init", side_effect=mock_wandb_init):
                    train.main()
                    # You can add assertions here about mocked_init being called if needed
                    # e.g., mocked_init.assert_called()
            else: # WANDB_MODE is "disabled"
                train.main() # wandb.init should effectively do nothing or be skipped inside train.py

    finally:
        # --- Restore original config values ---
        config.TOTAL_TIMESTEPS = original_total_timesteps
        config.STEPS_PER_EPOCH = original_steps_per_epoch
        config.EVAL_FREQ_EPISODES = original_eval_freq
        config.EVAL_NUM_GAMES = original_eval_episodes
        config.SAVE_FREQ_EPISODES = original_save_freq
        config.MAX_MOVES_PER_GAME_EVAL = original_max_moves_eval
        config.MINIBATCH_SIZE = original_minibatch_size_from_config_file
        # if hasattr(config, "RUN_NAME_PREFIX"): # Restore if you added it
            # config.RUN_NAME_PREFIX = original_run_name_prefix
        # if hasattr(config, "WANDB_NOTES"):
            # config.WANDB_NOTES = original_wandb_notes

        # --- Restore original environment variables ---
        # Note: mock.patch.dict handles restoration automatically when exiting the 'with' block.
        # If you manually set os.environ items, you'd need to restore them here.
        # For example:
        # if ORIGINAL_WANDB_PROJECT is None: os.environ.pop("WANDB_PROJECT", None)
        # else: os.environ["WANDB_PROJECT"] = ORIGINAL_WANDB_PROJECT
        # ... and so on for ENTITY, RUN_ID ...
        # (This manual restoration is NOT needed if using mock.patch.dict correctly)

        # Reload train again to ensure it's in its original state for subsequent tests
        # and picks up original config / environment.
        # os.environ.pop("SMOKE_TEST_ACTIVE", None) # Clean up test-specific env var
        importlib.reload(config)
        importlib.reload(train)
