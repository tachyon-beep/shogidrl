"""
training/utils.py: Helper functions for setup and configuration in the Shogi RL trainer.
"""

import glob
import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch

import wandb


def find_latest_checkpoint(model_dir_path):
    try:
        checkpoints = glob.glob(os.path.join(model_dir_path, "*.pth"))
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(model_dir_path, "*.pt"))
        if not checkpoints:
            return None
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    except (OSError, FileNotFoundError) as e:
        print(f"Error in find_latest_checkpoint: {e}", file=sys.stderr)
        return None


def serialize_config(config_obj: Any) -> str:
    if hasattr(config_obj, "dict"):
        conf_dict = config_obj.dict()
    else:
        conf_dict = {}
        if hasattr(config_obj, "__dict__"):
            source_dict = config_obj.__dict__
        elif isinstance(config_obj, dict):
            source_dict = config_obj
        else:
            source_dict = {
                key: getattr(config_obj, key)
                for key in dir(config_obj)
                if not key.startswith("__") and not callable(getattr(config_obj, key))
            }
        for k, v in source_dict.items():
            if isinstance(v, (int, float, str, bool, list, dict, tuple)) or v is None:
                conf_dict[k] = v
            elif hasattr(v, "dict"):
                conf_dict[k] = v.dict()
            elif hasattr(v, "__dict__"):
                conf_dict[k] = json.loads(serialize_config(v))
    try:
        return json.dumps(conf_dict, indent=4, sort_keys=True)
    except TypeError as e:
        print(f"Error serializing config: {e}", file=sys.stderr)
        return "{}"


def setup_directories(config, run_name):
    model_dir = config.logging.model_dir
    log_file = config.logging.log_file
    run_artifact_dir = os.path.join(model_dir, run_name)
    model_dir_path = run_artifact_dir
    log_file_path = os.path.join(run_artifact_dir, os.path.basename(log_file))
    eval_log_file_path = os.path.join(run_artifact_dir, "rich_periodic_eval_log.txt")
    os.makedirs(run_artifact_dir, exist_ok=True)
    return {
        "run_artifact_dir": run_artifact_dir,
        "model_dir": model_dir_path,
        "log_file_path": log_file_path,
        "eval_log_file_path": eval_log_file_path,
    }


def setup_seeding(config):
    seed = config.env.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def setup_wandb(config, run_name, run_artifact_dir):
    wandb_cfg = config.wandb
    is_active = wandb_cfg.enabled
    if is_active:
        try:
            config_dict_for_wandb = (
                json.loads(serialize_config(config)) if serialize_config(config) else {}
            )
            wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                name=run_name,
                config=config_dict_for_wandb,
                mode="online" if wandb_cfg.enabled else "disabled",
                dir=run_artifact_dir,
                resume="allow",
                id=run_name,
            )
        except (TypeError, ValueError, OSError) as e:
            print(
                f"Error initializing W&B: {e}. W&B logging disabled.", file=sys.stderr
            )
            is_active = False
    if not is_active:
        print(
            "Weights & Biases logging is disabled or failed to initialize.",
            file=sys.stderr,
        )
    return is_active
