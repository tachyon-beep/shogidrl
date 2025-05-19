"""
train2.py: Next-generation training interface for DRL Shogi Client

Features:
- Command-line interface (argparse)
- Checkpoint resume and auto-detection
- Config override from CLI
- Progress bar (tqdm)
- Advanced logging (human/JSON)
- Modular structure for future extensibility

This is a scaffold for Phase 1 of the training_update.md plan.
"""
import os
import sys
import argparse
import json
from datetime import datetime
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

import config
from keisei.ppo_agent import PPOAgent
from keisei.shogi.shogi_engine import ShogiGame
from keisei.utils import PolicyOutputMapper, TrainingLogger
from keisei.experience_buffer import ExperienceBuffer

# --- CLI ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(description="DRL Shogi Client Training (train2.py)")
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu/cuda)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default=None, help='Path to config override JSON')
    parser.add_argument('--logdir', type=str, default=None, help='Directory for logs and outputs')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--total-timesteps', type=int, default=None, help='Override total timesteps')
    parser.add_argument('--save-freq', type=int, default=None, help='Override save frequency (episodes)')
    parser.add_argument('--eval-freq', type=int, default=None, help='Override eval frequency (episodes)')
    parser.add_argument('--run-name', type=str, default=None, help='Custom run name')
    return parser.parse_args()

# --- CONFIG OVERRIDE ---
def apply_config_overrides(args):
    if args.total_timesteps is not None:
        config.TOTAL_TIMESTEPS = args.total_timesteps
    if args.save_freq is not None:
        config.SAVE_FREQ_EPISODES = args.save_freq
    if args.eval_freq is not None:
        config.EVAL_FREQ_EPISODES = args.eval_freq
    if args.device is not None:
        config.DEVICE = args.device
    if args.logdir is not None:
        config.MODEL_DIR = os.path.join(args.logdir, 'models')
        config.LOG_FILE = os.path.join(args.logdir, 'training_log.txt')
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    if args.config is not None:
        with open(args.config) as f:
            override = json.load(f)
        for k, v in override.items():
            if hasattr(config, k):
                setattr(config, k, v)

# --- CHECKPOINT AUTO-DETECTION ---
def find_latest_checkpoint(model_dir):
    if not os.path.exists(model_dir):
        return None
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, checkpoints[0])

# --- MAIN TRAINING LOOP (SCAFFOLD) ---
def main():
    args = parse_args()
    apply_config_overrides(args)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    logger = TrainingLogger(config.LOG_FILE)
    logger.log(f"Starting train2.py at {datetime.now().isoformat()}")
    logger.log(f"Config: {vars(config)}")
    # --- Agent and Env ---
    game = ShogiGame(max_moves_per_game=config.MAX_MOVES_PER_GAME)
    policy_mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=config.INPUT_CHANNELS, policy_output_mapper=policy_mapper,
        learning_rate=config.LEARNING_RATE, gamma=config.GAMMA, clip_epsilon=config.CLIP_EPSILON,
        ppo_epochs=config.PPO_EPOCHS, minibatch_size=config.MINIBATCH_SIZE,
        value_loss_coeff=config.VALUE_LOSS_COEFF, entropy_coef=config.ENTROPY_COEFF, device=config.DEVICE,
    )
    # --- Resume from checkpoint ---
    checkpoint_path = args.resume or find_latest_checkpoint(config.MODEL_DIR)
    if checkpoint_path:
        logger.log(f"Resuming from checkpoint: {checkpoint_path}")
        agent.load_model(checkpoint_path)
    else:
        logger.log("No checkpoint found. Starting fresh training.")
    buffer = ExperienceBuffer(
        buffer_size=config.STEPS_PER_EPOCH, gamma=config.GAMMA,
        lambda_gae=config.LAMBDA_GAE, device=config.DEVICE,
    )
    obs_np = game.reset()
    # --- Main loop (placeholder) ---
    pbar = tqdm(total=config.TOTAL_TIMESTEPS, desc="Training Progress")
    total_episodes_completed = 0
    for global_timestep in range(1, config.TOTAL_TIMESTEPS + 1):
        # ... Training logic goes here ...
        pbar.update(1)
        # Example: Save checkpoint
        if config.SAVE_FREQ_EPISODES > 0 and total_episodes_completed % config.SAVE_FREQ_EPISODES == 0:
            save_path = os.path.join(config.MODEL_DIR, f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth")
            agent.save_model(save_path)
        # ...
    pbar.close()
    logger.log("Training finished.")
    logger.close()

if __name__ == "__main__":
    main()
