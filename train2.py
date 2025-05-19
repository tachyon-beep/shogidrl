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
import argparse
import json
import os
import torch
import config
import numpy as np
from tqdm import tqdm
from datetime import datetime

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
    parser.add_argument('--run_name', type=str, default=None, help='Name for this training run (used for run subdirectory)')
    parser.add_argument('--savedir', type=str, default='logs', help='Base directory for logs and checkpoints (was --logdir)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--total-timesteps', type=int, default=None, help='Override total timesteps for training (for testing)')
    parser.add_argument('--dev', action='store_true', help='Run in short dev/test mode (quick training, minimal steps)')
    # Add more CLI args as needed for overrides
    return parser.parse_args()

# --- CONFIG OVERRIDE ---
def apply_config_overrides(args):
    # Load config from JSON if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_json = json.load(f)
        for k, v in config_json.items():
            if hasattr(config, k):
                setattr(config, k, v)
    # CLI overrides (device, etc.)
    if args.device:
        config.DEVICE = args.device
    if args.total_timesteps is not None:
        config.TOTAL_TIMESTEPS = args.total_timesteps
    # Add more CLI overrides as needed
    # --- PATCH: force short training for dev/test mode ---
    if hasattr(args, 'dev') and getattr(args, 'dev', False):
        config.TOTAL_TIMESTEPS = 10  # or a small number for quick runs
        config.SAVE_FREQ_EPISODES = 1
    return config

# --- CHECKPOINT AUTO-DETECTION ---
def find_latest_checkpoint(model_dir):
    if not os.path.isdir(model_dir):
        return None
    ckpts = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not ckpts:
        return None
    # Sort by global timestep or episode if present in filename
    def extract_ts(f):
        # Example: ppo_shogi_agent_episode_100_ts_41355.pth
        try:
            parts = f.split('_')
            ep = int(parts[4])
            ts = int(parts[6].split('.')[0])
            return (ep, ts)
        except Exception:
            print(f"Warning: Ignoring file in checkpoint dir that doesn't match expected pattern: {f}")
            return (0, 0)
    ckpts.sort(key=extract_ts, reverse=True)
    return os.path.join(model_dir, ckpts[0])

# --- CONFIG SERIALIZATION FOR LOGGING ---
def serialize_config(cfg):
    # Only log uppercase attributes (convention for config constants)
    d = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper() and not k.startswith('__')}
    return json.dumps(d, indent=2)

# --- MAIN TRAINING LOOP (SCAFFOLD) ---
def main():
    args = parse_args()
    cfg = apply_config_overrides(args)

    # Set up run directory (all outputs in one folder per run)
    run_name = args.run_name or f"shogi_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.savedir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    model_dir = run_dir  # Store checkpoints in the same run directory
    log_file = os.path.join(run_dir, 'training_log.txt')

    # Save effective config to file in run_dir
    with open(os.path.join(run_dir, 'effective_config.json'), 'w', encoding='utf-8') as f:
        f.write(serialize_config(cfg))

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Initialize logger
    logger = TrainingLogger(log_file)
    logger.log(f"Starting run: {run_name}")
    logger.log(f"Config: {serialize_config(cfg)}")

    # Initialize environment and agent
    game = ShogiGame(max_moves_per_game=cfg.MAX_MOVES_PER_GAME)
    policy_mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=cfg.INPUT_CHANNELS,
        policy_output_mapper=policy_mapper,
        learning_rate=cfg.LEARNING_RATE,
        gamma=cfg.GAMMA,
        clip_epsilon=cfg.CLIP_EPSILON,
        ppo_epochs=cfg.PPO_EPOCHS,
        minibatch_size=cfg.MINIBATCH_SIZE,
        value_loss_coeff=cfg.VALUE_LOSS_COEFF,
        entropy_coef=cfg.ENTROPY_COEFF,
        device=cfg.DEVICE,
    )
    experience_buffer = ExperienceBuffer(
        buffer_size=cfg.STEPS_PER_EPOCH,
        gamma=cfg.GAMMA,
        lambda_gae=cfg.LAMBDA_GAE,
        device=cfg.DEVICE,
    )

    # --- Checkpoint resume logic ---
    global_timestep = 0
    total_episodes_completed = 0
    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = find_latest_checkpoint(model_dir)
    if ckpt_path:
        logger.log(f"Resuming from checkpoint: {ckpt_path}")
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = agent.load_model(ckpt_path)
        if ckpt:
            global_timestep = ckpt.get('global_timestep', 0)
            total_episodes_completed = ckpt.get('total_episodes_completed', 0)
            if 'optimizer_state_dict' in ckpt:
                agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else:
        logger.log("No checkpoint found. Starting fresh training.")
        print("No checkpoint found. Starting fresh training.")

    # --- Main training loop (full logic) ---
    pbar = tqdm(total=cfg.TOTAL_TIMESTEPS, initial=global_timestep, desc="Training Progress")
    obs = game.reset()
    done = False
    episode_reward = 0.0
    move_count = 0
    while global_timestep < cfg.TOTAL_TIMESTEPS:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            done = True
            reward = 0.0
            action_idx = 0
            log_prob = 0.0
            value = 0.0
            action = None
            next_obs = game.reset()  # Ensure next_obs is always defined
        else:
            action, action_idx, log_prob, value = agent.select_action(obs, legal_moves)
            # Apply action to game using make_move
            next_obs, reward, done, info = game.make_move(action)
        # Only add experience if action is valid (skip if no legal moves and action is None)
        if action is not None or legal_moves:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            experience_buffer.add(obs_tensor, action_idx, reward, log_prob, value, done)
        obs = next_obs
        episode_reward += reward
        global_timestep += 1
        move_count += 1
        pbar.update(1)
        if done or (move_count >= cfg.MAX_MOVES_PER_GAME):
            total_episodes_completed += 1
            logger.log(f"Episode {total_episodes_completed} finished. Reward: {episode_reward}")
            obs = game.reset()
            episode_reward = 0.0
            move_count = 0
            done = False
            # Save checkpoint
            # Only save checkpoint if SAVE_FREQ_EPISODES > 0 and not at episode 0
            if cfg.SAVE_FREQ_EPISODES > 0 and total_episodes_completed > 0 and total_episodes_completed % cfg.SAVE_FREQ_EPISODES == 0:
                ckpt_path = os.path.join(model_dir, f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth")
                agent.save_model(ckpt_path, global_timestep=global_timestep, total_episodes_completed=total_episodes_completed)
            # Optionally: evaluation, logging, etc.
        # PPO update step
        if len(experience_buffer) >= cfg.STEPS_PER_EPOCH:
            agent.learn(experience_buffer)
            experience_buffer.clear()
    logger.close()

    # --- Ensure checkpoint is saved at end of training ---
    ckpt_path = os.path.join(model_dir, f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth")
    if not os.path.exists(ckpt_path):
        agent.save_model(ckpt_path, global_timestep=global_timestep, total_episodes_completed=total_episodes_completed)

if __name__ == "__main__":
    main()
