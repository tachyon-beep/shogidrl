"""
train.py: Next-generation training interface for DRL Shogi Client

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
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import re  # Add import for regular expressions

import config as app_config  # Import the config module directly
from keisei.experience_buffer import ExperienceBuffer
from keisei.ppo_agent import PPOAgent
from keisei.shogi.shogi_engine import ShogiGame
from keisei.utils import PolicyOutputMapper, TrainingLogger


# --- CLI ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="DRL Shogi Client Training (train2.py)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config override JSON"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run (used for run subdirectory)",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="logs",
        help="Base directory for logs and checkpoints (was --logdir)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps for training (for testing)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in short dev/test mode (quick training, minimal steps)",
    )
    return parser.parse_args()


# --- CONFIG OVERRIDE ---
def apply_config_overrides(args, cfg_module):
    # Load config from JSON if provided
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_json = json.load(f)
        for k, v in config_json.items():
            if hasattr(cfg_module, k):
                setattr(cfg_module, k, v)
            else:
                # Use logger for warnings, assuming logger is available or will be passed
                # For now, this print will be removed or replaced by logger if logger is accessible here
                # Consider passing logger to this function or handling warnings differently.
                # For this step, we'll remove the direct sys.stderr usage.
                # print(
                # f\"Warning: Config key '{k}' from JSON not found in base config module.\",
                # file=sys.stderr,
                # )
                pass  # Placeholder: Warning should be logged if a logger is available

    # CLI overrides (device, etc.)
    if args.device:
        cfg_module.DEVICE = args.device
    if args.total_timesteps is not None:
        cfg_module.TOTAL_TIMESTEPS = args.total_timesteps
    if hasattr(args, "dev") and getattr(args, "dev", False):
        cfg_module.TOTAL_TIMESTEPS = 100
        cfg_module.SAVE_FREQ_EPISODES = 1
        cfg_module.STEPS_PER_EPOCH = 50
        cfg_module.MINIBATCH_SIZE = 16
        cfg_module.PPO_EPOCHS = 2
    return cfg_module


# --- CHECKPOINT AUTO-DETECTION ---
def find_latest_checkpoint(model_dir):
    if not os.path.isdir(model_dir):
        return None
    ckpts = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not ckpts:
        return None

    # Regex to capture episode and timestep from filenames like:
    # ppo_shogi_agent_episode_100_ts_41355.pth
    # ppo_shogi_ep100_ts41355.pth
    # ppo_shogi_ep_100_ts_41355_extra_info.pth
    # ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth (from save logic)
    # ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}_final.pth (from final save logic)
    checkpoint_pattern = re.compile(r"episode_(\d+).*ts_(\d+)|ep(\d+).*ts(\d+)")

    def extract_ts(filename):
        match = checkpoint_pattern.search(filename)
        if match:
            # The pattern has two groups for episode and two for timestep due to the OR | condition
            # e.g. (ep_v1, ts_v1, None, None) or (None, None, ep_v2, ts_v2)
            # We need to check which group matched.
            if match.group(1) is not None and match.group(2) is not None:
                try:
                    ep = int(match.group(1))
                    ts = int(match.group(2))
                    return (ep, ts)
                except ValueError:
                    print(
                        f"Warning: Checkpoint file {filename} matched pattern but had non-integer episode/timestep."
                    )
                    return (0, 0)  # Treat as lowest priority
            elif match.group(3) is not None and match.group(4) is not None:
                try:
                    ep = int(match.group(3))
                    ts = int(match.group(4))
                    return (ep, ts)
                except ValueError:
                    print(
                        f"Warning: Checkpoint file {filename} matched pattern but had non-integer episode/timestep."
                    )
                    return (0, 0)  # Treat as lowest priority
        else:
            print(
                f"Warning: Ignoring file in checkpoint dir that doesn't match expected pattern: {filename}"
            )
            return (0, 0)  # Treat as lowest priority if no match

    # Filter out files that don't match and sort the rest
    valid_ckpts = []
    for f in ckpts:
        ep_ts = extract_ts(f)
        if ep_ts != (0, 0):  # Only consider if successfully parsed
            valid_ckpts.append((f, ep_ts))

    if not valid_ckpts:
        return None

    # Sort by episode, then by timestep, descending
    valid_ckpts.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(model_dir, valid_ckpts[0][0])


# --- CONFIG SERIALIZATION FOR LOGGING ---
def serialize_config(cfg_module):
    # Only log uppercase attributes (convention for config constants)
    d = {
        k: getattr(cfg_module, k)
        for k in dir(cfg_module)
        if k.isupper() and not k.startswith("__")
    }
    return json.dumps(d, indent=2)


# --- MAIN TRAINING LOOP (SCAFFOLD) ---
def main():
    args = parse_args()
    cfg = app_config  # Use the module directly
    cfg = apply_config_overrides(args, cfg)  # Pass the module to be modified

    # Set up run directory (all outputs in one folder per run)
    run_name = args.run_name or f"shogi_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.savedir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    model_dir = run_dir  # Store checkpoints in the same run directory
    log_file = os.path.join(run_dir, "training_log.txt")

    # Save effective config to file in run_dir
    with open(
        os.path.join(run_dir, "effective_config.json"), "w", encoding="utf-8"
    ) as f:
        f.write(serialize_config(cfg))

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Initialize logger (now handled by with statement below)

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
        buffer_size=cfg.STEPS_PER_EPOCH,  # Use STEPS_PER_EPOCH for buffer_size
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

    # Use TrainingLogger with a context manager
    with TrainingLogger(log_file) as logger:
        logger.log(f"Starting run: {run_name}")
        logger.log(f"Effective Config: {serialize_config(cfg)}")  # Pass the module

        if ckpt_path and os.path.exists(ckpt_path):
            logger.log(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = agent.load_model(ckpt_path)
            if ckpt:
                global_timestep = ckpt.get("global_timestep", 0)
                total_episodes_completed = ckpt.get("total_episodes_completed", 0)
                logger.log(
                    f"Resumed at timestep {global_timestep}, episodes {total_episodes_completed}"
                )
                if agent.last_kl_div is not None:
                    logger.log(
                        f"Resumed with last KL divergence: {agent.last_kl_div:.4f}"
                    )
            else:
                logger.log(
                    f"Failed to load checkpoint data from {ckpt_path}. Starting fresh."
                )
        else:
            logger.log(
                "No checkpoint found or specified path invalid. Starting fresh training."
            )

        # --- Main training loop (full logic) ---
        pbar = tqdm(
            total=cfg.TOTAL_TIMESTEPS, initial=global_timestep, desc="Training Progress"
        )
        obs = game.reset()  # game.reset() now returns obs directly (np.ndarray)
        current_episode_reward = 0.0  # Initialize as float
        current_episode_length = 0
        # Initialize reward and next_obs for the first iteration of the loop
        reward = 0.0
        next_obs = obs  # Initialize next_obs with the initial observation
        # done = False # done is initialized before each episode, so this line is not strictly needed here

        for _ in range(global_timestep, cfg.TOTAL_TIMESTEPS):  # Changed t_step to _
            done = False  # Reset done at the beginning of each step
            pbar.update(1)
            global_timestep += 1
            current_episode_length += 1

            # Ensure obs is correctly shaped for the agent (e.g., add batch dim if needed by agent)
            # Removed commented-out: # obs_for_agent = np.expand_dims(obs, axis=0) if obs.ndim == 3 else obs
            # The PPOAgent.select_action expects obs without batch dim, it adds it internally.
            obs_for_buffer = obs  # Store the original observation state for the buffer

            legal_moves = game.get_legal_moves()
            action_idx = (
                -1
            )  # Ensure action_idx is initialized before the conditional block
            log_prob = 0.0  # Ensure log_prob is initialized
            value = 0.0  # Ensure value is initialized
            selected_shogi_move = None  # Ensure selected_shogi_move is initialized
            # Initialize legal_mask_for_buffer with a default, e.g., all False if no legal moves.
            # The actual legal_mask will be determined by game.get_legal_moves() and policy_mapper.
            # If there are no legal_moves, this default might be used if we were to add to buffer in that case.
            # However, we only add to buffer if action_idx != -1.
            # The legal_mask returned by agent.select_action will be the one used.
            legal_mask_for_buffer = torch.zeros(
                policy_mapper.get_total_actions(), dtype=torch.bool, device=agent.device
            )

            if not legal_moves:
                # This case should ideally be handled by game termination logic
                logger.log(
                    f"Episode {total_episodes_completed + 1}: No legal moves available at timestep {current_episode_length}. Game is considered done."
                )
                done = True  # Treat as done
                # No action is selected, agent.select_action is bypassed.
                # selected_shogi_move remains None, action_idx remains -1.
            else:  # Only call agent if there are legal moves and game is not already done
                # The 'done' flag might have been set by game logic in the previous step,
                # but if there are legal moves, the agent should still select an action
                # for the current state. The 'done' status from the *previous* step
                # will be used for the buffer.
                # However, the current logic already sets done = False at the start of each step.
                # So, if legal_moves exist, 'done' here reflects the current game state before this move.
                (
                    selected_shogi_move,
                    action_idx,
                    log_prob,
                    value,
                    legal_mask_for_buffer,
                ) = agent.select_action(obs, legal_moves, is_training=True)
                # If, after selecting an action, the agent somehow returns no move (e.g. a bug in agent or mask handling)
                if selected_shogi_move is None:
                    logger.log(
                        f"Warning: Agent selected action_idx {action_idx} but returned no Shogi move. Treating as if no legal moves."
                    )
                    done = True  # Treat as done, similar to no legal moves initially.
                    action_idx = -1  # Ensure this is set to prevent adding to buffer.

            if (
                selected_shogi_move is not None and action_idx != -1
            ):  # Check action_idx as well
                game.make_move(selected_shogi_move)  # Call make_move, returns None
                next_obs = game.get_observation()  # Get observation after move
                done = game.game_over  # Update done status

                # Determine reward - placeholder logic
                if game.game_over:
                    # Sparse rewards: Reward is only given at the end of the game.
                    # This is a common approach in games with clear win/loss/draw outcomes.
                    if (
                        game.winner == game.current_player
                    ):  # Current player is the one who just moved
                        reward = 1.0
                    elif game.winner is None:  # Draw
                        # Currently, a draw is treated the same as a loss for the player who didn't win.
                        # (i.e., if the opponent didn't win, and it's a draw, the reward is 0.0,
                        # which is less than the win reward of 1.0).
                        # Consider if a different reward for draws (e.g., small positive, or distinct from loss)
                        # would be beneficial. For now, 0.0 is used.
                        reward = 0.0
                    else:  # Opponent won
                        reward = -1.0
                else:
                    reward = 0.0  # No reward if game is not over

                game_info = {
                    "termination_reason": game.termination_reason,
                    "winner": game.winner,
                }
                current_player_name = (
                    "Sente" if game.current_player == 0 else "Gote"
                )  # Note: current_player is now the *next* player
                # For logging the player who *made* the move, we might need to track previous player or adjust logic
                # For now, this logs the player whose turn it is *after* the move.
                usi_move = policy_mapper.shogi_move_to_usi(selected_shogi_move)
                log_message = f"Time: {global_timestep}, Ep: {total_episodes_completed + 1}, Step: {current_episode_length}, Player: {current_player_name}, Move (USI): {usi_move}, Reward: {reward:.2f}, Done: {done}"
                if done and game_info.get("termination_reason"):
                    log_message += f", Termination: {game_info['termination_reason']}"
                logger.log(log_message)
            else:  # No move was made (e.g., if done or no legal moves)
                # If already done, next_obs might not be valid or needed for buffer if we don\'t add this step.
                # If we must add to buffer, use current obs as next_obs, and reward is likely 0 or a penalty.
                # For simplicity, if no move is made because it\'s done, we might not add this to buffer,
                # or ensure reward/next_obs are sensible. Assuming `done` is true here.
                # obs_for_buffer is already set
                reward = 0.0  # Or some terminal reward/penalty if applicable
                # `done` is already true or set above

            # Add to experience buffer
            # ExperienceBuffer.add expects: obs, action, reward, log_prob, value, done
            # Only add to buffer if an action was actually taken (action_idx != -1).
            # This prevents adding entries when `done` is set true because no legal moves were available *before* action selection.
            if action_idx != -1:
                experience_buffer.add(
                    torch.tensor(
                        obs_for_buffer, dtype=torch.float32, device=agent.device
                    ),  # Ensure obs is a tensor on the correct device
                    action_idx,
                    reward,
                    log_prob,
                    value,
                    done,  # This `done` reflects the state *after* the move or if no legal moves (but action_idx would be -1)
                    legal_mask_for_buffer,  # Pass the legal_mask to the buffer
                )

            obs = next_obs
            current_episode_reward += reward

            if done:
                total_episodes_completed += 1
                logger.log(
                    f"Episode {total_episodes_completed} finished. Length: {current_episode_length}, Reward: {current_episode_reward:.2f}"
                )
                pbar.set_description(
                    f"Training Progress | Last Ep Reward: {current_episode_reward:.2f}"
                )
                obs = game.reset()  # game.reset() now returns obs directly (np.ndarray)
                current_episode_reward = 0.0
                current_episode_length = 0
                # Save model periodically by episode count
                if total_episodes_completed % cfg.SAVE_FREQ_EPISODES == 0:
                    ckpt_path_ep = os.path.join(
                        model_dir,
                        f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth",
                    )
                    agent.save_model(
                        ckpt_path_ep, global_timestep, total_episodes_completed
                    )
                    logger.log(f"Model saved to {ckpt_path_ep}")

            # Perform PPO update if buffer is full
            # The buffer collects cfg.STEPS_PER_EPOCH valid transitions before an update.
            # Transitions are only added if action_idx != -1 (i.e., a valid action was taken).
            # This means the number of environment interactions to fill the buffer might be
            # slightly more than STEPS_PER_EPOCH if some steps result in no action being added
            # (e.g., terminal state reached due to no legal moves before agent selection).
            # However, the PPO update itself will always use STEPS_PER_EPOCH valid samples.
            if len(experience_buffer) >= cfg.STEPS_PER_EPOCH:  # Check buffer length
                next_value_np = agent.get_value(next_obs)
                experience_buffer.compute_advantages_and_returns(
                    next_value_np
                )  # Corrected method name
                learn_metrics = agent.learn(experience_buffer)
                experience_buffer.clear()
                kl_div = learn_metrics.get("ppo/kl_divergence_approx", 0.0)
                logger.log(
                    f"PPO Update at Timestep {global_timestep}. Metrics: {json.dumps(learn_metrics)}"
                )
                # KL divergence is a key PPO metric. It measures the difference between the old and new policies.
                # If it gets too high, the policy updates are too drastic and can destabilize training.
                # Some PPO implementations use adaptive KL penalties or early stopping for updates if KL exceeds a threshold.
                pbar_postfix = {
                    "kl_div": f"{kl_div:.3f}",
                    "policy_loss": f"{learn_metrics.get('ppo/policy_loss', 0.0):.3f}",
                    "value_loss": f"{learn_metrics.get('ppo/value_loss', 0.0):.3f}",
                    "entropy": f"{learn_metrics.get('ppo/entropy', 0.0):.3f}",
                    "lr": f"{learn_metrics.get('ppo/learning_rate', 0.0):.1e}",
                }
                pbar.set_postfix(pbar_postfix)

        pbar.close()
        logger.log("Training finished.")

    # --- Ensure checkpoint is saved at end of training ---
    final_ckpt_path = os.path.join(
        model_dir,
        f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}_final.pth",
    )
    agent.save_model(final_ckpt_path, global_timestep, total_episodes_completed)
    logger.log(f"Final model saved to {final_ckpt_path}")  # Log instead of print


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"An error occurred in main: {e}")
        print(traceback.format_exc())
        raise  # Re-raise the exception to ensure the script still exits with an error code
