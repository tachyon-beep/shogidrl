"""
Main training script for Keisei Shogi RL agent.
This file contains the actual training logic and exposes a main() function.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import config as app_config
import wandb  # Import wandb at the top level

# Import the callable evaluation function
from evaluate import execute_full_evaluation_run
from keisei.experience_buffer import ExperienceBuffer
from keisei.ppo_agent import PPOAgent
from keisei.shogi.shogi_engine import Color, ShogiGame
from keisei.utils import PolicyOutputMapper, TrainingLogger

# Add datetime and os if not already imported (os and datetime are already imported)
# import os
# from datetime import datetime


# Expose main at the module level for import by the root-level shim


def parse_args():
    parser = argparse.ArgumentParser(description="DRL Shogi Client Training (train.py)")
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


def apply_config_overrides(args, cfg_module):
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_json = json.load(f)
        for k, v in config_json.items():
            if hasattr(cfg_module, k):
                setattr(cfg_module, k, v)
            else:
                # Added warning for unknown keys
                print(
                    f"Warning: Config key '{k}' from JSON not found in base config module.",
                    file=sys.stderr,
                )
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


def find_latest_checkpoint(model_dir):
    if not os.path.isdir(model_dir):
        return None
    ckpts = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not ckpts:
        return None
    # import re # Removed redundant import re

    checkpoint_pattern = re.compile(
        r"episode_(\d+).*ts_(\d+)|ep(\d+).*ts(\d+)|agent_episode_(\d+)_ts_(\d+)"
    )

    def extract_ts(filename):
        match = checkpoint_pattern.search(filename)
        if match:
            if match.group(1) is not None and match.group(2) is not None:
                try:
                    ep = int(match.group(1))
                    ts = int(match.group(2))
                    return (ep, ts)
                except ValueError:
                    return (0, 0)
            elif match.group(3) is not None and match.group(4) is not None:
                try:
                    ep = int(match.group(3))
                    ts = int(match.group(4))
                    return (ep, ts)
                except ValueError:
                    return (0, 0)
            elif (
                match.group(5) is not None and match.group(6) is not None
            ):  # Check for the new pattern
                try:
                    ep = int(match.group(5))
                    ts = int(match.group(6))
                    return (ep, ts)
                except ValueError:
                    return (0, 0)
        return (0, 0)

    valid_ckpts = []
    for f in ckpts:
        ep_ts = extract_ts(f)
        if ep_ts != (0, 0):
            valid_ckpts.append((f, ep_ts))
    if not valid_ckpts:
        return None
    valid_ckpts.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(model_dir, valid_ckpts[0][0])


def serialize_config(cfg_module):
    d = {
        k: getattr(cfg_module, k)
        for k in dir(cfg_module)
        if k.isupper() and not k.startswith("__")
    }
    return json.dumps(d, indent=2)


# --- MAIN TRAINING LOOP (SCAFFOLD) ---
def main():
    # Set multiprocessing start method
    if __name__ == "keisei.train":  # Check if running as the main module of the package
        try:
            mp.set_start_method("spawn", force=True)  # Added force=True
            print("Successfully set multiprocessing start method to 'spawn'")
        except RuntimeError as e:
            print(f"Could not set multiprocessing start method: {e}", file=sys.stderr)

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

    # Print key log messages to both stdout and logger for CLI discoverability
    def log_both(msg):
        print(msg)
        if "logger" in locals():
            logger.log(msg)

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
        log_both(f"Starting run: {run_name}")
        log_both(f"Effective Config: {serialize_config(cfg)}")

        # Initialize W&B for the main training run if enabled in config
        is_train_wandb_active = False
        if getattr(
            cfg, "WANDB_LOG_TRAIN", False
        ):  # Assuming a config like WANDB_LOG_TRAIN
            try:
                # import wandb # Moved to top
                wandb.init(
                    project=getattr(cfg, "WANDB_PROJECT_TRAIN", "keisei-training"),
                    entity=getattr(cfg, "WANDB_ENTITY_TRAIN", None),
                    name=run_name,  # Use the main run_name
                    config=json.loads(
                        serialize_config(cfg)
                    ),  # Log the effective config
                    reinit=True,  # In case of multiple main() calls in one process (e.g. testing)
                    group="TrainingRuns",
                )
                log_both(
                    f"Weights & Biases logging enabled for main training run: {run_name}"
                )
                is_train_wandb_active = True
            except Exception as e:
                log_both(
                    f"Error initializing W&B for training: {e}. W&B logging for training disabled."
                )
                is_train_wandb_active = False

        if ckpt_path and os.path.exists(ckpt_path):
            log_both(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = agent.load_model(ckpt_path)
            if ckpt:
                global_timestep = ckpt.get("global_timestep", 0)
                total_episodes_completed = ckpt.get("total_episodes_completed", 0)
                log_both(
                    f"Resumed at timestep {global_timestep}, episodes {total_episodes_completed}"
                )
                if agent.last_kl_div is not None:
                    log_both(
                        f"Resumed with last KL divergence: {agent.last_kl_div:.4f}"
                    )
            else:
                log_both(
                    f"Failed to load checkpoint data from {ckpt_path}. Starting fresh."
                )
        else:
            log_both(
                "No checkpoint found or specified path invalid. Starting fresh training."
            )

        # --- MAIN TRAINING LOOP (Ported and adapted from train2.py) ---
        pbar = tqdm(
            total=cfg.TOTAL_TIMESTEPS, initial=global_timestep, desc="Training Progress"
        )

        obs_reset_val = game.reset()
        if (
            isinstance(obs_reset_val, tuple)
            and len(obs_reset_val) > 0
            and isinstance(obs_reset_val[0], np.ndarray)
        ):
            obs = obs_reset_val[0]
        elif isinstance(obs_reset_val, np.ndarray):
            obs = obs_reset_val
        else:
            logger.log(
                f"CRITICAL: Initial game.reset() did not return a valid observation. Type: {type(obs_reset_val)}. Aborting."
            )
            pbar.close()
            return

        current_episode_reward = 0.0
        current_episode_length = 0
        done = False

        # Initialize variables that might not be set in all paths before buffer.add
        selected_shogi_move = None
        action_idx = -1
        log_prob = 0.0
        value = 0.0
        obs_for_buffer = obs  # Initialize with current obs
        legal_mask_for_buffer = torch.zeros(
            policy_mapper.get_total_actions(), dtype=torch.bool, device=agent.device
        )  # Dummy mask

        for _ in range(global_timestep, cfg.TOTAL_TIMESTEPS):
            pbar.update(1)
            global_timestep += 1
            current_episode_length += 1

            if not isinstance(obs, np.ndarray):
                logger.log(
                    f"Error: obs is not a numpy array at timestep {global_timestep}. Type: {type(obs)}. Ending episode."
                )
                done = True
                selected_shogi_move, action_idx, log_prob, value = None, -1, 0.0, 0.0
                obs_for_buffer = obs  # Keep last valid obs or handle appropriately
                legal_mask_for_buffer = torch.zeros(
                    policy_mapper.get_total_actions(),
                    dtype=torch.bool,
                    device=agent.device,
                )  # Dummy mask
            else:
                obs_for_buffer = obs  # This is s_t
                legal_moves = game.get_legal_moves()
                legal_mask_for_buffer = policy_mapper.get_legal_mask(
                    legal_moves, agent.device
                )

                if not legal_moves:  # Equivalent to not legal_mask_for_buffer.any()
                    logger.log(
                        f"Episode {total_episodes_completed + 1}: No legal moves. Game might have ended."
                    )
                    done = True

                if not done:
                    try:
                        # PPOAgent.select_action now expects legal_mask and returns 4 items
                        (
                            selected_shogi_move,
                            action_idx,
                            log_prob,
                            value,
                        ) = agent.select_action(
                            obs,
                            legal_moves,
                            legal_mask_for_buffer,
                            is_training=True,  # Pass legal_mask
                        )
                        # The 5th item (value_tensor) was removed from select_action's direct return
                        # It was originally `move_tuple` then `selected_shogi_move, action_idx, log_prob, value, _ = move_tuple`
                    except (IndexError, ValueError, RuntimeError) as e:  # MODIFIED_LINE
                        logger.log(
                            f"Error in agent.select_action: {e}. Treating as no move."
                        )
                        selected_shogi_move, action_idx, log_prob, value = (
                            None,
                            -1,
                            0.0,
                            0.0,
                        )
                        done = True  # End episode if agent fails

                if done:  # If already done by no legal moves or agent error
                    selected_shogi_move, action_idx, log_prob, value = (
                        None,
                        -1,
                        0.0,
                        0.0,
                    )

            reward = 0.0  # Default reward, will be updated by make_move
            next_obs = (
                obs  # Default next_obs (s_t+1), will be updated. If done, this is fine.
            )
            game_info = {}  # Default game_info

            if selected_shogi_move is not None:
                # Store the turn before the move for validation
                last_turn = game.current_player
                move_outcome = game.make_move(selected_shogi_move)
                # Assert that the turn has alternated after the move
                assert (
                    game.current_player != last_turn
                ), f"Turn did not alternate at move #{game.move_count}. Last turn: {last_turn}, Current turn: {game.current_player}"

                if not (isinstance(move_outcome, tuple) and len(move_outcome) == 4):
                    logger.log(
                        f"Error: game.make_move did not return a valid 4-tuple. Got: {move_outcome}. Ending episode."
                    )
                    # s_t+1 is current obs, reward 0, done True
                    next_obs, reward, done, game_info = (
                        obs,
                        0.0,
                        True,
                        {"termination_reason": "make_move_bad_return"},
                    )
                else:
                    next_obs_full, reward_raw, done_raw, game_info_raw = move_outcome

                    next_obs = (
                        next_obs_full if isinstance(next_obs_full, np.ndarray) else obs
                    )
                    if not isinstance(next_obs_full, np.ndarray):
                        logger.log(
                            f"Warning: next_obs_full from make_move is {type(next_obs_full)}. Using current obs as fallback for s_t+1."
                        )
                        done = True  # If next state is invalid, consider episode done.

                    reward = (
                        float(reward_raw)
                        if isinstance(reward_raw, (int, float))
                        else 0.0
                    )
                    if not isinstance(reward_raw, (int, float)):
                        logger.log(
                            f"Warning: reward_raw from make_move is {type(reward_raw)} ('{reward_raw}'). Using 0.0."
                        )

                    done = bool(done_raw)  # Update done state from game
                    game_info = game_info_raw if isinstance(game_info_raw, dict) else {}

                # Use the player who MADE the move (last_turn), not the next player to move (game.current_player)
                player_name = "Sente" if last_turn == Color.BLACK else "Gote"
                usi_move = policy_mapper.shogi_move_to_usi(selected_shogi_move)
                log_msg = f"Time: {global_timestep}, Ep: {total_episodes_completed + 1}, Step: {current_episode_length}, Player: {player_name}, Move (USI): {usi_move}, Reward: {reward:.2f}, Done: {done}"
                if game_info.get("termination_reason"):
                    log_msg += f", Termination: {game_info['termination_reason']}"
                elif isinstance(
                    game_info, str
                ):  # Should not happen if game_info_raw is dict
                    log_msg += f", Info: {game_info}"
                logger.log(log_msg)
            # else: No move was made (e.g. if done or no legal moves initially)
            # obs_for_buffer (s_t) is already set
            # next_obs (s_t+1) remains current obs (obs_for_buffer)
            # reward is 0.0, done is as determined earlier

            # Add to experience buffer: (s_t, a_t, r_t, log_prob_t, value_t, done_t+1, legal_mask_t)
            # Note: `done` here is the done signal *after* taking the action, so it corresponds to the next state.
            if isinstance(obs_for_buffer, np.ndarray):
                experience_buffer.add(
                    torch.tensor(
                        obs_for_buffer, dtype=torch.float32, device=agent.device
                    ),
                    action_idx,
                    reward,
                    log_prob,
                    value,
                    done,
                    legal_mask_for_buffer,  # Pass legal_mask_for_buffer
                )
            else:
                # This case should ideally be prevented by the check at the start of the loop
                logger.log(
                    f"Error: obs_for_buffer was not ndarray (type: {type(obs_for_buffer)}). Skipping add to buffer."
                )

            obs = next_obs  # obs becomes s_t+1 for the next iteration
            current_episode_reward += reward

            if done:
                total_episodes_completed += 1
                logger.log(
                    f"Episode {total_episodes_completed} finished. Length: {current_episode_length}, Reward: {current_episode_reward:.2f}"
                )
                pbar.set_description(
                    f"Training Progress | Last Ep Reward: {current_episode_reward:.2f}"
                )

                obs_reset_val = game.reset()
                if (
                    isinstance(obs_reset_val, tuple)
                    and len(obs_reset_val) > 0
                    and isinstance(obs_reset_val[0], np.ndarray)
                ):
                    obs = obs_reset_val[0]
                elif isinstance(obs_reset_val, np.ndarray):
                    obs = obs_reset_val
                else:
                    logger.log(
                        f"CRITICAL: game.reset() after episode did not return valid observation. Type: {type(obs_reset_val)}. Aborting training."
                    )
                    pbar.close()
                    break  # Exit training loop

                # Reset for new episode
                current_episode_reward = 0.0
                current_episode_length = 0
                done = False  # Important: reset done flag for the new episode
                # Re-initialize for the new episode start, obs is now the new s_0
                selected_shogi_move = None
                action_idx = -1
                log_prob = 0.0
                value = 0.0
                obs_for_buffer = obs
                legal_mask_for_buffer = torch.zeros(
                    policy_mapper.get_total_actions(),
                    dtype=torch.bool,
                    device=agent.device,
                )  # Dummy mask

                if total_episodes_completed % cfg.SAVE_FREQ_EPISODES == 0:
                    ckpt_path_ep = os.path.join(
                        model_dir,  # model_dir is run_dir here
                        f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth",
                    )
                    agent.save_model(
                        ckpt_path_ep, global_timestep, total_episodes_completed
                    )
                    logger.log(f"Model saved to {ckpt_path_ep}")  # Main training logger

                    # === New Periodic Evaluation Call ===
                    if getattr(cfg, "EVAL_DURING_TRAINING", False):
                        log_both(
                            f"--- Triggering Periodic Evaluation for: {ckpt_path_ep} ---"
                        )

                        eval_log_filename = f"periodic_eval_ts{global_timestep}_ep{total_episodes_completed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                        # Store eval logs in a subfolder of the run_dir
                        eval_log_dir = os.path.join(run_dir, "eval_logs")
                        os.makedirs(eval_log_dir, exist_ok=True)
                        eval_log_path = os.path.join(eval_log_dir, eval_log_filename)

                        # policy_mapper is already instantiated in keisei/train.py main()

                        eval_results = execute_full_evaluation_run(
                            agent_checkpoint_path=ckpt_path_ep,
                            opponent_type=getattr(cfg, "EVAL_OPPONENT_TYPE", "random"),
                            opponent_checkpoint_path=getattr(
                                cfg, "EVAL_OPPONENT_CHECKPOINT_PATH", None
                            ),
                            num_games=getattr(cfg, "EVAL_NUM_GAMES", 10),
                            max_moves_per_game=getattr(
                                cfg, "MAX_MOVES_PER_GAME_EVAL", 300
                            ),  # Use the eval specific config
                            device_str=getattr(
                                cfg, "EVAL_DEVICE", cfg.DEVICE
                            ),  # Default to main training device if EVAL_DEVICE not set
                            log_file_path_eval=eval_log_path,
                            policy_mapper=policy_mapper,  # Pass existing instance
                            seed=getattr(
                                cfg, "SEED", None
                            ),  # Optionally use the main training seed or a different one for eval
                            wandb_log_eval=getattr(cfg, "EVAL_WANDB_LOG", False),
                            wandb_project_eval=getattr(
                                cfg, "EVAL_WANDB_PROJECT", "keisei-periodic-evals"
                            ),
                            wandb_entity_eval=getattr(cfg, "EVAL_WANDB_ENTITY", None),
                            # Construct a unique run name for this specific evaluation W&B run
                            wandb_run_name_eval=f"{getattr(cfg, 'EVAL_WANDB_RUN_NAME_PREFIX', 'eval_')}{run_name}_ts{global_timestep}_ep{total_episodes_completed}",
                        )

                        if eval_results:
                            log_both(
                                f"--- Periodic Evaluation Summary (vs {eval_results.get('opponent_name', 'N/A')} for agent {eval_results.get('agent_name', 'N/A')}) ---"
                            )
                            log_both(
                                f"  Games: {eval_results.get('num_games')}, Wins: {eval_results.get('wins')}, Losses: {eval_results.get('losses')}, Draws: {eval_results.get('draws')}"
                            )
                            log_both(
                                f"  Win Rate: {eval_results.get('win_rate', 0):.2%}, Avg Game Length: {eval_results.get('avg_game_length', 0):.2f}"
                            )
                            # Optionally log key eval metrics to the main training W&B run if desired (prefix them)
                            if (
                                is_train_wandb_active and wandb.run
                            ):  # is_train_wandb_active from main W&B init
                                wandb.log(
                                    {
                                        f"eval_vs_{eval_results.get('opponent_name', 'opponent')}/win_rate": eval_results.get(
                                            "win_rate", 0
                                        ),
                                        f"eval_vs_{eval_results.get('opponent_name', 'opponent')}/avg_game_length": eval_results.get(
                                            "avg_game_length", 0
                                        ),
                                        f"eval_vs_{eval_results.get('opponent_name', 'opponent')}/wins": eval_results.get(
                                            "wins", 0
                                        ),
                                        f"eval_vs_{eval_results.get('opponent_name', 'opponent')}/losses": eval_results.get(
                                            "losses", 0
                                        ),
                                        f"eval_vs_{eval_results.get('opponent_name', 'opponent')}/draws": eval_results.get(
                                            "draws", 0
                                        ),
                                    },
                                    step=global_timestep,
                                )
                        else:
                            log_both(
                                "--- Periodic Evaluation did not return results or failed. ---"
                            )
                        log_both(
                            f"--- Finished Periodic Evaluation for: {ckpt_path_ep} ---"
                        )

            if len(experience_buffer) >= cfg.STEPS_PER_EPOCH:
                # `obs` here is s_t+N (the state *after* the last transition collected in the buffer)
                # This is the state for which we need to estimate the value for GAE calculation.
                value_for_gae_calc_obs = obs
                next_value_for_gae = 0.0  # Default if episode ended or error

                if (
                    done
                ):  # If the episode ended with the last transition added to the buffer.
                    next_value_for_gae = 0.0
                elif not isinstance(value_for_gae_calc_obs, np.ndarray):
                    logger.log(
                        f"Warning: obs for get_value (GAE) is not ndarray: {type(value_for_gae_calc_obs)}. Using 0.0 for next value."
                    )
                    next_value_for_gae = 0.0
                else:
                    try:
                        next_value_for_gae = agent.get_value(value_for_gae_calc_obs)
                    except (RuntimeError, ValueError) as e:  # MODIFIED_LINE
                        logger.log(
                            f"Error in agent.get_value for GAE calc: {e}. Using 0.0 for next value."
                        )
                        next_value_for_gae = 0.0

                experience_buffer.compute_advantages_and_returns(next_value_for_gae)
                learn_metrics = agent.learn(experience_buffer)
                experience_buffer.clear()

                kl_div = learn_metrics.get("ppo/kl_divergence_approx", 0.0)
                policy_loss = learn_metrics.get("ppo/policy_loss", 0.0)
                value_loss = learn_metrics.get("ppo/value_loss", 0.0)
                entropy = learn_metrics.get("ppo/entropy", 0.0)
                lr = learn_metrics.get("ppo/learning_rate", 0.0)

                logger.log(
                    f"PPO Update at Timestep {global_timestep}. Metrics: {json.dumps(learn_metrics)}"
                )
                pbar_postfix = {
                    "kl_div": f"{kl_div:.3f}",
                    "policy_loss": f"{policy_loss:.3f}",
                    "value_loss": f"{value_loss:.3f}",
                    "entropy": f"{entropy:.3f}",
                    "lr": f"{lr:.1e}",
                }
                pbar.set_postfix(pbar_postfix)

        pbar.close()
        if global_timestep >= cfg.TOTAL_TIMESTEPS:  # Check if loop completed normally
            logger.log("Training finished.")
        else:  # Loop may have been broken due to critical error
            logger.log("Training interrupted.")
        # End of ported training loop

        # --- Ensure checkpoint is saved at end of training ---
        final_ckpt_path = os.path.join(
            model_dir,
            f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}_final.pth",
        )
        agent.save_model(final_ckpt_path, global_timestep, total_episodes_completed)
        log_both(f"Final model saved to {final_ckpt_path}")  # Log instead of print

        # For minimal runs, also save a checkpoint with the standard pattern if none exists
        # This ensures test_train_runs_minimal passes even if no episode boundary was hit
        std_ckpt_pattern = os.path.join(model_dir, "ppo_shogi_ep*_ts*.pth")

        if not glob.glob(std_ckpt_pattern):
            std_ckpt_path = os.path.join(
                model_dir,
                f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth",
            )
            agent.save_model(std_ckpt_path, global_timestep, total_episodes_completed)
            log_both(f"Minimal run: extra checkpoint saved to {std_ckpt_path}")


__all__ = ["main"]

if __name__ == "__main__":
    # This block is for when keisei/train.py is run directly as a script
    try:
        mp.set_start_method("spawn", force=True)  # Added force=True
        print("Successfully set multiprocessing start method to 'spawn' (direct run)")
    except RuntimeError as e:
        print(
            f"Could not set multiprocessing start method (direct run): {e}",
            file=sys.stderr,
        )
    main()
