"""
Minimal train.py main loop for DRL Shogi Client with Weights & Biases integration.
"""

import torch
import os  # Ensure os is imported for path operations
from dotenv import load_dotenv  # For loading .env file
import wandb  # Weights & Biases

from keisei.shogi.shogi_engine import ShogiGame
from keisei.shogi.shogi_core_definitions import Color  # Added import
from keisei.utils import PolicyOutputMapper, TrainingLogger
from keisei.ppo_agent import PPOAgent
from keisei.experience_buffer import ExperienceBuffer
import config


def evaluate_agent(agent, num_games=5, logger=None):
    """Run evaluation games with the current agent and log win/draw/loss stats."""

    wins = 0
    draws = 0
    losses = 0
    for i in range(num_games):
        game = ShogiGame()
        obs_np = game.reset()  # Get initial observation
        for _ in range(100):  # Max moves per game
            legal_shogi_moves = game.get_legal_moves()
            if not legal_shogi_moves:
                # print(f"No legal moves for player {game.current_player} in eval. Game likely ended.")
                break

            # Pass observation as np.ndarray and legal_shogi_moves
            selected_shogi_move, _, _, _ = agent.select_action(
                obs_np, legal_shogi_moves=legal_shogi_moves, is_training=False
            )

            # Step the environment with the selected move
            # obs_np, reward, done, _ = game.make_move(selected_shogi_move)
            obs_np, reward, done, info = game.make_move(
                selected_shogi_move
            )  # Corrected unpacking

            if game.game_over:
                # Determine winner based on game.winner and game.current_player
                # This logic needs to be robust based on how ShogiGame sets winner
                if game.winner is not None:
                    # Assuming agent is always playing as BLACK for simplicity in eval
                    # or that we check whose turn it was when game ended if agent plays both
                    # For now, let's assume the agent is BLACK (player 0)
                    # and the opponent is WHITE (player 1)
                    # This part needs refinement based on actual game play during eval
                    # A simple check: if current_player made the last move and won.
                    # Or, if game.winner is set, use that.
                    # Let's assume agent is player 0 (BLACK)
                    # If game.winner is BLACK, agent wins.
                    # If game.winner is WHITE, agent loses.
                    # If draw (sennichite), it's a draw.
                    # This needs to be more robust: who is the agent playing as?
                    # For now, let's simplify: if agent is current_player and game ends, it's a loss unless checkmate
                    # This is tricky without knowing which color the agent is.
                    # Let's assume for now, if game.winner is set, we use it.
                    # And we need to define which player the agent is.
                    # For simplicity, let's assume agent is always BLACK in evaluation.
                    if game.winner == Color.BLACK:
                        wins += 1
                    elif game.winner == Color.WHITE:
                        losses += 1
                    else:  # Should be a draw if winner is None but game_over is True
                        draws += 1
                else:  # No winner explicitly set, but game over (e.g. sennichite, max_moves)
                    draws += 1  # Or handle based on specific game end condition
                break
        # If loop finishes due to max_moves without game.game_over, count as draw
        if not game.game_over:
            draws += 1

    # msg = f"Evaluation: {num_games} games | Wins: {wins} | Draws: {draws} | Losses: {losses}"
    # More detailed logging
    total_played = wins + losses + draws
    if (
        total_played == 0
    ):  # Avoid division by zero if num_games was 0 or all games aborted early
        win_rate = 0
        loss_rate = 0
        draw_rate = 0
    else:
        win_rate = wins / total_played
        loss_rate = losses / total_played
        draw_rate = draws / total_played

    msg = (
        f"Evaluation Complete: Ran {total_played}/{num_games} games.\n"
        f"  Wins: {wins} ({win_rate:.2%})\n"
        f"  Losses: {losses} ({loss_rate:.2%})\n"
        f"  Draws: {draws} ({draw_rate:.2%})"
    )

    if logger:
        logger.log(msg)
    else:
        print(msg)


def main():
    """Main training loop for PPO DRL Shogi Client."""  # Updated docstring

    # --- Weights & Biases Setup ---
    # Load environment variables from .env file
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if not wandb_api_key:
        print("WANDB_API_KEY not found in environment variables or .env file.")
        print(
            "Please set it or run 'wandb login'. For now, W&B logging will be disabled."
        )
        wandb_enabled = False
    else:
        # Ensure the key is in the environment for wandb to pick up if loaded from .env
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb_enabled = True

    if wandb_enabled:
        try:
            wandb.init(
                project="shogi-drl",  # Or your preferred project name
                config={
                    "total_timesteps": config.TOTAL_TIMESTEPS,
                    "steps_per_epoch": config.STEPS_PER_EPOCH,
                    "ppo_epochs": config.PPO_EPOCHS,
                    "minibatch_size": config.MINIBATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "gamma": config.GAMMA,
                    "clip_epsilon": config.CLIP_EPSILON,
                    "lambda_gae": config.LAMBDA_GAE,
                    "entropy_coeff": config.ENTROPY_COEFF,
                    "value_loss_coeff": config.VALUE_LOSS_COEFF,
                    "max_moves_per_game": config.MAX_MOVES_PER_GAME,
                    "input_channels": config.INPUT_CHANNELS,
                    "num_actions_total": config.NUM_ACTIONS_TOTAL,
                    "save_freq_episodes": config.SAVE_FREQ_EPISODES,
                    "device": config.DEVICE,
                    # Add any other relevant hyperparameters from config.py
                },
                name=f"ppo_run_{os.getpid()}",  # Example run name
                # Optional: resume="allow", id=YOUR_RUN_ID # For resuming runs
            )
            print("Weights & Biases initialized successfully.")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}")
            print("Proceeding without W&B logging.")
            wandb_enabled = False
    # --- End W&B Setup ---

    logger = TrainingLogger(config.LOG_FILE)
    logger.log("Starting DRL Shogi Client Training")
    logger.log(f"Using device: {config.DEVICE}")

    # Ensure MODEL_DIR exists
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
        logger.log(f"Created model directory: {config.MODEL_DIR}")

    game = ShogiGame()
    policy_mapper = PolicyOutputMapper()

    agent = PPOAgent(
        input_channels=config.INPUT_CHANNELS,
        policy_output_mapper=policy_mapper,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        clip_epsilon=config.CLIP_EPSILON,
        ppo_epochs=config.PPO_EPOCHS,
        minibatch_size=config.MINIBATCH_SIZE,
        value_loss_coeff=config.VALUE_LOSS_COEFF,
        entropy_coef=config.ENTROPY_COEFF,
        device=config.DEVICE,
    )
    logger.log("PPO Agent initialized.")

    buffer = ExperienceBuffer(
        buffer_size=config.STEPS_PER_EPOCH,
        gamma=config.GAMMA,
        lambda_gae=config.LAMBDA_GAE,
        device=config.DEVICE,
    )
    logger.log(f"Experience buffer initialized with size: {config.STEPS_PER_EPOCH}")

    obs_np = game.reset()

    episode_num = 0
    episode_reward = 0
    episode_steps = 0
    total_episodes_completed = 0  # For SAVE_FREQ_EPISODES

    # Main training loop driven by global timesteps
    for global_timestep in range(1, config.TOTAL_TIMESTEPS + 1):
        legal_shogi_moves = game.get_legal_moves()
        info = {}  # Initialize info dictionary

        if not legal_shogi_moves:
            logger.log(
                f"Warning: No legal moves for player {game.current_player} at timestep {global_timestep}. Game should be over."
            )
            if not game.game_over:
                logger.log(
                    "CRITICAL: No legal moves but game not marked over. Forcing done=True."
                )
                done = True
                info["reason"] = "forced_done_no_legal_moves"
            else:
                done = True  # Game is already over, make_move below will likely confirm this with its own info
                # If game is already over, we effectively skip to the episode end logic
                # We need to ensure obs_np, reward, etc. are consistent if we bypass make_move
                # For simplicity, let make_move be called if possible, or handle this state carefully
                # This path implies the episode ended on the previous step or due to external factor.
                # Let's assume make_move won't be called if done is True from here.
                # The episode end logic will handle reset.
        else:
            selected_shogi_move, policy_idx, log_prob_action, value_pred = (
                agent.select_action(
                    obs_np, legal_shogi_moves=legal_shogi_moves, is_training=True
                )
            )
            next_obs_np, reward, done, info_from_move = game.make_move(
                selected_shogi_move
            )
            info.update(info_from_move if info_from_move else {})  # Merge info
            episode_reward += reward
            buffer.add(obs_np, policy_idx, reward, log_prob_action, value_pred, done)
            obs_np = next_obs_np

        episode_steps += 1

        if done or episode_steps >= config.MAX_MOVES_PER_GAME:
            reason = info.get("reason", "unknown")
            if episode_steps >= config.MAX_MOVES_PER_GAME and not done:
                reason = "max_moves_reached"
                done = True  # Ensure done is true if max_moves is the reason

            log_message = (
                f"Episode {episode_num + 1} finished after {episode_steps} steps. "
                f"Total Timesteps: {global_timestep}. Reward: {episode_reward}. "
                f"Game Over: {done}. Reason: {reason}"
            )
            logger.log(log_message)
            if wandb_enabled:
                wandb.log(
                    {
                        "episode_reward": episode_reward,
                        "episode_steps": episode_steps,
                        "episode_num": episode_num + 1,
                        "total_episodes_completed": total_episodes_completed
                        + 1,  # Log before increment
                        "game_over_reason": reason,
                    },
                    step=global_timestep,
                )

            current_obs_before_reset = obs_np  # Save current obs for potential last_value calculation if buffer fills now
            obs_np = game.reset()
            episode_num += 1
            total_episodes_completed += 1
            episode_reward = 0
            episode_steps = 0

            # Save model checkpoint based on episodes completed
            if total_episodes_completed % config.SAVE_FREQ_EPISODES == 0:
                model_path = os.path.join(
                    config.MODEL_DIR,
                    f"ppo_shogi_agent_episode_{total_episodes_completed}_ts_{global_timestep}.pth",
                )
                agent.save_model(model_path)
                logger.log(f"Saved model checkpoint to {model_path}")
                if wandb_enabled:
                    try:
                        artifact = wandb.Artifact(
                            name=f"model-checkpoint-ep{total_episodes_completed}",
                            type="model",
                            description=f"PPO Shogi Agent model checkpoint after {total_episodes_completed} episodes, {global_timestep} timesteps.",
                            metadata={
                                "episode": total_episodes_completed,
                                "timestep": global_timestep,
                                "path": model_path,
                            },
                        )
                        artifact.add_file(model_path)
                        wandb.log_artifact(artifact)
                        logger.log(f"Logged model artifact to W&B: {artifact.name}")
                    except Exception as e:
                        logger.log(f"Error logging model artifact to W&B: {e}")

        # If buffer is full, perform learning update
        if len(buffer) == config.STEPS_PER_EPOCH:
            logger.log(
                f"Buffer full at timestep {global_timestep}. Performing PPO update."
            )
            last_value: float
            if done:  # If the episode ended exactly when the buffer got full
                last_value = 0.0
            else:  # Episode is still ongoing, get value of current_obs_before_reset or obs_np
                # If done was true, obs_np is already the reset state.
                # If done was false, obs_np is the current state from which we'd get last_value.
                # The variable current_obs_before_reset was saved before reset if an episode ended.
                # If the episode did not end, obs_np is the correct state.
                # Let's clarify: if an episode ended, current_obs_before_reset holds the state *before* reset.
                # If the episode is ongoing, obs_np is the current state.
                # The key is, what was the state *just before* the buffer became full?
                # It's `obs_np` if the episode is ongoing, or `current_obs_before_reset` if an episode just ended.
                # However, `compute_advantages_and_returns` needs the value of the *next* state after the last one added to buffer.
                # If `done` is true (episode ended), the value of the terminal state is 0.
                # If `done` is false (episode ongoing), the value is V(s_T) where s_T is `obs_np`.
                # The logic in `train.py` for `last_value` seems correct: if `done` (meaning the *last* state in buffer was terminal), `last_value` is 0.
                # Otherwise, `last_value` is `agent.get_value(obs_np)` (the state *after* the last one in buffer).
                last_value = agent.get_value(
                    obs_np
                )  # Get value of the current state (s_T)

            buffer.compute_advantages_and_returns(last_value)
            avg_policy_loss, avg_value_loss, avg_entropy = agent.learn(
                buffer
            )  # Capture losses
            buffer.clear()
            log_message_ppo = (
                f"PPO Update complete. Avg Policy Loss: {avg_policy_loss:.4f}, "
                f"Avg Value Loss: {avg_value_loss:.4f}, Avg Entropy: {avg_entropy:.4f}"
            )
            logger.log(log_message_ppo)
            if wandb_enabled:
                wandb.log(
                    {
                        "avg_policy_loss": avg_policy_loss,
                        "avg_value_loss": avg_value_loss,
                        "avg_entropy": avg_entropy,
                    },
                    step=global_timestep,
                )

        # Evaluation runs periodically (e.g., every N global timesteps or M episodes)
        if global_timestep % config.EVAL_FREQ_TIMESTEPS == 0:
            logger.log(f"Running periodic evaluation at timestep {global_timestep}.")
            evaluate_agent(agent, num_games=config.EVAL_NUM_GAMES, logger=logger)

    # Final cleanup or summary
    logger.log("Training finished.")
    if wandb_enabled:
        wandb.finish()
        logger.log("Weights & Biases run finished.")


if __name__ == "__main__":
    main()
