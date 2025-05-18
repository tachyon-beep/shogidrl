"""
Minimal train.py main loop for DRL Shogi Client with Weights & Biases integration.
"""

import os  # Ensure os is imported for path operations

import torch
from dotenv import load_dotenv  # For loading .env file

import config
import wandb  # Weights & Biases
from keisei.experience_buffer import ExperienceBuffer
from keisei.ppo_agent import PPOAgent
from keisei.shogi.shogi_core_definitions import Color  # Added import
from keisei.shogi.shogi_engine import ShogiGame
from keisei.utils import PolicyOutputMapper, TrainingLogger


def evaluate_agent(agent, num_games=5, logger=None):
    """Run evaluation games with the current agent and log win/draw/loss stats."""

    wins = 0
    draws = 0
    losses = 0
    game_lengths = []
    termination_reasons_counts = {}

    for i in range(num_games):
        game = ShogiGame(max_moves_per_game=config.MAX_MOVES_PER_GAME_EVAL) # Use a potentially different limit for eval
        obs_np = game.reset()  # Get initial observation
        current_game_moves = 0
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
            current_game_moves +=1

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
                game_lengths.append(current_game_moves)
                reason = info.get("termination_reason", "unknown")
                termination_reasons_counts[reason] = termination_reasons_counts.get(reason, 0) + 1
                break
        # If loop finishes due to max_moves without game.game_over, count as draw
        if not game.game_over:
            draws += 1
            game_lengths.append(current_game_moves) # Log length even if max_moves hit in eval loop
            reason = "eval_max_moves" # Specific reason for eval loop termination
            termination_reasons_counts[reason] = termination_reasons_counts.get(reason, 0) + 1


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

    avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    msg = (
        f"Evaluation Complete: Ran {total_played}/{num_games} games.\\n"
        f"  Wins: {wins} ({win_rate:.2%})\\n"
        f"  Losses: {losses} ({loss_rate:.2%})\\n"
        f"  Draws: {draws} ({draw_rate:.2%})\\n"
        f"  Average Game Length: {avg_game_length:.2f} moves"
    )
    if logger:
        logger.log(msg)
        # Log to WandB if enabled
        if wandb.run is not None:
            wandb.log({
                "eval/wins": wins,
                "eval/losses": losses,
                "eval/draws": draws,
                "eval/win_rate": win_rate,
                "eval/loss_rate": loss_rate,
                "eval/draw_rate": draw_rate,
                "eval/avg_game_length": avg_game_length,
                "eval/total_games_evaluated": total_played,
                **{f"eval/termination_{k}": v for k,v in termination_reasons_counts.items()}
            }, commit=False) # Commit with main training logs
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

    game = ShogiGame(max_moves_per_game=config.MAX_MOVES_PER_GAME) # Pass max_moves_per_game
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
        if not legal_shogi_moves:
            # This case implies the game ended before the agent could make a move (e.g., checkmate on spawn)
            # Or, more likely, an issue with game state or legal move generation.
            logger.log(f"CRITICAL: No legal moves for player {game.current_player} at start of a turn. Game state: {game.to_string()}")
            # Handle game reset or error appropriately
            # For now, let's assume this means the episode ended abruptly.
            if wandb_enabled and wandb.run is not None:
                wandb.log({
                    "episode/reward": episode_reward, # Log existing reward
                    "episode/length": episode_steps, # Log existing steps
                    "episode/number": total_episodes_completed,
                    "episode/termination/no_legal_moves_at_turn_start": 1,
                }, commit=False)
            obs_np = game.reset()
            episode_reward = 0
            episode_steps = 0
            total_episodes_completed += 1 # Count this as a completed (albeit problematic) episode
            continue # Skip to next timestep

        info = {}  # Initialize info dictionary
        obs_tensor_for_buffer = torch.tensor(obs_np, dtype=torch.float32, device=agent.device) # Prepare obs for buffer

        # Select action
        selected_shogi_move, selected_policy_index, log_prob, value_estimate = agent.select_action(
            obs_np, legal_shogi_moves=legal_shogi_moves, is_training=True
        )
        
        # Step the environment with the selected move
        next_obs_np, reward, done, info = game.make_move(selected_shogi_move)
        
        episode_reward += reward
        episode_steps += 1

        buffer.add(
            obs=obs_tensor_for_buffer, # Use the tensor version of obs_np
            action=selected_policy_index,
            reward=reward,
            value=value_estimate, 
            log_prob=log_prob,
            done=done,
        )
        
        obs_np = next_obs_np

        if done:
            total_episodes_completed += 1
            logger.log(
                f"Episode {total_episodes_completed} finished after {episode_steps} steps. "
                f"Reward: {episode_reward:.2f}. Termination: {info.get('termination_reason', 'N/A')}"
            )
            if wandb_enabled and wandb.run is not None:
                wandb.log({
                    "episode/reward": episode_reward,
                    "episode/length": episode_steps,
                    "episode/number": total_episodes_completed,
                    f"episode/termination/{info.get('termination_reason', 'unknown')}": 1,
                 }, commit=False) 

            obs_np = game.reset()
            episode_reward = 0
            episode_steps = 0

        # PPO Update Phase
        if global_timestep % config.STEPS_PER_EPOCH == 0:
            last_value = agent.get_value(obs_np) 
            buffer.compute_advantages_and_returns(last_value)
            
            metrics = agent.learn(buffer) 
            
            buffer.clear() 

            logger.log(f"PPO Update at timestep {global_timestep}")
            if wandb_enabled and wandb.run is not None:
                # Log PPO metrics
                # Ensure metrics is a dictionary before logging
                if isinstance(metrics, dict):
                    wandb.log(metrics, commit=False) 
                else:
                    logger.log(f"Warning: PPO metrics not a dict: {metrics}")

                # Evaluate agent periodically based on episodes
                if config.EVAL_FREQ_EPISODES > 0 and \
                   total_episodes_completed % config.EVAL_FREQ_EPISODES == 0 and \
                   total_episodes_completed > 0:
                    logger.log(f"Starting evaluation at episode {total_episodes_completed}...")
                    evaluate_agent(agent, num_games=config.EVAL_NUM_GAMES, logger=logger)

                # Save model periodically based on episodes
                if config.SAVE_FREQ_EPISODES > 0 and \
                   total_episodes_completed % config.SAVE_FREQ_EPISODES == 0 and \
                   total_episodes_completed > 0:
                    save_path = os.path.join(
                        config.MODEL_DIR,
                        f"ppo_shogi_agent_episode_{total_episodes_completed}_ts_{global_timestep}.pth",
                    )
                    agent.save_model(save_path)
                    logger.log(f"Model saved to {save_path}")
        
        if wandb_enabled and wandb.run is not None:
            wandb.log({"global_timestep": global_timestep}) 

    # --- End of Training ---
    logger.log("Training finished.")
    if wandb_enabled and wandb.run is not None:
        wandb.finish()
    logger.close()


if __name__ == "__main__":
    main()
