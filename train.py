"""
Minimal train.py main loop for DRL Shogi Client with Weights & Biases integration.
"""
import os
import time # For real-time print delay
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque # For rolling window statistics

from dotenv import load_dotenv
import torch
import numpy as np

import wandb # Weights & Biases
import wandb.errors # For handling W&B errors

import config # Your project's configuration file

from keisei.experience_buffer import ExperienceBuffer
from keisei.ppo_agent import PPOAgent
from keisei.shogi.shogi_core_definitions import Color, MoveTuple
from keisei.shogi.shogi_engine import ShogiGame
from keisei.utils import PolicyOutputMapper, TrainingLogger


def evaluate_agent(
    agent: PPOAgent,
    agent_color_to_eval_as: Color,
    num_games: int = 5,
    logger: Optional[TrainingLogger] = None
):
    """
    Run evaluation games with the current agent playing as a specific color
    against a random opponent. Logs win/draw/loss stats.
    """
    results = {
        "wins": 0, "draws": 0, "losses": 0,
        "game_lengths": [], "termination_reasons_counts": {},
    } # type: Dict[str, Any]


    def _select_eval_move_internal(
        current_agent: PPOAgent, current_game: ShogiGame, eval_as_color: Color,
        current_obs_np: np.ndarray, current_legal_shogi_moves: List[MoveTuple]
    ) -> Optional[MoveTuple]: # Can return None if opponent has no moves
        if current_game.current_player == eval_as_color:
            selected_move, _, _, _ = current_agent.select_action(
                current_obs_np, legal_shogi_moves=current_legal_shogi_moves, is_training=False
            )
        else: # Opponent's turn (plays randomly)
            if not current_legal_shogi_moves: # Should be caught by the main game loop's check
                return None
            random_index = int(torch.randint(len(current_legal_shogi_moves), (1,)).item())
            selected_move = current_legal_shogi_moves[random_index]
        return selected_move

    def _update_results_after_game_internal(
        current_game: ShogiGame, eval_as_color: Color, game_moves: int,
        game_results: Dict[str, Any], game_info: Dict[str, Any],
        game_idx_for_log: int, current_logger: Optional[TrainingLogger]
    ):
        final_reason = game_info.get("termination_reason", "unknown_eval_termination")

        # If game loop broke due to external limits but game itself didn't declare an end
        if not current_game.game_over:
            if game_moves >= config.MAX_MOVES_PER_GAME_EVAL:
                final_reason = "eval_max_moves_hit_in_loop"
            else:
                final_reason = "eval_loop_ended_unexpectedly"
            # Consider such games draws if no winner determined by game logic
            if current_game.winner is None:
                game_results["draws"] += 1
            elif current_game.winner == eval_as_color: # Should not happen if game not over
                game_results["wins"] +=1
            else:
                game_results["losses"] +=1
            if current_logger:
                current_logger.log(f"Eval game {game_idx_for_log+1} ended by eval loop limit ({final_reason}) at {game_moves} moves.")
        else: # Game ended naturally
            if current_game.winner == eval_as_color:
                game_results["wins"] += 1
            elif current_game.winner is None: # Draw (sennichite, max_moves by game itself, etc.)
                game_results["draws"] += 1
            else: # Opponent won
                game_results["losses"] += 1

        game_results["game_lengths"].append(game_moves)
        game_results["termination_reasons_counts"][final_reason] = game_results["termination_reasons_counts"].get(final_reason, 0) + 1


    def _play_single_evaluation_game_internal(
        current_agent: PPOAgent, eval_as_color: Color, game_idx: int,
        current_results_dict: Dict[str, Any], current_logger: Optional[TrainingLogger]
    ):
        single_game = ShogiGame(max_moves_per_game=config.MAX_MOVES_PER_GAME_EVAL)
        current_obs_np = single_game.reset()
        game_moves = 0
        current_info: Dict[str, Any] = {} # Store info from the last make_move

        while not single_game.game_over:
            current_legal_shogi_moves = single_game.get_legal_moves()

            if not current_legal_shogi_moves:
                if current_logger:
                    current_logger.log(f"Warning: No legal moves for player {single_game.current_player} in eval game {game_idx + 1}. Forcing game over.")
                if not single_game.game_over:
                    single_game.game_over = True # Game ends here
                    current_info["termination_reason"] = current_info.get("termination_reason", "stalemate_or_no_moves_eval")
                    # ShogiGame should determine winner (e.g. if stalemate or checkmate leading to no moves)
                break

            selected_move = _select_eval_move_internal(current_agent, single_game, eval_as_color, current_obs_np, current_legal_shogi_moves)
            if selected_move is None: # Opponent had no moves
                if not single_game.game_over: # Should have been caught by the game ending
                    single_game.game_over = True
                    current_info["termination_reason"] = "opponent_no_legal_moves_eval"
                break

            current_obs_np, _, _, current_info = single_game.make_move(selected_move)
            game_moves += 1

            # Safety break for the eval game loop, ShogiGame.max_moves_per_game should handle it first
            if game_moves >= config.MAX_MOVES_PER_GAME_EVAL + 10: # Generous safety margin
                if current_logger:
                    current_logger.log(f"Warning: Eval game {game_idx + 1} hit external safety break at {game_moves} moves.")
                if not single_game.game_over: # If game didn't self-terminate via its own max_moves
                    single_game.game_over = True
                    current_info["termination_reason"] = "eval_loop_safety_timeout"
                break

        _update_results_after_game_internal(single_game, eval_as_color, game_moves, current_results_dict, current_info, game_idx, current_logger)

    def _log_final_evaluation_summary_internal(
        current_results_dict: Dict[str, Any], eval_as_color: Color,
        total_games_to_play: int, current_logger: Optional[TrainingLogger]
    ):
        total_games_played = current_results_dict["wins"] + current_results_dict["losses"] + current_results_dict["draws"]
        denominator = total_games_played if total_games_played > 0 else 1.0 # Avoid division by zero

        win_r = current_results_dict["wins"] / denominator
        loss_r = current_results_dict["losses"] / denominator
        draw_r = current_results_dict["draws"] / denominator
        avg_len = sum(current_results_dict["game_lengths"]) / len(current_results_dict["game_lengths"]) \
            if current_results_dict["game_lengths"] else 0.0

        msg_summary = (
            f"Evaluation Complete (Agent as {eval_as_color.name}): Played {total_games_played}/{total_games_to_play} games.\n"
            f"  Wins: {current_results_dict['wins']} ({win_r:.2%}), Losses: {current_results_dict['losses']} ({loss_r:.2%}), Draws: {current_results_dict['draws']} ({draw_r:.2%})\n"
            f"  Avg Game Length: {avg_len:.2f} moves"
        )
        if current_logger:
            current_logger.log(msg_summary)
        else: print(msg_summary)

        if wandb.run is not None:
            log_d = {
                f"eval/{eval_as_color.name}/wins_count": current_results_dict["wins"],
                f"eval/{eval_as_color.name}/losses_count": current_results_dict["losses"],
                f"eval/{eval_as_color.name}/draws_count": current_results_dict["draws"],
                f"eval/{eval_as_color.name}/win_rate": win_r,
                f"eval/{eval_as_color.name}/loss_rate": loss_r,
                f"eval/{eval_as_color.name}/draw_rate": draw_r,
                f"eval/{eval_as_color.name}/avg_game_length": avg_len,
                f"eval/{eval_as_color.name}/total_games_evaluated": total_games_played,
            }
            for k,v_count in current_results_dict["termination_reasons_counts"].items():
                log_d[f"eval/{eval_as_color.name}/termination_{k}_count"] = v_count # Add _count for clarity
            wandb.log(log_d, commit=False)


    if logger:
        logger.log(f"Starting evaluation: Agent plays as {agent_color_to_eval_as.name} for {num_games} games.")
    for i in range(num_games):
        _play_single_evaluation_game_internal(agent, agent_color_to_eval_as, i, results, logger)
    _log_final_evaluation_summary_internal(results, agent_color_to_eval_as, num_games, logger)


def main():
    """Main training loop for PPO DRL Shogi Client."""

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_enabled = bool(wandb_api_key)

    logger = TrainingLogger(config.LOG_FILE)
    logger.log("Starting DRL Shogi Client Training")

    if not wandb_enabled:
        logger.log("WANDB_API_KEY not found. W&B logging will be disabled.")
    elif wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    wandb_config_params = {
        key: getattr(config, key) for key in dir(config)
        if not key.startswith("__") and not callable(getattr(config, key)) and key.isupper()
    }
    # Add any other dynamic configs if needed
    wandb_config_params["script_start_time"] = datetime.now().isoformat()


    if wandb_enabled:
        try:
            run_name = f"ppo_shogi_{datetime.now().strftime('%y%m%d-%H%M%S')}"
            if hasattr(config, 'LEARNING_RATE') and hasattr(config, 'STEPS_PER_EPOCH'): # Check if attributes exist
                run_name = f"ppo_lr{config.LEARNING_RATE}_batch{config.STEPS_PER_EPOCH}_{datetime.now().strftime('%y%m%d-%H%M%S')}"

            wandb.init(
                project=os.getenv("WANDB_PROJECT", "shogi-drl"),
                entity=os.getenv("WANDB_ENTITY"), # Set your W&B entity (username or team)
                config=wandb_config_params,
                name=run_name,
                tags=["PPO", "Shogi", "SelfPlay"],
                notes="Training PPO agent for Shogi with enhanced logging.",
                resume="allow", # Allow resuming runs if an ID is provided or auto-resumed
                # id=wandb.util.generate_id(), # Optionally generate a fixed ID for resuming
            )
            logger.log(f"Weights & Biases initialized successfully for run: {wandb.run.name if wandb.run else 'Unknown'}")
            if wandb.run:
                logger.log(f"W&B Run URL: {wandb.run.url}")
        except (wandb.errors.AuthenticationError,
                wandb.errors.CommError,
                wandb.errors.UsageError) as e: # Catch specific W&B initialization errors
            logger.log(f"Error initializing Weights & Biases: {e}. Proceeding without W&B logging.")
            wandb_enabled = False

    logger.log(f"Using device: {config.DEVICE}")

    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
        logger.log(f"Created model directory: {config.MODEL_DIR}")

    game = ShogiGame(max_moves_per_game=config.MAX_MOVES_PER_GAME)
    policy_mapper = PolicyOutputMapper()

    agent = PPOAgent(
        input_channels=config.INPUT_CHANNELS, policy_output_mapper=policy_mapper,
        learning_rate=config.LEARNING_RATE, gamma=config.GAMMA, clip_epsilon=config.CLIP_EPSILON,
        ppo_epochs=config.PPO_EPOCHS, minibatch_size=config.MINIBATCH_SIZE,
        value_loss_coeff=config.VALUE_LOSS_COEFF, entropy_coef=config.ENTROPY_COEFF, device=config.DEVICE,
    )
    logger.log("PPO Agent initialized.")
    if wandb_enabled and wandb.run:
        wandb.watch(agent.model, log="all", log_freq=config.STEPS_PER_EPOCH) # Watch model gradients

    buffer = ExperienceBuffer(
        buffer_size=config.STEPS_PER_EPOCH, gamma=config.GAMMA,
        lambda_gae=config.LAMBDA_GAE, device=config.DEVICE,
    )
    logger.log(f"Experience buffer initialized with size: {config.STEPS_PER_EPOCH}")

    obs_np = game.reset()
    episode_reward = 0.0
    episode_steps = 0
    total_episodes_completed = 0

    # For rolling window statistics
    rolling_window_size = getattr(config, "ROLLING_WINDOW_SIZE", 100) # Default to 100 if not in config
    recent_episode_rewards: deque[float] = deque(maxlen=rolling_window_size)
    recent_episode_lengths: deque[int] = deque(maxlen=rolling_window_size)
    recent_training_outcomes: deque[int] = deque(maxlen=rolling_window_size) # Store 1 for Black win, -1 for White win, 0 for Draw

    last_eval_episode_count = 0 # Track episodes for eval frequency
    last_save_episode_count = 0 # Track episodes for save frequency

    for global_timestep in range(1, config.TOTAL_TIMESTEPS + 1):
        current_player_for_action = game.current_player
        legal_shogi_moves = game.get_legal_moves()

        if not legal_shogi_moves:
            logger.log(f"CRITICAL: No legal moves for player {current_player_for_action.name} at timestep {global_timestep}. Game state: {game.to_string()}")
            termination_reason = "no_legal_moves_at_turn_start"
            if wandb_enabled and wandb.run is not None:
                wandb.log({
                    "episode/reward": episode_reward, "episode/length": episode_steps,
                    "episode/number": total_episodes_completed,
                    f"episode/termination/{termination_reason}_count": 1, # Add _count for consistency
                }, commit=False)

            obs_np = game.reset()
            episode_reward = 0.0
            episode_steps = 0
            total_episodes_completed += 1
            # Log outcome for rolling stats if this counts as an episode end
            recent_episode_rewards.append(0) # Or appropriate reward for this outcome
            recent_episode_lengths.append(0) # Or actual steps if any
            recent_training_outcomes.append(0) # Assuming draw or special case
            continue

        obs_tensor_for_buffer = torch.from_numpy(obs_np).float().to(agent.device)

        selected_shogi_move, selected_policy_index, log_prob, value_estimate = agent.select_action(
            obs_np, legal_shogi_moves=legal_shogi_moves, is_training=True
        )

        if getattr(config, 'PRINT_GAME_REAL_TIME', False): # Safely access config
            try:
                move_str = policy_mapper.shogi_move_to_usi(selected_shogi_move)
            except (ValueError, TypeError): # Be more specific about expected errors from conversion
                move_str = str(selected_shogi_move)
            print(f"AI Train Move ({current_player_for_action.name}): {move_str}")
            if getattr(config, 'REAL_TIME_PRINT_DELAY', 0) > 0:
                time.sleep(config.REAL_TIME_PRINT_DELAY)


        next_obs_np, reward, done, move_info = game.make_move(selected_shogi_move)
        episode_reward += reward
        episode_steps += 1

        buffer.add(
            obs=obs_tensor_for_buffer, action=selected_policy_index, reward=reward,
            value=value_estimate, log_prob=log_prob, done=done,
        )
        obs_np = next_obs_np

        if done:
            total_episodes_completed += 1
            termination_reason = move_info.get('termination_reason', 'unknown_episode_end')
            logger.log(
                f"Episode {total_episodes_completed} finished after {episode_steps} steps. "
                f"Reward: {episode_reward:.2f}. Last move by {current_player_for_action.name}. "
                f"Termination: {termination_reason}"
            )

            recent_episode_rewards.append(episode_reward)
            recent_episode_lengths.append(episode_steps)
            outcome_val = 0 # Draw
            if game.winner == Color.BLACK:
                outcome_val = 1
            elif game.winner == Color.WHITE:
                outcome_val = -1
            recent_training_outcomes.append(outcome_val)

            if wandb_enabled and wandb.run is not None:
                log_data_ep = {
                    "episode/reward_raw": episode_reward, # Log raw reward
                    "episode/length_raw": episode_steps,  # Log raw length
                    "episode/number_cumulative": total_episodes_completed,
                    f"episode/termination/{termination_reason}_count": 1,
                }
                if game.winner == Color.BLACK:
                    log_data_ep["episode/black_wins_count_training"] = 1
                elif game.winner == Color.WHITE:
                    log_data_ep["episode/white_wins_count_training"] = 1
                elif game.winner is None:
                    log_data_ep["episode/draws_count_training"] = 1
                wandb.log(log_data_ep, commit=False)

            obs_np = game.reset()
            episode_reward = 0.0
            episode_steps = 0

        # PPO Update Phase (and periodic logging/eval/save)
        if global_timestep % config.STEPS_PER_EPOCH == 0:
            last_value = agent.get_value(obs_np)
            buffer.compute_advantages_and_returns(last_value)

            ppo_metrics = agent.learn(buffer)
            buffer.clear()
            logger.log(f"PPO Update at timestep {global_timestep}")

            # Prepare and log training metrics
            train_metrics_to_log = {}
            if isinstance(ppo_metrics, dict):
                for key, value in ppo_metrics.items():
                    train_metrics_to_log[f"train/{key.replace('ppo/', '')}"] = value
            else:
                logger.log(f"Warning: PPO metrics not a dict: {ppo_metrics}")

            if wandb_enabled and wandb.run is not None:
                if train_metrics_to_log:
                    wandb.log(train_metrics_to_log, commit=False) # Commit happens later

                # Log rolling window stats
                if len(recent_episode_rewards) > 0:
                    wandb.log({
                        "episode/reward_rolling_avg": sum(recent_episode_rewards) / len(recent_episode_rewards),
                        "episode/length_rolling_avg": sum(recent_episode_lengths) / len(recent_episode_lengths),
                    }, commit=False)
                if len(recent_training_outcomes) > 0:
                    total_in_window = len(recent_training_outcomes)
                    black_wins_in_window = sum(1 for o in recent_training_outcomes if o == 1)
                    white_wins_in_window = sum(1 for o in recent_training_outcomes if o == -1)
                    draws_in_window = sum(1 for o in recent_training_outcomes if o == 0)
                    wandb.log({
                        "episode/rolling/black_win_rate": black_wins_in_window / total_in_window,
                        "episode/rolling/white_win_rate": white_wins_in_window / total_in_window,
                        "episode/rolling/draw_rate": draws_in_window / total_in_window,
                    }, commit=False)

            # Evaluation
            if config.EVAL_FREQ_EPISODES > 0 and \
               total_episodes_completed >= last_eval_episode_count + config.EVAL_FREQ_EPISODES:
                logger.log(f"Running evaluation at episode {total_episodes_completed} (timestep {global_timestep})")
                evaluate_agent(agent, agent_color_to_eval_as=Color.BLACK, num_games=config.EVAL_NUM_GAMES, logger=logger)
                evaluate_agent(agent, agent_color_to_eval_as=Color.WHITE, num_games=config.EVAL_NUM_GAMES, logger=logger)
                last_eval_episode_count = total_episodes_completed

            # Model Saving
            if config.SAVE_FREQ_EPISODES > 0 and \
               total_episodes_completed >= last_save_episode_count + config.SAVE_FREQ_EPISODES:
                save_path = os.path.join(
                    config.MODEL_DIR,
                    f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth"
                )
                agent.save_model(save_path) # Includes optimizer state

                if wandb_enabled and wandb.run is not None:
                    artifact_name = f"{wandb.run.name}-model-ep{total_episodes_completed}"
                    # Check if artifact already exists with this name to version it
                    try:
                        # Try to fetch to see if it exists, though this doesn't directly give versions.
                        # Better to use aliases or increment version numbers in name if needed.
                        # For simplicity, allow overwriting or new artifact based on name.
                        pass
                    except wandb.errors.CommError: # If artifact doesn't exist
                        pass

                    model_artifact = wandb.Artifact(
                        name=artifact_name,
                        type="model",
                        description=f"PPO Shogi Agent model after {total_episodes_completed} episodes.",
                        metadata={**wandb_config_params, "episode": total_episodes_completed, "global_timestep": global_timestep}
                    )
                    model_artifact.add_file(save_path)
                    wandb.log_artifact(model_artifact)
                    logger.log(f"Model saved as WandB Artifact: {model_artifact.name}")
                last_save_episode_count = total_episodes_completed

            # Commit all logs for this PPO update step
            if wandb_enabled and wandb.run is not None:
                wandb.log({"global_timestep": global_timestep}, commit=True)

        # Log global_timestep and buffer fill ratio on non-update steps too for progress tracking
        elif wandb_enabled and wandb.run is not None and global_timestep % 100 == 0: # Log less frequently here
            wandb.log({"global_timestep": global_timestep,
                        "progress/buffer_fill_ratio": len(buffer) / config.STEPS_PER_EPOCH if config.STEPS_PER_EPOCH > 0 else 0.0
                       }, commit=True)

    logger.log("Training finished.")
    if wandb_enabled and wandb.run is not None:
        logger.log("Starting final evaluation (Agent as BLACK)...")
        evaluate_agent(agent, agent_color_to_eval_as=Color.BLACK, num_games=config.EVAL_NUM_GAMES * 2, logger=logger)
        logger.log("Starting final evaluation (Agent as WHITE)...")
        evaluate_agent(agent, agent_color_to_eval_as=Color.WHITE, num_games=config.EVAL_NUM_GAMES * 2, logger=logger)

        wandb.log({}, commit=True) # Final commit for any pending eval logs
        wandb.finish()
    logger.close()


if __name__ == "__main__":
    main()
