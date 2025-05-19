"""
Minimal train.py main loop for DRL Shogi Client with Weights & Biases integration.
"""
from datetime import datetime
import os
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from dotenv import load_dotenv

import config
import wandb
from keisei.experience_buffer import ExperienceBuffer
from keisei.ppo_agent import PPOAgent
from keisei.shogi.shogi_core_definitions import Color, MoveTuple
from keisei.shogi.shogi_engine import ShogiGame
from keisei.utils import PolicyOutputMapper, TrainingLogger

# (Assuming evaluate_agent and its helpers are defined as in my previous correct response)
# Ensure evaluate_agent and its helpers are correctly structured without indentation issues
# and that the duplicated logging block at its end is removed.
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
    }

    def _select_eval_move_internal(
        current_agent: PPOAgent, current_game: ShogiGame, eval_as_color: Color,
        current_obs_np: np.ndarray, current_legal_shogi_moves: List[MoveTuple]
    ) -> MoveTuple:
        if current_game.current_player == eval_as_color:
            selected_move, _, _, _ = current_agent.select_action(
                current_obs_np, legal_shogi_moves=current_legal_shogi_moves, is_training=False
            )
        else:
            random_index = int(torch.randint(len(current_legal_shogi_moves), (1,)).item())
            selected_move = current_legal_shogi_moves[random_index]
        return selected_move

    def _update_results_after_game_internal(
        current_game: ShogiGame, eval_as_color: Color, game_moves: int,
        game_results: Dict[str, Any], game_info: Dict[str, Any]
    ):
        if current_game.winner == eval_as_color:
            game_results["wins"] += 1
        elif current_game.winner is None:
            game_results["draws"] += 1
        else:
            game_results["losses"] += 1
        game_results["game_lengths"].append(game_moves)
        reason = game_info.get("termination_reason", "unknown_eval_termination")
        if not current_game.game_over and game_moves >= config.MAX_MOVES_PER_GAME_EVAL:
            reason = "eval_max_moves_exceeded"
        game_results["termination_reasons_counts"][reason] = game_results["termination_reasons_counts"].get(reason, 0) + 1

    def _play_single_evaluation_game_internal(
        current_agent: PPOAgent, eval_as_color: Color, game_idx: int,
        current_results_dict: Dict[str, Any], current_logger: Optional[TrainingLogger]
    ):
        single_game = ShogiGame(max_moves_per_game=config.MAX_MOVES_PER_GAME_EVAL)
        current_obs_np = single_game.reset()
        game_moves = 0
        current_info: Dict[str, Any] = {}
        while not single_game.game_over:
            current_legal_shogi_moves = single_game.get_legal_moves()
            if not current_legal_shogi_moves:
                if current_logger:
                    current_logger.log(f"Warning: No legal moves for player {single_game.current_player} in eval game {game_idx + 1}.")
                if not single_game.game_over:
                    single_game.game_over = True
                    current_info["termination_reason"] = current_info.get("termination_reason", "stalemate_no_legal_moves")
                break
            selected_move = _select_eval_move_internal(current_agent, single_game, eval_as_color, current_obs_np, current_legal_shogi_moves)
            current_obs_np, _, _, current_info = single_game.make_move(selected_move)
            game_moves += 1
            if game_moves >= config.MAX_MOVES_PER_GAME_EVAL + 10: # Safety break
                if current_logger:
                    current_logger.log(f"Warning: Eval game {game_idx + 1} safety break at {game_moves} moves.")
                if not single_game.game_over:
                    single_game.game_over = True
                    current_info["termination_reason"] = "eval_loop_safety_timeout"
                break
        _update_results_after_game_internal(single_game, eval_as_color, game_moves, current_results_dict, current_info)

    def _log_final_evaluation_summary_internal(
        current_results_dict: Dict[str, Any], eval_as_color: Color,
        total_games_to_play: int, current_logger: Optional[TrainingLogger]
    ):
        total_games_played = current_results_dict["wins"] + current_results_dict["losses"] + current_results_dict["draws"]
        denom = total_games_played if total_games_played > 0 else 1.0
        win_r, loss_r, draw_r = current_results_dict["wins"]/denom, current_results_dict["losses"]/denom, current_results_dict["draws"]/denom
        avg_len = sum(current_results_dict["game_lengths"])/len(current_results_dict["game_lengths"]) if current_results_dict["game_lengths"] else 0.0
        msg = (
            f"Evaluation Complete (Agent as {eval_as_color.name}): Played {total_games_played}/{total_games_to_play} games.\n"
            f"  Wins: {current_results_dict['wins']} ({win_r:.2%}), Losses: {current_results_dict['losses']} ({loss_r:.2%}), Draws: {current_results_dict['draws']} ({draw_r:.2%})\n"
            f"  Avg Game Length: {avg_len:.2f} moves"
        )
        if current_logger:
            current_logger.log(msg)
        else:
            print(msg)
        if wandb.run is not None:
            log_d = {
                f"eval/{eval_as_color.name}/wins": current_results_dict["wins"], f"eval/{eval_as_color.name}/losses": current_results_dict["losses"],
                f"eval/{eval_as_color.name}/draws": current_results_dict["draws"], f"eval/{eval_as_color.name}/win_rate": win_r,
                f"eval/{eval_as_color.name}/loss_rate": loss_r, f"eval/{eval_as_color.name}/draw_rate": draw_r,
                f"eval/{eval_as_color.name}/avg_game_length": avg_len, f"eval/{eval_as_color.name}/total_games_evaluated": total_games_played,
            }
            for k,v in current_results_dict["termination_reasons_counts"].items():
                log_d[f"eval/{eval_as_color.name}/termination_{k}"] = v
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

    if wandb_enabled:
        try:
            run_name = f"ppo_lr{config.LEARNING_RATE}_batch{config.STEPS_PER_EPOCH}_{datetime.now().strftime('%y%m%d-%H%M%S')}"
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "shogi-drl"),
                entity=os.getenv("WANDB_ENTITY"),
                config=wandb_config_params, name=run_name, tags=["PPO", "Shogi", "SelfPlay"],
                notes="Training PPO agent for Shogi.", resume="allow",
            )
            logger.log(f"Weights & Biases initialized successfully for run: {run_name}")
            if wandb.run:
                logger.log(f"W&B Run URL: {wandb.run.url}")
        except Exception as e:
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

    buffer = ExperienceBuffer(
        buffer_size=config.STEPS_PER_EPOCH, gamma=config.GAMMA,
        lambda_gae=config.LAMBDA_GAE, device=config.DEVICE,
    )
    logger.log(f"Experience buffer initialized with size: {config.STEPS_PER_EPOCH}")

    obs_np = game.reset()
    episode_reward = 0.0
    episode_steps = 0
    total_episodes_completed = 0
    last_eval_episode_trigger = 0
    last_save_episode_trigger = 0

    for global_timestep in range(1, config.TOTAL_TIMESTEPS + 1):
        current_player_for_action = game.current_player
        legal_shogi_moves = game.get_legal_moves()

        if not legal_shogi_moves:
            logger.log(f"CRITICAL: No legal moves for player {current_player_for_action.name} at timestep {global_timestep}. Game state: {game.to_string()}") # <<<<< FIX 3: Removed perspective
            termination_reason = "no_legal_moves_at_turn_start"
            if wandb_enabled and wandb.run is not None:
                wandb.log({
                    "episode/reward": episode_reward, "episode/length": episode_steps,
                    "episode/number": total_episodes_completed,
                    f"episode/termination/{termination_reason}": 1,
                }, commit=False)
            obs_np = game.reset()
            episode_reward = 0.0
            episode_steps = 0
            total_episodes_completed += 1
            continue

        obs_tensor_for_buffer = torch.from_numpy(obs_np).float().to(agent.device) # Shape (46, 9, 9)

        selected_shogi_move, selected_policy_index, log_prob, value_estimate = agent.select_action(
            obs_np, legal_shogi_moves=legal_shogi_moves, is_training=True
        )

        if config.PRINT_GAME_REAL_TIME:
            try:
                move_str = policy_mapper.shogi_move_to_usi(selected_shogi_move)
            except AttributeError:
                move_str = str(selected_shogi_move) # Fallback
            print(f"AI Train Move ({current_player_for_action.name}): {move_str}")

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
            if wandb_enabled and wandb.run is not None:
                log_data_ep = {
                    "episode/reward": episode_reward, "episode/length": episode_steps,
                    "episode/number": total_episodes_completed,
                    f"episode/termination/{termination_reason}": 1,
                }
                if game.winner == Color.BLACK:
                    log_data_ep["episode/black_wins_training"] = 1
                elif game.winner == Color.WHITE:
                    log_data_ep["episode/white_wins_training"] = 1
                elif game.winner is None:
                    log_data_ep["episode/draws_training"] = 1
                wandb.log(log_data_ep, commit=False)
            obs_np = game.reset()
            episode_reward = 0.0
            episode_steps = 0

        if global_timestep % config.STEPS_PER_EPOCH == 0:
            # Ensure obs_np is NumPy array for get_value, as Pylance error might be misleading
            # but it should be correct given current flow.
            last_value = agent.get_value(obs_np) # <<<<< FIX 6: Confirm obs_np is ndarray
            buffer.compute_advantages_and_returns(last_value)

            ppo_metrics = agent.learn(buffer)
            buffer.clear()
            logger.log(f"PPO Update at timestep {global_timestep}")

            # For FIX 2 (SyntaxError): Carefully inspect this block and lines before it in your actual file.
            # The simplified version from your paste:
            # wandb_metrics_log = {f"train/{k}": v for k, v in metrics.items()} if metrics else {}
            # My more verbose but potentially safer version:
            train_metrics_to_log = {}
            if isinstance(ppo_metrics, dict):
                for key, value in ppo_metrics.items():
                    train_metrics_to_log[f"train/{key.replace('ppo/', '')}"] = value
            else:
                logger.log(f"Warning: PPO metrics not a dict: {ppo_metrics}")

            if wandb_enabled and wandb.run is not None:
                if train_metrics_to_log: # Log only if not empty
                    wandb.log(train_metrics_to_log, commit=False)

            if config.EVAL_FREQ_EPISODES > 0 and \
               total_episodes_completed > 0 and \
               total_episodes_completed % config.EVAL_FREQ_EPISODES == 0 : # Colon was present in my generated, check user's file
                logger.log(f"Running evaluation at episode {total_episodes_completed} (timestep {global_timestep})")
                evaluate_agent(agent, agent_color_to_eval_as=Color.BLACK, num_games=config.EVAL_NUM_GAMES, logger=logger)
                evaluate_agent(agent, agent_color_to_eval_as=Color.WHITE, num_games=config.EVAL_NUM_GAMES, logger=logger)
                # This logic for last_eval_episode_trigger was in my suggestion, ensure it's there or adapted
                # last_eval_episode_trigger = total_episodes_completed

            if config.SAVE_FREQ_EPISODES > 0 and \
               total_episodes_completed > 0 and \
               total_episodes_completed % config.SAVE_FREQ_EPISODES == 0:
                save_path = os.path.join(
                    config.MODEL_DIR,
                    f"ppo_shogi_ep{total_episodes_completed}_ts{global_timestep}.pth"
                )
                agent.save_model(save_path)
                if wandb_enabled and wandb.run is not None:
                    model_artifact = wandb.Artifact(
                        name=f"{wandb.run.name}-model-ep{total_episodes_completed}", # Or your preferred naming
                        type="model",
                        description=f"PPO Shogi Agent model checkpoint after {total_episodes_completed} episodes.",
                        metadata={
                            **wandb_config_params,  # <<< THIS IS THE CHANGE
                            "episode": total_episodes_completed,
                            "global_timestep": global_timestep
                        }
                    )
                    model_artifact.add_file(save_path)
                    wandb.log_artifact(model_artifact)
                    logger.log(f"Model saved as WandB Artifact: {model_artifact.name}")
                # This logic for last_save_episode_trigger was in my suggestion, ensure it's there or adapted
                # last_save_episode_trigger = total_episodes_completed

            if wandb_enabled and wandb.run is not None:
                wandb.log({"global_timestep": global_timestep}, commit=True)

        elif wandb_enabled and wandb.run is not None: # Log some progress even if not an update step
            wandb.log({"global_timestep": global_timestep,
                        "progress/buffer_fill_ratio": len(buffer) / config.STEPS_PER_EPOCH if config.STEPS_PER_EPOCH > 0 else 0
                       }, commit=True)

    logger.log("Training finished.")
    if wandb_enabled and wandb.run is not None:
        logger.log("Starting final evaluation (Agent as BLACK)...")
        evaluate_agent(agent, agent_color_to_eval_as=Color.BLACK, num_games=config.EVAL_NUM_GAMES * 2, logger=logger)
        logger.log("Starting final evaluation (Agent as WHITE)...")
        evaluate_agent(agent, agent_color_to_eval_as=Color.WHITE, num_games=config.EVAL_NUM_GAMES * 2, logger=logger)
        wandb.log({}, commit=True) # Final commit for eval logs
        wandb.finish()
    logger.close()

if __name__ == "__main__":
    main()
