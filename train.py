"""
Minimal train.py main loop for DRL Shogi Client (random agent, no learning).
"""

import torch
from keisei.shogi.shogi_engine import ShogiGame
from keisei.shogi.shogi_core_definitions import Color # Added import
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
        obs_np = game.reset() # Get initial observation
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
            obs_np, reward, done, info = game.make_move(selected_shogi_move) # Corrected unpacking

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
                    else: # Should be a draw if winner is None but game_over is True
                        draws +=1
                else: # No winner explicitly set, but game over (e.g. sennichite, max_moves)
                    draws += 1 # Or handle based on specific game end condition
                break
        # If loop finishes due to max_moves without game.game_over, count as draw
        if not game.game_over:
            draws += 1

    # msg = f"Evaluation: {num_games} games | Wins: {wins} | Draws: {draws} | Losses: {losses}"
    # More detailed logging
    total_played = wins + losses + draws
    if total_played == 0: # Avoid division by zero if num_games was 0 or all games aborted early
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
    """Minimal main loop for random-agent DRL Shogi Client with logging and evaluation."""
    game = ShogiGame()
    policy_mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=config.INPUT_CHANNELS,
        policy_output_mapper=policy_mapper,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
    )
    buffer = ExperienceBuffer(buffer_size=10)
    logger = TrainingLogger("logs/train.log")
    # Main training loop
    for episode_num in range(2):
        obs_np = game.reset()
        # Ensure obs_np is correctly shaped if ShogiGame.reset() doesn't return it directly
        # For example, if reset just resets state and observation is fetched separately:
        # obs_np = game.get_observation() 
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        done = False
        current_episode_steps = 0
        episode_reward = 0

        while not done and current_episode_steps < config.MAX_MOVES_PER_GAME:
            legal_shogi_moves = game.get_legal_moves()
            if not legal_shogi_moves:
                # Handle game end states like checkmate or stalemate
                # print(f"No legal moves for player {game.current_player}. Game likely ended.")
                break # Exit inner loop if no legal moves

            # Get action from agent
            # Ensure obs is in the correct format (np.ndarray) for select_action
            # If obs is already a tensor, convert it: obs.squeeze(0).cpu().numpy()
            # Assuming obs_np is the correct NumPy array observation:
            selected_shogi_move, policy_idx, log_prob_action, value_pred = agent.select_action(
                obs_np, legal_shogi_moves=legal_shogi_moves, is_training=True
            )
    
            # Step the environment
            # next_obs_np, reward, done, _ = game.make_move(selected_shogi_move)
            next_obs_np, reward, done, info = game.make_move(selected_shogi_move) # Corrected unpacking
            episode_reward += reward

            # Add experience to buffer
            # Ensure obs_np is the observation *before* the action
            buffer.add(obs_np, policy_idx, reward, log_prob_action, value_pred)

            obs_np = next_obs_np
            obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            current_episode_steps += 1

        logger.log(f"Episode {episode_num+1}: Buffer size: {len(buffer)}")
    evaluate_agent(agent, num_games=5, logger=logger)
    logger.close()


if __name__ == "__main__":
    main()
