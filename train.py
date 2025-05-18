"""
Minimal train.py main loop for DRL Shogi Client (random agent, no learning).
"""

import torch
from keisei.shogi.shogi_engine import ShogiGame
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
        game.reset()
        for _ in range(100):  # Max moves per game
            obs = torch.tensor(game.get_observation(), dtype=torch.float32).unsqueeze(0)
            idx = agent.select_action(obs)
            # For now, use a dummy move mapping (since PolicyOutputMapper is minimal)
            # In a real setup, map idx to a legal move
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            move = legal_moves[0]  # Always pick the first legal move for demo
            game.make_move(move)
            if game.game_over:
                break
        # For demo, treat all games as draws
        draws += 1
    msg = f"Evaluation: {num_games} games | Wins: {wins} | Draws: {draws} | Losses: {losses}"
    if logger:
        logger.log(msg)
    else:
        print(msg)


def main():
    """Minimal main loop for random-agent DRL Shogi Client with logging and evaluation."""
    game = ShogiGame()
    policy_mapper = PolicyOutputMapper()
    agent = PPOAgent(
        config.INPUT_CHANNELS, policy_mapper.get_total_actions(), policy_mapper
    )
    buffer = ExperienceBuffer(buffer_size=10)
    logger = TrainingLogger("logs/train.log")
    for episode in range(2):
        game.reset()
        obs = torch.tensor(game.get_observation(), dtype=torch.float32).unsqueeze(0)
        for step in range(5):
            idx = agent.select_action(obs)
            buffer.add(obs, idx, 0.0)
        logger.log(f"Episode {episode+1}: Buffer size: {len(buffer)}")
    evaluate_agent(agent, num_games=5, logger=logger)
    logger.close()


if __name__ == "__main__":
    main()
