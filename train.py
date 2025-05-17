"""
Minimal train.py main loop for DRL Shogi Client (random agent, no learning).
"""

import torch
from keisei.shogi_engine import ShogiGame
from keisei.utils import PolicyOutputMapper
from keisei.ppo_agent import PPOAgent
from keisei.experience_buffer import ExperienceBuffer
import config


def main():
    """Minimal main loop for random-agent DRL Shogi Client."""
    game = ShogiGame()
    policy_mapper = PolicyOutputMapper()
    agent = PPOAgent(
        config.INPUT_CHANNELS, policy_mapper.get_total_actions(), policy_mapper
    )
    buffer = ExperienceBuffer(buffer_size=10)
    for episode in range(2):
        game.reset()
        obs = torch.tensor(game.get_observation(), dtype=torch.float32).unsqueeze(0)
        for step in range(5):
            idx = agent.select_action(obs)
            buffer.add(obs, idx, 0.0)
    print(f"Buffer size: {len(buffer)}")


if __name__ == "__main__":
    main()
