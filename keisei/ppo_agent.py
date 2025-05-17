"""
Minimal PPOAgent for DRL Shogi Client (random action selection for now).
"""

import torch
import numpy as np
from keisei.neural_network import ActorCritic
from typing import Optional


class PPOAgent:
    """Proximal Policy Optimization agent for Shogi (random action selection for now)."""

    def __init__(
        self,
        input_channels: int,
        num_actions_total: int,
        policy_output_mapper,
        learning_rate: float = 3e-4,
    ):
        self.model = ActorCritic(input_channels, num_actions_total)
        self.policy_output_mapper = policy_output_mapper
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_actions_total = num_actions_total

    def select_action(
        self, obs: torch.Tensor, legal_indices: Optional[np.ndarray] = None
    ) -> int:
        """Select a random legal action (or any action if legal_indices is None)."""
        if legal_indices is not None and len(legal_indices) > 0:
            idx = np.random.choice(legal_indices)
        else:
            idx = np.random.randint(self.num_actions_total)
        return int(idx)
