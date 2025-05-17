"""
Minimal ActorCritic neural network for DRL Shogi Client (dummy forward pass).
"""

import torch.nn as nn


class ActorCritic(nn.Module):
    """Actor-Critic neural network for Shogi RL agent."""

    def __init__(self, input_channels: int, num_actions_total: int):
        """Initialize the ActorCritic network with convolutional and linear layers."""
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.policy_head = nn.Linear(16 * 9 * 9, num_actions_total)
        self.value_head = nn.Linear(16 * 9 * 9, 1)

    def forward(self, x):
        """Forward pass: returns policy logits and value estimate."""
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
