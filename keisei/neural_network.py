"""
Minimal ActorCritic neural network for DRL Shogi Client (dummy forward pass).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Actor-Critic neural network for Shogi RL agent (PPO-ready)."""

    def __init__(self, input_channels: int, num_actions_total: int):
        """Deeper CNN with batch norm and dropout for Shogi."""
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.policy_head = nn.Linear(64 * 9 * 9, num_actions_total)
        self.value_head = nn.Linear(64 * 9 * 9, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def get_action_and_value(self, obs, legal_indices=None):
        """
        Given an observation (and optional legal action indices), return a sampled action, its log probability,
        and value estimate.
        """
        policy_logits, value = self.forward(obs)
        if legal_indices is not None:
            mask = torch.zeros_like(policy_logits)
            mask[:, legal_indices] = 1
            policy_logits = policy_logits.masked_fill(mask == 0, float("-inf"))
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate the log probabilities, entropy, and value for given observations and actions.
        """
        policy_logits, value = self.forward(obs)
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value.squeeze(-1)
