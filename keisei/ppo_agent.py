"""
Minimal PPOAgent for DRL Shogi Client.
"""

from typing import Optional
import torch
import numpy as np
from keisei.neural_network import ActorCritic


class PPOAgent:
    """Proximal Policy Optimization agent for Shogi (PPO logic)."""

    def __init__(
        self,
        input_channels: int,
        num_actions_total: int,
        policy_output_mapper,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """
        Initialize the PPOAgent with model, optimizer, and PPO hyperparameters.
        """
        self.model = ActorCritic(input_channels, num_actions_total)
        self.policy_output_mapper = policy_output_mapper
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_actions_total = num_actions_total
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(
        self, obs: torch.Tensor, legal_indices: Optional[np.ndarray] = None
    ) -> int:
        """
        Select an action index given an observation and optional legal action indices.
        """
        self.model.eval()
        with torch.no_grad():
            legal_idx_tensor = None
            if legal_indices is not None:
                legal_idx_tensor = torch.as_tensor(legal_indices, dtype=torch.long)
            action, _, _ = self.model.get_action_and_value(obs, legal_idx_tensor)
        return int(action.item())

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE) returns for PPO.
        """
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_update(
        self, obs, actions, old_log_probs, returns, advantages, epochs=4, batch_size=32
    ):
        """
        Perform PPO update over multiple epochs and mini-batches.
        """
        obs = torch.stack(obs)
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(old_log_probs)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        n = len(obs)
        for _ in range(epochs):
            idxs = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_idx = idxs[start:end]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                log_probs, entropy, values = self.model.evaluate_actions(
                    batch_obs, batch_actions
                )
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (batch_returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
