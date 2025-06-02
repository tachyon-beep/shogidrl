"""
resnet_tower.py: ActorCriticResTower model for Keisei Shogi with SE block support.
"""
import sys  # For stderr
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: Optional[float] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio) if se_ratio else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se:
            out = self.se(out)  # pylint: disable=not-callable
        out += x
        return F.relu(out)

class ActorCriticResTower(nn.Module):
    def __init__(self, input_channels: int, num_actions_total: int, tower_depth: int = 9, tower_width: int = 256, se_ratio: Optional[float] = None):
        super().__init__()
        self.stem = nn.Conv2d(input_channels, tower_width, 3, padding=1)
        self.bn_stem = nn.BatchNorm2d(tower_width)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(tower_width, se_ratio) for _ in range(tower_depth)
        ])
        # Slim policy head: 2 planes, then flatten, then linear
        self.policy_head = nn.Sequential(
            nn.Conv2d(tower_width, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, num_actions_total)
        )
        # Slim value head: 2 planes, then flatten, then linear
        self.value_head = nn.Sequential(
            nn.Conv2d(tower_width, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 1)
        )
    def forward(self, x):
        x = F.relu(self.bn_stem(self.stem(x)))
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an observation (and optional legal action mask), return a sampled or deterministically chosen action,
        its log probability, and value estimate.
        Args:
            obs: Input observation tensor.
            legal_mask: Optional boolean tensor indicating legal actions.
                        If provided, illegal actions will be masked out before sampling/argmax.
            deterministic: If True, choose the action with the highest probability (argmax).
                           If False, sample from the distribution.
        Returns:
            action: Chosen action tensor.
            log_prob: Log probability of the chosen action.
            value: Value estimate tensor.
        """
        policy_logits, value = self.forward(obs)

        if legal_mask is not None:
            # Apply the legal mask: set logits of illegal actions to -infinity
            # Ensure legal_mask has the same shape as policy_logits or is broadcastable
            if (
                legal_mask.ndim == 1
                and policy_logits.ndim == 2
                and policy_logits.shape[0] == 1
            ):
                legal_mask = legal_mask.unsqueeze(0)  # Adapt for batch size 1

            masked_logits = torch.where(
                legal_mask,
                policy_logits,
                torch.tensor(float("-inf"), device=policy_logits.device),
            )
            # Handle case where all masked_logits are -inf to prevent NaN in softmax
            if not torch.any(legal_mask):  # Or check if masked_logits are all -inf
                # If no legal moves, softmax over original logits might be one option,
                # or let it produce NaNs which should be caught upstream.
                # For now, this will lead to NaNs if all are masked.
                # This situation should ideally be caught by the caller (e.g. PPOAgent)
                pass  # Let it proceed, PPOAgent should handle no legal moves.
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)

        # Check for NaNs in probs, which can happen if all logits were -inf
        if torch.isnan(probs).any():
            # This is a fallback: if probs are NaN (e.g. all legal actions masked out and all logits became -inf),
            # distribute probability uniformly over all actions to avoid erroring out in Categorical.
            # A better solution is for the caller to handle "no legal actions" gracefully.
            print(
                "Warning: NaNs in probabilities in ActorCriticResTower.get_action_and_value. Check legal_mask and logits. Defaulting to uniform.",
                file=sys.stderr,
            )
            probs = torch.ones_like(policy_logits) / policy_logits.shape[-1]

        dist = torch.distributions.Categorical(probs=probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value  # value is already squeezed in forward()

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probabilities, entropy, and value for given observations and actions.
        Args:
            obs: Input observation tensor.
            actions: Actions taken tensor.
            legal_mask: Optional boolean tensor indicating legal actions.
                        If provided, it can be used to ensure probabilities are calculated correctly,
                        though for evaluating *taken* actions, it's often assumed they were legal.
                        It primarily affects entropy calculation if used to restrict the distribution.
        Returns:
            log_probs: Log probabilities of the taken actions.
            entropy: Entropy of the action distribution.
            value: Value estimate tensor.
        """
        policy_logits, value = self.forward(obs)

        # If legal_mask is None (e.g., when called from PPOAgent.learn during batch processing),
        # the policy distribution (probs) and entropy are calculated over all possible actions,
        # not just those that were legal in the specific states from which 'actions' were sampled.
        # This is a common approach. To calculate entropy strictly over the legal action space
        # for each state in the batch, legal masks for each observation in obs_minibatch
        # would need to be stored in the experience buffer and passed here.
        if legal_mask is not None:
            # Apply the legal mask for calculating probabilities and entropy correctly
            # Ensure legal_mask has the same shape as policy_logits or is broadcastable

            # The shape of legal_mask should be (batch_size, num_actions)
            # The shape of policy_logits is (batch_size, num_actions)
            # No unsqueezing or broadcasting adjustment should be needed here if shapes are consistent.
            # The previous check for legal_mask.ndim == 1 was more for get_action_and_value with batch_size=1.
            # Here, we expect legal_mask to match policy_logits if provided.

            masked_logits = torch.where(
                legal_mask,
                policy_logits,
                torch.tensor(float("-inf"), device=policy_logits.device),
            )
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)

        # Check for NaNs in probs (e.g. if all logits in a row were -inf due to masking)
        # Replace NaNs with uniform distribution for stability in entropy calculation for those rows
        if torch.isnan(probs).any():
            print(
                "Warning: NaNs in probabilities in ActorCriticResTower.evaluate_actions. Check legal_mask and logits. Defaulting to uniform for affected rows.",
                file=sys.stderr,
            )
            nan_rows = torch.isnan(probs).any(dim=1)
            probs[nan_rows] = torch.ones_like(probs[nan_rows]) / policy_logits.shape[-1]

        dist = torch.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value  # value is already squeezed in forward()
