"""
Minimal ActorCritic neural network for DRL Shogi Client (dummy forward pass).
"""

import sys  # For stderr
from typing import Optional, Tuple  # Added Optional

import torch
import torch.nn.functional as F
from torch import nn  # Corrected import


class ActorCritic(nn.Module):
    """Actor-Critic neural network for Shogi RL agent (PPO-ready)."""

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

    def get_action_and_value(self, obs: torch.Tensor, legal_mask: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if legal_mask.ndim == 1 and policy_logits.ndim == 2 and policy_logits.shape[0] == 1:
                legal_mask = legal_mask.unsqueeze(0) # Adapt for batch size 1

            masked_logits = torch.where(legal_mask, policy_logits, torch.tensor(float("-inf"), device=policy_logits.device))
            # Handle case where all masked_logits are -inf to prevent NaN in softmax
            if not torch.any(legal_mask): # Or check if masked_logits are all -inf
                # If no legal moves, softmax over original logits might be one option,
                # or let it produce NaNs which should be caught upstream.
                # For now, this will lead to NaNs if all are masked.
                # This situation should ideally be caught by the caller (e.g. PPOAgent)
                pass # Let it proceed, PPOAgent should handle no legal moves.
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)

        # Check for NaNs in probs, which can happen if all logits were -inf
        if torch.isnan(probs).any():
            # This is a fallback: if probs are NaN (e.g. all legal actions masked out and all logits became -inf),
            # distribute probability uniformly over all actions to avoid erroring out in Categorical.
            # A better solution is for the caller to handle "no legal actions" gracefully.
            print("Warning: NaNs in probabilities in ActorCritic.get_action_and_value. Check legal_mask and logits. Defaulting to uniform.", file=sys.stderr)
            probs = torch.ones_like(policy_logits) / policy_logits.shape[-1]


        dist = torch.distributions.Categorical(probs=probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)  # Squeeze value to match typical use

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if legal_mask.ndim == 1 and policy_logits.ndim == 2 and policy_logits.shape[0] == actions.shape[0]:
                # If policy_logits is (batch, num_actions) and legal_mask is (batch, num_actions)
                # or if legal_mask is (num_actions) and needs to be broadcasted.
                # Assuming legal_mask is (batch_size, num_actions) if provided for a batch.
                pass # Shape should be compatible

            masked_logits = torch.where(legal_mask, policy_logits, torch.tensor(float("-inf"), device=policy_logits.device))
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)
        
        # Check for NaNs in probs (e.g. if all logits in a row were -inf due to masking)
        # Replace NaNs with uniform distribution for stability in entropy calculation for those rows
        if torch.isnan(probs).any():
            print("Warning: NaNs in probabilities in ActorCritic.evaluate_actions. Check legal_mask and logits. Defaulting to uniform for affected rows.", file=sys.stderr)
            nan_rows = torch.isnan(probs).any(dim=1)
            probs[nan_rows] = torch.ones_like(probs[nan_rows]) / policy_logits.shape[-1]


        dist = torch.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value.squeeze(-1)  # Squeeze value
