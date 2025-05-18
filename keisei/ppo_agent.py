"""
Minimal PPOAgent for DRL Shogi Client.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from keisei.experience_buffer import ExperienceBuffer  # Added import
from keisei.neural_network import ActorCritic
from keisei.shogi.shogi_core_definitions import MoveTuple
from keisei.utils import PolicyOutputMapper


class PPOAgent:
    """Proximal Policy Optimization agent for Shogi (PPO logic)."""

    def __init__(
        self,
        input_channels: int,
        policy_output_mapper: PolicyOutputMapper,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        # gae_lambda: float = 0.95, # This will be used by ExperienceBuffer
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 10,  # Added from plan
        minibatch_size: int = 64,  # Added from plan
        value_loss_coeff: float = 0.5,  # Renamed from value_coef for clarity and consistency
        entropy_coef: float = 0.01,  # Added from plan (was already in old init)
        device: str = "cpu",
    ):
        """
        Initialize the PPOAgent with model, optimizer, and PPO hyperparameters.
        """
        self.device = torch.device(device)
        self.policy_output_mapper = policy_output_mapper
        self.num_actions_total = self.policy_output_mapper.get_total_actions()
        self.model = ActorCritic(input_channels, self.num_actions_total).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        # self.gae_lambda = gae_lambda # gae_lambda is used by ExperienceBuffer
        self.clip_epsilon = clip_epsilon
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def select_action(
        self,
        obs: np.ndarray,  # Changed to np.ndarray to match ShogiGame.get_observation()
        legal_shogi_moves: List[MoveTuple],  # Added legal_shogi_moves
        is_training: bool = True,  # Added is_training flag
    ) -> Tuple[MoveTuple, int, float, float]:  # Return type updated
        """
        Select an action index given an observation and optional legal action indices.
        Returns the selected Shogi move, its policy index, log probability, and value estimate.
        """
        self.model.train(is_training)  # Set model to train or eval mode

        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_logits, value = self.model(obs_tensor)

        # Create a mask for legal actions
        legal_mask = self.policy_output_mapper.get_legal_mask(
            legal_shogi_moves, self.device
        )

        # Apply the mask to the logits (set illegal actions to -infinity)
        masked_logits = torch.where(
            legal_mask, action_logits, torch.tensor(float("-inf"), device=self.device)
        )

        selected_policy_index_val: int
        log_prob_val: float

        if not legal_mask.any():
            # This case should be handled by the game (e.g. checkmate/stalemate)
            # If we must return, define behavior. For now, error or specific return.
            # This path indicates an issue if the game isn't over.
            # Raising an error might be more appropriate here.
            # For now, to prevent crashes if this state is reached unexpectedly:
            print(
                "CRITICAL WARNING: No legal moves available in select_action. This indicates a game state issue or a bug."
            )
            # Fallback: select action 0, log_prob 0. This is arbitrary and problematic.
            selected_policy_index_val = 0
            log_prob_val = 0.0  # log(1) for a dummy probability
            # It might be better to raise an exception here:
            # raise ValueError("No legal moves available for action selection.")
        else:
            probs = F.softmax(masked_logits, dim=-1)

            if torch.isnan(probs).any():
                print(
                    "Warning: NaN in probabilities. Using uniform over legal, or all if mask is empty."
                )
                if legal_mask.any():
                    probs = legal_mask.float() / legal_mask.sum()
                else:  # Should ideally not be reached if the above `if not legal_mask.any()` handles it
                    probs = (
                        torch.ones_like(action_logits, device=self.device)
                        / self.num_actions_total
                    )
                # Ensure no NaNs after fallback
                probs = torch.nan_to_num(
                    probs, nan=0.0
                )  # Replace any remaining NaNs with 0
                if (
                    probs.sum() == 0
                ):  # If all probs became 0 (e.g. all legal moves had NaN issues)
                    # Fallback to uniform over all actions if sum is zero to avoid division by zero in Categorical
                    probs = (
                        torch.ones_like(action_logits, device=self.device)
                        / self.num_actions_total
                    )

            if is_training:
                action_distribution = torch.distributions.Categorical(probs=probs)
                selected_policy_index_tensor = action_distribution.sample()
                log_prob_tensor = action_distribution.log_prob(
                    selected_policy_index_tensor
                )
            else:
                selected_policy_index_tensor = torch.argmax(probs, dim=-1)
                # Create a distribution to get log_prob for the argmax action
                # Ensure probs is not all zero before creating Categorical
                if probs.sum() == 0:
                    # This can happen if all logits were -inf and fallbacks failed to create valid probs
                    print(
                        "Warning: All probabilities are zero in eval mode. Defaulting log_prob."
                    )
                    log_prob_tensor = torch.tensor(
                        0.0, device=self.device
                    )  # Default log_prob
                else:
                    action_distribution = torch.distributions.Categorical(probs=probs)
                    log_prob_tensor = action_distribution.log_prob(
                        selected_policy_index_tensor
                    )

            selected_policy_index_val = int(selected_policy_index_tensor.item())
            log_prob_val = float(log_prob_tensor.item())

        selected_shogi_move = self.policy_output_mapper.policy_index_to_shogi_move(
            selected_policy_index_val
        )

        return (
            selected_shogi_move,
            selected_policy_index_val,
            log_prob_val,
            float(value.item()),
        )

    def get_value(self, obs_np: np.ndarray) -> float:
        """Get the value prediction from the critic for a given NumPy observation."""
        self.model.eval()  # Set model to eval mode for inference
        obs_tensor = torch.tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(obs_tensor)  # Assuming model returns (logits, value)
        return float(value.item())

    def learn(self, experience_buffer: ExperienceBuffer):
        """
        Perform PPO update using experiences from the buffer.
        """
        self.model.train()  # Set model to train mode
        batch_data = experience_buffer.get_batch()
        obs_batch = batch_data["obs"]
        actions_batch = batch_data["actions"]
        old_log_probs_batch = batch_data["log_probs"]
        # old_values_batch = batch_data["values"] # Values from buffer, can be used for clipped value loss
        advantages_batch = batch_data["advantages"]
        returns_batch = batch_data["returns"]

        # Normalize advantages (optional but often helpful)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        num_samples = len(obs_batch)
        indices = np.arange(num_samples)

        # Placeholder for logging losses
        total_policy_loss, total_value_loss, total_entropy = (
            0,
            0,
            0,
        )  # Renamed and initialized
        num_updates = 0  # Initialized

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                minibatch_indices = indices[start_idx:end_idx]

                # Get minibatch data
                mb_obs = obs_batch[minibatch_indices]
                mb_actions = actions_batch[minibatch_indices]
                mb_old_log_probs = old_log_probs_batch[minibatch_indices]
                mb_advantages = advantages_batch[minibatch_indices]
                mb_returns = returns_batch[minibatch_indices]
                # mb_old_values = old_values_batch[minibatch_indices] # For clipped value loss

                # Evaluate actions and values from the current policy
                # Assumes self.model.evaluate_actions(obs, actions) returns (log_probs, entropy, values)
                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    mb_obs, mb_actions
                )

                # Calculate policy ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Calculate surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss
                # Standard PPO value loss:
                value_loss = ((new_values - mb_returns) ** 2).mean()
                # Optional: Clipped value loss (from some PPO implementations)
                # value_pred_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -self.clip_epsilon, self.clip_epsilon)
                # vf_loss1 = (new_values - mb_returns).pow(2)
                # vf_loss2 = (value_pred_clipped - mb_returns).pow(2)
                # value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                # Total loss
                loss = (
                    policy_loss
                    - self.entropy_coef * entropy.mean()
                    + self.value_loss_coeff * value_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0

        return avg_policy_loss, avg_value_loss, avg_entropy

    def save_model(self, path: str):
        """Save the model and optimizer state to the given path."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str, map_location=None):
        """Load the model and optimizer state from the given path."""
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
