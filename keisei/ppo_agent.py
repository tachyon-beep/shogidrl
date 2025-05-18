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
        self.last_kl_div = 0.0  # Initialize KL divergence tracker

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
                "CRITICAL WARNING: No legal moves available in select_action. "
                "This indicates a game state issue or a bug."
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
        Returns a dictionary of logging metrics.
        """
        self.model.train()  # Set model to train mode
        
        # Retrieve data from buffer. Ensure tensors are on the correct device.
        # ExperienceBuffer should handle device placement, but good to be explicit if needed.
        batch_data = experience_buffer.get_batch() 

        # Get current learning rate early, in case we return early
        current_lr = self.optimizer.param_groups[0]["lr"]

        if not batch_data:
            # Return default metrics if buffer is empty or provides no data
            return {
                "ppo/policy_loss": 0.0,
                "ppo/value_loss": 0.0,
                "ppo/entropy": 0.0,
                "ppo/kl_divergence_approx": 0.0,
                "ppo/learning_rate": current_lr,
            }

        obs_batch = batch_data["obs"].to(self.device)
        actions_batch = batch_data["actions"].to(self.device)
        old_log_probs_batch = batch_data["log_probs"].to(self.device)
        advantages_batch = batch_data["advantages"].to(self.device)
        returns_batch = batch_data["returns"].to(self.device)

        if obs_batch.shape[0] == 0: # Check if tensors are empty after potential filtering
            return {
                "ppo/policy_loss": 0.0,
                "ppo/value_loss": 0.0,
                "ppo/entropy": 0.0,
                "ppo/kl_divergence_approx": 0.0,
                "ppo/learning_rate": current_lr,
            }

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        num_samples = obs_batch.shape[0]
        indices = np.arange(num_samples)

        total_policy_loss_epoch = 0.0
        total_value_loss_epoch = 0.0
        total_entropy_epoch = 0.0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                minibatch_indices = indices[start_idx:end_idx]

                if len(minibatch_indices) == 0:
                    continue

                obs_minibatch = obs_batch[minibatch_indices]
                actions_minibatch = actions_batch[minibatch_indices]
                old_log_probs_minibatch = old_log_probs_batch[minibatch_indices]
                advantages_minibatch = advantages_batch[minibatch_indices]
                returns_minibatch = returns_batch[minibatch_indices]
                
                if obs_minibatch.shape[0] == 0:
                    continue

                new_logits, new_values = self.model(obs_minibatch)
                new_values = new_values.squeeze(-1) # Squeeze the last dimension

                # Ensure shapes are compatible for loss calculation
                if new_values.shape != returns_minibatch.shape:
                    # This can happen if batch_size is 1 and squeeze makes new_values scalar
                    if new_values.numel() == returns_minibatch.numel():
                        new_values = new_values.reshape(returns_minibatch.shape)
                    else:
                        # print(f"Warning: Shape mismatch in learn(). new_values: {new_values.shape}, returns: {returns_minibatch.shape}. Skipping minibatch.")
                        continue # Skip this minibatch if shapes are incompatible

                probs = F.softmax(new_logits, dim=-1)
                action_distribution = torch.distributions.Categorical(probs=probs)
                new_log_probs = action_distribution.log_prob(actions_minibatch)
                entropy = action_distribution.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_minibatch)
                surr1 = ratio * advantages_minibatch
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * advantages_minibatch
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns_minibatch)
                loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss_epoch += policy_loss.item()
                total_value_loss_epoch += value_loss.item()
                total_entropy_epoch += entropy.item()
                num_updates += 1
        
        avg_policy_loss = 0.0
        avg_value_loss = 0.0
        avg_entropy = 0.0
        kl_divergence_final_approx = 0.0

        if num_updates > 0:
            avg_policy_loss = total_policy_loss_epoch / num_updates
            avg_value_loss = total_value_loss_epoch / num_updates
            avg_entropy = total_entropy_epoch / num_updates

            with torch.no_grad():
                final_new_logits, _ = self.model(obs_batch)
                final_new_probs = F.softmax(final_new_logits, dim=-1)
                final_action_dist = torch.distributions.Categorical(probs=final_new_probs)
                final_new_log_probs = final_action_dist.log_prob(actions_batch)
                kl_divergence_final_approx = (old_log_probs_batch - final_new_log_probs).mean().item()
        
        self.last_kl_div = kl_divergence_final_approx

        metrics = {
            "ppo/policy_loss": avg_policy_loss,
            "ppo/value_loss": avg_value_loss,
            "ppo/entropy": avg_entropy,
            "ppo/kl_divergence_approx": self.last_kl_div,
            "ppo/learning_rate": current_lr,
        }
        return metrics

    def save_model(self, file_path: str):
        """Saves the model state dictionary to a file."""
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path: str):
        """Loads the model state dictionary from a file."""
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.to(self.device) # Ensure model is on the correct device after loading
