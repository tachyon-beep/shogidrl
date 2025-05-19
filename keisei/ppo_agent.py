"""
Minimal PPOAgent for DRL Shogi Client.
"""

import os # For checking file existence in load_model
from typing import List, Tuple, Dict, Any # Added Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from keisei.experience_buffer import ExperienceBuffer
from keisei.neural_network import ActorCritic
from keisei.shogi.shogi_core_definitions import MoveTuple # Assuming MoveTuple is defined
from keisei.utils import PolicyOutputMapper


class PPOAgent:
    """Proximal Policy Optimization agent for Shogi (PPO logic)."""

    def __init__(
        self,
        input_channels: int,
        policy_output_mapper: PolicyOutputMapper,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 10,
        minibatch_size: int = 64,
        value_loss_coeff: float = 0.5,
        entropy_coef: float = 0.01,
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
        self.clip_epsilon = clip_epsilon
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.last_kl_div = 0.0  # Initialize KL divergence tracker

    def select_action(
        self,
        obs: np.ndarray,
        legal_shogi_moves: List[MoveTuple], # Type hint for Shogi moves
        is_training: bool = True,
    ) -> Tuple[Any, int, float, float]: # Return Shogi move (Any for now), policy index, log_prob, value
        """
        Select an action given an observation and legal Shogi moves.
        Returns the selected Shogi move, its policy index, log probability, and value estimate.
        """
        self.model.train(is_training)

        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_logits, value = self.model(obs_tensor)

        legal_mask = self.policy_output_mapper.get_legal_mask(
            legal_shogi_moves, self.device
        )

        masked_logits = torch.where(
            legal_mask, action_logits, torch.tensor(float("-inf"), device=self.device)
        )

        selected_policy_index_val: int
        log_prob_val: float

        if not legal_mask.any():
            error_msg = (
                "PPOAgent.select_action() called with no legal moves available. "
                "This usually means the game has already ended or there's an issue "
                "in the game state or legal move generation."
            )
            # Consider logging this with a proper logger if available
            print(f"CRITICAL ERROR in select_action: {error_msg}")
            raise ValueError(error_msg)

        probs = F.softmax(masked_logits, dim=-1)

        if torch.isnan(probs).any():
            print("Warning: NaN detected in action probabilities. Attempting fallback to uniform over legal moves.")
            if legal_mask.any():
                probs = legal_mask.float() / legal_mask.sum()
            else: # Should have been caught by the earlier `if not legal_mask.any():`
                  # This fallback is a last resort to prevent crashes with Categorical
                print("Critical Warning: No legal moves and NaN in probs. Uniform over ALL actions.")
                probs = torch.ones_like(action_logits, device=self.device) / self.num_actions_total

            probs = torch.nan_to_num(probs, nan=0.0) # Clean any remaining NaNs
            if probs.sum() < 1e-8: # If all probs became zero or sum is too small
                print("Critical Warning: Probs sum to zero after NaN fallback. Uniform over ALL actions.")
                probs = torch.ones_like(action_logits, device=self.device) / self.num_actions_total

        if is_training:
            action_distribution = torch.distributions.Categorical(probs=probs)
            selected_policy_index_tensor = action_distribution.sample()
            log_prob_tensor = action_distribution.log_prob(selected_policy_index_tensor)
        else: # Evaluation mode
            selected_policy_index_tensor = torch.argmax(probs, dim=-1)
            if probs.sum() > 1e-8: # Avoid issues with Categorical if probs are effectively all zero
                action_distribution = torch.distributions.Categorical(probs=probs)
                log_prob_tensor = action_distribution.log_prob(selected_policy_index_tensor)
            else:
                print("Warning: Probs sum to zero in eval mode. Defaulting log_prob.")
                log_prob_tensor = torch.tensor(0.0, device=self.device)

        selected_policy_index_val = int(selected_policy_index_tensor.item())
        log_prob_val = float(log_prob_tensor.item())

        selected_shogi_move = self.policy_output_mapper.policy_index_to_shogi_move(
            selected_policy_index_val
        )

        if selected_shogi_move is None:
             # This should ideally not happen if legal_mask.any() was true and policy_output_mapper is robust
            error_msg = (
                f"PolicyOutputMapper returned None for a selected policy index {selected_policy_index_val}. "
                f"This implies an issue with the action mapping or that the selected index was unexpectedly invalid "
                f"despite legal_mask checks. Logits: {action_logits.cpu().numpy().tolist()}, "
                f"MaskedLogits: {masked_logits.cpu().numpy().tolist()}, Probs: {probs.cpu().numpy().tolist()}"
            )
            print(f"CRITICAL ERROR in select_action: {error_msg}")
            raise ValueError(error_msg)


        return (
            selected_shogi_move,
            selected_policy_index_val,
            log_prob_val,
            float(value.item()),
        )

    def get_value(self, obs_np: np.ndarray) -> float:
        """Get the value prediction from the critic for a given NumPy observation."""
        self.model.eval()
        obs_tensor = torch.tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, value_estimate = self.model(obs_tensor) # Model returns (action_logits, value)
        return float(value_estimate.item())

    def learn(self, experience_buffer: ExperienceBuffer) -> Dict[str, float]:
        """
        Perform PPO update using experiences from the buffer.
        Returns a dictionary of logging metrics.
        """
        self.model.train()

        batch_data = experience_buffer.get_batch()
        # It's good practice for ExperienceBuffer.get_batch() to already place tensors on self.device
        # or for the buffer itself to live on self.device if memory allows.
        # If not, the .to(self.device) calls below are necessary.

        current_lr = self.optimizer.param_groups[0]["lr"]

        if not batch_data or batch_data["obs"].shape[0] == 0:
            print("Warning: PPO learn called with no data in buffer.")
            return {
                "ppo/policy_loss": 0.0, "ppo/value_loss": 0.0, "ppo/entropy": 0.0,
                "ppo/kl_divergence_approx": self.last_kl_div,
                "ppo/learning_rate": current_lr,
            }

        obs_batch = batch_data["obs"].to(self.device)
        actions_batch = batch_data["actions"].to(self.device)
        old_log_probs_batch = batch_data["log_probs"].to(self.device)
        advantages_batch = batch_data["advantages"].to(self.device)
        returns_batch = batch_data["returns"].to(self.device)

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        num_samples = obs_batch.shape[0]
        indices = np.arange(num_samples)

        total_policy_loss_epoch, total_value_loss_epoch, total_entropy_epoch = 0.0, 0.0, 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                minibatch_indices = indices[start_idx:end_idx]

                if len(minibatch_indices) == 0:
                    continue

                # Slice minibatch data
                obs_minibatch = obs_batch[minibatch_indices]
                actions_minibatch = actions_batch[minibatch_indices]
                old_log_probs_minibatch = old_log_probs_batch[minibatch_indices]
                advantages_minibatch = advantages_batch[minibatch_indices]
                returns_minibatch = returns_batch[minibatch_indices]

                if obs_minibatch.shape[0] == 0: # Should not happen if len(minibatch_indices) > 0
                    continue

                new_logits, new_values_raw = self.model(obs_minibatch)
                new_values = new_values_raw.squeeze(-1) # Expected shape: (minibatch_size,)

                # Ensure returns_minibatch and new_values have compatible shapes for loss calculation
                # Typically, both should be 1D tensors of shape (current_minibatch_size,).
                # ExperienceBuffer should ensure returns_minibatch is already 1D.
                if new_values.shape != returns_minibatch.shape:
                    # This can happen if minibatch_size is 1 and squeeze turns new_values into a 0-D scalar.
                    # Or if returns_minibatch was (N, 1) and new_values became (N,).
                    if new_values.numel() == returns_minibatch.numel():
                        try:
                            returns_minibatch_view = returns_minibatch.view_as(new_values)
                            value_loss = F.mse_loss(new_values, returns_minibatch_view)
                        except RuntimeError as e:
                            print(f"Warning: Shape mismatch & reshape error in learn(). new_values: {new_values.shape}, "
                                  f"returns: {returns_minibatch.shape}. Error: {e}. Skipping minibatch.")
                            continue
                    else:
                        print(f"Warning: Incompatible elements in learn(). new_values: {new_values.shape} ({new_values.numel()}), "
                              f"returns: {returns_minibatch.shape} ({returns_minibatch.numel()}). Skipping minibatch.")
                        continue
                else:
                    value_loss = F.mse_loss(new_values, returns_minibatch)

                probs = F.softmax(new_logits, dim=-1)
                action_distribution = torch.distributions.Categorical(probs=probs)
                new_log_probs = action_distribution.log_prob(actions_minibatch)
                entropy = action_distribution.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_minibatch)
                surr1 = ratio * advantages_minibatch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_minibatch

                policy_loss = -torch.min(surr1, surr2).mean()
                # value_loss already calculated above with shape handling
                loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss_epoch += policy_loss.item()
                total_value_loss_epoch += value_loss.item() # Use the calculated value_loss
                total_entropy_epoch += entropy.item()
                num_updates += 1

        avg_policy_loss = total_policy_loss_epoch / num_updates if num_updates > 0 else 0.0
        avg_value_loss = total_value_loss_epoch / num_updates if num_updates > 0 else 0.0
        avg_entropy = total_entropy_epoch / num_updates if num_updates > 0 else 0.0
        kl_divergence_final_approx = 0.0

        if num_updates > 0:
            with torch.no_grad():
                final_new_logits, _ = self.model(obs_batch)
                final_new_probs = F.softmax(final_new_logits, dim=-1)
                final_action_dist = torch.distributions.Categorical(probs=final_new_probs)
                final_new_log_probs = final_action_dist.log_prob(actions_batch)
                kl_divergence_final_approx = (old_log_probs_batch - final_new_log_probs).mean().item()

        self.last_kl_div = kl_divergence_final_approx

        metrics: Dict[str, float] = {
            "ppo/policy_loss": avg_policy_loss,
            "ppo/value_loss": avg_value_loss,
            "ppo/entropy": avg_entropy,
            "ppo/kl_divergence_approx": self.last_kl_div,
            "ppo/learning_rate": current_lr,
        }
        return metrics

    def save_model(self, file_path: str) -> None:
        """Saves the model and optimizer state dictionaries to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, file_path)
        print(f"PPOAgent model and optimizer saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """Loads the model and optimizer state dictionaries from a file."""
        if not os.path.exists(file_path):
            print(f"Warning: Model checkpoint not found at {file_path}. Agent not loaded.")
            return

        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # If optimizer states need to be moved to device explicitly (usually not if model is on device)
            # for state in self.optimizer.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, torch.Tensor):
            #             state[k] = v.to(self.device)
        else:
            print(f"Warning: Optimizer state not found in checkpoint {file_path}. Optimizer not loaded/reset.")

        self.model.to(self.device) # Ensure model is on the correct device
        print(f"PPOAgent model and optimizer loaded from {file_path}")
