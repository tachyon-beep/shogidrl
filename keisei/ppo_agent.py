"""
Minimal PPOAgent for DRL Shogi Client.
"""

import os
import sys  # For stderr
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple  # Removed Any

import numpy as np
import torch
import torch.nn.functional as F  # For F.softmax

from keisei.experience_buffer import ExperienceBuffer
from keisei.neural_network import ActorCritic
from keisei.utils import PolicyOutputMapper

if TYPE_CHECKING:
    from keisei.shogi.shogi_core_definitions import MoveTuple


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
        legal_shogi_moves: List[
            "MoveTuple"
        ],  # Still useful for policy_index_to_shogi_move
        legal_mask: torch.Tensor,  # INPUT PARAMETER
        is_training: bool = True,
    ) -> Tuple[
        Optional["MoveTuple"],
        int,
        float,
        float,
        # No longer returns legal_mask
    ]:  # Return Shogi move (Any for now), policy index, log_prob, value
        """
        Select an action given an observation, legal Shogi moves, and a precomputed legal_mask.
        Returns the selected Shogi move, its policy index, log probability, and value estimate.
        """
        self.model.train(is_training)  # Set model to train or eval mode

        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # legal_mask is now passed as an argument.
        # The internal creation of legal_mask is removed.
        # Example:
        # legal_mask = self.policy_output_mapper.get_legal_mask(
        # legal_shogi_moves, self.device
        # ) # REMOVED

        if not legal_mask.any():
            print(
                "Warning: PPOAgent.select_action called with no legal moves (based on input legal_mask).",
                file=sys.stderr,
            )
            # Fallback behavior might be needed if model.get_action_and_value can't handle all-False mask.
            # neural_network.py's get_action_and_value attempts to handle this.
            # If this path is hit, it implies the caller might not have checked for no legal moves.
            # The train.py logic should ideally prevent calling select_action if no legal_moves.
            pass  # Let it proceed, model.get_action_and_value will use the all-false mask.

        # Get action, log_prob, and value from the ActorCritic model
        # Pass deterministic based on not is_training
        selected_policy_index_tensor, log_prob_tensor, value_tensor = (
            self.model.get_action_and_value(  # Pass the provided legal_mask
                obs_tensor, legal_mask=legal_mask, deterministic=not is_training
            )
        )

        selected_policy_index_val = int(selected_policy_index_tensor.item())
        log_prob_val = float(log_prob_tensor.item())
        value_float = float(
            value_tensor.item()
        )  # Value is already squeezed in get_action_and_value

        selected_shogi_move: Optional["MoveTuple"] = None  # Quoted
        try:
            selected_shogi_move = self.policy_output_mapper.policy_index_to_shogi_move(
                selected_policy_index_val
            )
        except IndexError as e:
            print(
                f"Error in PPOAgent.select_action: Policy index {selected_policy_index_val} out of bounds. {e}",
                file=sys.stderr,
            )
            # Handle by returning no move or re-raising, depending on desired robustness.
            return None, -1, 0.0, value_float  # Return 4 items
            # Or raise the error

        return (
            selected_shogi_move,
            selected_policy_index_val,
            log_prob_val,
            value_float,
            # legal_mask, # REMOVED from return
        )

    def get_value(self, obs_np: np.ndarray) -> float:
        """Get the value prediction from the critic for a given NumPy observation."""
        self.model.eval()  # Set model to evaluation mode
        obs_tensor = torch.tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, _, value_estimate = self.model.get_action_and_value(
                obs_tensor, deterministic=True
            )  # Get value deterministically
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
            print(
                "Warning: PPOAgent.learn called with empty batch_data.", file=sys.stderr
            )
            return {
                "ppo/policy_loss": 0.0,
                "ppo/value_loss": 0.0,
                "ppo/entropy": 0.0,
                "ppo/kl_divergence_approx": self.last_kl_div,
                "ppo/learning_rate": current_lr,
            }

        obs_batch = batch_data["obs"].to(self.device)
        actions_batch = batch_data["actions"].to(self.device)
        old_log_probs_batch = batch_data["log_probs"].to(self.device)
        advantages_batch = batch_data["advantages"].to(self.device)
        returns_batch = batch_data["returns"].to(self.device)
        legal_masks_batch = batch_data["legal_masks"].to(
            self.device
        )  # Added legal_masks_batch

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        num_samples = obs_batch.shape[0]
        indices = np.arange(num_samples)

        total_policy_loss_epoch, total_value_loss_epoch, total_entropy_epoch = (
            0.0,
            0.0,
            0.0,
        )
        num_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                minibatch_indices = indices[start_idx:end_idx]

                obs_minibatch = obs_batch[minibatch_indices]
                actions_minibatch = actions_batch[minibatch_indices]
                old_log_probs_minibatch = old_log_probs_batch[minibatch_indices]
                advantages_minibatch = advantages_batch[minibatch_indices]
                returns_minibatch = returns_batch[minibatch_indices]
                legal_masks_minibatch = legal_masks_batch[
                    minibatch_indices
                ]  # Added legal_masks_minibatch

                # Get new log_probs, entropy, and value from the model
                # Note on entropy: legal_mask is now passed here. Entropy is calculated
                # over legal actions only.
                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    obs_minibatch,
                    actions_minibatch,
                    legal_mask=legal_masks_minibatch,  # Pass legal_masks_minibatch
                )

                # PPO Loss Calculation
                ratio = torch.exp(new_log_probs - old_log_probs_minibatch)

                # Clipped surrogate objective
                surr1 = ratio * advantages_minibatch
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * advantages_minibatch
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(
                    new_values.squeeze(), returns_minibatch.squeeze()
                )

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=0.5
                )  # Optional: gradient clipping
                self.optimizer.step()

                total_policy_loss_epoch += policy_loss.item()
                total_value_loss_epoch += value_loss.item()
                total_entropy_epoch += entropy_loss.item()
                num_updates += 1

        avg_policy_loss = (
            total_policy_loss_epoch / num_updates if num_updates > 0 else 0.0
        )
        avg_value_loss = (
            total_value_loss_epoch / num_updates if num_updates > 0 else 0.0
        )
        avg_entropy = total_entropy_epoch / num_updates if num_updates > 0 else 0.0
        kl_divergence_final_approx = 0.0

        if num_updates > 0:
            with torch.no_grad():
                # For KL divergence, we need to evaluate actions with the current policy
                # considering the legal masks that were active when those actions were chosen.
                # The ActorCritic.evaluate_actions method handles the legal_mask internally
                # for calculating log_probs and entropy. For KL, we need the log_probs
                # from the current policy for the actions taken, using the same legal_masks.
                # The call to evaluate_actions for the full batch (if needed for KL) should also pass legal_masks_batch.
                # However, the current KL approximation uses model(obs_batch) which doesn't take legal_mask.
                # For a more accurate KL involving legal actions, the distribution from model()
                # would need to be masked before calculating log_prob.
                # For simplicity, current KL approx is kept, but note this subtlety.

                # Re-evaluate log_probs for the entire batch with current policy and original legal masks
                # to get a consistent comparison for KL divergence.
                current_log_probs_for_kl, _, _ = self.model.evaluate_actions(
                    obs_batch, actions_batch, legal_mask=legal_masks_batch
                )
                kl_divergence_final_approx = (
                    (old_log_probs_batch - current_log_probs_for_kl).mean().item()
                )

        self.last_kl_div = kl_divergence_final_approx

        metrics: Dict[str, float] = {
            "ppo/policy_loss": avg_policy_loss,
            "ppo/value_loss": avg_value_loss,
            "ppo/entropy": avg_entropy,
            "ppo/kl_divergence_approx": self.last_kl_div,
            "ppo/learning_rate": current_lr,
        }
        return metrics

    def save_model(
        self,
        file_path: str,
        global_timestep: int = 0,
        total_episodes_completed: int = 0,
    ) -> None:
        """Saves the model, optimizer, and training state to a file."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_timestep": global_timestep,
                "total_episodes_completed": total_episodes_completed,
            },
            file_path,
        )
        print(f"PPOAgent model, optimizer, and state saved to {file_path}")

    def load_model(self, file_path: str) -> dict:
        """Loads the model, optimizer, and training state from a file. Returns the checkpoint dict."""
        if not os.path.exists(file_path):
            print(
                f"Warning: Model checkpoint not found at {file_path}. Agent not loaded."
            )
            return {}
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Ensure optimizer state is also moved to the correct device if necessary
            # This is often handled by PyTorch if model parameters are on the device already
            # or by explicitly iterating through optimizer state and moving tensors.
            # For Adam, this might involve: for state in self.optimizer.state.values():
            # for k, v in state.items(): if isinstance(v, torch.Tensor): state[k] = v.to(self.device)
        else:
            print(
                "Warning: Optimizer state not found in checkpoint. Initializing optimizer from scratch.",
                file=sys.stderr,
            )
        self.model.to(self.device)
        print(f"PPOAgent model, optimizer, and state loaded from {file_path}")
        return checkpoint  # Return the loaded checkpoint dictionary
