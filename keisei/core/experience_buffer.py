"""
Minimal ExperienceBuffer for DRL Shogi Client.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch  # Ensure torch is imported


@dataclass
class Experience:
    """Single experience transition for parallel collection."""

    obs: torch.Tensor
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool
    legal_mask: torch.Tensor


class ExperienceBuffer:
    """Experience buffer for storing transitions during RL training."""

    def __init__(
        self, buffer_size: int, gamma: float, lambda_gae: float, device: str = "cpu"
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.device = torch.device(device)  # Store as torch.device

        # Initialize buffers as empty lists
        self.obs: list[torch.Tensor] = []  # Assuming obs are stored as tensors
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.legal_masks: list[torch.Tensor] = []  # Added to store legal masks
        self.advantages: list[torch.Tensor] = (
            []
        )  # Populated by compute_advantages_and_returns
        self.returns: list[torch.Tensor] = (
            []
        )  # Populated by compute_advantages_and_returns
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        legal_mask: torch.Tensor,  # Added legal_mask
    ):
        """
        Add a transition to the buffer.
        'obs' is expected to be a PyTorch tensor of shape (C, H, W) on self.device.
        'legal_mask' is expected to be a PyTorch tensor on self.device.
        """
        if self.ptr < self.buffer_size:
            # obs should already be on self.device and have shape (C,H,W) when passed from train.py
            self.obs.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)  # Storing scalar value estimates
            self.dones.append(done)
            self.legal_masks.append(legal_mask)  # Store legal_mask
            self.ptr += 1
        else:
            # This case should ideally be handled by the training loop,
            # which calls learn() and then clear() when buffer is full.
            print("Warning: ExperienceBuffer is full. Cannot add new experience.")

    def compute_advantages_and_returns(
        self, last_value: float
    ):  # last_value is a float
        """
        Computes Generalized Advantage Estimation (GAE) and returns for the collected experiences.
        This should be called after the buffer is full (i.e., self.ptr == self.buffer_size).
        Uses PyTorch tensor operations for GAE calculation.
        """
        if self.ptr == 0:
            print("Warning: compute_advantages_and_returns called on an empty buffer.")
            self.advantages = []
            self.returns = []
            return

        # Convert lists to tensors for GAE computation
        rewards_tensor = torch.tensor(
            self.rewards[: self.ptr], dtype=torch.float32, device=self.device
        )
        values_tensor = torch.tensor(
            self.values[: self.ptr], dtype=torch.float32, device=self.device
        )
        # Dones tensor: 1.0 if not done, 0.0 if done (for mask)
        dones_tensor = torch.tensor(
            self.dones[: self.ptr], dtype=torch.float32, device=self.device
        )
        masks_tensor = 1.0 - dones_tensor

        advantages_list: list[torch.Tensor] = [
            torch.tensor(0.0, device=self.device)
        ] * self.ptr
        returns_list: list[torch.Tensor] = [
            torch.tensor(0.0, device=self.device)
        ] * self.ptr
        gae = torch.tensor(0.0, device=self.device)  # Ensure gae is always a tensor

        # last_value is V(S_t+1) for the last state in the buffer
        # If the last state was terminal, next_value should be 0, handled by mask.
        next_value_tensor = torch.tensor(
            last_value, dtype=torch.float32, device=self.device
        )

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                current_next_value = next_value_tensor
            else:
                current_next_value = values_tensor[t + 1]

            delta = (
                rewards_tensor[t]
                + self.gamma * current_next_value * masks_tensor[t]
                - values_tensor[t]
            )
            gae = delta + self.gamma * self.lambda_gae * masks_tensor[t] * gae

            # Ensure scalar tensors by squeezing any extra dimensions
            advantages_list[t] = gae.squeeze()  # Store as scalar tensor
            returns_list[t] = (
                gae + values_tensor[t]
            ).squeeze()  # Store as scalar tensor

        self.advantages = advantages_list
        self.returns = returns_list

    def get_batch(self) -> dict:
        """
        Returns all collected experiences as a dictionary of PyTorch tensors on self.device.
        Assumes compute_advantages_and_returns has been called.
        """
        if self.ptr == 0:
            # This should be handled by PPOAgent.learn() checking for empty batch_data
            # For safety, one might return structured empty tensors if needed upstream.
            print("Warning: get_batch called on an empty or not-yet-computed buffer.")
            return {}  # PPOAgent.learn already checks for this

        num_samples = self.ptr

        # --- Efficient Batching for Observations ---
        # self.obs is a list of tensors, each (C,H,W), already on self.device.
        # Stack them along a new batch dimension (dim=0).
        try:
            obs_tensor = torch.stack(self.obs[:num_samples], dim=0)
        except RuntimeError as e:
            # This might happen if tensors in self.obs are not on the same device or have inconsistent shapes.
            # Should not happen if train.py\'s add() is consistent.
            raise ValueError(
                f"Failed to stack observation tensors: {e}. Check tensor shapes and devices."
            ) from e

        # --- Convert other lists to tensors ---
        actions_tensor = torch.tensor(
            self.actions[:num_samples], dtype=torch.int64, device=self.device
        )
        log_probs_tensor = torch.tensor(
            self.log_probs[:num_samples], dtype=torch.float32, device=self.device
        )
        # self.values are scalar V(s_t) estimates stored as floats
        values_tensor = torch.tensor(
            self.values[:num_samples], dtype=torch.float32, device=self.device
        )

        advantages_tensor = torch.stack(self.advantages[:num_samples])
        returns_tensor = torch.stack(self.returns[:num_samples])

        # Dones can be bool or float, PPO often uses float for masking.
        dones_tensor = torch.tensor(
            self.dones[:num_samples], dtype=torch.bool, device=self.device
        )

        # Stack legal_masks (list of tensors) into a single tensor
        # Assuming legal_masks are already on self.device
        try:
            legal_masks_tensor = torch.stack(self.legal_masks[:num_samples], dim=0)
        except RuntimeError as e:
            raise ValueError(
                f"Failed to stack legal_mask tensors: {e}. Check tensor shapes and devices."
            ) from e

        return {
            "obs": obs_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,  # These are old_log_probs for PPO
            "values": values_tensor,  # These are old_V(s_t) for PPO, used with GAE
            "advantages": advantages_tensor,
            "returns": returns_tensor,  # These are the GAE-based returns (targets for value func)
            "dones": dones_tensor,  # For record keeping or if needed by learn()
            "legal_masks": legal_masks_tensor,  # Added legal_masks_tensor
        }

    def clear(self):
        """Clears all stored experiences from the buffer."""
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.legal_masks.clear()  # Clear legal_masks
        self.advantages.clear()
        self.returns.clear()
        self.ptr = 0

    def __len__(self):
        """Return the number of transitions currently in the buffer."""
        return self.ptr

    def add_batch(self, experiences: List[Experience]) -> None:
        """
        Add a batch of experiences to the buffer for parallel collection.

        Args:
            experiences: List of Experience objects to add
        """
        for exp in experiences:
            if self.ptr >= self.buffer_size:
                break  # Don't exceed buffer size

            self.add(
                obs=exp.obs,
                action=exp.action,
                reward=exp.reward,
                log_prob=exp.log_prob,
                value=exp.value,
                done=exp.done,
                legal_mask=exp.legal_mask,
            )

    def add_from_worker_batch(self, worker_data: Dict[str, torch.Tensor]) -> None:
        """
        Add experiences from worker batch data (optimized for parallel collection).

        Args:
            worker_data: Dictionary containing batched tensors from workers
        """
        batch_size = worker_data["obs"].shape[0]

        for i in range(batch_size):
            if self.ptr >= self.buffer_size:
                break

            self.add(
                obs=worker_data["obs"][i],
                action=int(worker_data["actions"][i].item()),
                reward=worker_data["rewards"][i].item(),
                log_prob=worker_data["log_probs"][i].item(),
                value=worker_data["values"][i].item(),
                done=bool(worker_data["dones"][i].item()),
                legal_mask=worker_data["legal_masks"][i],
            )

    def get_worker_batch_format(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get current buffer contents in worker batch format for validation.

        Returns:
            Dictionary with batched tensors or None if buffer is empty
        """
        if self.ptr == 0:
            return None

        return {
            "obs": torch.stack(self.obs[: self.ptr], dim=0),
            "actions": torch.tensor(
                self.actions[: self.ptr], dtype=torch.int64, device=self.device
            ),
            "rewards": torch.tensor(
                self.rewards[: self.ptr], dtype=torch.float32, device=self.device
            ),
            "log_probs": torch.tensor(
                self.log_probs[: self.ptr], dtype=torch.float32, device=self.device
            ),
            "values": torch.tensor(
                self.values[: self.ptr], dtype=torch.float32, device=self.device
            ),
            "dones": torch.tensor(
                self.dones[: self.ptr], dtype=torch.bool, device=self.device
            ),
            "legal_masks": torch.stack(self.legal_masks[: self.ptr], dim=0),
        }

    def merge_from_parallel_buffers(
        self, parallel_buffers: List["ExperienceBuffer"]
    ) -> None:
        """
        Merge experiences from multiple parallel buffers.

        Args:
            parallel_buffers: List of ExperienceBuffer instances from workers
        """
        for buffer in parallel_buffers:
            if buffer.ptr == 0:
                continue

            # Add all experiences from this buffer
            for i in range(buffer.ptr):
                if self.ptr >= self.buffer_size:
                    return  # Main buffer is full

                self.add(
                    obs=buffer.obs[i],
                    action=buffer.actions[i],
                    reward=buffer.rewards[i],
                    log_prob=buffer.log_probs[i],
                    value=buffer.values[i],
                    done=buffer.dones[i],
                    legal_mask=buffer.legal_masks[i],
                )
