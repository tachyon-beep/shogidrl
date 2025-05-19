"""
Minimal ExperienceBuffer for DRL Shogi Client.
"""

#import numpy as np
import torch # Ensure torch is imported

class ExperienceBuffer:
    """Experience buffer for storing transitions during RL training."""

    def __init__(
        self, buffer_size: int, gamma: float, lambda_gae: float, device: str = "cpu"
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.device = torch.device(device) # Store as torch.device

        # Initialize buffers as empty lists
        self.obs: list[torch.Tensor] = [] # Assuming obs are stored as tensors
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.advantages: list[float] = [] # Populated by compute_advantages_and_returns
        self.returns: list[float] = []    # Populated by compute_advantages_and_returns
        self.ptr = 0

    def add(self, obs: torch.Tensor, action: int, reward: float, log_prob: float, value: float, done: bool):
        """
        Add a transition to the buffer.
        'obs' is expected to be a PyTorch tensor of shape (C, H, W) on self.device.
        """
        if self.ptr < self.buffer_size:
            # obs should already be on self.device and have shape (C,H,W) when passed from train.py
            self.obs.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value) # Storing scalar value estimates
            self.dones.append(done)
            self.ptr += 1
        else:
            # This case should ideally be handled by the training loop,
            # which calls learn() and then clear() when buffer is full.
            print("Warning: ExperienceBuffer is full. Cannot add new experience.")


    def compute_advantages_and_returns(self, last_value: float): # last_value is a float
        """
        Computes Generalized Advantage Estimation (GAE) and returns for the collected experiences.
        This should be called after the buffer is full (i.e., self.ptr == self.buffer_size).
        """
        if self.ptr == 0:
            print("Warning: compute_advantages_and_returns called on an empty buffer.")
            self.advantages = []
            self.returns = []
            return

        # Ensure advantages and returns lists are sized correctly for the current ptr
        self.advantages = [0.0] * self.ptr
        self.returns = [0.0] * self.ptr
        gae = 0.0

        # Values and rewards are Python floats/lists of floats. last_value is float.
        # GAE calculation can remain largely as Python floats for now, converted to tensor in get_batch.
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value # Value of S_t+1 (state after last action in buffer)
            else:
                next_value = self.values[t + 1] # V(S_t+1)

            # If self.dones[t] is True, it means S_t was the terminal state of an episode.
            # So, the value of any subsequent state V(S_{t+1}) should be 0.
            mask = 1.0 - float(self.dones[t]) # 1.0 if not done, 0.0 if done

            delta = self.rewards[t] + self.gamma * next_value * mask - self.values[t]
            gae = delta + self.gamma * self.lambda_gae * mask * gae

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batch(self) -> dict:
        """
        Returns all collected experiences as a dictionary of PyTorch tensors on self.device.
        Assumes compute_advantages_and_returns has been called.
        """
        if self.ptr == 0:
            # This should be handled by PPOAgent.learn() checking for empty batch_data
            # For safety, one might return structured empty tensors if needed upstream.
            print("Warning: get_batch called on an empty or not-yet-computed buffer.")
            return {} # PPOAgent.learn already checks for this

        num_samples = self.ptr

        # --- Efficient Batching for Observations ---
        # self.obs is a list of tensors, each (C,H,W), already on self.device.
        # Stack them along a new batch dimension (dim=0).
        try:
            obs_tensor = torch.stack(self.obs[:num_samples], dim=0)
        except RuntimeError as e:
            # This might happen if tensors in self.obs are not on the same device or have inconsistent shapes.
            # Should not happen if train.py's add() is consistent.
            print(f"Error stacking observation tensors in ExperienceBuffer: {e}")
            # Fallback or re-raise, for now, let's indicate a problem by returning empty.
            return {}


        # --- Convert other lists to tensors ---
        actions_tensor = torch.tensor(self.actions[:num_samples], dtype=torch.int64, device=self.device)
        log_probs_tensor = torch.tensor(self.log_probs[:num_samples], dtype=torch.float32, device=self.device)
        # self.values are scalar V(s_t) estimates stored as floats
        values_tensor = torch.tensor(self.values[:num_samples], dtype=torch.float32, device=self.device)

        advantages_tensor = torch.tensor(self.advantages[:num_samples], dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(self.returns[:num_samples], dtype=torch.float32, device=self.device)

        # Dones can be bool or float, PPO often uses float for masking.
        dones_tensor = torch.tensor(self.dones[:num_samples], dtype=torch.bool, device=self.device)

        return {
            "obs": obs_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,      # These are old_log_probs for PPO
            "values": values_tensor,          # These are old_V(s_t) for PPO, used with GAE
            "advantages": advantages_tensor,
            "returns": returns_tensor,          # These are the GAE-based returns (targets for value func)
            "dones": dones_tensor               # For record keeping or if needed by learn()
        }

    def clear(self):
        """Clears all stored experiences from the buffer."""
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
        self.ptr = 0

    def __len__(self):
        """Return the number of transitions currently in the buffer."""
        return self.ptr
