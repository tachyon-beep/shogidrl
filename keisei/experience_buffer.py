"""
Minimal ExperienceBuffer for DRL Shogi Client.
"""

import torch  # Add torch import for tensor conversion
import numpy as np  # Add numpy import for optimized tensor conversion


class ExperienceBuffer:
    """Experience buffer for storing transitions during RL training."""

    def __init__(
        self, buffer_size: int, gamma: float, lambda_gae: float, device: str = "cpu"
    ):  # Added gamma and lambda_gae
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.device = device
        self.obs: list = []
        self.actions: list = []
        self.rewards: list = []
        self.log_probs: list = []
        self.values: list = []
        self.dones: list = []  # Added dones
        self.advantages: list = []  # Added advantages
        self.returns: list = []  # Added returns
        self.ptr = 0

    def add(self, obs, action, reward, log_prob, value, done):  # Added done
        """Add a transition to the buffer if not full."""
        if self.ptr < self.buffer_size:
            self.obs.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.dones.append(done)  # Added
            self.ptr += 1
        # else:
        # Optionally, handle buffer full case, e.g., by logging or raising error
        # For PPO, the buffer is typically filled up to buffer_size and then processed.

    def compute_advantages_and_returns(self, last_value: float):
        """
        Computes Generalized Advantage Estimation (GAE) and returns for the collected experiences.
        This should be called after the buffer is full (i.e., self.ptr == self.buffer_size).
        """
        if self.ptr != self.buffer_size:
            # Or raise an error, or log a warning.
            # This method assumes a full buffer of STEPS_PER_EPOCH.
            print(
                "Warning: compute_advantages_and_returns called on a buffer not "
                "yet full. Size: {self.ptr}/{self.buffer_size}"
            )
            # Decide if we should proceed or return. For now, let's proceed if there's anything.
            if self.ptr == 0:
                return

        # Ensure internal lists for advantages and returns are reset if this can be called multiple times
        # on the same buffer instance before clearing (though standard PPO clears after learning).
        self.advantages = [0.0] * self.ptr
        self.returns = [0.0] * self.ptr

        gae = 0.0
        # Iterate backwards from the last collected experience
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:  # Last step in the buffer
                next_non_terminal = (
                    1.0 - self.dones[t]
                )  # If last step was 'done', next_non_terminal is 0
                next_value = last_value  # Value of the state *after* the last action in the buffer
            else:
                next_non_terminal = (
                    1.0 - self.dones[t]
                )  # This is actually for the current step's 'done' status
                # to determine if V(s_t+1) should be zeroed out.
                # More accurately, it's (1 - self.dones[t+1]) if we think about future steps,
                # but for GAE, we use dones[t] to mask future values if current state t is terminal.
                # Let's re-evaluate: delta uses V(s_t+1). If s_t is terminal, V(s_t+1) is 0.
                # So, next_non_terminal should be based on self.dones[t] for the delta calculation.
                next_value = self.values[t + 1]

            # Delta: R_t + gamma * V(S_{t+1}) * (1 - D_t) - V(S_t)
            # Here, self.dones[t] refers to whether state S_t led to termination.
            # If S_t is terminal, then the value of S_{t+1} is 0.
            # So, (1 - self.dones[t]) correctly masks V(S_{t+1}) if S_t was terminal.
            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1 - self.dones[t])
                - self.values[t]
            )

            # GAE: delta_t + gamma * lambda * (1 - D_t) * GAE_{t+1}
            # Again, (1 - self.dones[t]) masks future GAE if S_t was terminal.
            gae = delta + self.gamma * self.lambda_gae * (1 - self.dones[t]) * gae

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]  # Return = Advantage + Value

    def get_batch(self) -> dict:
        """
        Returns all collected experiences as a dictionary of PyTorch tensors.
        Assumes compute_advantages_and_returns has been called.
        """
        if not self.advantages or not self.returns:
            # Or raise an error. This indicates compute_advantages_and_returns wasn't called.
            print("Warning: get_batch called before compute_advantages_and_returns.")
            # Potentially return empty or handle error. For now, proceed if data exists.

        # Ensure all lists are of the same length (self.ptr)
        # This should be guaranteed if add and compute_advantages_and_returns are used correctly.
        num_samples = self.ptr

        # Convert lists to tensors
        # Note: self.obs contains numpy arrays. Stack them.
        # Optimized conversion: list of np.ndarrays -> single np.ndarray -> torch.tensor
        obs_np_array = np.array(self.obs[:num_samples], dtype=np.float32)
        obs_tensor = torch.tensor(obs_np_array, dtype=torch.float32, device=self.device)

        actions_tensor = torch.tensor(
            self.actions[:num_samples], dtype=torch.int64, device=self.device
        )  # Assuming actions are indices
        log_probs_tensor = torch.tensor(
            self.log_probs[:num_samples], dtype=torch.float32, device=self.device
        )
        values_tensor = torch.tensor(
            self.values[:num_samples], dtype=torch.float32, device=self.device
        )
        # Old values
        # dones_tensor = torch.tensor(self.dones[:num_samples], dtype=torch.float32,
        # device=self.device) # Not directly used by PPO loss but good to have
        advantages_tensor = torch.tensor(
            self.advantages[:num_samples], dtype=torch.float32, device=self.device
        )
        returns_tensor = torch.tensor(
            self.returns[:num_samples], dtype=torch.float32, device=self.device
        )

        return {
            "obs": obs_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,  # These are the log_probs of actions taken (old_log_probs)
            "values": values_tensor,  # These are the value estimates at the time of action (old_values)
            "advantages": advantages_tensor,
            "returns": returns_tensor,  # These are the TD(lambda) returns / GAE-based returns
            # "dones": dones_tensor # Optional, if needed by the learning algorithm directly
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
