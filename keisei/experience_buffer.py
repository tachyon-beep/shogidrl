"""
Minimal ExperienceBuffer for DRL Shogi Client.
"""


class ExperienceBuffer:
    """Experience buffer for storing transitions during RL training."""

    def __init__(self, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.obs: list = []
        self.actions: list = []
        self.rewards: list = []
        self.log_probs: list = []  # Added
        self.values: list = []     # Added
        self.ptr = 0

    def add(self, obs, action, reward, log_prob, value):
        """Add a transition to the buffer if not full."""
        if self.ptr < self.buffer_size:
            self.obs.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)  # Added
            self.values.append(value)        # Added
            self.ptr += 1

    def __len__(self):
        """Return the number of transitions currently in the buffer."""
        return self.ptr
