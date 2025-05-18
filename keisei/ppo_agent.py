"""
Minimal PPOAgent for DRL Shogi Client.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np

from keisei.neural_network import ActorCritic
from keisei.utils import PolicyOutputMapper
from keisei.shogi.shogi_core_definitions import MoveTuple


class PPOAgent:
    """Proximal Policy Optimization agent for Shogi (PPO logic)."""

    def __init__(
        self,
        input_channels: int,
        policy_output_mapper: PolicyOutputMapper,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
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
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(
        self,
        obs: np.ndarray, # Changed to np.ndarray to match ShogiGame.get_observation()
        legal_shogi_moves: List[MoveTuple], # Added legal_shogi_moves
        is_training: bool = True, # Added is_training flag
    ) -> Tuple[MoveTuple, int, float, float]: # Return type updated
        """
        Select an action index given an observation and optional legal action indices.
        Returns the selected Shogi move, its policy index, log probability, and value estimate.
        """
        self.model.train(is_training) # Set model to train or eval mode

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value = self.model(obs_tensor)

        # Create a mask for legal actions
        legal_mask = self.policy_output_mapper.get_legal_mask(legal_shogi_moves, self.device)
        
        # Apply the mask to the logits (set illegal actions to -infinity)
        masked_logits = torch.where(legal_mask, action_logits, torch.tensor(float('-inf'), device=self.device))
        
        selected_policy_index_val: int
        log_prob_val: float

        if not legal_mask.any():
            # This case should be handled by the game (e.g. checkmate/stalemate)
            # If we must return, define behavior. For now, error or specific return.
            # This path indicates an issue if the game isn't over.
            # Raising an error might be more appropriate here.
            # For now, to prevent crashes if this state is reached unexpectedly:
            print("CRITICAL WARNING: No legal moves available in select_action. This indicates a game state issue or a bug.")
            # Fallback: select action 0, log_prob 0. This is arbitrary and problematic.
            selected_policy_index_val = 0
            log_prob_val = 0.0 # log(1) for a dummy probability
            # It might be better to raise an exception here:
            # raise ValueError("No legal moves available for action selection.")
        else:
            probs = F.softmax(masked_logits, dim=-1)

            if torch.isnan(probs).any():
                print("Warning: NaN in probabilities. Using uniform over legal, or all if mask is empty.")
                if legal_mask.any():
                    probs = legal_mask.float() / legal_mask.sum()
                else: # Should ideally not be reached if the above `if not legal_mask.any()` handles it
                    probs = torch.ones_like(action_logits, device=self.device) / self.num_actions_total
                # Ensure no NaNs after fallback
                probs = torch.nan_to_num(probs, nan=0.0) # Replace any remaining NaNs with 0
                if probs.sum() == 0: # If all probs became 0 (e.g. all legal moves had NaN issues)
                    # Fallback to uniform over all actions if sum is zero to avoid division by zero in Categorical
                    probs = torch.ones_like(action_logits, device=self.device) / self.num_actions_total


            if is_training:
                action_distribution = torch.distributions.Categorical(probs=probs)
                selected_policy_index_tensor = action_distribution.sample()
                log_prob_tensor = action_distribution.log_prob(selected_policy_index_tensor)
            else:
                selected_policy_index_tensor = torch.argmax(probs, dim=-1)
                # Create a distribution to get log_prob for the argmax action
                # Ensure probs is not all zero before creating Categorical
                if probs.sum() == 0:
                     # This can happen if all logits were -inf and fallbacks failed to create valid probs
                    print("Warning: All probabilities are zero in eval mode. Defaulting log_prob.")
                    log_prob_tensor = torch.tensor(0.0, device=self.device) # Default log_prob
                else:
                    action_distribution = torch.distributions.Categorical(probs=probs)
                    log_prob_tensor = action_distribution.log_prob(selected_policy_index_tensor)
            
            selected_policy_index_val = int(selected_policy_index_tensor.item())
            log_prob_val = float(log_prob_tensor.item())

        selected_shogi_move = self.policy_output_mapper.policy_index_to_shogi_move(selected_policy_index_val)
        
        return selected_shogi_move, selected_policy_index_val, log_prob_val, float(value.item())

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE) returns for PPO.
        """
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_update(
        self, obs, actions, old_log_probs, returns, advantages, epochs=4, batch_size=32
    ):
        """
        Perform PPO update over multiple epochs and mini-batches.
        """
        obs = torch.stack(obs)
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(old_log_probs)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        n = len(obs)
        for _ in range(epochs):
            idxs = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_idx = idxs[start:end]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                log_probs, entropy, values = self.model.evaluate_actions(
                    batch_obs, batch_actions
                )
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (batch_returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
