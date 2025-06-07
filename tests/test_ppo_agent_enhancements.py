import numpy as np
import pytest
import torch

from keisei.core.actor_critic_protocol import ActorCriticProtocol
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper


class DummyScaler:
    """Simple scaler that adds 1 to observations and records calls."""

    def __init__(self):
        self.last_input = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.clone()
        return x + 1.0


def _make_dummy_model(return_value: float) -> ActorCriticProtocol:
    class DummyModel(torch.nn.Module):
        def __init__(self, num_actions: int, value: float):
            super().__init__()
            self.num_actions = num_actions
            self.value = torch.tensor([value], dtype=torch.float32)
            self.param = torch.nn.Parameter(torch.ones(1))
            self.last_obs_forward = None
            self.last_obs_eval = None
            self.grad_enabled = None

        def forward(self, obs: torch.Tensor):
            self.last_obs_forward = obs.detach().clone()
            batch = obs.shape[0]
            logits = torch.zeros(batch, self.num_actions, device=obs.device)
            value = self.value.expand(batch) * self.param
            return logits, value

        def get_action_and_value(self, obs, legal_mask=None, deterministic=False):
            self.grad_enabled = torch.is_grad_enabled()
            logits, value = self.forward(obs)
            action = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
            return action, log_prob, value

        def evaluate_actions(self, obs, actions, legal_mask=None):
            self.last_obs_eval = obs.detach().clone()
            logits, value = self.forward(obs)
            log_probs = torch.zeros(obs.shape[0], device=obs.device)
            entropy = torch.zeros(obs.shape[0], device=obs.device)
            return log_probs, entropy, value

    mapper = PolicyOutputMapper()
    return DummyModel(mapper.get_total_actions(), return_value)


def test_select_action_no_grad_and_obs_norm(minimal_app_config):
    scaler = DummyScaler()
    model = _make_dummy_model(0.0)
    agent = PPOAgent(
        model=model,
        config=minimal_app_config,
        device=torch.device("cpu"),
        scaler=scaler,
    )
    obs = np.zeros((minimal_app_config.env.input_channels, 9, 9), dtype=np.float32)
    legal_mask = torch.ones(agent.num_actions_total, dtype=torch.bool)
    agent.select_action(obs, legal_mask, is_training=True)
    assert model.grad_enabled is False
    assert torch.allclose(
        model.last_obs_forward, torch.from_numpy(obs).unsqueeze(0) + 1
    )


def _create_buffer(
    old_value: float, reward: float, channels: int, device: str = "cpu"
) -> ExperienceBuffer:
    buffer = ExperienceBuffer(buffer_size=1, gamma=1.0, lambda_gae=1.0, device=device)
    obs = torch.zeros(channels, 9, 9, device=device)
    mask = torch.ones(
        PolicyOutputMapper().get_total_actions(), dtype=torch.bool, device=device
    )
    buffer.add(
        obs=obs,
        action=0,
        reward=reward,
        log_prob=0.0,
        value=old_value,
        done=True,
        legal_mask=mask,
    )
    buffer.compute_advantages_and_returns(0.0)
    return buffer


def test_value_function_clipping_enabled(minimal_app_config):
    config = minimal_app_config.model_copy()
    config.training.ppo_epochs = 1
    config.training.minibatch_size = 1
    config.training.enable_value_clipping = True
    model = _make_dummy_model(1.5)
    scaler = DummyScaler()
    agent = PPOAgent(
        model=model, config=config, device=torch.device("cpu"), scaler=scaler
    )
    buffer = _create_buffer(1.0, 3.0, channels=config.env.input_channels)
    metrics = agent.learn(buffer)
    assert torch.allclose(
        model.last_obs_eval, torch.zeros_like(model.last_obs_eval) + 1
    )
    assert abs(metrics["ppo/value_loss"] - 3.24) < 1e-2


def test_value_function_clipping_disabled(minimal_app_config):
    config = minimal_app_config.model_copy()
    config.training.ppo_epochs = 1
    config.training.minibatch_size = 1
    config.training.enable_value_clipping = False
    model = _make_dummy_model(1.5)
    agent = PPOAgent(model=model, config=config, device=torch.device("cpu"))
    buffer = _create_buffer(1.0, 3.0, channels=config.env.input_channels)
    metrics = agent.learn(buffer)
    assert abs(metrics["ppo/value_loss"] - 2.25) < 1e-2
