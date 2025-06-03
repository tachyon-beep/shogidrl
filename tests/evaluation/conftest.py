"""
Common fixtures and utilities for evaluation tests.
"""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from keisei.config_schema import (
    AppConfig,
    DemoConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_core_definitions import MoveTuple
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import BaseOpponent, EvaluationLogger, PolicyOutputMapper

INPUT_CHANNELS = 46  # Use the default from config for tests


# A mock PPOAgent for testing purposes
# Inherit from PPOAgent to satisfy type hints for run_evaluation_loop, and BaseOpponent for other uses.
class MockPPOAgent(PPOAgent, BaseOpponent):
    def __init__(
        self,
        config,
        device,
        name="MockPPOAgentForTest",
    ):
        # Create mock model first for dependency injection
        from keisei.core.neural_network import ActorCritic

        policy_mapper = PolicyOutputMapper()
        mock_model = ActorCritic(
            config.env.input_channels, policy_mapper.get_total_actions()
        )

        # Call parent constructors with model parameter
        PPOAgent.__init__(
            self, model=mock_model, config=config, device=device, name=name
        )
        BaseOpponent.__init__(self, name=name)

        # Override with MagicMock for testing
        self.model = MagicMock()
        self._is_ppo_agent_mock = True  # Flag to identify this mock
        # self.name is set by PPOAgent's __init__ via BaseOpponent

    def load_model(
        self, file_path: str
    ) -> dict:  # Parameter name changed to file_path, return type to dict
        # print(f"MockPPOAgent: Pretending to load model from {file_path}")
        return {}  # Return an empty dict as per PPOAgent

    def select_action(
        self,
        obs: np.ndarray,
        legal_mask: torch.Tensor,
        *,
        is_training: bool = True,
    ):
        # For test compatibility, always return a dummy move and values
        # Assume legal_mask is a tensor of bools, pick the first True index
        idx = int(legal_mask.nonzero(as_tuple=True)[0][0]) if legal_mask.any() else 0
        return (None, idx, 0.0, 0.0)

    def get_value(
        self, obs_np: np.ndarray
    ) -> float:  # obs_np type changed to np.ndarray
        """Mocked get_value method."""
        return 0.0  # Return a dummy float value

    # If used as a BaseOpponent directly (e.g. PPO vs PPO where one is simplified)
    def select_move(
        self, game_instance: ShogiGame
    ) -> MoveTuple:  # Return type changed back to MoveTuple
        legal_moves = game_instance.get_legal_moves()
        if not legal_moves:
            raise ValueError("MockPPOAgent.select_move: No legal moves available.")
        # Simplified for BaseOpponent interface, actual PPO logic is in select_action
        obs_np = MagicMock(
            spec=np.ndarray
        )  # Dummy observation, spec for type hint if needed
        legal_mask_tensor = MagicMock(
            spec=torch.Tensor
        )  # Dummy mask, spec for type hint
        action_result = self.select_action(obs_np, legal_mask_tensor, is_training=False)
        selected_move = action_result[0]
        if selected_move is None:
            # This should ideally not happen if legal_moves is not empty.
            # Handle cases where select_action might return None for the move.
            raise ValueError(
                "MockPPOAgent.select_move: select_action returned None for a move despite legal moves being available."
            )
        return selected_move


@pytest.fixture
def policy_mapper():
    return PolicyOutputMapper()


@pytest.fixture
def eval_logger_setup(tmp_path):
    log_file = tmp_path / "test_eval.log"
    logger = EvaluationLogger(str(log_file), also_stdout=False)
    with logger:  # Ensure logger is used as a context manager
        yield logger, str(log_file)
    # logger.close() is handled by the context manager's __exit__


@pytest.fixture
def shogi_game_initial():
    return ShogiGame()


@pytest.fixture
def mock_app_config():
    """Returns a mock AppConfig for testing."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=13527,  # Example value
            seed=42,
            max_moves_per_game=500,  # Added missing parameter
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=200,
            ppo_epochs=4,
            minibatch_size=32,
            learning_rate=0.0003,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=2,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
            enable_periodic_evaluation=False,  # Added missing parameter
            max_moves_per_game=500,  # Added missing parameter
            log_file_path_eval="/tmp/eval.log",  # Added missing parameter
            wandb_log_eval=False,  # Added missing parameter
        ),
        logging=LoggingConfig(
            log_file="logs/test_evaluate_log.txt",
            model_dir="models/test_evaluate_models/",
            run_name="test_evaluate_run",
        ),
        wandb=WandBConfig(
            enabled=False,
            project="keisei-shogi-rl",
            entity=None,
            run_name_prefix="keisei",
            watch_model=True,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
    )


@pytest.fixture
def mock_app_config_parallel(tmp_path):
    """Returns a mock AppConfig with parallel enabled for testing."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=13527,
            seed=42,
            max_moves_per_game=500,  # Added missing parameter
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=200,
            ppo_epochs=4,
            minibatch_size=32,
            learning_rate=0.0003,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=2,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
            enable_periodic_evaluation=False,  # Added missing parameter
            max_moves_per_game=500,  # Added missing parameter
            log_file_path_eval="/tmp/eval_parallel.log",  # Added missing parameter
            wandb_log_eval=False,  # Added missing parameter
        ),
        logging=LoggingConfig(
            log_file=str(tmp_path / "logs/test_evaluate_log_parallel.txt"),
            model_dir=str(tmp_path / "models/test_evaluate_models_parallel/"),
            run_name="test_evaluate_run_parallel",
        ),
        wandb=WandBConfig(
            enabled=False,
            project="keisei-shogi-rl",
            entity=None,
            run_name_prefix="keisei",
            watch_model=True,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        parallel=ParallelConfig(  # Corrected ParallelConfig
            enabled=True,
            num_workers=2,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
    )


# Helper to create a minimal AppConfig for test agents
def make_test_config(device_str, input_channels, policy_mapper):
    # If policy_mapper is a pytest fixture function, raise an error to prevent direct calls
    if hasattr(policy_mapper, "_pytestfixturefunction"):
        raise RuntimeError(
            "policy_mapper fixture was passed directly; pass an instance instead."
        )
    try:
        num_actions_total = policy_mapper.get_total_actions()
    except (AttributeError, TypeError) as e:
        raise ValueError(
            "policy_mapper must provide a valid get_total_actions() method returning an int."
        ) from e
    return AppConfig(
        env=EnvConfig(
            device=device_str,
            input_channels=input_channels,
            num_actions_total=num_actions_total,
            seed=42,
            max_moves_per_game=500,  # Added missing parameter
        ),
        training=TrainingConfig(
            total_timesteps=1,
            steps_per_epoch=1,
            ppo_epochs=1,
            minibatch_size=1,
            learning_rate=1e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            input_features="core46",
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            weight_decay=0.0,  # Added missing argument
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=1,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
            enable_periodic_evaluation=False,  # Added missing parameter
            max_moves_per_game=500,  # Added missing parameter
            log_file_path_eval="/tmp/eval.log",  # Added missing parameter
            wandb_log_eval=False,  # Added missing parameter
        ),
        logging=LoggingConfig(
            log_file="/tmp/eval.log", model_dir="/tmp/", run_name="test-eval-run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="eval",
            entity=None,
            run_name_prefix="test-eval-run",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
        parallel=ParallelConfig(
            enabled=False,
            num_workers=4,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
    )
