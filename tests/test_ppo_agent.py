"""
Unit tests for PPOAgent in ppo_agent.py
"""

from typing import List  # Add this import

INPUT_CHANNELS = 46  # Use the default from config_schema for tests

import numpy as np
import pytest
import torch

from keisei.core.experience_buffer import ExperienceBuffer  # Added import
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi import ShogiGame  # Corrected import for ShogiGame
from keisei.shogi.shogi_core_definitions import (  # Ensure MoveTuple is imported
    MoveTuple,
)
from keisei.utils import PolicyOutputMapper


def create_test_config():
    """Create a properly configured AppConfig for testing."""
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

    mapper = PolicyOutputMapper()
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=1e-3,
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
        ),
        evaluation=EvaluationConfig(
            num_games=1, opponent_type="random", evaluation_interval_timesteps=50000
        ),
        logging=LoggingConfig(
            log_file="/tmp/test.log", model_dir="/tmp/", run_name="test_run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
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


def test_ppo_agent_init_and_select_action():
    """Test PPOAgent initializes and select_action returns a valid index."""
    mapper = PolicyOutputMapper()
    config = create_test_config()
    agent = PPOAgent(config=config, device=torch.device("cpu"))
    rng = np.random.default_rng(42)
    obs = rng.random((INPUT_CHANNELS, 9, 9)).astype(np.float32)
    game = ShogiGame(max_moves_per_game=512)
    legal_moves: List[MoveTuple] = game.get_legal_moves()

    # Ensure there's at least one legal move for the test to proceed
    if not legal_moves:
        # If ShogiGame starts with no legal moves (e.g. before first player acts, or specific setup)
        # and PolicyOutputMapper is populated, we need a known valid move.
        # This is a fallback for test robustness.
        # A standard opening move for Black (Sente)
        default_move: MoveTuple = (6, 7, 5, 7, False)  # Example: Pawn 7g->6g
        if default_move in mapper.move_to_idx:  # Check if mapper knows this move
            legal_moves.append(default_move)
        else:
            # If even this default isn't in mapper, the mapper or test setup is problematic.
            # For now, try to find *any* move the mapper knows to avoid crashing select_action.
            if mapper.idx_to_move:
                legal_moves.append(mapper.idx_to_move[0])
            else:
                pytest.skip(
                    "PolicyOutputMapper has no moves, cannot test select_action effectively."
                )

    if not legal_moves:  # If still no legal_moves, skip test
        pytest.skip("No legal moves could be determined for select_action test.")

    # Create legal_mask based on legal_moves
    legal_mask = mapper.get_legal_mask(legal_moves, device=agent.device)

    (
        selected_move,
        idx,
        log_prob,
        value,
    ) = agent.select_action(
        obs,
        legal_mask,
        is_training=True,
    )
    assert isinstance(idx, int)
    assert 0 <= idx < agent.num_actions_total
    assert (
        isinstance(selected_move, tuple) or selected_move is None
    )  # select_action can return None if no legal moves (though guarded by caller)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    # The legal_mask is an input to select_action, not an output.
    # We can assert properties of the input legal_mask if needed, or the one used by the model.
    # For example, check the one created above:
    assert isinstance(legal_mask, torch.Tensor)
    assert legal_mask.shape[0] == agent.num_actions_total
    assert legal_mask.dtype == torch.bool


def test_ppo_agent_learn():
    """Test PPOAgent's learn method with dummy data from an ExperienceBuffer."""
    config = create_test_config()
    agent = PPOAgent(config=config, device=torch.device("cpu"))

    buffer_size = 4  # Small buffer for testing
    experience_buffer = ExperienceBuffer(
        buffer_size=buffer_size,
        gamma=0.99,
        lambda_gae=0.95,
        device="cpu",  # Use CPU for testing
    )

    # Populate buffer with some dummy data
    rng = np.random.default_rng(42)
    dummy_obs_np = rng.random((INPUT_CHANNELS, 9, 9)).astype(np.float32)
    dummy_obs_tensor = torch.from_numpy(dummy_obs_np).to(
        torch.device("cpu")
    )  # Convert to tensor on CPU

    # Create a dummy legal_mask. For this test, its content might not be critical,
    # but its shape should match num_actions_total.
    dummy_legal_mask = torch.ones(
        agent.num_actions_total, dtype=torch.bool, device="cpu"
    )
    # Make at least one action illegal if num_actions_total > 0 to test masking, if desired
    if agent.num_actions_total > 0:
        dummy_legal_mask[0] = False

    for i in range(buffer_size):
        experience_buffer.add(
            obs=dummy_obs_tensor,  # <<< PASS THE TENSOR HERE
            action=i % agent.num_actions_total,
            reward=float(i),
            log_prob=0.1 * i,
            value=0.5 * i,
            done=(i == buffer_size - 1),
            legal_mask=dummy_legal_mask,  # Added dummy_legal_mask
        )

    assert len(experience_buffer) == buffer_size

    # Compute advantages and returns
    last_value = 0.0  # Assuming terminal state after buffer is full for simplicity
    experience_buffer.compute_advantages_and_returns(last_value)

    # Call the learn method
    try:
        metrics = agent.learn(experience_buffer)
        assert (
            metrics is not None
        ), "learn() should return a metrics dictionary, not None"
        # Check if losses are returned and are floats (or can be zero)
        assert isinstance(metrics["ppo/policy_loss"], float)
        assert isinstance(metrics["ppo/value_loss"], float)
        assert isinstance(metrics["ppo/entropy"], float)
        assert isinstance(metrics["ppo/kl_divergence_approx"], float)
        assert isinstance(metrics["ppo/learning_rate"], float)
    except (
        RuntimeError
    ) as e:  # Catch a more specific exception if possible, or document why general Exception is needed.
        pytest.fail(f"agent.learn() raised an exception: {e}")

    # Optionally, check if buffer is cleared after learn (if that's the intended behavior of learn or a subsequent step)
    # For now, just ensuring it runs and returns losses.
    # If learn is supposed to clear the buffer, add:
    # assert len(experience_buffer) == 0
    # However, current PPO plan has clear after learn in train.py, not in agent.learn() itself.


# Further tests could include:
# - Testing select_action in eval mode (is_training=False)
# - Testing model saving and loading (if not covered elsewhere)
# - More specific checks on loss values if expected behavior is known for dummy data
#   (though this can be complex and brittle)


def test_ppo_agent_learn_loss_components():
    """Test that PPOAgent.learn correctly computes and returns individual loss components."""
    config = create_test_config()
    # Override specific settings for this test
    config.training.ppo_epochs = 2  # Multiple epochs to test learning behavior
    agent = PPOAgent(config=config, device=torch.device("cpu"))

    buffer_size = 8  # Larger buffer for more realistic training
    experience_buffer = ExperienceBuffer(
        buffer_size=buffer_size,
        gamma=0.99,
        lambda_gae=0.95,
        device="cpu",
    )

    # Create deterministic data for more predictable testing
    torch.manual_seed(42)
    np.random.seed(42)

    dummy_obs_tensor = torch.randn(INPUT_CHANNELS, 9, 9, device="cpu")
    dummy_legal_mask = torch.ones(
        agent.num_actions_total, dtype=torch.bool, device="cpu"
    )

    # Create varied rewards and values to test advantage calculation
    rewards = [1.0, -0.5, 2.0, 0.0, 1.5, -1.0, 0.5, 2.5]
    values = [0.8, 0.2, 1.5, 0.1, 1.0, -0.3, 0.3, 2.0]

    for i in range(buffer_size):
        experience_buffer.add(
            obs=dummy_obs_tensor,
            action=i % agent.num_actions_total,
            reward=rewards[i],
            log_prob=0.1 * (i + 1),  # Varied log probs
            value=values[i],
            done=(i == buffer_size - 1),
            legal_mask=dummy_legal_mask,
        )

    # Compute advantages with realistic last value
    last_value = 1.2
    experience_buffer.compute_advantages_and_returns(last_value)

    # Capture initial model parameters for change verification
    initial_params = [p.clone() for p in agent.model.parameters()]

    # Call learn method
    metrics = agent.learn(experience_buffer)

    # Verify all expected metrics are present
    expected_metrics = [
        "ppo/policy_loss",
        "ppo/value_loss",
        "ppo/entropy",
        "ppo/kl_divergence_approx",
        "ppo/learning_rate",
    ]

    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert isinstance(metrics[metric], float), f"Metric {metric} should be float"
        assert not np.isnan(metrics[metric]), f"Metric {metric} is NaN"
        assert not np.isinf(metrics[metric]), f"Metric {metric} is infinite"

    # Verify reasonable metric ranges
    assert metrics["ppo/learning_rate"] == config.training.learning_rate
    assert (
        metrics["ppo/entropy"] <= 0.0
    ), "Entropy loss should be negative (entropy bonus)"
    assert metrics["ppo/policy_loss"] >= 0.0, "Policy loss should be non-negative"
    assert metrics["ppo/value_loss"] >= 0.0, "Value loss should be non-negative"

    # Verify model parameters changed (learning occurred)
    final_params = [p.clone() for p in agent.model.parameters()]
    params_changed = any(
        not torch.allclose(initial, final, atol=1e-6)
        for initial, final in zip(initial_params, final_params)
    )
    assert params_changed, "Model parameters should change after learning"


def test_ppo_agent_learn_advantage_normalization():
    """Test that advantages are properly normalized during learning."""
    mapper = PolicyOutputMapper()
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

    config = AppConfig(
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
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=1,
            minibatch_size=4,  # Ensure single minibatch
            learning_rate=1e-3,
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
        ),
        evaluation=EvaluationConfig(
            num_games=1, opponent_type="random", evaluation_interval_timesteps=50000
        ),
        logging=LoggingConfig(
            log_file="/tmp/test.log", model_dir="/tmp/", run_name="test_run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )
    agent = PPOAgent(config=config, device=torch.device("cpu"))

    buffer_size = 4
    experience_buffer = ExperienceBuffer(
        buffer_size=buffer_size,
        gamma=0.99,
        lambda_gae=0.95,
        device="cpu",
    )

    # Create data with known advantage distribution
    dummy_obs_tensor = torch.randn(INPUT_CHANNELS, 9, 9, device="cpu")
    dummy_legal_mask = torch.ones(
        agent.num_actions_total, dtype=torch.bool, device="cpu"
    )

    # Set high variance in rewards to test normalization
    rewards = [10.0, -5.0, 15.0, -8.0]
    values = [1.0, 1.0, 1.0, 1.0]  # Constant values for simpler advantage calc

    for i in range(buffer_size):
        experience_buffer.add(
            obs=dummy_obs_tensor,
            action=i % agent.num_actions_total,
            reward=rewards[i],
            log_prob=0.1,
            value=values[i],
            done=(i == buffer_size - 1),
            legal_mask=dummy_legal_mask,
        )

    experience_buffer.compute_advantages_and_returns(0.0)  # Terminal value = 0

    # Verify advantages computed correctly in buffer
    batch_data = experience_buffer.get_batch()
    raw_advantages = batch_data["advantages"]

    # Advantages should have non-zero variance before normalization
    assert (
        torch.std(raw_advantages, dim=0) > 0.1
    ), "Advantages should have significant variance"

    # Call learn - this will normalize advantages internally
    metrics = agent.learn(experience_buffer)

    # Learning should succeed with normalized advantages
    assert metrics["ppo/policy_loss"] >= 0.0
    assert not np.isnan(metrics["ppo/policy_loss"])


def test_ppo_agent_learn_gradient_clipping():
    """Test that gradient clipping is applied during learning."""
    mapper = PolicyOutputMapper()
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

    # Use high learning rate to potentially create large gradients
    config = AppConfig(
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
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=1.0,  # High learning rate
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            gradient_clip_max_norm=0.5,  # Explicit gradient clipping
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
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(
            num_games=1, opponent_type="random", evaluation_interval_timesteps=50000
        ),
        logging=LoggingConfig(
            log_file="/tmp/test.log", model_dir="/tmp/", run_name="test_run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )
    agent = PPOAgent(config=config, device=torch.device("cpu"))

    buffer_size = 4
    experience_buffer = ExperienceBuffer(
        buffer_size=buffer_size,
        gamma=0.99,
        lambda_gae=0.95,
        device="cpu",
    )

    # Create data that might produce large gradients
    dummy_obs_tensor = torch.randn(INPUT_CHANNELS, 9, 9, device="cpu")
    dummy_legal_mask = torch.ones(
        agent.num_actions_total, dtype=torch.bool, device="cpu"
    )

    # Extreme reward values to potentially create large policy updates
    rewards = [100.0, -100.0, 50.0, -50.0]

    for i in range(buffer_size):
        experience_buffer.add(
            obs=dummy_obs_tensor,
            action=i % agent.num_actions_total,
            reward=rewards[i],
            log_prob=0.1,
            value=0.0,
            done=(i == buffer_size - 1),
            legal_mask=dummy_legal_mask,
        )

    experience_buffer.compute_advantages_and_returns(0.0)

    # Learn should complete without exploding gradients due to clipping
    metrics = agent.learn(experience_buffer)

    # Verify learning completed successfully
    assert metrics is not None
    assert all(not np.isnan(v) for v in metrics.values()), "No metrics should be NaN"
    assert all(
        not np.isinf(v) for v in metrics.values()
    ), "No metrics should be infinite"


def test_ppo_agent_learn_empty_buffer_handling():
    """Test PPOAgent.learn behavior with empty experience buffer."""
    mapper = PolicyOutputMapper()
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

    config = AppConfig(
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
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=1e-3,
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
        ),
        evaluation=EvaluationConfig(
            num_games=1, opponent_type="random", evaluation_interval_timesteps=50000
        ),
        logging=LoggingConfig(
            log_file="/tmp/test.log", model_dir="/tmp/", run_name="test_run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )
    agent = PPOAgent(config=config, device=torch.device("cpu"))

    # Create empty buffer
    experience_buffer = ExperienceBuffer(
        buffer_size=4,
        gamma=0.99,
        lambda_gae=0.95,
        device="cpu",
    )

    # Don't add any experiences - buffer remains empty

    # Learn should handle empty buffer gracefully
    metrics = agent.learn(experience_buffer)

    # Should return default/zero metrics without crashing
    assert metrics is not None
    assert isinstance(metrics, dict)
    expected_metrics = [
        "ppo/policy_loss",
        "ppo/value_loss",
        "ppo/entropy",
        "ppo/kl_divergence_approx",
        "ppo/learning_rate",
    ]

    for metric in expected_metrics:
        assert metric in metrics
        # Should be zero or default values for empty buffer
        assert isinstance(metrics[metric], (int, float))


def test_ppo_agent_learn_kl_divergence_tracking():
    """Test that KL divergence is properly computed and tracked."""
    mapper = PolicyOutputMapper()
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

    config = AppConfig(
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
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=2,  # Multiple epochs to see KL divergence change
            minibatch_size=2,
            learning_rate=1e-3,
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
        ),
        evaluation=EvaluationConfig(
            num_games=1, opponent_type="random", evaluation_interval_timesteps=50000
        ),
        logging=LoggingConfig(
            log_file="/tmp/test.log", model_dir="/tmp/", run_name="test_run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )
    agent = PPOAgent(config=config, device=torch.device("cpu"))

    buffer_size = 4
    experience_buffer = ExperienceBuffer(
        buffer_size=buffer_size,
        gamma=0.99,
        lambda_gae=0.95,
        device="cpu",
    )

    dummy_obs_tensor = torch.randn(INPUT_CHANNELS, 9, 9, device="cpu")
    dummy_legal_mask = torch.ones(
        agent.num_actions_total, dtype=torch.bool, device="cpu"
    )

    for i in range(buffer_size):
        experience_buffer.add(
            obs=dummy_obs_tensor,
            action=i % agent.num_actions_total,
            reward=float(i),
            log_prob=0.1,
            value=0.5,
            done=(i == buffer_size - 1),
            legal_mask=dummy_legal_mask,
        )

    experience_buffer.compute_advantages_and_returns(0.0)

    # First learn call
    metrics1 = agent.learn(experience_buffer)
    kl_div_1 = metrics1["ppo/kl_divergence_approx"]

    # Verify KL divergence is tracked in agent
    assert hasattr(agent, "last_kl_div")
    assert agent.last_kl_div == kl_div_1

    # KL divergence should be a reasonable value
    assert isinstance(kl_div_1, float)
    assert not np.isnan(kl_div_1)
    assert not np.isinf(kl_div_1)

    # For multiple epochs with the same data, KL should generally be small
    # (policy shouldn't diverge dramatically from itself)
    assert abs(kl_div_1) < 10.0, f"KL divergence {kl_div_1} seems too large"


def test_ppo_agent_learn_minibatch_processing():
    """Test that minibatch processing works correctly with different batch sizes."""
    mapper = PolicyOutputMapper()
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

    # Test with buffer size that doesn't divide evenly by minibatch size
    config = AppConfig(
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
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=32,
            ppo_epochs=1,
            minibatch_size=3,  # Buffer size 5 / minibatch size 3 = uneven split
            learning_rate=1e-3,
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
        ),
        evaluation=EvaluationConfig(
            num_games=1, opponent_type="random", evaluation_interval_timesteps=50000
        ),
        logging=LoggingConfig(
            log_file="/tmp/test.log", model_dir="/tmp/", run_name="test_run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )
    agent = PPOAgent(config=config, device=torch.device("cpu"))

    buffer_size = 5  # Odd size to test uneven minibatch splitting
    experience_buffer = ExperienceBuffer(
        buffer_size=buffer_size,
        gamma=0.99,
        lambda_gae=0.95,
        device="cpu",
    )

    dummy_obs_tensor = torch.randn(INPUT_CHANNELS, 9, 9, device="cpu")
    dummy_legal_mask = torch.ones(
        agent.num_actions_total, dtype=torch.bool, device="cpu"
    )

    for i in range(buffer_size):
        experience_buffer.add(
            obs=dummy_obs_tensor,
            action=i % agent.num_actions_total,
            reward=float(i),
            log_prob=0.1,
            value=0.5,
            done=(i == buffer_size - 1),
            legal_mask=dummy_legal_mask,
        )

    experience_buffer.compute_advantages_and_returns(0.0)

    # Learn should handle uneven minibatch split correctly
    metrics = agent.learn(experience_buffer)

    # Should complete successfully
    assert metrics is not None
    assert "ppo/policy_loss" in metrics
    assert not np.isnan(metrics["ppo/policy_loss"])
    assert not np.isinf(metrics["ppo/policy_loss"])
