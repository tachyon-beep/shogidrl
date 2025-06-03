"""
Tests for the main execution functions in the evaluation system.

This module contains tests for execute_full_evaluation_run function which orchestrates
the complete evaluation process including agent loading, opponent initialization,
W&B integration, and seeding.
"""

from unittest.mock import MagicMock, patch

import pytest

from keisei.core.ppo_agent import PPOAgent
from keisei.evaluate import execute_full_evaluation_run
from keisei.evaluation.evaluation_logger import EvaluationLogger
from keisei.evaluation.opponents.simple_heuristic_opponent import SimpleHeuristicOpponent
from keisei.evaluation.opponents.simple_random_opponent import SimpleRandomOpponent
from keisei.evaluation.policy_output_mapper import PolicyOutputMapper


# Mock decorators for execute_full_evaluation_run tests
COMMON_MAIN_MOCKS = [
    "keisei.evaluate.load_config",
    "keisei.evaluate.EvaluationLogger",  
    "keisei.evaluate.run_evaluation_loop",
    "keisei.evaluate.initialize_opponent",
    "keisei.evaluate.load_evaluation_agent",
    "keisei.evaluate.PolicyOutputMapper",
    "random.seed",
    "numpy.random.seed", 
    "torch.manual_seed",
]


def apply_mocks(mock_list):
    """Helper decorator to apply multiple patches consistently."""
    def decorator(func):
        for mock_path in reversed(mock_list):
            func = patch(mock_path)(func)
        return func
    return decorator


@apply_mocks(COMMON_MAIN_MOCKS)
def test_execute_full_evaluation_run_basic_random(
    mock_load_config,
    mock_eval_logger_class,
    mock_run_loop,
    mock_init_opponent,
    mock_load_agent,
    mock_policy_output_mapper_class,
    mock_random_seed,
    mock_np_seed,
    mock_torch_seed,
    mock_wandb_disabled,
    tmp_path,
):
    """
    Test basic functionality of execute_full_evaluation_run with random opponent.

    Verifies the complete evaluation pipeline works correctly including:
    - Agent loading from checkpoint
    - Random opponent initialization  
    - Evaluation loop execution
    - Result aggregation
    - No W&B integration when disabled
    """
    # Set up mock_load_config to return a config with the required structure
    mock_config = MagicMock()
    mock_config.env.input_channels = 46  # INPUT_CHANNELS
    mock_load_config.return_value = mock_config

    # Create a real PolicyOutputMapper instance for the test
    policy_mapper_instance = PolicyOutputMapper()

    mock_agent_instance = MagicMock(spec=PPOAgent)
    mock_agent_instance.name = "LoadedEvalAgent"
    mock_load_agent.return_value = mock_agent_instance

    mock_opponent_instance = SimpleRandomOpponent(name="EvalRandomOpponent")
    mock_init_opponent.return_value = mock_opponent_instance

    mock_logger_instance = MagicMock(spec=EvaluationLogger)
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    expected_run_loop_results = {
        "games_played": 1,
        "agent_wins": 1,
        "opponent_wins": 0,
        "draws": 0,
        "game_results": [],
        "win_rate": 1.0,
        "loss_rate": 0.0,
        "draw_rate": 0.0,
        "avg_game_length": 10.0,
    }
    mock_run_loop.return_value = expected_run_loop_results

    log_file = tmp_path / "eval_basic_run.log"
    agent_ckpt_path = "./agent_for_eval.pth"
    num_games_to_run = 1
    max_moves_for_game = 250

    # Call the function being tested
    results = execute_full_evaluation_run(
        agent_checkpoint_path=agent_ckpt_path,
        opponent_type="random",
        opponent_checkpoint_path=None,
        num_games=num_games_to_run,
        max_moves_per_game=max_moves_for_game,
        device_str="cpu",
        log_file_path_eval=str(log_file),
        policy_mapper=policy_mapper_instance,
        seed=None,
        wandb_log_eval=False,
    )

    # Verify mocks were called correctly
    mock_policy_output_mapper_class.assert_not_called()

    mock_load_agent.assert_called_once_with(
        agent_ckpt_path, "cpu", policy_mapper_instance, 46
    )
    mock_init_opponent.assert_called_once_with(
        "random", None, "cpu", policy_mapper_instance, 46
    )
    mock_eval_logger_class.assert_called_once_with(str(log_file), also_stdout=True)

    mock_run_loop.assert_called_once()
    pos_args, kw_args = mock_run_loop.call_args
    assert pos_args[0] == mock_agent_instance
    assert pos_args[1] == mock_opponent_instance
    assert pos_args[2] == num_games_to_run
    assert pos_args[3] == mock_logger_instance
    assert pos_args[4] == max_moves_for_game
    assert not kw_args

    mock_wandb_disabled["init"].assert_not_called()
    mock_wandb_disabled["log"].assert_not_called()
    mock_wandb_disabled["finish"].assert_not_called()
    mock_wandb_disabled["run"].assert_not_called()

    assert results == expected_run_loop_results


@apply_mocks(COMMON_MAIN_MOCKS)
def test_execute_full_evaluation_run_heuristic_opponent_with_wandb(
    mock_load_config,
    mock_torch_seed,
    mock_np_seed,
    mock_random_seed,
    mock_eval_logger_class,
    mock_run_loop,
    mock_init_opponent,
    mock_load_agent,
    mock_policy_output_mapper_class,
    mock_wandb_active,
    tmp_path,
):
    """
    Test execute_full_evaluation_run with heuristic opponent and W&B integration.

    Verifies:
    - Heuristic opponent initialization
    - W&B integration with proper logging and configuration
    - Wandb project, entity, and run name configuration
    - Final summary metrics logging to W&B
    """
    # Set up mock_load_config
    mock_config = MagicMock()
    mock_config.env.input_channels = 46
    mock_load_config.return_value = mock_config

    policy_mapper_instance = PolicyOutputMapper()

    mock_agent_instance = MagicMock(spec=PPOAgent)
    mock_agent_instance.name = "LoadedMainAgentWandb"
    mock_load_agent.return_value = mock_agent_instance

    mock_opponent_instance = SimpleHeuristicOpponent(name="MainHeuristicOpponent")
    mock_init_opponent.return_value = mock_opponent_instance

    mock_logger_instance = MagicMock(spec=EvaluationLogger)
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    expected_run_loop_results = {
        "games_played": 2,
        "agent_wins": 1,
        "opponent_wins": 1,
        "draws": 0,
        "game_results": [],
        "win_rate": 0.5,
        "loss_rate": 0.5,
        "draw_rate": 0.0,
        "avg_game_length": 20.0,
    }
    mock_run_loop.return_value = expected_run_loop_results

    log_file = tmp_path / "eval_wandb.log"
    agent_ckpt_path = "./agent_wandb.pth"
    num_games_to_run = 2
    max_moves_for_game = 300
    seed_to_use = None

    # W&B setup
    wandb_project_name = "test_project"
    wandb_entity_name = "test_entity"
    wandb_run_name_custom = "custom_eval_run"

    mock_wandb_active["init"].return_value.name = wandb_run_name_custom
    mock_wandb_active["run"].name = wandb_run_name_custom

    # Call function
    results = execute_full_evaluation_run(
        agent_checkpoint_path=agent_ckpt_path,
        opponent_type="heuristic",
        opponent_checkpoint_path=None,
        num_games=num_games_to_run,
        max_moves_per_game=max_moves_for_game,
        device_str="cpu",
        log_file_path_eval=str(log_file),
        policy_mapper=policy_mapper_instance,
        seed=seed_to_use,
        wandb_log_eval=True,
        wandb_project_eval=wandb_project_name,
        wandb_entity_eval=wandb_entity_name,
        wandb_run_name_eval=wandb_run_name_custom,
        wandb_reinit=True,
    )

    # Verify calls
    mock_policy_output_mapper_class.assert_not_called()
    mock_load_agent.assert_called_once_with(
        agent_ckpt_path, "cpu", policy_mapper_instance, 46
    )
    mock_init_opponent.assert_called_once_with(
        "heuristic", None, "cpu", policy_mapper_instance, 46
    )
    mock_eval_logger_class.assert_called_once_with(str(log_file), also_stdout=True)

    # Verify run_evaluation_loop call
    run_loop_pos_args, run_loop_kwargs = mock_run_loop.call_args
    assert run_loop_pos_args[0] == mock_agent_instance
    assert run_loop_pos_args[1] == mock_opponent_instance
    assert run_loop_pos_args[2] == num_games_to_run
    assert run_loop_pos_args[3] == mock_logger_instance
    assert run_loop_pos_args[4] == max_moves_for_game
    assert not run_loop_kwargs

    # Verify W&B integration
    expected_wandb_config = {
        "agent_checkpoint": agent_ckpt_path,
        "opponent_type": "heuristic",
        "opponent_checkpoint": None,
        "num_games": num_games_to_run,
        "max_moves_per_game": max_moves_for_game,
        "device": "cpu",
        "seed": seed_to_use,
    }
    mock_wandb_active["init"].assert_called_once_with(
        project=wandb_project_name,
        entity=wandb_entity_name,
        name=wandb_run_name_custom,
        config=expected_wandb_config,
        reinit=True,
    )
    mock_wandb_active["log"].assert_called_once_with(
        {
            "eval/final_win_rate": expected_run_loop_results["win_rate"],
            "eval/final_loss_rate": expected_run_loop_results["loss_rate"],
            "eval/final_draw_rate": expected_run_loop_results["draw_rate"],
            "eval/final_avg_game_length": expected_run_loop_results["avg_game_length"],
        }
    )
    mock_wandb_active["finish"].assert_called_once()

    # Verify seeding not called when seed=None
    mock_random_seed.assert_not_called()
    mock_np_seed.assert_not_called()
    mock_torch_seed.assert_not_called()

    assert results == expected_run_loop_results


@apply_mocks(COMMON_MAIN_MOCKS)
def test_execute_full_evaluation_run_ppo_vs_ppo_with_wandb(
    mock_load_config,
    mock_torch_seed,
    mock_np_seed,
    mock_random_seed,
    mock_eval_logger_class,
    mock_run_loop,
    mock_init_opponent,
    mock_load_agent,
    mock_policy_output_mapper_class,
    mock_wandb_active,
    tmp_path,
):
    """
    Test execute_full_evaluation_run with PPO vs PPO evaluation and W&B.

    This is a complex integration test that verifies:
    - Loading two separate PPO agents (one for evaluation, one as opponent)
    - Complex mock side effects for different checkpoint paths
    - PPO opponent initialization through the opponent system
    - W&B logging for agent vs agent battles
    """
    # Set up mock_load_config
    mock_config = MagicMock()
    mock_config.env.input_channels = 46
    mock_load_config.return_value = mock_config

    policy_mapper_instance = PolicyOutputMapper()

    # Create separate mock agents for evaluation and opponent
    mock_agent_to_eval = MagicMock(spec=PPOAgent)
    mock_agent_to_eval.name = "PPOAgentToEvaluate"
    mock_opponent_agent = MagicMock(spec=PPOAgent)
    mock_opponent_agent.name = "PPOAgentOpponent"

    agent_eval_path = "./agent_to_eval.pth"
    agent_opponent_path = "./agent_opponent.pth"

    # Set up side effects for loading different agents
    def load_agent_side_effect(checkpoint_path, device, pol_mapper, in_channels):
        if checkpoint_path == agent_eval_path:
            return mock_agent_to_eval
        if checkpoint_path == agent_opponent_path:
            return mock_opponent_agent
        pytest.fail(f"Unexpected call to load_evaluation_agent with {checkpoint_path}")
        return None

    mock_load_agent.side_effect = load_agent_side_effect

    # Set up opponent initialization side effect
    def init_opponent_side_effect(
        opponent_type, opponent_path, device, pol_mapper, in_channels
    ):
        if opponent_type == "ppo" and opponent_path == agent_opponent_path:
            return mock_opponent_agent
        if opponent_type == "random":
            return SimpleRandomOpponent()
        if opponent_type == "heuristic":
            return SimpleHeuristicOpponent()
        pytest.fail(
            f"Unexpected call to initialize_opponent with {opponent_type}, {opponent_path}"
        )
        return None

    mock_init_opponent.side_effect = init_opponent_side_effect

    mock_logger_instance = MagicMock(spec=EvaluationLogger)
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    expected_results = {
        "games_played": 1,
        "agent_wins": 0,
        "opponent_wins": 1,
        "draws": 0,
        "game_results": [],
        "win_rate": 0.0,
        "loss_rate": 1.0,
        "draw_rate": 0.0,
        "avg_game_length": 15.0,
    }
    mock_run_loop.return_value = expected_results

    log_file = tmp_path / "eval_ppo_vs_ppo.log"
    num_games_val = 1
    max_moves_val = 200
    wandb_project_val = "ppo_battle_project"

    # Configure W&B mock
    mock_wandb_active["init"].return_value.name = "test_ppo_run"
    mock_wandb_active["run"].name = "test_ppo_run"

    results = execute_full_evaluation_run(
        agent_checkpoint_path=agent_eval_path,
        opponent_type="ppo",
        opponent_checkpoint_path=agent_opponent_path,
        num_games=num_games_val,
        max_moves_per_game=max_moves_val,
        device_str="cpu",
        log_file_path_eval=str(log_file),
        policy_mapper=policy_mapper_instance,
        seed=None,
        wandb_log_eval=True,
        wandb_project_eval=wandb_project_val,
        wandb_entity_eval=None,
        wandb_run_name_eval="test_ppo_run",
        wandb_reinit=True,
    )

    # Verify calls
    mock_policy_output_mapper_class.assert_not_called()

    # Verify agent loading - called once directly for agent_to_eval
    assert mock_load_agent.call_count == 1
    mock_load_agent.assert_any_call(
        agent_eval_path, "cpu", policy_mapper_instance, 46
    )

    # Verify opponent initialization
    mock_init_opponent.assert_called_once_with(
        "ppo", agent_opponent_path, "cpu", policy_mapper_instance, 46
    )

    mock_eval_logger_class.assert_called_once_with(str(log_file), also_stdout=True)

    # Verify run_evaluation_loop call
    run_loop_pos_args, run_loop_kwargs = mock_run_loop.call_args
    assert run_loop_pos_args[0] == mock_agent_to_eval
    assert run_loop_pos_args[1] == mock_opponent_agent
    assert run_loop_pos_args[2] == num_games_val
    assert run_loop_pos_args[3] == mock_logger_instance
    assert run_loop_pos_args[4] == max_moves_val
    assert not run_loop_kwargs

    # Verify W&B integration
    expected_wandb_config = {
        "agent_checkpoint": agent_eval_path,
        "opponent_type": "ppo",
        "opponent_checkpoint": agent_opponent_path,
        "num_games": num_games_val,
        "max_moves_per_game": max_moves_val,
        "device": "cpu",
        "seed": None,
    }
    mock_wandb_active["init"].assert_called_once_with(
        project=wandb_project_val,
        entity=None,
        name="test_ppo_run",
        config=expected_wandb_config,
        reinit=True,
    )
    mock_wandb_active["log"].assert_called_once_with(
        {
            "eval/final_win_rate": expected_results["win_rate"],
            "eval/final_loss_rate": expected_results["loss_rate"],
            "eval/final_draw_rate": expected_results["draw_rate"],
            "eval/final_avg_game_length": expected_results["avg_game_length"],
        }
    )
    mock_wandb_active["finish"].assert_called_once()

    # Verify no seeding when seed=None
    mock_random_seed.assert_not_called()
    mock_np_seed.assert_not_called()
    mock_torch_seed.assert_not_called()

    assert results == expected_results


@apply_mocks(COMMON_MAIN_MOCKS)
def test_execute_full_evaluation_run_with_seed(
    mock_load_config,
    mock_torch_seed,
    mock_np_seed,
    mock_random_seed,
    mock_eval_logger_class,
    mock_run_loop,
    mock_init_opponent,
    mock_load_agent,
    mock_policy_output_mapper_class,
    mock_wandb_disabled,
    tmp_path,
):
    """
    Test execute_full_evaluation_run with explicit seeding.

    Verifies that when a seed is provided:
    - All three seeding functions are called with the correct seed value
    - Random, numpy, and torch seeds are set properly
    - The evaluation runs correctly with seeding applied
    """
    # Set up mock_load_config
    mock_config = MagicMock()
    mock_config.env.input_channels = 46
    mock_load_config.return_value = mock_config

    policy_mapper_instance = PolicyOutputMapper()

    mock_agent_instance = MagicMock(spec=PPOAgent)
    mock_agent_instance.name = "AgentForSeedTest"
    mock_load_agent.return_value = mock_agent_instance

    mock_opponent_instance = SimpleRandomOpponent(name="RandomOpponentForSeedTest")
    mock_init_opponent.return_value = mock_opponent_instance

    mock_logger_instance = MagicMock(spec=EvaluationLogger)
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    mock_run_loop.return_value = {
        "games_played": 1,
        "agent_wins": 1,
        "opponent_wins": 0,
        "draws": 0,
        "game_results": [],
        "win_rate": 1.0,
        "loss_rate": 0.0,
        "draw_rate": 0.0,
        "avg_game_length": 5.0,
    }

    log_file_path = tmp_path / "eval_seed_test.log"
    agent_checkpoint_file = "./agent_seed.pth"
    seed_value_to_test = 123

    # Call the function
    execute_full_evaluation_run(
        agent_checkpoint_path=agent_checkpoint_file,
        opponent_type="random",
        opponent_checkpoint_path=None,
        num_games=1,
        max_moves_per_game=100,
        device_str="cpu",
        log_file_path_eval=str(log_file_path),
        policy_mapper=policy_mapper_instance,
        seed=seed_value_to_test,
        wandb_log_eval=False,
    )

    # Verify seeding was called correctly
    mock_random_seed.assert_called_once_with(seed_value_to_test)
    mock_np_seed.assert_called_once_with(seed_value_to_test)
    mock_torch_seed.assert_called_once_with(seed_value_to_test)

    # Verify other basic functionality
    mock_policy_output_mapper_class.assert_not_called()
    mock_load_agent.assert_called_once()
    mock_init_opponent.assert_called_once()
    mock_eval_logger_class.assert_called_once()
    mock_run_loop.assert_called_once()
    mock_wandb_disabled["init"].assert_not_called()
    mock_wandb_disabled["log"].assert_not_called()
    mock_wandb_disabled["finish"].assert_not_called()
    mock_wandb_disabled["run"].assert_not_called()
