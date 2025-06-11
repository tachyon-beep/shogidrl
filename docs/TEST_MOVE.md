# Test File Reorganization Plan

This document outlines the plan to reorganize the test files within the `/home/john/keisei/tests/` directory for better structure and maintainability.

## Plan Steps

1.  **Create New Subdirectories**: Create new subdirectories within `/home/john/keisei/tests/` based on common testing categories.
2.  **Move Files**: Move existing test files into these new subdirectories based on their primary focus.
3.  **Update Imports**: Adjust Python `import` statements in all affected files (moved test files, `conftest.py` files, and potentially any main codebase files that might reference test utilities) to reflect the new file locations.
4.  **Handle `__init__.py`**: Ensure `__init__.py` files are present in all new test subdirectories to make them recognizable as Python packages.
5.  **Run Tests and Fix Errors**: Iteratively run the test suite, identify any import errors or other path-related issues, and fix them until all tests pass.

## Proposed Directory Structure & File Moves

*   **`tests/shogi/`** (For tests related to Shogi game logic, rules, board, moves, etc.)
    *   `test_shogi_utils.py`
    *   `test_move_formatting.py`
    *   `test_observation_constants.py`
    *   `test_legal_mask_generation.py`
    *   `test_shogi_engine_integration.py`
    *   `test_shogi_rules_and_validation.py`
    *   `test_shogi_game_mock_comprehensive.py`
    *   `test_shogi_game_rewards.py`
    *   `test_shogi_board_alignment.py`
    *   `test_reward_with_flipped_perspective.py`
    *   `test_shogi_game_core_logic.py`
    *   `test_shogi_game_io.py`
    *   `test_shogi_core_definitions.py`
    *   `test_features.py` (related to board features)

*   **`tests/core/`** (For tests of core PPO agent components, neural network, experience buffer, model management, etc.)
    *   `test_ppo_agent_learning.py`
    *   `test_neural_network.py`
    *   `test_ppo_agent_enhancements.py`
    *   `test_resnet_tower.py`
    *   `test_actor_critic_refactoring.py`
    *   `test_scheduler_factory.py`
    *   `test_ppo_agent_core.py`
    *   `test_ppo_agent_edge_cases.py`
    *   `test_experience_buffer.py`
    *   `test_model_manager_init.py`
    *   `test_model_manager_checkpoint_and_artifacts.py`
    *   `test_model_save_load.py`
    *   `test_checkpoint.py`

*   **`tests/training/`** (For tests related to the training loop, trainer, session management, seeding, metrics, etc.)
    *   `test_step_manager.py`
    *   `test_trainer_resume_state.py`
    *   `test_seeding.py`
    *   `test_trainer_config.py`
    *   `test_metrics_manager.py`
    *   `test_env_manager.py`
    *   `test_training_loop_manager.py`
    *   `test_session_manager.py`

*   **`tests/evaluation/`** (Consolidate existing and new evaluation-related tests)
    *   `__init__.py` (from `tests/evaluation/__init__.py`)
    *   `test_evaluation_manager.py` (from `tests/evaluation/test_evaluation_manager.py`)
    *   `test_in_memory_evaluation.py` (from `tests/evaluation/test_in_memory_evaluation.py`)
    *   `test_evaluate_evaluator_modern.py` (from `tests/evaluation/test_evaluate_evaluator_modern.py`)
    *   `test_evaluate_agent_loading.py` (from `tests/evaluation/test_evaluate_agent_loading.py`)
    *   `test_evaluate_main.py` (from `tests/evaluation/test_evaluate_main.py`)
    *   `test_evaluate_evaluator.py` (from `tests/evaluation/test_evaluate_evaluator.py`)
    *   `test_elo_registry.py` (from `tests/evaluation/test_elo_registry.py`)
    *   `test_evaluate_opponents.py` (from `tests/evaluation/test_evaluate_opponents.py`)
    *   `test_evaluation_callback_integration.py` (from `tests/evaluation/test_evaluation_callback_integration.py`)
    *   `strategies/test_ladder_evaluator.py` (from `tests/evaluation/strategies/test_ladder_evaluator.py`)
    *   `strategies/test_single_opponent_evaluator.py` (from `tests/evaluation/strategies/test_single_opponent_evaluator.py`)
    *   `strategies/test_tournament_evaluator.py` (from `tests/evaluation/strategies/test_tournament_evaluator.py`)
    *   `strategies/test_benchmark_evaluator.py` (from `tests/evaluation/strategies/test_benchmark_evaluator.py`)
    *   `test_previous_model_selector.py` (from `tests/evaluation/test_previous_model_selector.py`)
    *   `test_evaluate_evaluator_modern_fixed.py` (from `tests/evaluation/test_evaluate_evaluator_modern_fixed.py`)
    *   `test_performance_validation_simple.py` (from `tests/evaluation/test_performance_validation_simple.py`)
    *   `test_performance_validation.py` (from `tests/evaluation/test_performance_validation.py`)
    *   `test_opponent_pool.py` (from `tests/evaluation/test_opponent_pool.py`)
    *   `test_model_manager.py` (from `tests/evaluation/test_model_manager.py`)
    *   `test_core.py` (from `tests/evaluation/test_core.py` - to be reviewed for potential rename if conflicting with `tests/core/`)
    *   `conftest.py` (from `tests/evaluation/conftest.py`)
    *   `test_evaluate.py` (from `tests/test_evaluate.py`)
    *   `test_enhanced_evaluation_features.py` (from `tests/test_enhanced_evaluation_features.py`)

*   **`tests/integration/`** (For tests that check interactions between multiple components)
    *   `test_remediation_integration.py`
    *   `test_integration_smoke.py`
    *   `test_wandb_integration.py`
    *   `test_trainer_training_loop_integration.py`
    *   `test_trainer_session_integration.py`
    *   `test_configuration_integration.py`

*   **`tests/parallel/`** (For tests related to parallel processing)
    *   `test_parallel_system.py`
    *   `test_parallel_smoke.py`

*   **`tests/utils/`** (For tests of utility functions and test-specific utilities)
    *   `test_utils_checkpoint_validation.py`
    *   `test_logger.py`
    *   `mock_utilities.py`
    *   `test_dependencies.py`
    *   `test_profiling.py` (Categorized here for now)

*   **`tests/display/`** (For tests related to UI/display components)
    *   `test_display_components.py`
    *   `test_display_infrastructure.py`

*   **`tests/e2e/`** (For end-to-end tests of major functionalities)
    *   `test_train.py`

*   **Remaining in `tests/` (Root):**
    *   `__init__.py`
    *   `conftest.py` (Global conftest)

## `conftest.py` Handling

*   The global `tests/conftest.py` will remain for fixtures used across multiple test categories.
*   The existing `tests/evaluation/conftest.py` will be moved to the consolidated `tests/evaluation/` directory.
*   If other subdirectories require specific, localized fixtures, new `conftest.py` files will be created within them.

This reorganization aims to improve the clarity and organization of the test suite.
