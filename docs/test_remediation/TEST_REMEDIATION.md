Okay, here's a detailed tasking statement for an AI programmer based on the previous test file review.

---
## Tasking Statement: Test Suite Refinement and Enhancement (Keisei ShogiDRL)

**Objective:** Improve the robustness, coverage, clarity, and maintainability of the existing Pytest test suite for the Keisei ShogiDRL project.

**Project Lead:** AI Architect (via Gemini)
**Assigned To:** AI Programmer
**Due Date:** TBD
**Date Issued:** 29 May 2025

**Background:**
A recent peer review of the test suite has identified several areas for improvement. This tasking statement outlines specific actions to address these findings. The goal is to ensure the test suite provides reliable and comprehensive validation of the application's functionality.

---
### I. Test File Organization and Deduplication

1.  **Consolidate Shogi Game Logic Tests:**
    * Merge `test_shogi_game.py` and `test_shogi_game_core_logic.py`. Retain the name `test_shogi_game_core_logic.py`. Ensure all unique tests are preserved and any identical tests are deduplicated.
    * **File to modify/create:** `tests/test_shogi_game_core_logic.py`
    * **File(s) to remove:** `tests/test_shogi_game.py`
2.  **Consolidate Shogi Game I/O Tests:**
    * Merge `test_shogi_game_io.py` and `test_shogi_game_observation_and_io.py`. Retain the name `test_shogi_game_io.py`. Ensure all unique tests are preserved.
    * **File to modify/create:** `tests/test_shogi_game_io.py`
    * **File(s) to remove:** `tests/test_shogi_game_observation_and_io.py`
3.  **Centralize `Piece` Class Tests:**
    * Move all tests for the `Piece` class from `tests/test_shogi_engine_integration.py` to `tests/test_shogi_core_definitions.py`.
    * Ensure no redundancy with existing tests in `test_shogi_core_definitions.py`.
    * **Files to modify:** `tests/test_shogi_core_definitions.py`, `tests/test_shogi_engine_integration.py`
4.  **Relocate Remaining `test_shogi_engine_integration.py` Tests:**
    * Review any remaining tests in `tests/test_shogi_engine_integration.py` (after `Piece` tests are moved).
    * Relocate game logic tests to `tests/test_shogi_game_core_logic.py` or `tests/test_shogi_rules_and_validation.py` as appropriate.
    * Aim to deprecate and remove `tests/test_shogi_engine_integration.py` if all relevant tests can be better housed elsewhere.
    * **Files to modify:** `tests/test_shogi_rules_and_validation.py`, `tests/test_shogi_game_core_logic.py`
    * **File(s) to remove (potentially):** `tests/test_shogi_engine_integration.py`
5.  **Consolidate Trainer-Session Integration Tests:**
    * Review `tests/test_trainer_session_integration.py` and `tests/test_trainer_session_integration_fixed.py`.
    * Merge all unique and valid tests into `tests/test_trainer_session_integration_fixed.py`.
    * Rename `tests/test_trainer_session_integration_fixed.py` to `tests/test_trainer_session_integration.py`.
    * **File to modify/create:** `tests/test_trainer_session_integration.py`
    * **File(s) to remove:** The original `tests/test_trainer_session_integration.py` (if distinct from `_fixed`) and the `_fixed` suffixed file after renaming.
6.  **Remove Empty Test Files:**
    * Delete `tests/test_wandb_integration_clean.py` as it's currently empty.
    * **File(s) to remove:** `tests/test_wandb_integration_clean.py`

---
### II. Enhance Test Coverage (Missing Tests)

1.  **`ExperienceBuffer` (`tests/test_experience_buffer.py`):**
    * Implement detailed tests for `compute_advantages_and_returns()`, verifying the correctness of GAE calculations with sample data.
    * Implement tests for `get_batch()`, ensuring correct tensor shapes, dtypes, and device placement. Test the error path (empty dict return on stacking issues) and consider if an exception should be raised instead.
    * Implement tests for `clear()`.
    * Test behavior when `add()` is called on a full buffer.
2.  **`PPOAgent.learn` (`tests/test_ppo_agent.py`):**
    * Expand beyond a smoke test.
    * Verify individual loss components (policy, value, entropy) are calculated as expected given controlled inputs.
    * Mock `optimizer.step()` and check if gradients are computed (e.g., by checking `param.grad` is not None for model parameters before `optimizer.zero_grad()`).
    * Test KL divergence calculation.
3.  **Model `get_action_and_value`/`evaluate_actions`:**
    * For `tests/test_resnet_tower.py`: Add detailed tests for `ActorCriticResTower.get_action_and_value()` and `ActorCriticResTower.evaluate_actions()`. Focus on:
        * Correct application of `legal_mask`.
        * Behavior with an all-`False` `legal_mask` (robust NaN handling or defined error).
        * Deterministic vs. stochastic action selection.
    * For `tests/test_neural_network.py`: Add similar tests for the minimal `ActorCritic` if it's intended for any use beyond a placeholder. Clarify its role; if it's purely a placeholder for `PPOAgent`'s default, minimal testing is fine.
4.  **`Trainer` (`tests/test_trainer_session_integration.py` or a new `test_trainer.py`):**
    * **CRITICAL:** Implement tests to specifically verify the correct restoration of training state (`global_timestep`, `total_episodes_completed`, `black_wins`, `white_wins`, `draws`) in the `Trainer` instance when resuming from a checkpoint. This involves ensuring `ModelManager.handle_checkpoint_resume` effectively communicates loaded stats back to `Trainer`.
    * Consider a minimal end-to-end test of `Trainer.run_training_loop()` for a few steps, mocking out heavy components like actual model training and evaluation callbacks, but verifying the loop structure, step increments, and PPO update calls.
5.  **Utilities (`tests/test_utils.py`):**
    * Add dedicated unit tests for `utils.utils.load_config`, covering:
        * Loading default config.
        * Loading from YAML and JSON override files.
        * Correct application of CLI overrides (dot notation and flat key mapping).
        * Correct precedence of overrides.
    * Add dedicated unit tests for `utils.utils.generate_run_name`, covering:
        * Use of explicit `run_name`.
        * Auto-generation format with different prefixes, model types, and features.
        * Timestamp inclusion.
6.  **Shogi Game I/O (`tests/test_shogi_game_io.py`):**
    * Add more tests for `game_to_kif` focusing on KIF standard compliance for move notation (e.g., `７六歩` vs. USI-like `7g7f`) and edge cases (e.g., games with only drops, early terminations).
    * Add more tests for `sfen_to_move_tuple` with a wider variety of malformed SFEN move strings.
7.  **Shogi Rules Logic (`tests/test_shogi_rules_and_validation.py`):**
    * Expand tests for `is_in_check` with more diverse board positions.
    * Test `generate_piece_potential_moves` more thoroughly with various configurations of blocking pieces (friendly and opponent).
    * Review the `check_for_uchi_fu_zume` test (`test_uchifuzume_pawn_drop_check_but_not_mate_due_to_block_not_uchifuzume`) to ensure it covers scenarios where a block (not just king escape or pawn capture) negates uchi_fu_zume. If the current function doesn't support this, note it as a potential enhancement for the game logic.
    * Add specific tests for `must_promote_specific_piece`.

---
### III. Address Errors and Improve Existing Tests

1.  **`mock_utilities.MockPolicyOutputMapper`:**
    * Update hardcoded `action_space_size` from `2187` to `13527` to match the main application.
    * Make `get_move_from_policy_index` more flexible if tests require varied outputs based on the input index, or clearly document its fixed output.
2.  **`test_experience_buffer.py`:**
    * Modify `dummy_legal_mask` to use a size derived from `PolicyOutputMapper().get_total_actions()` or a mocked equivalent, instead of an arbitrary size like `10`.
3.  **`test_train.py` (`test_train_resume_autodetect`):**
    * Improve `run_dir` finding to be more robust, e.g., by predicting the generated run name based on the test's config and timestamp, then searching for that specific directory name.
4.  **`test_evaluate.py` (`test_execute_full_evaluation_run_*` tests):**
    * **CRITICAL:** Modify the tests and potentially the `evaluation/loop.py` logic (or ensure `Evaluator` handles it) so that a *correct* `legal_mask` (derived from `game.get_legal_moves()`) is passed to the agent's `select_action` method during evaluation, instead of a dummy all-ones mask. This is vital for meaningful evaluation.
5.  **`test_shogi_game_core_logic.py` (`test_undo_move_multiple_moves`):**
    * Investigate the need for the manual "yank" of a piece (`game.set_piece(5, 1, None)`). Ideally, `undo_move` should fully restore the state without requiring such manual interventions in the test. This might indicate a subtle bug in `undo_move` or `make_move` history tracking when `set_piece` is used mid-sequence in a test.
6.  **`test_shogi_game_core_logic.py` (Termination Tests):**
    * For tests like `test_game_termination_checkmate_stalemate`, strive to make the game reach its terminal state through `make_move` calls rather than manually setting `game.game_over` and `game.winner`. This will better test the game's own termination detection logic. If specific SFENs are meant to be *already* terminal when loaded by `from_sfen`, then `from_sfen` should ideally evaluate and set these flags, or a separate `game.evaluate_termination()` method could be called.
7.  **`test_trainer_session_integration_fixed.py` (to be `test_trainer_session_integration.py`):**
    * Ensure `mock_config` dynamically sets `env_config.input_channels` based on `training_config.input_features` and `FEATURE_SPECS` to avoid inconsistencies. For example, if `input_features="core46+all"`, `input_channels` should be 51, not a fixed 46.

---
### IV. General Test Quality Improvements

1.  **Logging in Tests:** Replace `print()` statements used for debugging within tests with a proper logging framework (e.g., Python's `logging` module configured for tests, or pytest's built-in capturing) if persistent debug output is needed.
2.  **Test Naming:** After file consolidation, ensure all test function names are clear, descriptive, and follow a consistent pattern (e.g., `test_component_scenario_expectedBehavior`).
3.  **Test Data and Setups:** For complex Shogi rule tests, continue using SFEN strings. Add comments to SFEN strings explaining the key aspects of the position being tested.
4.  **Fixtures:** Maximize the use of `pytest` fixtures for setting up common objects like `PolicyOutputMapper`, `ShogiGame` instances in specific states, and configurations. Ensure fixture scope (`function`, `module`, `session`) is appropriate.
5.  **Magic Numbers:** Review tests for hardcoded magic numbers (e.g., action space sizes, specific indices, plane counts). Where possible, derive these from configuration, constants defined in the source code, or from methods of the objects under test (e.g., `mapper.get_total_actions()`).
6.  **Assertion Clarity:** Ensure assertions are precise and error messages (e.g., in `f"Error: {variable}"`) are informative.

---

**Acceptance Criteria:**
* All specified refactoring and deduplication tasks are completed.
* New tests for missing coverage areas are implemented and pass.
* Identified errors/bugs in existing tests are fixed, and tests pass.
* General quality improvements are applied across the test suite.
* The overall test suite passes reliably.
* Code coverage metrics (if available) show an improvement or maintain a high level for the tested modules.

**Notes:**
* Prioritize the "CRITICAL" items.
* When mocking, ensure mocks accurately reflect the interface of the object being mocked.
* Write new tests to be independent and avoid unintended dependencies on the state of other tests.
* Remember to run linters and formatters on the updated test code.