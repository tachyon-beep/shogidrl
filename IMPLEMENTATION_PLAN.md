# DRL Shogi Client: Implementation Plan

## 1. Implementation Approach
- **Phased, Test-Driven Development (TDD):**
  - Implement each class and function as described in DESIGN.md.
  - Immediately write a unit test for each new class/function.
  - After completing each class (or each function for complex/mission-critical logic), write and run corresponding tests before proceeding.
  - Run linting and resolve any errors after each implementation and test.
- **Code Organization:**
  - All source code goes in the `keisei/` directory.
  - All tests go in the `tests/` directory.
  - Shared test fixtures and scaffolding are placed in `tests/conftest.py`.

## 2. Implementation Phases

### Phase 1: Core Game Mechanics (`shogi_engine.py`)
- [x] Implement `Piece` class.
- [x] Implement `ShogiGame` class with board setup, move logic, and state management.
- [x] Write unit tests for each method and class as they are implemented.
- [x] Ensure all code passes linting (e.g., flake8, black).

### Phase 2: State Representation & Action Mapping
- [x] Implement `get_observation()` in `ShogiGame`.
- [x] Implement `PolicyOutputMapper` in `utils.py`.
- [x] Write unit tests for state representation and action mapping.

### Phase 3: RL Loop Structure & Random Agent
- [x] Implement `ActorCritic` in `neural_network.py` (initial dummy forward pass).
- [x] Implement `PPOAgent` in `ppo_agent.py` (random action selection initially).
- [x] Implement `ExperienceBuffer` in `experience_buffer.py`.
- [x] Implement `train.py` main loop (random agent, no learning).
- [x] Write unit tests for all new classes and methods.

### Phase 4: Full PPO Implementation
- [x] Complete `ActorCritic` and `PPOAgent` with full PPO logic.
- [x] Integrate GAE and learning steps.
- [x] Write/expand unit tests for RL logic.

### Phase 5: Integration, Testing, and Refinement (Iterative)
- **I. Piece Drops Implementation (`shogi_engine.py`)**
  - [x] Manage Pieces in Hand (initialize `self.hands`, update `make_move`/`undo_move` for captures)
  - [x] Generate Legal Drop Moves (in `get_legal_moves`, handle `(None, None, r_to, c_to, piece_type, is_drop=True)`)
  - [x] Implement Legality Checks for Drops (Nifu, Illegal Drop Squares for P/L/N, Uchi Fu Zume integration)
  - [x] Execute Drop Moves (in `make_move`, remove from hand, place on board)
  - [x] Update `get_observation` to include hand pieces representation
- **II. Promotions Implementation (`shogi_engine.py`)**
  - [x] Define Promotion Zones
  - [x] Generate Promotion Moves (in `get_legal_moves`, handle optional and forced promotions)
  - [x] Execute Promotion Moves (in `make_move`, update piece state if `promote=True`)
  - [x] Update `undo_move` to correctly revert promotions
- **III. Advanced Legality and Game State (`shogi_engine.py`)**
  - [x] Implement `is_nifu` detection logic
  - [x] Integrate `is_nifu` into drop move legality checks
  - [x] Implement `is_uchi_fu_zume` detection logic
  - [x] Integrate `is_uchi_fu_zume` into pawn drop move legality checks
  - [x] Implement `detect_sennichite` logic
  - [x] Integrate `sennichite` into game over condition and repetition history
  - [x] Robust Checkmate Detection (`is_checkmate`)
  - [x] Stalemate Detection (`is_stalemate` - no legal moves, not in check)
- **IV. Expand `PolicyOutputMapper` and Integrate with `PPOAgent` (COMPLETE)**
  - [x] Expand `PolicyOutputMapper` in `keisei/utils.py` to include all possible moves (board moves with and without promotion, drop moves).
  - [x] Ensure `total_actions` in `PolicyOutputMapper` is correctly calculated (should be 13527).
  - [x] Implement `shogi_move_to_policy_index(move)` and `policy_index_to_shogi_move(idx)` in `PolicyOutputMapper`.
  - [x] Implement `get_legal_mask(legal_shogi_moves, device)` in `PolicyOutputMapper`.
  - [x] Update `PPOAgent` constructor to take `PolicyOutputMapper` instance and derive `num_actions_total`.
  - [x] Update `PPOAgent.select_action` to use `PolicyOutputMapper.get_legal_mask()` for filtering actions.
  - [x] Ensure `PPOAgent.select_action` returns the selected Shogi move, policy index, log probability, and value.
  - [x] Update `train.py` to correctly instantiate and use the updated `PPOAgent` and `PolicyOutputMapper`.
  - [x] Fix `ShogiGame.reset()` to return an observation.
  - [x] Fix `ShogiGame.make_move()` to return `(next_observation, reward, done, info_dict)`.
  - [x] Update `ExperienceBuffer.add` to accept and store `log_prob` and `value`.
  - [x] Run all tests (`pytest`) and ensure they pass.
- **V. Comprehensive Unit and Integration Testing** (In Progress)
  - [x] Ensure all existing advanced rule tests in `test_shogi_engine.py` pass.
  - [x] Add new unit tests for drop logic in `test_shogi_rules_logic.py` or `test_shogi_engine.py`.
  - [x] Add new unit tests for promotion logic (optional/forced, all promotable pieces, zone checks)
  - [x] Add new unit tests for `get_observation` with hand pieces
  - [x] Add new unit tests for `undo_move` with drops and promotions
  - [x] Tune hyperparameters and refine architecture as needed
  - [x] Expand tests for edge cases and advanced rules:
    - [x] Nifu edge cases (promoted pawns, after captures, pawn drops)
    - [x] Uchi Fu Zume edge cases (complex king escapes, non-pawn drops)
    - [x] Sennichite edge cases (repetition with drops, captures, and promotions)
    - [x] Illegal drops and move legality in rare board states
    - [x] Checkmate and stalemate detection edge cases

## 3. Test Strategy

- **Test-Driven Development:**
  - Write a unit test for every new function and class immediately after implementation.
  - Use `pytest` as the test runner.
- **Test Organization:**
  - All tests are placed in the `tests/` directory, mirroring the structure of the `keisei/` codebase.
  - Shared fixtures, mocks, and test utilities are placed in `tests/conftest.py` only.
- **Test Coverage:**
  - Cover all public methods and critical private methods.
  - Include tests for normal cases, edge cases, and error handling.
- **Best Practices:**
  - Read all relevant code before making a decision on the best way to proceed.
  - Do not change working live code to resolve a failing test; instead, evaluate if the test is testing meaningful functionality and follows best practice.
  - Use descriptive test names and docstrings.
  - Keep tests isolated and independent.
  - Use parameterized tests for repeated logic.
  - Mock external dependencies where appropriate.
- **Linting:**
  - Run linting (flake8, black) after every implementation and test addition.
  - Resolve all linting errors before proceeding.
- **Continuous Integration:**
  - (Optional) Set up CI to run tests and linting on every commit.

## 4. Directory Structure
```
keisei/
    shogi_engine.py
    neural_network.py
    ppo_agent.py
    experience_buffer.py
    utils.py
    train.py
    config.py
    shogi/
        __init__.py
        shogi_core_definitions.py
        shogi_game.py
        shogi_rules_logic.py
        shogi_move_execution.py
        shogi_game_io.py

tests/
    conftest.py
    test_shogi_engine.py
    test_neural_network.py
    test_ppo_agent.py
    test_experience_buffer.py
    test_utils.py
    test_train.py
    test_logger.py
    test_evaluate.py
    test_model_save_load.py
```

## 5. Next Steps
- Scaffold the `keisei/` and `tests/` directories.
- Begin with `Piece` and `ShogiGame` classes in `shogi_engine.py` and their corresponding tests.
- Follow the phased plan, ensuring tests and linting are always up to date.

## Coding Standards
- All functions and methods must have type annotations for arguments and return values.
- All class attributes must have type annotations in __init__.
- Use snake_case for function, method, and variable names.
- Use UpperCamelCase for class names.
- Maximum line length is 120 characters (flake8 enforced).
- Use docstrings for all public classes, methods, and functions.
- Use pytest for all tests, and place them in the tests/ directory.
- Use pylint and flake8 for linting; resolve all warnings and errors.
- Use black for code formatting.
- Protected/private members may be accessed in tests for foundational logic, with a comment or linter directive.
- Under no circumstances should you implement or add lines like `# pylint: disable=protected-access`, `# noqa`, `# type: ignore`, or any similar directive to bypass linting, type checking, or test errors. All code and tests must comply with linting and type checking requirements by design, not by disabling checks. This applies to all files, tests, and instructions in the project.
- All code and tests must pass type checking with mypy.
- All code and tests must pass linting and formatting before merging or committing.

<details>
<summary>IV. Fix `ShogiGame` and `ExperienceBuffer` APIs and ensure all tests pass (COMPLETE)</summary>

- **Task**: Modify `ShogiGame.reset()` to return an observation (DONE)
- **Task**: Modify `ShogiGame.make_move()` to return `(next_observation, reward, done, info_dict)` (DONE)
- **Task**: Update `ExperienceBuffer.add()` to accept `log_prob` and `value` (DONE)
- **Task**: Update `train.py` to correctly use the modified `ShogiGame` and `ExperienceBuffer` APIs. (DONE)
- **Task**: Ensure all existing unit tests pass after these changes. (DONE)
- **Task**: Add new unit tests for drop logic in `shogi_rules_logic.py` (e.g., nifu, uchi-fu-zume, cannot drop piece with no moves). (DONE)
- **Task**: Add new unit tests for promotion logic in `shogi_rules_logic.py` (e.g., can_promote, must_promote, promotion zones). (DONE)
</details>

<details>
<summary>V. Continue Test Coverage and Refinement</summary>

- **Task**: Add new unit tests for `ShogiGame.get_observation()` with pieces in hand.
- **Task**: Tune hyperparameters and refine architecture as needed
- **Task**: Expand tests for edge cases and advanced rules:
  - [ ] Nifu edge cases (promoted pawns, after captures, pawn drops)
  - [ ] Uchi Fu Zume edge cases (complex king escapes, non-pawn drops)
  - [ ] Illegal drops and move legality in rare board states
    # Nifu (two pawns on the same file) edge cases.
            # - [x] Test dropping a pawn is illegal if an unpromoted pawn of the same color is on the file.
            # - [x] Test dropping a pawn is legal if a *promoted* pawn (Tokin) of the same color is on the file.
            # Uchi Fu Zume (illegal pawn drop checkmate) edge cases.
            # - [x] Test pawn drop is illegal if it's mate, king has no moves, and no piece can capture the checking pawn.
            # - [x] Test pawn drop is legal if king can escape (diagonally, capture pawn).
            # - [x] Test pawn drop is legal if another piece can capture the checking pawn.
            # - [x] Test pawn drop is legal if check can be blocked by another piece (making it not mate).
            # Illegal drops and move legality in rare or complex board states.
            # - [x] Test pinned pieces (Rook, Bishop) cannot expose king.
            # - [x] Test king cannot move into check.
</details>
