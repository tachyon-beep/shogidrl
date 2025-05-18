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

### Phase 5: Advanced Rules & Engine Completion
- **I. Piece Drops Implementation (`shogi_engine.py`)**
  - [ ] Manage Pieces in Hand (initialize `self.hands`, update `make_move`/`undo_move` for captures)
  - [ ] Generate Legal Drop Moves (in `get_legal_moves`, handle `(None, None, r_to, c_to, piece_type, is_drop=True)`)
  - [ ] Implement Legality Checks for Drops (Nifu, Illegal Drop Squares for P/L/N, Uchi Fu Zume integration)
  - [ ] Execute Drop Moves (in `make_move`, remove from hand, place on board)
  - [ ] Update `get_observation` to include hand pieces representation
- **II. Promotions Implementation (`shogi_engine.py`)**
  - [ ] Define Promotion Zones
  - [ ] Generate Promotion Moves (in `get_legal_moves`, handle optional and forced promotions)
  - [ ] Execute Promotion Moves (in `make_move`, update piece state if `promote=True`)
  - [ ] Update `undo_move` to correctly revert promotions
- **III. Advanced Legality and Game State (`shogi_engine.py`)**
  - [ ] Robust Checkmate Detection (`is_checkmate`)
  - [ ] Stalemate Detection (`is_stalemate` - no legal moves, not in check)
  - [ ] Ensure Sennichite detection is robust and integrated into game over checks
- **IV. `PolicyOutputMapper` Expansion (`utils.py`)**
  - [ ] Design and implement a comprehensive mapping for all moves (normal, drops, promotions)
  - [ ] Implement `move_to_index`, `index_to_move`, `get_legal_mask`
  - [ ] Update `NUM_ACTIONS_TOTAL` in `config.py`
- **V. Test Coverage for Engine Completion**
  - [ ] Ensure all existing advanced rule tests in `test_shogi_engine.py` pass
  - [ ] Add new unit tests for drop logic (all piece types, valid/invalid squares, hand updates)
  - [ ] Add new unit tests for promotion logic (optional/forced, all promotable pieces, zone checks)
  - [ ] Add new unit tests for `get_observation` with hand pieces
  - [ ] Add new unit tests for `undo_move` with drops and promotions
- [ ] Add model saving/loading
- [x] Add model saving/loading
- [x] Add logging and evaluation
- [ ] Tune hyperparameters and refine architecture as needed
- [ ] Expand tests for edge cases and advanced rules:
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

tests/
    conftest.py
    test_shogi_engine.py
    test_neural_network.py
    test_ppo_agent.py
    test_experience_buffer.py
    test_utils.py
    test_train.py
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
