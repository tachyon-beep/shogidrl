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
- Implement `Piece` class.
- Implement `ShogiGame` class with board setup, move logic, and state management.
- Write unit tests for each method and class as they are implemented.
- Ensure all code passes linting (e.g., flake8, black).

### Phase 2: State Representation & Action Mapping
- Implement `get_observation()` in `ShogiGame`.
- Implement `PolicyOutputMapper` in `utils.py`.
- Write unit tests for state representation and action mapping.

### Phase 3: RL Loop Structure & Random Agent
- Implement `ActorCritic` in `neural_network.py` (initial dummy forward pass).
- Implement `PPOAgent` in `ppo_agent.py` (random action selection initially).
- Implement `ExperienceBuffer` in `experience_buffer.py`.
- Implement `train.py` main loop (random agent, no learning).
- Write unit tests for all new classes and methods.

### Phase 4: Full PPO Implementation
- Complete `ActorCritic` and `PPOAgent` with full PPO logic.
- Integrate GAE and learning steps.
- Write/expand unit tests for RL logic.

### Phase 5: Advanced Rules & Refinements
- Implement advanced Shogi rules (Nifu, Uchi Fu Zume, Sennichite, etc.).
- Add model saving/loading, logging, and evaluation.
- Tune hyperparameters and refine architecture as needed.
- Expand tests for edge cases and advanced rules.

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
- All code and tests must pass type checking with mypy.
- All code and tests must pass linting and formatting before merging or committing.
