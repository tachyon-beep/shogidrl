# Training System Refactoring Plan (TRAINING_REFACTOR.md)

**Date:** May 26, 2025

## 1. Goals

The primary goals of this refactoring are to:
-   **Improve Modularity:** Decouple components for better separation of concerns, making the codebase easier to understand, maintain, and extend.
-   **Enhance Configurability:** Centralize configurations, making it easier to experiment with different hyperparameters, models, and settings without code changes.
-   **Increase Maintainability:** A cleaner, more organized structure will simplify debugging, testing, and future development efforts.
-   **Promote Reusability:** Well-defined components can be more easily reused in different contexts or projects.

## 2. Overall Project Structure and Configuration Management

### 2.1. New Directory Structure

To achieve better organization, the following directory structure changes are proposed:

*   **`keisei/core/`**: This new directory will house the fundamental algorithmic components of the RL agent.
    *   Move `keisei/ppo_agent.py` to `keisei/core/ppo_agent.py`
    *   Move `keisei/neural_network.py` to `keisei/core/neural_network.py`
    *   Move `keisei/experience_buffer.py` to `keisei/core/experience_buffer.py`
*   **`keisei/training/`**: This existing directory will be more focused, containing components specifically related to the training process.
    *   Create `keisei/training/trainer.py`: This new file will contain a `Trainer` class responsible for orchestrating the main training loop and related logic.
    *   `keisei/train.py` (existing in `keisei/`) will be refactored to be a lightweight script that initializes and uses the `Trainer` class.
*   **`keisei/evaluation/`**: This new directory will house components specifically related to the agent evaluation process.
    *   Create `keisei/evaluation/evaluator.py`: This new file will contain an `Evaluator` class responsible for managing evaluation runs.
    *   `keisei/evaluate.py` (existing in `keisei/`) will be refactored to be a lightweight script that initializes and uses the `Evaluator` class.
*   **`keisei/configs/`**: (New directory)
    *   This directory will store default configuration files (e.g., using YAML or JSON format). Examples: `default_training_config.yaml`, `default_evaluation_config.yaml`.
    *   This centralizes default parameters, making them transparent and easier to manage.
*   **`keisei/utils/`**: (New directory, replacing `keisei/utils.py`)
    *   This directory will consolidate shared utility functions and classes, keeping the `keisei/` root cleaner.
    *   The contents of the current `keisei/utils.py` will be moved here, potentially split into multiple files for better organization (e.g., `loggers.py`, `mappers.py`).

### 2.2. Configuration Handling

**Objective:** Implement a robust and centralized configuration system. This allows for easy modification of parameters for training, evaluation, model architecture, and other aspects without altering the core codebase. It also improves reproducibility.

**Guiding Principle for Configuration Settings:**
*   **Pydantic Models as the Single Source of Truth for Settings and Defaults:** The Pydantic model definitions (e.g., in a new file like `keisei/configs/schemas.py`) will define all *available* configuration settings, their types, validation rules, and their reasonable default *values*.
*   **No Hidden Defaults:** The system must not introduce or rely on default/fallback values for configuration settings that are not explicitly defined in the Pydantic models. All configurable aspects of the system must be declared within these models.
*   **Transparency in Overrides:** While CLI arguments and specific config files (like `keisei/configs/my_experiment.yaml`) can override the *values* of these settings, they cannot introduce new, previously undeclared settings (i.e., settings not defined in the Pydantic models).
*   **Strict Validation:** Pydantic will handle the validation. If a loaded configuration (after merging YAML and CLI overrides) is found to be invalid against the Pydantic models (e.g., type errors, missing required fields without defaults, or containing unrecognized settings if `extra='forbid'` is used), Pydantic will raise a validation error. The system should:
    *   Clearly report these validation errors.
    *   Terminate execution (hard crash) rather than attempting to run with a partially valid or silently modified configuration.
*   **Adding New Settings:** Any new configurable parameter required by the system must first be added to the relevant Pydantic model definition with appropriate type hints and a sensible default value if applicable. This ensures that the "schema" of the configuration is always explicit, version-controlled, and type-safe.

**Actions:**

1.  **Configuration Files and Pydantic Schemas:**
    *   Utilize YAML for configuration files stored in `keisei/configs/` (e.g., `default_config.yaml`, `experiment_specific.yaml`).
    *   Create a new file, for example `keisei/configs/schemas.py`, to house all Pydantic model definitions.
    *   Define a hierarchical structure for Pydantic models. For instance, a main `AppConfig` model could contain nested models like `TrainingConfig`, `EvaluationConfig`, `AgentConfig` (itself potentially a base for specific agent configs like `PPOAgentConfig`), `EnvironmentConfig`, `LoggingConfig`, and `WandBConfig`.
    *   Default values for settings should primarily be defined directly within the Pydantic models. An initial `keisei/configs/default_config.yaml` can be generated from these Pydantic models to serve as a user-friendly template.
    *   Use Pydantic for data validation and settings management, loading data from the YAML files into these model instances. Consider a base Pydantic model for common settings and inherit from it for specific configurations (training, evaluation, agent).

2.  **Argument Parsing (in `keisei/train.py` and `keisei/evaluate.py`):**
    *   Refactor `parse_args()` in `keisei/train.py` (and create a similar one for `keisei/evaluate.py` or a shared one in `keisei/utils/config_utils.py`).
    *   The primary command-line argument should be `--config-path` pointing to a configuration file.
    *   Allow essential overrides via command-line arguments (e.g., `--run-name`, `--savedir`, `--device`, `--seed`, `--total-timesteps`). These CLI arguments will take precedence over the config file values.

3.  **Config Loading and Merging:**
    *   Implement a dedicated config loading function (e.g., in a new `keisei/utils/config_loader.py`).
    *   This function will:
        *   Load the base configuration from the specified YAML file into Pydantic models.
        *   Merge any overrides provided through command-line arguments into the Pydantic model instances.
        *   Pydantic will automatically validate the final configuration.
    *   The `apply_config_overrides` function in `keisei/train.py` will be updated or replaced by this new system.

4.  **Config Object Propagation:**
    *   The loaded and validated configuration (as a Pydantic model instance) will be passed as a parameter to the `Trainer`, `Evaluator`, `PPOAgent`, and other relevant components.

5.  **Serialization of Used Configuration (`serialize_config` in `keisei/train.py`):**
    *   Complete the `serialize_config` function.
    *   This function should save the *final, effective* configuration (after all overrides and defaults are applied) to the specific run's log directory (e.g., as `config.yaml`). This is crucial for reproducibility and debugging.

## 3. Core Component Refactoring (`keisei/core/`)

This directory will contain the heart of the RL agent's logic.

### 3.1. `ppo_agent.py` (Moved to `keisei/core/ppo_agent.py`)

**Objective:** Ensure the `PPOAgent` is a well-defined, configurable, and self-contained component.

**Actions:**

1.  **Configuration via `__init__`:**
    *   All PPO-specific hyperparameters (e.g., `learning_rate`, `gamma`, `clip_epsilon`, `ppo_epochs`, `minibatch_size`, `value_loss_coeff`, `entropy_coef`) must be passed to the `__init__` method.
    *   These values will be sourced from the global configuration object.
2.  **Agent Naming:**
    *   The `name` parameter in `__init__` is a good practice for identifying agent instances, especially if multiple agents are used or logged.
3.  **Base Agent Class (Optional but Recommended):**
    *   Consider defining an abstract base class `BaseAgent` in a new file `keisei/core/base_agent.py`.
    *   `PPOAgent` would then inherit from `BaseAgent`. This provides a common interface if other RL algorithms (e.g., DQN, A2C) are implemented in the future.
4.  **Model Instantiation:**
    *   The `PPOAgent` currently instantiates `ActorCritic`. Ensure that the parameters for `ActorCritic` (e.g., `input_channels`, `num_actions_total`) are also derived from the configuration object passed to the `PPOAgent`.

### 3.2. `neural_network.py` (Moved to `keisei/core/neural_network.py`)

**Objective:** Ensure the `ActorCritic` network is correctly implemented, configurable, and provides the necessary functionalities for PPO.

**Actions:**

1.  **CRITICAL BUG FIX - `forward` Method:**
    *   The `forward(self, x)` method currently has a `return` statement without any values. This is a critical bug.
    *   **It must be corrected to:** `return policy_logits, value`
2.  **Complete `get_action_and_value` Method:**
    *   This method is crucial for the agent to select actions during environment interaction.
    *   **Implementation Steps:**
        1.  Obtain `policy_logits, value = self.forward(obs)`.
        2.  If `legal_mask` is provided, apply it to `policy_logits` to prevent selection of illegal actions (e.g., by setting logits of illegal actions to a very small number like -infinity before softmax).
        3.  Apply `torch.nn.functional.softmax` to the (masked) `policy_logits` to get action probabilities.
        4.  Create a `torch.distributions.Categorical` distribution from these probabilities.
        5.  If `deterministic` is true, choose the action with the highest probability (`dist.mode` or `torch.argmax`).
        6.  Otherwise, sample an action from the distribution (`dist.sample()`).
        7.  Calculate the log probability of the chosen action (`dist.log_prob(action)`).
        8.  Calculate the entropy of the distribution (`dist.entropy()`).
        9.  **Return:** The chosen `action`, its `log_prob`, the `value` estimate, and the `entropy`.
3.  **Complete `evaluate_actions` Method:**
    *   This method is used during the PPO learning phase to re-evaluate actions taken from old policies.
    *   **Implementation Steps:**
        1.  Obtain `policy_logits, value = self.forward(obs)`.
        2.  If `legal_mask` is provided (it generally should be, corresponding to the state `obs`), apply it to `policy_logits`.
        3.  Apply `torch.nn.functional.softmax` to the (masked) `policy_logits`.
        4.  Create a `torch.distributions.Categorical` distribution.
        5.  Calculate the log probability of the provided `actions` under this new distribution (`dist.log_prob(actions)`).
        6.  Calculate the entropy of this distribution (`dist.entropy()`).
        7.  **Return:** The `log_probs` of the given `actions`, the new `value` estimates for `obs`, and the `entropy`.
4.  **Network Architecture Configuration:**
    *   Make the architecture of the `ActorCritic` network (e.g., number of convolutional filters, kernel sizes, number of units in linear layers) configurable via its `__init__` method, with parameters sourced from the global configuration object.

### 3.3. `experience_buffer.py` (Moved to `keisei/core/experience_buffer.py`)

**Objective:** Ensure the `ExperienceBuffer` correctly stores and retrieves experiences, and accurately computes advantages and returns for PPO.

**Actions:**

1.  **Complete `add()` Method:**
    *   Implement the logic to store the transition: `obs`, `action`, `reward`, `log_prob`, `value`, `done`, `legal_mask`.
    *   Handle buffer capacity:
        *   If `self.ptr < self.buffer_size`: Append new data to the respective lists.
        *   If the buffer is full and `ptr` would exceed `buffer_size`, implement wrap-around logic. For example, use the modulo operator: `self.obs[self.ptr % self.buffer_size] = obs`.
        *   Alternatively, consider using `collections.deque(maxlen=buffer_size)` for each list, which handles fixed-size and wrap-around automatically.
    *   Increment `self.ptr` correctly. If not using `deque`, `self.ptr` might just increment and then be used with modulo for indexing, or it could reset to 0 when `buffer_size` is reached if you are filling and then processing the whole buffer. The current PPO structure suggests filling then processing.
2.  **Complete `compute_advantages_and_returns()` Method:**
    *   This method calculates Generalized Advantage Estimation (GAE) and returns.
    *   Ensure `last_value` (which is a float representing the value of the terminal state or the bootstrapped value from the next state if not terminal) is correctly incorporated. It should be converted to a tensor for consistent tensor operations if other values are tensors.
    *   The GAE calculation loop should iterate backward from the last experience.
    *   Store the computed `advantages` and `returns` as PyTorch tensors in `self.advantages` and `self.returns` lists/tensors.
3.  **Complete `get_batch()` Method:**
    *   This method should yield minibatches of experiences for training.
    *   **Implementation Steps:**
        1.  Ensure all stored experiences (`obs`, `actions`, `log_probs`, `values`, `advantages`, `returns`, `legal_masks`) are converted to PyTorch tensors if they aren't already.
        2.  Generate a permutation of indices from `0` to `self.buffer_size - 1` for shuffling.
        3.  Iterate through these shuffled indices, yielding batches of `self.minibatch_size`.
        4.  Each batch should be a dictionary mapping keys (e.g., "obs", "actions") to tensors containing the minibatch data.
        5.  Ensure all tensors yielded are on `self.device`.
4.  **Refine `clear()` Method:**
    *   Ensure `self.ptr` is reset to `0`.
    *   Crucially, also clear `self.advantages` and `self.returns` lists, in addition to other experience lists, to prepare for the next round of experience collection.

## 4. Training Process Refactoring (`keisei/training/` and `keisei/train.py`)

### 4.1. `keisei/training/trainer.py` (New File)

**Objective:** Encapsulate the entire training lifecycle within a dedicated `Trainer` class, promoting separation of concerns.

**Actions:**

1.  **`Trainer` Class Definition:**
    *   `__init__(self, config, agent, environment, experience_buffer, policy_mapper, logger, wandb_run=None)`:
        *   Store these components as instance members (e.g., `self.config`, `self.agent`).
        *   The `config` object will provide all necessary training parameters (total timesteps, learning rates, save intervals, etc.).
2.  **`train(self)` Method:**
    *   This will be the main public method to start and manage the training process.
    *   **Outer Loop (Total Timesteps/Epochs):** Iterate for the configured number of total training timesteps or epochs.
    *   **Inner Loop (Experience Collection):** For each iteration of the outer loop (or PPO rollout):
        *   Loop until `experience_buffer.buffer_size` experiences are collected.
        *   Inside this loop (typically one or more game episodes):
            *   Reset the `environment` (`shogi_game.reset()`).
            *   Loop per episode step:
                *   Get current `observation` from the environment.
                *   Generate `legal_mask` using `self.policy_mapper` and current legal moves from the environment.
                *   `self.agent.select_action(observation, legal_shogi_moves, legal_mask)` to get action, log_prob, value.
                *   Take the `action` in the `environment` to get next_observation, reward, done.
                *   `self.experience_buffer.add(...)` the transition.
                *   Log per-step or per-episode statistics (e.g., episode reward, length) using `self.logger` and `wandb` (if enabled).
                *   If `done`, break episode loop or handle reset.
    *   **Advantage Calculation:** After the buffer is full, call `self.experience_buffer.compute_advantages_and_returns(last_value)`.
    *   **PPO Update Phase:**
        *   Loop for `self.config.ppo_epochs`.
        *   For each epoch, iterate through minibatches from `self.experience_buffer.get_batch()`.
        *   For each minibatch, call `self.agent.learn(minibatch_data)` to update the agent's policy and value networks.
        *   Log training metrics (e.g., policy loss, value loss, entropy) from the `learn` method.
    *   **Buffer Clearing:** Call `self.experience_buffer.clear()`.
    *   **Checkpointing:** Periodically save the agent's model using `self.agent.save_model(...)` based on configuration (e.g., every N timesteps or M iterations).
    *   **Periodic Evaluation:** Periodically, call an evaluation routine (potentially by instantiating and using an `Evaluator` object from `keisei/evaluation/evaluator.py`) to monitor agent performance against baseline opponents.
3.  **Helper Methods:**
    *   Consider private helper methods within the `Trainer` for tasks like checkpointing, detailed logging, or managing evaluation calls to keep the `train()` method cleaner.

### 4.2. `keisei/train.py` (Refactor Existing - Becomes a Lightweight Script)

**Objective:** Transform the main `train.py` into a lean script responsible only for setup, initialization of components, and invoking the `Trainer`.

**Actions:**

1.  **`main()` Function Refactoring:**
    *   Call `parse_args()` to get command-line arguments (primarily config path and overrides).
    *   Load the configuration using the new config loading mechanism.
    *   Initialize `wandb` if enabled in the configuration.
    *   Initialize `TrainingLogger` (from `keisei.utils.loggers` or similar).
    *   Initialize the environment (`ShogiGame`).
    *   Initialize `PolicyOutputMapper` (from `keisei.utils.mappers` or similar).
    *   Initialize the `PPOAgent` (or other agent based on config), passing the relevant part of the configuration.
    *   Initialize `ExperienceBuffer`, passing relevant configuration.
    *   Instantiate the `Trainer` class: `trainer = Trainer(config, agent, environment, buffer, policy_mapper, logger, wandb_run)`.
    *   Start training: `trainer.train()`.
    *   Handle `wandb.finish()` if applicable, and logger cleanup.
2.  **Utility Functions:**
    *   Ensure `find_latest_checkpoint(model_dir)` is robustly implemented, likely as a utility function (perhaps in `keisei.utils.checkpoint_utils`).

## 5. Evaluation Process Refactoring (`keisei/evaluation/` and `keisei/evaluate.py`)

### 5.1. `keisei/evaluation/evaluator.py` (New File)

**Objective:** Create a dedicated `Evaluator` class to manage and execute agent evaluation runs.

**Actions:**

1.  **`Evaluator` Class Definition:**
    *   `__init__(self, config, policy_mapper, device_str)`:
        *   Store the main `config` (or a relevant evaluation-specific sub-config), `policy_mapper`, and `device_str`.
2.  **`evaluate_agent(self, agent_checkpoint_path, opponent_type, opponent_checkpoint_path=None, num_games=None, ...)` Method:**
    *   This method will encapsulate the logic currently in `execute_full_evaluation_run`.
    *   Parameters will include the path to the agent's checkpoint, details about the opponent (type, path if it's another model), number of games, etc. These can be sourced from the main config or passed directly.
    *   **Steps:**
        1.  Load the agent to be evaluated using `load_evaluation_agent` (this function itself might become a method of `Evaluator` or a utility).
        2.  Initialize the opponent using `initialize_opponent` (similarly, could be part of `Evaluator` or a utility).
        3.  Set up `EvaluationLogger` (from `keisei.utils.loggers`) and `wandb` for this specific evaluation run (if configured, potentially as a separate W&B run or a part of a larger training run).
        4.  Execute the evaluation loop (the logic from `run_evaluation_loop` can be a private method of `Evaluator` or a standalone utility called by it). This loop plays `num_games` between the agent and the opponent.
        5.  Collect and aggregate evaluation metrics (win rate, average game length, etc.).
        6.  Log these metrics using the `EvaluationLogger` and `wandb`.
        7.  Return a dictionary of evaluation metrics.

### 5.2. `keisei/evaluate.py` (Refactor Existing - Becomes a Lightweight Script)

**Objective:** Transform `evaluate.py` into a lean script for setting up and invoking the `Evaluator`.

**Actions:**

1.  **`main()` Function (or a new CLI entry point):**
    *   Parse command-line arguments (e.g., path to evaluation config, agent checkpoint path, opponent details).
    *   Load the evaluation configuration.
    *   Initialize `PolicyOutputMapper`.
    *   Instantiate the `Evaluator`: `evaluator = Evaluator(config, policy_mapper, device_str)`.
    *   Call `evaluator.evaluate_agent(...)` with parameters derived from command-line arguments and the loaded configuration.
    *   Print or save the returned evaluation results.
2.  **Code Migration:**
    *   The core logic of `execute_full_evaluation_run`, `load_evaluation_agent`, `initialize_opponent`, and `run_evaluation_loop` will be moved into the `Evaluator` class as methods or be called by it from utility modules.

## 6. Utility Modules (`keisei/utils/`)

**Objective:** Centralize shared utility functions and classes into a dedicated `keisei/utils/` directory to improve project organization and reduce clutter in the main `keisei/` directory.

**Actions:**

1.  **Create `keisei/utils/` Directory:**
    *   This new directory will house all shared utilities.
2.  **Migrate and Organize `keisei/utils.py` Contents:**
    *   The existing `keisei/utils.py` file will be removed. Its contents will be moved into the `keisei/utils/` directory.
    *   Consider splitting the contents into thematically organized files for better clarity:
        *   `keisei/utils/loggers.py`: For `TrainingLogger` and `EvaluationLogger`.
        *   `keisei/utils/mappers.py`: For `PolicyOutputMapper`.
        *   `keisei/utils/opponents.py`: For `BaseOpponent` and potentially simple opponent implementations like `SimpleRandomOpponent` if they are broadly used.
        *   `keisei/utils/config_loader.py`: For configuration loading and validation logic (if not using a library like Hydra that handles this).
        *   `keisei/utils/checkpoint_utils.py`: For functions like `find_latest_checkpoint` and potentially model saving/loading helpers if they can be generalized.
        *   An `__init__.py` file within `keisei/utils/` can be used to re-export key classes/functions for easier importing, e.g., `from keisei.utils import TrainingLogger`.
3.  **Update Import Statements:**
    *   All import statements across the entire project that previously referenced `keisei.utils` (e.g., `from keisei.utils import TrainingLogger`) must be updated to reflect the new structure (e.g., `from keisei.utils.loggers import TrainingLogger` or `from keisei.utils import TrainingLogger` if re-exported via `__init__.py`). This is a critical step and requires careful find-and-replace or IDE refactoring tools.
4.  **`PolicyOutputMapper`:**
    *   This remains a complex and critical utility. Ensure it is thoroughly unit-tested after relocation.
5.  **Review `shogi_game_io.py`:**
    *   Re-evaluate if any functions within `keisei/shogi/shogi_game_io.py` are purely for neural network input/output mapping (and not general Shogi game I/O). If so, consider moving them to `keisei/utils/mappers.py` or a similar utility module to be closer to `PolicyOutputMapper` and network-related utilities.

## 7. Testing and Documentation

### 7.1. Testing

**Objective:** Ensure the refactored codebase maintains correctness and that new components are reliable.

**Actions:**

1.  **Update Existing Tests:**
    *   All existing unit and integration tests in the `tests/` directory must be updated to reflect the new file locations, class names, method signatures, and responsibilities.
    *   Pay close attention to tests for `ppo_agent.py`, `neural_network.py`, `experience_buffer.py`, `train.py`, and `evaluate.py`.
2.  **New Unit Tests:**
    *   Write new unit tests for the `Trainer` class (`tests/training/test_trainer.py`).
    *   Write new unit tests for the `Evaluator` class (`tests/evaluation/test_evaluator.py`).
    *   Write unit tests for any new utility modules or functions (e.g., in `tests/utils/`).
3.  **Test Critical Components:**
    *   Thoroughly test the configuration loading and overriding mechanism.
    *   Verify the correctness of the fixed/completed methods in `neural_network.py` (`forward`, `get_action_and_value`, `evaluate_actions`).
    *   Verify the correctness of the completed methods in `experience_buffer.py` (`add`, `compute_advantages_and_returns`, `get_batch`).
4.  **Integration Tests:**
    *   Ensure integration tests still cover the end-to-end training and evaluation workflows.

### 7.2. Documentation

**Objective:** Keep all documentation up-to-date with the refactored codebase.

**Actions:**

1.  **Docstrings:**
    *   Update module, class, and function/method docstrings for all changed and new files to accurately reflect their purpose, parameters, and return values.
2.  **High-Level Documentation:**
    *   Update `README.md` if necessary to reflect major structural changes or how to run training/evaluation.
    *   Update any existing design documents (e.g., in `docs/`) that describe the system architecture.
    *   This `TRAINING_REFACTOR.md` document will serve as a key piece of documentation for these changes.

## 8. Phased Execution Plan

To manage the complexity of this refactoring, the following phased approach is recommended. Each phase should conclude with updating and passing all relevant tests.

**Step 0: Baseline Configuration Schema Definition and Validation**
*   **Objective:** Document the current system's configuration (especially neural network architecture and key training parameters) by defining initial Pydantic models. Validate that these models accurately reflect the existing implementation before any code is changed.
*   **Actions:**
    1.  **Inspect Current Configuration Sources:**
        *   Thoroughly review `keisei/neural_network.py` to identify all hardcoded or implicitly defined architectural parameters (e.g., number of convolutional filters, kernel sizes, strides, padding, number of units in linear layers, activation functions).
        *   Review `config.py` and other relevant files (e.g., `ppo_agent.py`, `experience_buffer.py`, `train.py`) to identify all current configuration values (e.g., learning rates, gamma, buffer sizes, PPO hyperparameters, batch sizes, total timesteps, save directories, device settings, etc.).
    2.  **Define Initial Pydantic Schemas:**
        *   Create the new file `keisei/configs/schemas.py`.
        *   Based on the inspection, define the initial Pydantic models. This should include a main `AppConfig` model, which in turn might contain nested models like `NetworkConfig`, `PPOAgentConfig`, `TrainingConfig`, `ExperienceBufferConfig`, `EnvironmentConfig`, `LoggingConfig`, and `WandBConfig`.
        *   Populate these models with fields corresponding to all identified parameters. Crucially, set their **default values to exactly match the current, existing implementation** (whether hardcoded or from `config.py`).
    3.  **Cross-Verify Pydantic Schema Against Current Implementation:**
        *   Manually cross-reference the defined Pydantic models and their default values against the actual values used in `keisei/neural_network.py`, `config.py`, and other relevant parts of the current codebase.
        *   The goal is to confirm that if the new Pydantic-based configuration system were hypothetically used *today* (with these defaults and no YAML/CLI overrides), it would produce the *exact same configuration values* that the system currently operates with. This is a manual validation of the schema's accuracy against the current state.
    4.  **Document Baseline:**
        *   Add a note or section in this `TRAINING_REFACTOR.md` document (or as comments within `keisei/configs/schemas.py`) confirming the completion of this validation. Highlight any specific parameters whose current values were captured, any ambiguities encountered, or assumptions made during this process. This establishes the validated baseline configuration schema.

**Phase 1: Configuration Infrastructure Implementation** (was Phase 1)
*   **Objective:** Build the infrastructure to load, merge, and validate configurations using the Pydantic schemas defined in Step 0.
*   **Actions:**
    1.  Create the new directory structures: `keisei/core/`, `keisei/evaluation/`, `keisei/configs/` (if not already created for `schemas.py`), `keisei/utils/`.
    2.  Implement the YAML loading and Pydantic model instantiation logic in a new `keisei/utils/config_loader.py`. This loader should:
        *   Accept a path to a YAML configuration file (e.g., via a CLI argument).
        *   Load the YAML data.
        *   Instantiate the main `AppConfig` Pydantic model (defined in `keisei/configs/schemas.py`), populating it from the YAML data. Values in YAML will override the defaults defined in the Pydantic models.
        *   Implement CLI argument parsing (e.g., using `argparse` in `train.py` and `evaluate.py`, potentially calling a shared parsing function in `config_loader.py`) for essential overrides (like `--config-path`, `--run-name`, `--savedir`, `--device`, `--seed`, `--total-timesteps`).
        *   Merge CLI overrides into the Pydantic model instance. Pydantic will perform validation upon instantiation and any subsequent updates.
    3.  Create an initial `keisei/configs/default_config.yaml`. This file should ideally be minimal or even empty initially if all defaults are well-defined in the Pydantic models (Step 0). It can serve as a template for users to copy and customize for their experiments.
    4.  Refactor `keisei/train.py` (and `keisei/evaluate.py` if time permits for basic config loading) to:
        *   Use the new `config_loader.py` to load the `AppConfig`.
        *   Begin passing relevant parts of the `AppConfig` object to components (even if those components don't fully utilize them yet, preparing for later phases).
    5.  Write unit tests for:
        *   The Pydantic models themselves (e.g., ensuring defaults are correct, validation rules for types and constraints work as expected).
        *   The configuration loading and merging logic in `config_loader.py` (e.g., test loading from YAML, applying CLI overrides correctly, handling missing files or invalid YAML).

**Phase 2: Core Component Refactoring (Iterative)** (was Phase 2)
*   **2a. `neural_network.py`:**
        *   Move `keisei/neural_network.py` to `keisei/core/neural_network.py`.
        *   Refactor `ActorCritic` to accept its architecture parameters (input channels, num_actions, layer sizes, etc.) from the `NetworkConfig` part of the main `AppConfig` object.
        *   Implement the critical bug fix for the `forward` method.
        *   Complete the `get_action_and_value` and `evaluate_actions` methods as specified.
        *   Update existing tests and write new unit tests for `ActorCritic`, focusing on the new methods and configurability via the config object.
    *   **2b. `experience_buffer.py`:**
        *   Move `keisei/experience_buffer.py` to `keisei/core/experience_buffer.py`.
        *   Refactor `ExperienceBuffer` to accept its parameters (buffer_size, minibatch_size, device) from the `ExperienceBufferConfig` (or `TrainingConfig`) part of the `AppConfig`.
        *   Complete the `add`, `compute_advantages_and_returns`, and `get_batch` methods.
        *   Refine the `clear` method.
        *   Update existing tests and write new unit tests for `ExperienceBuffer`.
    *   **2c. `ppo_agent.py`:**
        *   Move `keisei/ppo_agent.py` to `keisei/core/ppo_agent.py`.
        *   Refactor `PPOAgent.__init__` to take the `PPOAgentConfig` (and potentially `NetworkConfig` or the main `AppConfig`) and the instantiated `ActorCritic` model.
        *   Ensure all hyperparameters (learning_rate, gamma, clip_epsilon, ppo_epochs, value_loss_coeff, entropy_coef, etc.) are sourced from the config object.
        *   Update the `learn` method to use the completed `ActorCritic` and `ExperienceBuffer` methods.
        *   Consider creating `keisei/core/base_agent.py` with `BaseAgent` if time permits, and have `PPOAgent` inherit from it.
        *   Update existing tests and write new unit tests for `PPOAgent`.

**Phase 3: Training System Refactoring** (was Phase 3)
    *   Create `keisei/training/trainer.py` with the `Trainer` class.
    *   The `Trainer.__init__` should accept the main `AppConfig` (or relevant sub-configs like `TrainingConfig`, `AgentConfig`), the instantiated `PPOAgent`, `ShogiGame` environment, `ExperienceBuffer`, `PolicyOutputMapper`, and `TrainingLogger`.
    *   Implement the `Trainer.train()` method, orchestrating the training loop, experience collection, PPO updates, checkpointing, and periodic evaluation calls as detailed in the plan, sourcing all parameters from the config object.
    *   Refactor `keisei/train.py` to be a lightweight script: initialize components (loading config, creating agent, environment, buffer, logger, policy_mapper, all configured via `AppConfig`) and then create and run the `Trainer`.
    *   Update existing tests for `train.py` and write new unit tests for the `Trainer` class.

**Phase 4: Evaluation System Refactoring** (was Phase 4)
    *   Create `keisei/evaluation/evaluator.py` with the `Evaluator` class.
    *   The `Evaluator.__init__` should accept the `AppConfig` (or `EvaluationConfig`), `PolicyOutputMapper`, and device string (from config).
    *   Implement the `Evaluator.evaluate_agent()` method, incorporating logic from `execute_full_evaluation_run`, `load_evaluation_agent`, `initialize_opponent`, and `run_evaluation_loop`, sourcing parameters from the config.
    *   Refactor `keisei/evaluate.py` to be a lightweight script: initialize components (loading config) and then create and run the `Evaluator`.
    *   Update existing tests for `evaluate.py` and write new unit tests for the `Evaluator` class.

**Phase 5: Utilities Refactoring & Import Updates** (was Phase 5)
    *   Move the contents of the old `keisei/utils.py` into the `keisei/utils/` directory, splitting into logical files: `loggers.py` (for `TrainingLogger`, `EvaluationLogger`, configured via `LoggingConfig`), `mappers.py` (for `PolicyOutputMapper`), `opponents.py`, `checkpoint_utils.py`. (Note: `config_loader.py` would already be in `keisei/utils/` from Phase 1).
    *   Ensure `keisei/utils/__init__.py` is created if needed to re-export key utilities for cleaner imports.
    *   Carefully update all import statements across the entire project to reflect the new file locations and module structure. This is a critical and potentially error-prone step.
    *   Review `keisei/shogi/shogi_game_io.py` for any functions that should be moved to `keisei/utils/mappers.py`.
    *   Update/write unit tests for all refactored utility modules.

**Phase 6: Finalization, Documentation & Full System Test** (was Phase 6)
    *   Conduct a thorough review of all changes.
    *   Ensure all docstrings (module, class, function) are updated and comprehensive.
    *   Update `README.md` and any other high-level documentation to reflect the new architecture and usage instructions (including how to use YAML configs and CLI overrides).
    *   Perform a full integration test of the training and evaluation pipelines with a sample configuration.
    *   Ensure all tests in the suite are passing reliably.

## 9. Conclusion (was 8)

This refactoring represents a significant effort to modernize the `keisei` codebase. The outcome will be a more robust, flexible, and maintainable system, better suited for ongoing research and development in Shogi reinforcement learning. Incremental changes and thorough testing at each step will be crucial for a successful transition.

**Overarching Constraint for Each Step:**
