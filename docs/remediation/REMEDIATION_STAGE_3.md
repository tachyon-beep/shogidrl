### **Executable Plan: Modernisation & Automation**

#### **Task 3.1: Configuration System Overhaul**

**Objective:** Replace the fragile `config.py` and CLI argument system with a robust, centralized, and validated configuration system using Pydantic, as recommended by the audit[cite: 191, 303].

**Steps:**

1.  **Define Pydantic Schema:**
    - [x] Create a new file, `keisei/config_schema.py`.
    - [x] Define Pydantic `BaseModel` classes for logical groups:
        * `EnvConfig` (for `ShogiGame` settings like max moves).
        * `TrainingConfig` (for hyperparameters like `LEARNING_RATE`, `PPO_EPOCHS`, `TOTAL_TIMESTEPS`).
        * `EvaluationConfig` (for evaluation-specific settings like `NUM_GAMES`, `OPPONENT_TYPE`).
        * A top-level `AppConfig` model that contains the nested config objects.
    - [x] Migrate all default values from `config.py` into the fields of these Pydantic models.
    - [x] Add Pydantic validators to enforce constraints (e.g., ensure learning rates are positive)[cite: 305].

2.  **Integrate Pydantic Models into the Code:**
    - [x] Refactor the `Trainer` and `Evaluator` classes to accept a single, typed `AppConfig` object in their `__init__` methods, replacing the untyped `SimpleNamespace` `cfg` object[cite: 306].
    - [x] Update all code to access parameters via the new structured config object (e.g., `self.config.training.learning_rate`).

3.  **Implement Config Loading from Files and CLI:**
    - [x] Create a central utility function responsible for loading configuration.
    - [x] Support loading settings from a YAML or JSON file, which Pydantic can parse into the `AppConfig` model[cite: 307].
    - [x] Modify the `argparse` setup in the main entry points (`train.py`, `evaluate.py`) to accept a path to a config file. Allow a few key CLI flags (like `--total-timesteps`) to override values in the config object after it has been loaded from the file[cite: 309].

4.  **Deprecate and Remove Legacy `config.py`:**
    - [ ] Once all values have been migrated to the Pydantic schema and file-based configs, delete `config.py` to create a single source of truth and avoid confusion[cite: 310].
    - [ ] Update all tests and documentation to use the new system. Provide example `config.yaml` files for users[cite: 313].

**Verification:**

1.  - [x] **Update Tests:** All unit and integration tests must be updated to construct and pass the new Pydantic `AppConfig` object instead of mocking the old `cfg`.
2.  - [x] **End-to-End Test:** Run a full training session using a YAML configuration file to ensure all components correctly receive their parameters.

---

#### **Task 3.2: Finalize Package Reorganization**

**Objective:** Execute the planned package reorganization to align the code structure with its logical domains, making the project easier for contributors to navigate[cite: 320].

**Steps:**

1.  - [ ] **Create New Directory Structure:**
    * Create the subpackages within the `keisei/` directory as outlined in the `TRAINING_REFACTOR.md` plan[cite: 6]:
        * `keisei/core/`: For core RL algorithms (agent, network, buffer)[cite: 314].
        * `keisei/training/`: For the `Trainer` and related training utilities[cite: 317].
        * `keisei/evaluation/`: For the `Evaluator` and opponent definitions[cite: 318].
        * `keisei/shogi/`: (Already exists) For the game engine.
        * `keisei/utils/`: For shared, general-purpose utilities (loggers, etc.)[cite: 316].

2.  - [ ] **Move Modules to New Locations:**
    * Use `git mv` to move the existing Python files into their new subpackage directories to preserve file history.
        * `git mv keisei/ppo_agent.py keisei/core/ppo_agent.py`
        * `git mv keisei/trainer.py keisei/training/trainer.py`
        * ... and so on for all relevant modules.

3.  - [ ] **Update All `import` Statements:**
    * Systematically search the entire project and update all `import` statements to reflect the new, deeper module paths. For example, `from keisei.ppo_agent import PPOAgent` will become `from keisei.core.ppo_agent import PPOAgent`.

**Verification:**

1.  - [ ] **Run Test Suite:** The entire test suite must be run and pass after the reorganization. This confirms that all import paths have been correctly updated.
2.  - [ ] **CI Pipeline:** The CI pipeline must successfully execute on the refactored code structure.

---

#### **Task 3.3: Enhance Automation and Reproducibility**

**Objective:** Introduce automation for common research workflows like hyperparameter tuning and model management[cite: 302].

**Steps:**

1.  - [ ] **Automate Hyperparameter Sweeps:**
    * Integrate with Weights & Biases Sweeps[cite: 322].
    * Create a `sweep.yaml` configuration file defining the parameters to search (e.g., learning rate, number of PPO epochs) and the search strategy (e.g., bayes, random).
    * Modify the training script so that `wandb.init()` can be configured by the W&B agent, allowing it to automatically run experiments with different hyperparameter combinations.

2.  - [ ] **Formalize Model Versioning with Artifacts:**
    * Integrate W&B Artifacts to version and store trained models[cite: 324, 327].
    * At the end of a training run, modify the code to save the final (or best) model checkpoint as a W&B artifact.
    * This will tie the model weights directly to the run that produced them, including its code version, configuration, and results, which is a significant improvement over storing checkpoints in a loose directory structure[cite: 328].

**Verification:**

1.  - [ ] **Run a Test Sweep:** Execute a small W&B sweep with 2-3 runs to confirm that experiments are being created with different hyperparameters.
2.  - [ ] **Check Artifacts:** After a successful test run, verify that a new model artifact appears in the W&B project UI and that it can be downloaded and used for an evaluation run.

---

#### Progress Update (May 27, 2025)

- Renamed `example_config.yaml` to `default_config.yaml` for clarity and convention.
- Updated all code references in `keisei/utils.py` to use `default_config.yaml` as the default config file.
- Updated all documentation and usage examples in `HOW_TO_USE.md` to reference `default_config.yaml`.
- Verified that the CLI and training system work out-of-the-box with the new default config filename.
- No legacy references to `example_config.yaml` remain in the codebase or documentation.

---