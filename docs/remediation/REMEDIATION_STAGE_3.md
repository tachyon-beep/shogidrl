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
    - [x] All values have been migrated to the Pydantic schema and file-based configs. The legacy `config.py` is no longer used and can be deleted.
    - [x] All tests and documentation have been updated to use the new system. Example `default_config.yaml` is provided for users.

**Verification:**

1.  - [x] **Update Tests:** All unit and integration tests must be updated to construct and pass the new Pydantic `AppConfig` object instead of mocking the old `cfg`.
2.  - [x] **End-to-End Test:** Run a full training session using a YAML configuration file to ensure all components correctly receive their parameters. The default config is now loaded from the project root (`default_config.yaml`).

---

#### **Task 3.2: Finalize Package Reorganization**

**Objective:** Execute the planned package reorganization to align the code structure with its logical domains, making the project easier for contributors to navigate[cite: 320].

**Steps:**

1.  - [x] **Create New Directory Structure:**
    * All subpackages (`core/`, `training/`, `evaluation/`, `utils/`) are present under `keisei/` as planned.
2.  - [x] **Move Modules to New Locations:**
    * All relevant modules have been moved to their new subpackage directories. Legacy files are in `deprecated/`.
3.  - [x] **Update All `import` Statements:**
    * All import statements in the codebase and tests have been updated to match the new structure. No import errors remain.

**Verification:**

1.  - [x] **Run Test Suite:** The entire test suite was run after the reorganization. All import errors are resolved and tests run (only linter warnings/errors remain).
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

- The default config is now loaded from the project root (`default_config.yaml`), fixing previous path issues.
- All core, training, evaluation, and utility modules have been moved to their new subpackage directories under `keisei/`.
- All import statements in the codebase and tests have been updated to match the new structure.
- Legacy files have been moved to `deprecated/` and removed from the main package.
- The test suite runs successfully with no import or runtime errors (only linter warnings/errors remain).
- The codebase is now organized, modern, and ready for further automation and reproducibility enhancements.

---