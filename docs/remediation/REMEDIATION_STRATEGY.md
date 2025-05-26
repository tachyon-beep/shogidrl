### **ShogiDRL Project Remediation Plan**

This plan operationalizes the findings from the software audit, prioritizing the highest-impact changes first. The primary goal is to refactor the core training system for modularity, address critical bugs and technical debt, and establish a robust development infrastructure.

***

### **1. Core System Refactor: Modularize Training & Evaluation**

This is the highest priority task, as it addresses the most critical architectural flaw: the monolithic training script[cite: 3, 11]. This refactor will create a more maintainable, extensible, and testable codebase by introducing class-based abstractions for training and evaluation, as outlined in the `TRAINING_REFACTOR.md` document[cite: 6, 112].

* **1.1. Create `Trainer` Class:**
    * In a new `keisei/trainer.py` module, create a `Trainer` class[cite: 264].
    * Move the setup logic (argument parsing, configuration assembly, directory creation, component initialization) from `keisei/train.py` into the `Trainer`'s `__init__` method[cite: 265, 461, 462].
    * Move the main `while` loop from `keisei/train.py` into a `train()` method within the `Trainer` class[cite: 265, 468]. State currently managed by local variables (e.g., `global_timestep`, win counters) should become instance attributes of the `Trainer` class[cite: 111, 492].
    * Refactor logical blocks within the loop (e.g., episode termination, PPO updates, checkpointing, evaluation triggers) into smaller, private helper methods within the `Trainer` class to improve clarity[cite: 167, 173].

* **1.2. Create `Evaluator` Class:**
    * In `keisei/evaluate.py`, create an `Evaluator` class to encapsulate evaluation logic, similar to the `Trainer`[cite: 269, 484].
    * The `execute_full_evaluation_run` function's logic should be moved into methods within this new class[cite: 269]. This will improve symmetry with the training module and centralize evaluation state management[cite: 270].

* **1.3. Update Entry Points:**
    * Modify the `if __name__ == "__main__":` blocks in `keisei/train.py` and `keisei/evaluate.py` to be thin wrappers that instantiate and run the `Trainer` and `Evaluator` classes, respectively[cite: 268, 494]. This ensures the command-line interface remains unchanged.

* **1.4. Relocate Utility Functions:**
    * Move helper functions currently in `train.py`, such as `find_latest_checkpoint` and `serialize_config`, into `keisei/utils.py` or make them static methods of the `Trainer` class to declutter the main script[cite: 271, 272].

***

### **2. Critical Fixes & Technical Debt Reduction**

These tasks should be addressed to resolve immediate bugs and inconsistencies. Some can be done in parallel with the refactor, while others are best done just after.

* **2.1. Resolve Non-Functional Configuration Override:**
    * The `--override` CLI flag is a stub and silently ignores user input[cite: 14, 38, 178].
    * **Action:** Either fully implement the override functionality in `apply_config_overrides` or remove the flag entirely to prevent user confusion[cite: 237]. The audit notes that a full Pydantic-based system is planned for the future, so a simple, temporary implementation may suffice[cite: 191].

* **2.2. Correct Configuration Inconsistencies:**
    * The `NUM_ACTIONS_TOTAL` constant is inconsistent between `config.py` (31,592) and the actual number of moves generated and used (13,527)[cite: 39, 40].
    * **Action:** Unify the constant to the correct value of 13,527 across the codebase to ensure consistency and prevent potential errors[cite: 239].

* **2.3. Implement Deterministic Environment Seeding:**
    * The game environment (`ShogiGame`) is not currently seeded, which hinders full reproducibility for self-play experiments[cite: 66]. A code comment notes this as a "TODO"[cite: 68].
    * **Action:** Implement a `seed()` method in the `ShogiGame` class and call it from the training script using the provided seed from the configuration[cite: 240].

* **2.4. Prune Unused Dependencies:**
    * The `requirements.txt` file includes packages that do not appear to be used in the codebase, such as `networkx`, `sympy`, and `GitPython`[cite: 101, 413].
    * **Action:** Remove these unused dependencies from `requirements.txt` to reduce the environment's installation footprint and potential security surface[cite: 241].

***

### **3. Infrastructure & Code Quality Tooling**

This set of tasks establishes a safety net to prevent regressions and improve long-term code health. It should be implemented as soon as possible.

* **3.1. Establish Continuous Integration (CI):**
    * The project currently lacks a CI pipeline to automatically run tests[cite: 7, 139].
    * **Action:** Implement a CI workflow using a tool like GitHub Actions[cite: 248]. This pipeline should, at a minimum, install dependencies and run the full test suite (`pytest`) on every push and pull request[cite: 249]. The sample configuration in the audit appendix is a good starting point[cite: 450].

* **3.2. Integrate Code Quality Tools:**
    * The audit recommends integrating automated tools to enforce code quality and style[cite: 274].
    * **Action:**
        * Configure a linter (e.g., `flake8` or `pylint`) to automatically detect issues like unused imports and undefined variables[cite: 275].
        * Adopt a code formatter (e.g., `black`) to ensure a consistent style across the project[cite: 278].
        * Gradually introduce static type checking with `mypy`, leveraging the project's existing type hints to catch potential type-related bugs[cite: 279, 280].

***

### **4. Strategic Enhancements (Post-Refactor)**

Once the codebase is stabilized and refactored, these larger architectural improvements can be undertaken.

* **4.1. Implement a Robust Configuration System:**
    * The current system of using `config.py` and ad-hoc overrides is fragile[cite: 13, 116]. The audit strongly endorses the documented plan to use Pydantic[cite: 118, 191].
    * **Action:**
        * Define explicit Pydantic models for configuration sections (e.g., `TrainingConfig`, `EnvConfig`, `EvaluationConfig`)[cite: 304].
        * Refactor the `Trainer` and `Evaluator` classes to be initialized with these Pydantic config objects, which can be loaded from YAML/JSON files[cite: 306, 307].
        * Deprecate and eventually remove `config.py`[cite: 310].

* **4.2. Finalize Package Reorganization:**
    * The audit supports the documented plan to reorganize modules into a clearer directory structure (e.g., `keisei/core`, `keisei/training`)[cite: 112, 314].
    * **Action:** After the initial `Trainer`/`Evaluator` refactor is complete, execute the file moves to create the final package structure, updating all imports accordingly[cite: 314, 315].

* **4.3. Investigate Performance and Scalability:**
    * The current single-process design may become a bottleneck for larger-scale experiments[cite: 92, 124].
    * **Action:** Once the system is stable, profile the application to identify performance hotspots[cite: 134]. If necessary, explore parallelism strategies, such as using `multiprocessing` for data generation (self-play) or wrapping the `ShogiGame` in a standard vectorized environment interface like OpenAI Gym[cite: 333, 340].