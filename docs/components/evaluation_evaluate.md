# Software Documentation Template for Subsystems - Evaluation Orchestrator

## 📘 evaluate.py as of 2025-05-31

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/evaluation/`
**Documentation Version:** `1.0`
**Date:** `2025-05-31`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Orchestrates comprehensive evaluation of trained PPO Shogi agents through a configurable evaluation framework. This module manages the complete evaluation pipeline from agent loading to result reporting, with integrated logging, metrics tracking, and W&B (Weights & Biases) integration for experiment management.

* **Key Responsibilities:**
  - Coordinate agent and opponent loading with proper device management
  - Set up evaluation environment with logging, seeding, and metrics tracking
  - Orchestrate evaluation execution through the evaluation loop
  - Manage W&B integration for experiment tracking and result visualization
  - Provide CLI interface for standalone evaluation runs
  - Handle error recovery and resource cleanup
  - Support various opponent types (random, heuristic, PPO agents)

* **Domain Context:**
  Comprehensive evaluation system for deep reinforcement learning agents in Shogi, providing systematic performance assessment with statistical analysis, experiment tracking, and reproducible evaluation protocols.

* **High-Level Architecture / Interaction Summary:**
  The evaluation orchestrator serves as the main entry point for agent evaluation, coordinating between agent loading utilities, evaluation loop execution, logging systems, and experiment tracking platforms. It provides both programmatic API and CLI interface for flexible evaluation deployment in research and production environments.

---

### 2. Modules 📦

* **Module Name:** `evaluate.py`

  * **Purpose:** Implement comprehensive evaluation orchestration with experiment management and reporting.
  * **Design Patterns Used:** Orchestrator pattern for evaluation coordination, Facade pattern for simplified API, Builder pattern for configuration setup.
  * **Key Functions/Classes Provided:** 
    - `Evaluator` - Main evaluation orchestration class
    - `execute_full_evaluation_run()` - Legacy-compatible evaluation function
    - `main_cli()` - Command-line interface entry point
  * **Configuration Surface:**
    - Agent and opponent checkpoint paths
    - Evaluation parameters (num_games, max_moves)
    - Device configuration and random seeding
    - Logging configuration and file paths
    - W&B integration parameters (project, entity, run name)
    - CLI argument parsing and validation
  * **Dependencies:**
    * **Internal:**
      - `keisei.core.ppo_agent.PPOAgent`: Agent implementation
      - `keisei.evaluation.loop.run_evaluation_loop`, `ResultsDict`: Core evaluation logic
      - `keisei.utils.BaseOpponent`: Opponent interface
      - `keisei.utils.EvaluationLogger`: Logging utilities
      - `keisei.utils.PolicyOutputMapper`: Action space mapping
      - `keisei.utils.agent_loading.initialize_opponent`, `load_evaluation_agent`: Agent loading
      - `keisei.utils.utils.load_config`: Configuration management
    * **External:**
      - `torch`: PyTorch for neural networks and device management
      - `numpy`: NumPy for random seeding and numerical operations
      - `wandb`: Weights & Biases for experiment tracking
      - `dotenv`: Environment variable loading
      - `argparse`: CLI argument parsing
      - `os`, `random`: System utilities
  * **External API Contracts:**
    - Returns structured ResultsDict for programmatic use
    - CLI interface follows standard argument conventions
    - W&B integration compatible with MLOps workflows
  * **Side Effects / Lifecycle Considerations:**
    - Creates log directories and files
    - Initializes and manages W&B runs
    - Loads model checkpoints and sets device states
    - Sets global random seeds for reproducibility
  * **Usage Examples:**
    ```python
    from keisei.evaluation.evaluate import Evaluator
    
    evaluator = Evaluator(
        agent_checkpoint_path="models/best_agent.pt",
        opponent_type="random",
        opponent_checkpoint_path=None,
        num_games=100,
        max_moves_per_game=300,
        device_str="cuda",
        log_file_path_eval="eval_results.log",
        policy_mapper=PolicyOutputMapper(),
        wandb_log_eval=True
    )
    results = evaluator.evaluate()
    ```

---

### 3. Classes 🏛️

* **Class Name:** `Evaluator`

  * **Defined In Module:** `evaluate.py`
  * **Purpose:** Orchestrate comprehensive agent evaluation with experiment management and robust error handling.
  * **Design Role:** Central coordinator implementing facade pattern for complex evaluation pipeline with resource management.
  * **Inheritance:**
    * **Extends:** `object` (implicit)
    * **Subclasses (internal only):** None
  * **Key Attributes/Properties:**
    - `agent_checkpoint_path: str` – Path to trained agent checkpoint
    - `opponent_type: str` – Type of opponent ("random", "heuristic", "ppo")
    - `opponent_checkpoint_path: Optional[str]` – Path to opponent checkpoint if PPO
    - `num_games: int` – Number of evaluation games to play
    - `max_moves_per_game: int` – Maximum moves per game
    - `device_str: str` – Device for neural network operations
    - `log_file_path_eval: str` – Path for evaluation log file
    - `policy_mapper: PolicyOutputMapper` – Action space mapping utility
    - `seed: Optional[int]` – Random seed for reproducibility
    - `wandb_log_eval: bool` – Enable W&B logging
    - `wandb_project_eval: Optional[str]` – W&B project name
    - `wandb_entity_eval: Optional[str]` – W&B entity/team name
    - `wandb_run_name_eval: Optional[str]` – Custom run name
    - `logger_also_stdout: bool` – Log to stdout in addition to file
    - `wandb_extra_config: Optional[dict]` – Additional W&B configuration
    - `wandb_reinit: Optional[bool]` – W&B reinitialize flag
    - `wandb_group: Optional[str]` – W&B run group
    - Private state: `_wandb_active`, `_wandb_run`, `_logger`, `_agent`, `_opponent`
  * **Key Methods:**
    - `evaluate()` - Main evaluation execution method
    - `_setup()` - Internal setup for agents, logging, and W&B
  * **Interconnections:**
    * **Internal Class/Module Calls:** Coordinates with evaluation loop and utilities
    * **External Systems:** W&B platform, file system, PyTorch ecosystem
  * **Lifecycle & State:**
    - Initialization: Configure all parameters and dependencies
    - Setup: Load agents, initialize logging and W&B
    - Execution: Run evaluation loop with monitoring
    - Cleanup: Close logging and W&B resources
  * **Threading/Concurrency:**
    - Not thread-safe; designed for single-threaded evaluation
    - W&B and logging operations are sequential
  * **Usage Example:**
    ```python
    evaluator = Evaluator(
        agent_checkpoint_path="model.pt",
        opponent_type="random",
        num_games=50,
        device_str="cuda",
        log_file_path_eval="eval.log",
        policy_mapper=mapper
    )
    results = evaluator.evaluate()
    ```

---

### 4. Functions/Methods ⚙️

* **Function/Method Name:** `__init__`

  * **Defined In:** `evaluate.py`
  * **Belongs To:** `Evaluator`
  * **Purpose:** Initialize evaluator with comprehensive configuration for evaluation pipeline.
  * **Parameters:**
    - `agent_checkpoint_path: str` – Path to agent model checkpoint
    - `opponent_type: str` – Opponent type identifier
    - `opponent_checkpoint_path: Optional[str]` – Opponent checkpoint path
    - `num_games: int` – Number of evaluation games
    - `max_moves_per_game: int` – Game move limit
    - `device_str: str` – PyTorch device specification
    - `log_file_path_eval: str` – Evaluation log file path
    - `policy_mapper: PolicyOutputMapper` – Action mapping utility
    - `seed: Optional[int] = None` – Random seed for reproducibility
    - `wandb_log_eval: bool = False` – Enable W&B logging
    - Additional W&B and logging parameters
  * **Returns:**
    - `None`
  * **Raises/Exceptions:**
    - No direct exceptions; validation handled in setup
  * **Side Effects:**
    - Stores configuration parameters as instance attributes
    - Initializes private state variables
  * **Calls To:**
    - Attribute assignment only
  * **Preconditions:** Valid configuration parameters
  * **Postconditions:** Evaluator ready for setup and execution
  * **Algorithmic Note:**
    - Simple parameter storage with deferred validation
  * **Usage Example:**
    ```python
    evaluator = Evaluator(
        agent_checkpoint_path="agent.pt",
        opponent_type="random",
        num_games=10,
        device_str="cpu",
        log_file_path_eval="eval.log",
        policy_mapper=PolicyOutputMapper()
    )
    ```

* **Function/Method Name:** `_setup`

  * **Defined In:** `evaluate.py`
  * **Belongs To:** `Evaluator`
  * **Purpose:** Initialize all evaluation components including agents, logging, W&B, and random seeding.
  * **Parameters:**
    - None (uses instance attributes)
  * **Returns:**
    - `None`
  * **Raises/Exceptions:**
    - `RuntimeError` for setup failures (seed, directory creation, agent/opponent loading)
  * **Side Effects:**
    - Sets global random seeds
    - Creates log directories
    - Initializes W&B run
    - Loads neural network models
    - Creates logger instance
  * **Calls To:**
    - `random.seed()`, `torch.manual_seed()`, `np.random.seed()` - Seeding
    - `wandb.init()` - W&B initialization
    - `os.makedirs()` - Directory creation
    - `EvaluationLogger()` - Logger setup
    - `load_config()` - Configuration loading
    - `load_evaluation_agent()`, `initialize_opponent()` - Agent loading
  * **Preconditions:** Valid configuration parameters
  * **Postconditions:** All components ready for evaluation
  * **Algorithmic Note:**
    - Sequential setup with comprehensive error handling
    - W&B initialization with fallback for failures
  * **Usage Example:**
    ```python
    evaluator._setup()  # Internal method called by evaluate()
    ```

* **Function/Method Name:** `evaluate`

  * **Defined In:** `evaluate.py`
  * **Belongs To:** `Evaluator`
  * **Purpose:** Execute complete evaluation pipeline with setup, loop execution, and cleanup.
  * **Parameters:**
    - None (uses configured instance attributes)
  * **Returns:**
    - `Optional[ResultsDict]` – Evaluation results or None on failure
  * **Raises/Exceptions:**
    - `RuntimeError` for setup or evaluation failures
    - Handles evaluation loop exceptions gracefully
  * **Side Effects:**
    - Performs complete evaluation setup
    - Executes evaluation loop
    - Logs results to W&B and file system
    - Cleans up W&B run
  * **Calls To:**
    - `_setup()` - Component initialization
    - `run_evaluation_loop()` - Core evaluation execution
    - `wandb.log()` - Metrics logging
    - `wandb.finish()` - W&B cleanup
  * **Preconditions:** Evaluator properly initialized
  * **Postconditions:** Complete evaluation results available
  * **Algorithmic Note:**
    - Orchestrates entire evaluation pipeline with error recovery
    - Ensures proper resource cleanup even on failures
  * **Usage Example:**
    ```python
    results = evaluator.evaluate()
    if results:
        print(f"Win rate: {results['win_rate']:.2%}")
    ```

* **Function/Method Name:** `execute_full_evaluation_run`

  * **Defined In:** `evaluate.py`
  * **Belongs To:** Module-level function
  * **Purpose:** Legacy-compatible wrapper for Evaluator class providing functional interface.
  * **Parameters:**
    - All Evaluator initialization parameters
  * **Returns:**
    - `Optional[ResultsDict]` – Evaluation results
  * **Raises/Exceptions:**
    - Passes through Evaluator exceptions
  * **Side Effects:**
    - Creates and executes Evaluator instance
  * **Calls To:**
    - `Evaluator()` - Class instantiation
    - `evaluator.evaluate()` - Evaluation execution
  * **Preconditions:** Valid configuration parameters
  * **Postconditions:** Complete evaluation results
  * **Algorithmic Note:**
    - Simple wrapper maintaining backward compatibility
  * **Usage Example:**
    ```python
    results = execute_full_evaluation_run(
        agent_checkpoint_path="agent.pt",
        opponent_type="random",
        num_games=10,
        device_str="cpu",
        log_file_path_eval="eval.log",
        policy_mapper=mapper
    )
    ```

* **Function/Method Name:** `main_cli`

  * **Defined In:** `evaluate.py`
  * **Belongs To:** Module-level function
  * **Purpose:** Provide command-line interface for standalone evaluation execution.
  * **Parameters:**
    - None (uses argparse for CLI arguments)
  * **Returns:**
    - `None` (prints results to stdout)
  * **Raises/Exceptions:**
    - Argparse exceptions for invalid arguments
  * **Side Effects:**
    - Parses command-line arguments
    - Executes evaluation
    - Prints results to stdout
  * **Calls To:**
    - `argparse.ArgumentParser()` - Argument parsing
    - `execute_full_evaluation_run()` - Evaluation execution
  * **Preconditions:** Valid CLI arguments
  * **Postconditions:** Evaluation completed with results output
  * **Algorithmic Note:**
    - Standard CLI pattern with argument validation
  * **Usage Example:**
    ```bash
    python -m keisei.evaluation.evaluate --agent_checkpoint_path agent.pt --opponent_type random --num_games 10
    ```

---

### 5. Shared or Complex Data Structures 📊

* **Structure Name:** `W&B Configuration Dictionary`
  * **Type:** `Dict[str, Any]`
  * **Purpose:** Structured configuration for Weights & Biases experiment tracking
  * **Format:** Dictionary with standard W&B configuration keys
  * **Fields:**
    - `agent_checkpoint: str` – Agent model path for tracking
    - `opponent_type: str` – Opponent configuration
    - `num_games: int` – Evaluation scale
    - `device: str` – Compute infrastructure
    - `seed: int` – Reproducibility parameter
    - Custom fields from `wandb_extra_config`
  * **Validation Constraints:**
    - All fields must be JSON-serializable
    - Required fields must be present
  * **Used In:** W&B run initialization, experiment tracking

* **Structure Name:** `CLI Argument Configuration`
  * **Type:** `argparse.Namespace`
  * **Purpose:** Command-line argument parsing and validation
  * **Format:** Standard argparse structure
  * **Fields:**
    - Required: `agent_checkpoint_path`, `opponent_type`
    - Optional: `num_games`, `device_str`, `seed`, W&B parameters
    - Flags: `wandb_log_eval`, `logger_also_stdout`, `wandb_reinit`
  * **Validation Constraints:**
    - Opponent type must be in ["random", "heuristic", "ppo"]
    - Numeric parameters must be positive
    - Paths must be valid when specified
  * **Used In:** CLI interface, parameter validation

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  evaluate.py
    ├── uses → evaluation/loop.py (core evaluation)
    ├── uses → core/ppo_agent.py (agent implementation)
    ├── uses → utils/agent_loading.py (model loading)
    ├── uses → utils/EvaluationLogger (logging)
    ├── uses → utils/PolicyOutputMapper (action mapping)
    ├── uses → utils/BaseOpponent (opponent interface)
    └── provides → CLI and API interfaces
  ```

* **Cross-Folder Imports:**
  - Core agent functionality from `/core/`
  - Evaluation loop from `/evaluation/`
  - Utility functions from `/utils/`
  - Configuration from project root

* **Data Flow Summary:**
  - CLI arguments → argument parsing → evaluator configuration
  - Agent checkpoints → model loading → evaluation ready agents
  - Evaluation execution → loop results → statistical computation
  - Results → W&B logging → experiment tracking
  - Results → file logging → persistent records

---

### 7. Non-Functional Aspects 🛠️

* **Performance:**
  - Setup overhead minimal compared to evaluation time
  - Model loading time depends on checkpoint size and device
  - W&B logging has network overhead but runs asynchronously

* **Security:**
  - Checkpoint loading validates file existence and format
  - No direct user input validation (assumes trusted environment)
  - W&B API key handling through environment variables

* **Error Handling & Logging:**
  - Comprehensive exception handling with graceful degradation
  - W&B failures don't prevent evaluation completion
  - Detailed error messages for debugging
  - Resource cleanup guaranteed even on failures

* **Scalability Concerns:**
  - Single-threaded evaluation design
  - Memory usage scales with model size, not evaluation length
  - W&B logging scales well with experiment volume

* **Testing & Instrumentation:**
  - Unit tests for configuration validation
  - Integration tests with mock W&B and file systems
  - Performance monitoring for large-scale evaluations

---

### 8. Configuration & Environment ♻️

* **Environment Variables:**
  - `WANDB_API_KEY` – W&B authentication (via dotenv)
  - PyTorch CUDA environment variables
  - Configuration file paths via environment

* **CLI Interfaces / Entry Points:**
  - Primary CLI: `python -m keisei.evaluation.evaluate`
  - Standard argument format with help documentation
  - Example: `--agent_checkpoint_path models/best.pt --opponent_type random --num_games 100`

* **Config File Schema:**
  - Evaluation parameters in main configuration:
    ```yaml
    evaluation:
      default_num_games: 100
      default_max_moves: 300
      default_device: "cuda"
    wandb:
      project: "keisei-evaluation"
      entity: "research-team"
    ```

---

### 9. Glossary 📖

* **Orchestrator:** Component that coordinates multiple subsystems for complex workflows
* **W&B (Weights & Biases):** Machine learning experiment tracking and visualization platform
* **Checkpoint:** Saved model state including weights and training metadata
* **Evaluation Loop:** Core game execution engine for systematic agent assessment
* **Legacy Wrapper:** Functional interface maintaining backward compatibility

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:**
  - W&B errors could be more specific for different failure modes
  - No validation for opponent checkpoint compatibility
  - Limited error recovery for partial evaluation failures

* **TODOs / Deferred Features:**
  - Implement resume capability for interrupted evaluations
  - Add support for multi-opponent tournaments
  - Implement parallel evaluation execution
  - Add real-time evaluation progress tracking
  - Support for custom evaluation metrics

* **Suggested Refactors:**
  - Extract W&B management to separate utility class
  - Add configuration validation with Pydantic
  - Implement evaluation checkpointing for long runs
  - Add support for different result output formats (JSON, CSV)

---

## Notes for AI/Agent Developers 🧠

1. **Comprehensive Setup:** The evaluator handles all aspects of evaluation environment preparation including seeding, logging, and experiment tracking
2. **Error Resilience:** Robust error handling ensures evaluation can continue even if non-critical components (like W&B) fail
3. **Experiment Tracking:** Built-in W&B integration provides professional-grade experiment management and result visualization
4. **Flexible Interface:** Both programmatic API and CLI interface support different usage patterns from research to production
