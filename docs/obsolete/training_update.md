# DRL Shogi Client: Training Interface Upgrade Plan

## Objective

Transform the current `train.py` into a comprehensive, user-friendly, and robust training interface for deep RL experiments. The new script (`train2.py`) will address usability, reproducibility, experiment management, and extensibility.

---

## Feature Roadmap & Detailed Plan

### 1. Checkpoint Management & Resume
- **Add CLI/config option to specify a checkpoint to resume from.**
- **Auto-detect and load the latest checkpoint if none specified.**
- **Allow custom tags/notes for checkpoints.**
- **Log checkpoint metadata (episode, timestep, config hash, etc).**

### 2. Command-Line Interface (CLI)
- **Use `argparse` or `click` for flexible CLI:**
  - Set hyperparameters (learning rate, batch size, etc.) at runtime.
  - Choose device (CPU/GPU).
  - Enable/disable W&B or other logging.
  - Specify output directories for logs and models.
  - Select mode: train, eval, test, sweep.
  - Set random seed.

### 3. Training Progress & Visualization
- **Add real-time progress bar (e.g., `tqdm`).**
- **Print periodic summaries of key metrics (reward, loss, win rate).**
- **Optionally plot training curves live (matplotlib, tensorboard, or W&B).**

### 4. Advanced Logging
- **Granular logging (per step, episode, update).**
- **Log system info (CPU/GPU usage, memory, etc).**
- **Save logs in both human-readable and machine-readable (JSON/CSV) formats.**
- **Snapshot config and code with each run.**

### 5. Evaluation & Validation
- **Scheduled evaluation against baselines or random agents.**
- **Option to run evaluation only, or at custom intervals.**
- **Save best-performing models based on evaluation metrics.**

### 6. Hyperparameter Sweeps
- **Integrate with W&B sweeps or similar tools.**
- **Support grid/random search via CLI/config.**

### 7. Robustness & Error Handling
- **Graceful interruption handling (Ctrl+C): save state and allow resume.**
- **Warnings for config mismatches or missing files.**
- **Automatic backup of config and code snapshot with each run.**

### 8. Reproducibility
- **Save random seeds and environment info with each run.**
- **Option to set seeds via CLI/config.**

### 9. Multi-Run & Experiment Management
- **Support for running multiple experiments in parallel or sequence.**
- **Organize outputs by experiment/run name.**

### 10. Extensibility
- **Plugin system or hooks for custom callbacks (e.g., logging, early stopping).**
- **Modularize code to allow easy swapping of agent, environment, or network.**

---

## Implementation Phases

### Phase 1: CLI & Checkpointing
- Add argparse/click CLI to `train2.py`.
- Implement checkpoint resume and auto-detection.
- Refactor config loading to support overrides from CLI.

### Phase 2: Logging & Progress
- Integrate tqdm for progress bars.
- Add advanced logging (JSON, CSV, human-readable).
- Print periodic summaries.

### Phase 3: Evaluation & Validation
- Add scheduled evaluation and best-model saving.
- Support evaluation-only mode.

### Phase 4: Hyperparameter Sweeps & Experiment Management
- Integrate with W&B sweeps or similar.
- Add experiment/run organization and metadata.

### Phase 5: Robustness, Reproducibility, Extensibility
- Add graceful interruption handling.
- Save seeds, config, and code snapshot.
- Add plugin/callback system.
- Modularize agent/environment/model selection.

---

## Deliverables
- `train2.py` with all above features.
- Updated documentation (`HOW_TO_USE.md`, `README.md`).
- Example configs and CLI usage.
- Test cases for new features.

---

## Notes
- Prioritize CLI, checkpointing, and logging first for immediate usability.
- Each phase should be tested and documented before moving to the next.
- Maintain backward compatibility where possible.

<!-- PHASED UPGRADE PLAN: CRITICAL FIXES, REPRODUCIBILITY, AND TESTING -->

## Phase 1: Critical Bug Fixes & Core Training Loop Enablement

**Goal:** Make `train2.py` correctly run, save, and resume basic training progression.

1. **Implement Accurate Checkpoint Resume Logic**
    - **PPOAgent:**
        - Extend `save_model()` to save:
            - `global_timestep`
            - `total_episodes_completed`
            - Optimizer state (`agent.optimizer.state_dict()`)
        - Extend `load_model()` to load and return/restore these values.
    - **train2.py:**
        - Receive restored `global_timestep`, `total_episodes_completed` from `agent.load_model()`.
        - Initialize the main loop, `tqdm` progress bar, and counters using these restored values.
        - Load optimizer state into `agent.optimizer`.

2. **Implement Episode Completion Tracking**
    - In the main training loop placeholder:
        - Add logic to detect episode end (e.g., after `config.MAX_MOVES_PER_GAME` or a `done` flag from `game.step()`).
        - Increment `total_episodes_completed` when an episode finishes.
        - Use this variable for `SAVE_FREQ_EPISODES` and `EVAL_FREQ_EPISODES` logic.

3. **Correct Configuration Logging**
    - Replace `vars(config)` in `logger.log()` with a method that iterates `dir(config)`, filters for relevant (e.g., uppercase) attributes, and serializes them (e.g., to a JSON string) for logging.

---

## Phase 2: Enhancing Reproducibility & Output Management

**Goal:** Make experiments easier to reproduce and outputs more organized.

4. **Standardize Log Directory & Run Name Structure**
    - In `apply_config_overrides` or `main()`:
        - If `args.run_name` is provided, create a subdirectory named `args.run_name` inside `args.logdir`. All outputs (models, logs, etc.) should go here.
        - If `args.run_name` is *not* provided, generate a default unique run name (e.g., timestamp-based: `shogi_run_YYYYMMDD_HHMMSS`) and create that subdirectory within `args.logdir`.
        - Update `config.MODEL_DIR`, `config.LOG_FILE` to point to this new run-specific directory.

5. **Save Effective Configuration to File**
    - After all config overrides are applied (CLI, JSON file), save the final, effective configuration dictionary (from Task 3's logging improvement) to a JSON file (e.g., `effective_config.json`) within the run-specific log directory created in Task 4.

6. **Implement Full Seeding for Reproducibility**
    - If `args.seed` is provided and CUDA is active, add:
        - `torch.cuda.manual_seed_all(args.seed)`
        - `torch.backends.cudnn.deterministic = True`
        - `torch.backends.cudnn.benchmark = False` (document performance trade-off)

---

## Phase 3: Essential Testing for New Functionality

**Goal:** Ensure core features and fixes are covered by automated tests.

7. **Develop Unit Tests for Key `train2.py` Features**
    - Add test for JSON config override (`--config`), verifying correct values are used (e.g., by checking log output or saved `effective_config.json`).
    - Add test for specific CLI argument overrides (e.g., `--total-timesteps`), verifying via logs or saved config.
    - Add test for explicit checkpoint resume (`--resume <specific_file_path>`), ensuring it overrides auto-detection.
    - Add test for the `logdir`/`run_name` directory structure creation.
    - (Post-Phase 1): Once checkpointing is more complete, attempt a test that saves a checkpoint, then resumes, and verifies that `global_timestep` and `total_episodes_completed` continue correctly (this might be tricky with `subprocess` and might require more advanced test fixtures or direct function calls if feasible).

---

## Phase 4: Iterative Improvements & Advanced Features (Future)

**Goal:** Add quality-of-life and advanced research features once the core is stable.

8. Integrate with an Experiment Tracking Service (e.g., W&B, TensorBoard)
9. Implement a Dedicated Evaluation Loop
10. Enhance Progress Bar & Logging Verbosity Options

This plan front-loads the critical fixes to make `train2.py` functional and reliable, then moves to usability and testing, leaving more advanced features for later. Each task is relatively self-contained.

# (The original roadmap and phases are preserved above for reference.)
# (This section is the actionable, detailed plan for immediate implementation.)
# (Update as each phase is completed.)
