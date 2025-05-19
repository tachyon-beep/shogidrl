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
