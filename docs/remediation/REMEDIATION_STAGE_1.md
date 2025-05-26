### **Executable Plan: Core System Refactor**

#### **Phase 1.1: Refactor the Training Module**

This phase focuses on migrating the logic from the monolithic `keisei/train.py` script into a class-based structure.

**Step 1: Create the `Trainer` Class Structure**

1.  Create a new file: `keisei/trainer.py`.
2.  In this new file, define the `Trainer` class. The audit's proposed refactor provides a strong template for this class[cite: 461].
3.  Define the `__init__(self, cfg, args)` method. This method will take the configuration namespace and parsed command-line arguments as input and will be responsible for all setup tasks.

**Step 2: Migrate Setup and Initialization Logic into `Trainer.__init__`**

Move the initialization code from the top of the `main()` function in `keisei/train.py` into the `Trainer.__init__` method.

1.  **Store Config and Arguments:**
    * Set `self.cfg = cfg` and `self.args = args`.
    * Store core parameters like `self.run_name = args.run_name`.

2.  **Create Directories:**
    * Move the `os.makedirs` logic for creating the run-specific output directory into `__init__`[cite: 462].

3.  **Initialize RL Components:**
    * Instantiate `ShogiGame`, `PolicyOutputMapper`, `PPOAgent`, and `ExperienceBuffer` and assign them to instance attributes (e.g., `self.game`, `self.agent`, `self.buffer`)[cite: 462]. Pass the required configuration values from `self.cfg`.

4.  **Initialize State Variables:**
    * Initialize all state-tracking variables, which currently "float around in the function scope," as instance attributes[cite: 111].
    * These include: `self.global_step = 0`, `self.episodes_completed = 0`, `self.black_wins = 0`, `self.white_wins = 0`, and `self.draws = 0`[cite: 159, 462].

5.  **Implement Checkpoint Resumption:**
    * Create a private helper method, `_resume_from_checkpoint(self, resume_path)`.
    * Move the logic for finding and loading a checkpoint from `PPOAgent.load_model` into this method[cite: 466].
    * This method should update the state variables (`self.global_step`, win counters, etc.) with the values loaded from the checkpoint file.
    * Call `self._resume_from_checkpoint(args.resume)` from within `__init__`.

6.  **Initialize Logging and UI:**
    * Move the setup for `rich.console.Console`, `TrainingLogger`, and the `rich.progress.Progress` bar into `__init__`[cite: 463].
    * Move the Weights & Biases setup logic into a separate private method, `_init_wandb(self, args)`, and call it from `__init__`[cite: 467].

**Step 3: Migrate the Main Loop into `Trainer.train()`**

1.  Define the main public method, `train(self)`.
2.  Move the entire `while self.global_step < self.cfg.TOTAL_TIMESTEPS:` loop and its surrounding `with Live(...)` context manager from `keisei/train.py` into this `train()` method[cite: 99, 156].
3.  Update all references to local variables (e.g., `global_timestep`, `agent`) to use the instance attributes (e.g., `self.global_step`, `self.agent`).

**Step 4: Refactor the `train()` Method into Smaller Logical Units**

To improve the clarity of the training loop, break down the monolithic `while` loop into calls to smaller, more focused private methods within the `Trainer` class. The audit identifies several candidates for this[cite: 167].

1.  Create a `_handle_episode_end(self, info)` method. Move the ~50 lines of logic for logging, updating win/loss/draw counters, and resetting the game state into this method[cite: 173].
2.  Create a `_perform_ppo_update(self, next_obs)` method. Move the logic for computing advantages and calling `self.agent.learn(self.buffer)` into this method.
3.  Create a `_save_checkpoint(self)` method to handle saving the model and training state.
4.  Create a `_run_periodic_evaluation(self)` method to encapsulate saving an evaluation checkpoint, calling `execute_full_evaluation_run`, and logging the results. This makes managing the agent's `train()` and `eval()` modes more explicit and contained[cite: 168].

**Step 5: Update the `keisei/train.py` Entry Point**

1.  Delete the now-redundant `main()` function from `keisei/train.py`.
2.  In the `if __name__ == "__main__":` block:
    * Keep the existing `argparse` and configuration setup logic.
    * Instantiate the `Trainer`: `trainer = Trainer(cfg, args)`.
    * Start the training process: `trainer.train()`.
    * Import the new `Trainer` class at the top of the file.

---

#### **Phase 1.2: Refactor the Evaluation Module (Symmetrically)**

Apply the same class-based pattern to the evaluation script to improve consistency and structure.

**Step 1: Create the `Evaluator` Class**

1.  In `keisei/evaluate.py`, define an `Evaluator` class as proposed in the audit's refactor example[cite: 484].
2.  Define an `__init__` method that takes evaluation-specific configuration (e.g., agent checkpoint path, opponent type, number of games).
3.  Move the setup logic from `execute_full_evaluation_run` (loading the agent, initializing the opponent, setting up logging) into the `Evaluator.__init__` method.

**Step 2: Implement the `evaluate()` Method**

1.  Define an `evaluate(self)` method.
2.  Move the game-playing loop (`for game_idx in range(self.num_games):`) from `execute_full_evaluation_run` into this method.
3.  The method should return a dictionary of results (wins, losses, draws, win rate)[cite: 491].

**Step 3: Update the `keisei/evaluate.py` Entry Point**

1.  The audit notes that `evaluate.py` currently has no CLI[cite: 371]. Add an `if __name__ == "__main__":` block.
2.  Add `argparse` logic to accept command-line arguments for the agent checkpoint, opponent type, etc., as suggested in the refactor proposal[cite: 491].
3.  This block should instantiate the `Evaluator` class with the parsed arguments and call its `evaluate()` method, printing the final results.

---

#### **Phase 1.3: Verification**

1.  **Run the Test Suite:** After completing the refactoring, run the entire existing test suite. Per the audit, key tests in `test_train.py` cover CLI execution and checkpoint/resume behavior[cite: 8]. These tests should pass without any changes to the test code itself, confirming that the refactor did not alter the external behavior of the scripts.
2.  **Manual Smoke Test:** Manually run a short training session using the `python keisei/train.py` command to ensure the TUI, logging, and overall flow function as before.