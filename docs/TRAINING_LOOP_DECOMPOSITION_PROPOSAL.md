# Proposal: Decompose Main Training Loop from Trainer

**Date:** May 29, 2025
**Author:** GitHub Copilot
**Context:** This proposal addresses the need to further reduce the size and complexity of the `Trainer` class by decomposing the main training loop, as identified in the `TRAINER_STATUS_REPORT.md` and in alignment with Phase 6 of `TRAINER_REFACTOR.md`.

## 1. Objective

To extract the primary iteration logic of the training loop from `keisei/training/trainer.py` into a new, dedicated `TrainingLoopManager` class. This will:
- Significantly reduce the line count and complexity of the `Trainer` class, bringing it closer to the ~200-300 line target.
- Improve separation of concerns, with `Trainer` focusing on initialization and high-level orchestration, and `TrainingLoopManager` managing the iterative training process.
- Enhance the testability of the training loop's mechanics.

## 2. Current State

The `Trainer.run_training_loop()` method, along with its direct helper methods like `_execute_training_step` (which delegates to `StepManager`), `_perform_ppo_update`, and the loop control structure itself, currently resides within `Trainer`. While `StepManager` handles individual step execution, the overall loop orchestration, PPO update decisions, callback invocation, and display updates are still managed directly by `Trainer` methods.

The `TRAINER_REFACTOR.md` (Phase 6) already envisions a leaner `Trainer` that delegates loop execution. This proposal details the implementation of that vision.

## 3. Proposed Solution: `TrainingLoopManager`

Introduce a new class: `keisei/training/training_loop_manager.py:TrainingLoopManager`.

### 3.1. Responsibilities of `TrainingLoopManager`

-   **Orchestrate Training Iterations:** Manage the main `while` loop that runs for `config.training.total_timesteps`.
-   **Global Timestep Management:** Own and increment the `global_timestep` counter.
-   **Coordinate Step Execution:** Invoke `StepManager` to execute individual training steps and handle episode progression.
-   **Trigger PPO Updates:** Determine when PPO updates should occur (e.g., based on `steps_per_epoch`) and call the appropriate method on the `Trainer` (or `PPOAgent`) to perform the update.
-   **Invoke Callbacks:** Trigger registered callbacks at appropriate points (e.g., `on_step_end`).
-   **Manage UI/Display Updates:** Coordinate with `TrainingDisplay` to refresh the UI with progress and metrics, potentially using a `MetricsManager`.
-   **Graceful Start and Termination:** Handle the initialization of the loop state and the conditions for ending the loop.

### 3.2. Proposed Class Structure for `TrainingLoopManager`

```python
# keisei/training/training_loop_manager.py

from typing import TYPE_CHECKING, Any, Callable
from keisei.config_schema import AppConfig

if TYPE_CHECKING:
    from .trainer import Trainer # To access methods like _perform_ppo_update, _finalize_training
    from .step_manager import StepManager, EpisodeState
    from .display import TrainingDisplay
    from .callbacks import Callback # Assuming a base Callback type
    # from .metrics_manager import MetricsManager # If a dedicated metrics manager is used

class TrainingLoopManager:
    def __init__(
        self,
        trainer: "Trainer", # Pass the trainer instance for coordination
        config: AppConfig,
        step_manager: "StepManager",
        training_display: "TrainingDisplay",
        callbacks_list: list["Callback"],
        # metrics_manager: "MetricsManager", # Optional, if metrics are centralized
        logger_func: Callable, # For logging within the loop manager
    ):
        self.trainer = trainer
        self.config = config
        self.step_manager = step_manager
        self.training_display = training_display
        self.callbacks = callbacks_list
        # self.metrics_manager = metrics_manager
        self.logger = logger_func

        self.global_timestep: int = trainer.global_timestep # Initialize from Trainer's state (e.g., after resume)

    def _should_perform_ppo_update(self) -> bool:
        """Determines if a PPO update should be performed at the current timestep."""
        return (
            (self.global_timestep + 1) % self.config.training.steps_per_epoch == 0
            and self.trainer.experience_buffer.ptr == self.config.training.steps_per_epoch # Access buffer via trainer
        )

    def _trigger_callbacks_on_step_end(self):
        """Invokes all registered on_step_end callbacks."""
        for callback in self.callbacks:
            # Callback might need access to the trainer instance or specific parts of it
            callback.on_step_end(self.trainer)

    def _update_display_and_metrics(self):
        """Updates the training display with current metrics and progress."""
        # This logic would mirror what's currently in Trainer's loop
        # for updating display, potentially using self.trainer.pending_progress_updates
        # or a MetricsManager.

        # Example using pending_progress_updates from Trainer:
        if self.trainer.pending_progress_updates:
            self.training_display.update_progress(
                current_step=self.global_timestep,
                total_steps=self.config.training.total_timesteps,
                completed_episodes=self.trainer.total_episodes_completed, # Access via trainer
                # Pass other metrics from self.trainer.pending_progress_updates
                **self.trainer.pending_progress_updates 
            )
            self.trainer.pending_progress_updates.clear()


    def run(self):
        """Executes the main training loop."""
        self.logger(f"Starting training loop. Initial global_timestep: {self.global_timestep}")

        # Initialize episode state via StepManager (coordinated by Trainer initially)
        # Trainer should call _initialize_game_state which returns initial episode_state
        episode_state: "EpisodeState" = self.trainer.current_episode_state

        try:
            with self.training_display.get_context_manager(): # If display uses a context manager
                while self.global_timestep < self.config.training.total_timesteps:
                    # Execute training step via StepManager (as currently done in Trainer)
                    # The result of this step execution (new episode_state, done, info etc.)
                    # will be handled similarly to how Trainer._execute_training_step does.
                    # For simplicity, assume Trainer has a method that wraps this.
                    episode_state = self.trainer._execute_training_step_delegated(episode_state)
                    
                    # PPO Update
                    if self._should_perform_ppo_update():
                        # Call Trainer's method to perform the PPO update
                        self.trainer._perform_ppo_update(episode_state.current_obs, self.logger)

                    # Update displays and metrics
                    self._update_display_and_metrics()

                    # Run callbacks
                    self._trigger_callbacks_on_step_end()

                    self.global_timestep += 1
                    self.trainer.global_timestep = self.global_timestep # Keep trainer's count in sync

                    # Check for termination signals if any (e.g., from a callback)

        except KeyboardInterrupt:
            self.logger("Training loop interrupted by user (KeyboardInterrupt).")
            # Perform any necessary cleanup before finalizing
        except Exception as e:
            self.logger(f"Exception in training loop: {e}", exc_info=True)
            # Perform cleanup
            raise
        finally:
            self.logger(f"Training loop attempting to finalize. Global_timestep: {self.global_timestep}")
            # Finalize training (save model, etc.) by calling Trainer's method
            self.trainer._finalize_training(self.logger)
            self.trainer.session_manager.finalize_session() # Ensure session is finalized

```

### 3.3. Refined `Trainer` Class Interaction

```python
# keisei/training/trainer.py (conceptual changes)

# ... imports ...
from .training_loop_manager import TrainingLoopManager

class Trainer:
    def __init__(self, config: AppConfig, args: Any):
        # ... (existing manager initializations: Session, Env, Model) ...
        # self.metrics_manager = MetricsManager() # If used

        self._setup_components() # Sets up game, agent, buffer, etc.

        # Initialize TrainingLoopManager
        self.training_loop_manager = TrainingLoopManager(
            trainer=self,
            config=self.config,
            step_manager=self.step_manager,
            training_display=self.display,
            callbacks_list=self.callbacks, # self.callbacks is already a list
            # metrics_manager=self.metrics_manager,
            logger_func=self.logger.log_both # or self.logger.log
        )
        
        # Initial episode state needed by the loop manager
        self.current_episode_state: Optional[EpisodeState] = None


    def _setup_components(self):
        # ... (existing setup logic for session, env, model, agent, buffer, step_manager, display, callbacks) ...
        # Ensure self.global_timestep is initialized (e.g. from checkpoint or to 0)
        # Ensure self.callbacks is initialized as a list
        pass

    def _initialize_game_state_for_loop(self) -> EpisodeState:
        """Wrapper to initialize/reset game state for the training loop."""
        # This method would call self.env_manager.reset_game() and self.step_manager.reset_episode()
        # and store/return the initial EpisodeState.
        # This is called before starting the loop.
        self.env_manager.reset_game()
        self.current_episode_state = self.step_manager.reset_episode()
        return self.current_episode_state

    def _execute_training_step_delegated(self, episode_state: EpisodeState) -> EpisodeState:
        """
        Called by TrainingLoopManager to execute one step.
        This method encapsulates the logic previously in Trainer._execute_training_step,
        which primarily delegates to self.step_manager and handles episode completion logic
        (updating trainer stats like black_wins, white_wins, draws, total_episodes_completed,
        and pending_progress_updates).
        """
        # ... (Logic from current Trainer._execute_training_step)
        # This will use self.step_manager, update self.black_wins, self.white_wins, etc.
        # and self.pending_progress_updates.
        # It returns the new episode_state.
        # For brevity, the full logic of _execute_training_step is not duplicated here.
        # It would involve calling self.step_manager.execute_step and self.step_manager.handle_episode_end
        # and updating trainer's own statistics.
        
        # Simplified placeholder for the complex logic:
        step_result = self.step_manager.execute_step(
            episode_state=episode_state,
            global_timestep=self.global_timestep,
            logger_func=self.logger.log_both, # or self.logger.log
        )

        if not step_result.success:
            return self.step_manager.reset_episode()

        updated_episode_state = self.step_manager.update_episode_state(episode_state, step_result)

        if step_result.done:
            # Update game statistics based on outcome
            if step_result.info and "winner" in step_result.info:
                winner = step_result.info["winner"]
                if winner == "black": self.black_wins += 1
                elif winner == "white": self.white_wins += 1
                else: self.draws += 1
            else: self.draws += 1
            
            game_stats_for_sm = {"black_wins": self.black_wins, "white_wins": self.white_wins, "draws": self.draws}
            new_episode_state = self.step_manager.handle_episode_end(
                updated_episode_state, step_result, game_stats_for_sm, self.total_episodes_completed, self.logger.log_both
            )
            self.total_episodes_completed += 1
            
            # Update pending_progress_updates (as in current Trainer)
            # ...
            return new_episode_state
        
        return updated_episode_state


    # _perform_ppo_update remains in Trainer, called by TrainingLoopManager
    def _perform_ppo_update(self, current_obs_np, logger_func: Callable):
        # ... (existing logic) ...
        pass

    # _finalize_training remains in Trainer, called by TrainingLoopManager
    def _finalize_training(self, logger_func: Callable):
        # ... (existing logic) ...
        pass

    def run_training_loop(self):
        """Main entry point to start training."""
        self.session_manager.log_session_info( # Log session info before loop
            logger_func=self.logger.log_both,
            # ... other args for log_session_info
        )
        if self.resumed_from_checkpoint: # Log resume info
            self.logger.log_both(f"Resumed training from checkpoint: {self.resumed_from_checkpoint}")
        
        self.model_manager.log_model_info(self.logger.log_both) # Log model info

        self.current_episode_state = self._initialize_game_state_for_loop() # Initialize first episode

        self.training_loop_manager.run()
```

## 4. Benefits

-   **Reduced `Trainer` Complexity:** `Trainer.py` will be significantly smaller and easier to understand.
-   **Clear Separation of Concerns:**
    -   `Trainer`: Handles overall setup, holds shared state/managers, and provides high-level actions (like PPO update, finalization).
    -   `TrainingLoopManager`: Manages the iterative process of training, step-by-step.
-   **Improved Testability:** The logic of the training loop itself (iteration, conditions for PPO update, callback triggering, display updates) can be unit-tested more effectively within `TrainingLoopManager`.
-   **Maintainability:** Changes to the loop mechanics will be localized to `TrainingLoopManager`.

## 5. Impact on Existing Code

-   `keisei/training/trainer.py`: Will be refactored. The `run_training_loop` method will delegate to `TrainingLoopManager`. Helper methods like `_execute_training_step` will either be moved/adapted into `TrainingLoopManager` or remain as helper methods called by it (as `_execute_training_step_delegated`). Methods like `_perform_ppo_update` and `_finalize_training` will be called by `TrainingLoopManager`.
-   A new file `keisei/training/training_loop_manager.py` will be created.
-   Initialization of `Trainer` will include creating an instance of `TrainingLoopManager`.
-   State like `global_timestep`, `total_episodes_completed`, game win/loss/draw counts, and `pending_progress_updates` will likely remain attributes of `Trainer` as they represent the overall training session state, potentially accessed or updated by `TrainingLoopManager` via the `trainer` instance.

## 6. Next Steps

1.  Create `keisei/training/training_loop_manager.py` with the proposed class structure.
2.  Incrementally move logic from `Trainer.run_training_loop()` and its direct helpers into `TrainingLoopManager`.
3.  Update `Trainer` to initialize and use `TrainingLoopManager`.
4.  Ensure all existing tests pass and add new unit tests for `TrainingLoopManager`.
