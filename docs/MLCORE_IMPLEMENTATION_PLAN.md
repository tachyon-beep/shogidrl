# ML Core Systems - Implementation Plan (June 1, 2025)

This document outlines an implementation plan to address the findings and recommendations from the `MLCORE_REVIEW.md` document dated June 1, 2025. The plan prioritizes critical fixes and improvements to enhance the robustness, maintainability, and performance of Keisei's Machine Learning core systems.

## Prioritization Key:
-   **Critical**: Must be addressed immediately. Significant impact on correctness or stability.
-   **High**: Important for robust training and correctness.
-   **Medium**: Improves best practices, maintainability, or adds valuable features.
-   **Low**: Minor enhancements or ongoing tasks.

---

## I. Critical Priority Items

### 1. Silent Failure in Experience Buffer Batching
-   **Recommendation**: `ExperienceBuffer.get_batch()` should explicitly raise an error on tensor-stack failures or if batch construction results in an invalid/empty state, instead of returning an empty dict.
-   **Affected System(s)/File(s)**: System 5 / `core/experience_buffer.py`
-   **Actionable Steps**:
    1.  Modify `ExperienceBuffer.get_batch()` method.
    2.  Add try-except blocks around `torch.stack` operations or other potential failure points during batch creation.
    3.  If an error occurs or if the resultant batch is empty/invalid, raise a specific, informative `RuntimeError` or `ValueError`.
    4.  Add unit tests to verify that errors are raised correctly under simulated failure conditions (e.g., empty buffer, inconsistent tensor shapes).
-   **Estimated Effort**: Small
-   **Impact**: Prevents silent training skips, ensuring training integrity.

---

## II. High Priority Items

### 1. Handling "No Legal Moves" & Model NaN Issues
-   **Recommendation**:
    -   Shift responsibility for handling "no legal moves" scenarios upstream from the model to `StepManager` or `PPOAgent`.
    -   `StepManager` should detect if `ShogiGame.get_legal_moves()` returns an empty list. If so, it should treat this as a terminal condition for the episode or signal an error, rather than proceeding to `agent.select_action` with an all-false mask.
-   **Affected System(s)/File(s)**:
    -   System 4 / `training/models/resnet_tower.py` (and potentially `core/neural_network.py`)
    -   System 6 / `training/step_manager.py`
    -   System 5 / `core/ppo_agent.py` (potentially, for how it handles signals from `StepManager`)
-   **Actionable Steps**:
    1.  **In `StepManager.execute_step()`**:
        -   After calling `game.get_legal_moves()`, check if the returned list is empty.
        -   If empty:
            -   Log a warning/error.
            -   Set `done = True` for the current step.
            -   Assign a suitable reward (e.g., loss or draw, depending on game rules for such states if not explicitly checkmate/stalemate).
            -   Ensure `StepResult` reflects this termination.
            -   Prevent calling `agent.select_action()`.
    2.  **In Model (`ActorCriticResTower`)**:
        -   The existing NaN handling (warning + uniform fallback) can remain as a last-resort defense, but it should be triggered less frequently if upstream handling is correct.
        -   Review if the `pass` statement when `not torch.any(legal_mask)` in `get_action_and_value` can be made more robust or if it's fully covered by the subsequent NaN check. With upstream changes, this path should ideally not be hit with an empty *true* legal mask.
    3.  Add integration tests to verify behavior when a game state results in no legal moves.
-   **Estimated Effort**: Medium
-   **Impact**: Prevents model from attempting to process invalid states, reduces NaN occurrences, improves training stability and correctness.

### 2. Implement Observation and Advantage Normalization
-   **Recommendation**: Implement and ensure proper use of observation normalization (e.g., running mean/std) and advantage normalization (batch-wise mean/std subtraction and division) in PPO.
-   **Affected System(s)/File(s)**: System 5 / `core/ppo_agent.py`, `core/experience_buffer.py` (for collecting stats or applying normalization)
-   **Actionable Steps**:
    1.  **Advantage Normalization (in `PPOAgent.learn`)**:
        -   After computing advantages for a batch in `ExperienceBuffer` (or when retrieved in `PPOAgent`), normalize them: `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)`.
    2.  **Observation Normalization (Optional but Recommended)**:
        -   Decide on a strategy:
            -   **Running Mean/Std**: Maintain running statistics of observations. Apply normalization before feeding observations to the model. This requires careful state management, especially with checkpoints.
            -   **Batch Normalization**: Already part of the ResNet tower, which helps.
            -   **Input Scaling**: Simpler scaling (e.g., dividing by 255 if observations are pixel-like, though Shogi features are different) might be a first step if full running mean/std is complex. Given the 46-channel feature set, analyze the range and distribution of these features to determine appropriate normalization.
        -   If implementing running mean/std:
            -   Create a wrapper around the environment or a utility class to update and apply normalization.
            -   Ensure these running stats are saved and loaded with checkpoints.
    3.  Add configuration options to enable/disable these normalizations.
    4.  Test the impact on training performance and stability.
-   **Estimated Effort**: Medium to Large (especially for robust observation normalization)
-   **Impact**: Significantly improves PPO stability and performance.

---

## III. Medium Priority Items

### 1. Refactor Model Code Duplication
-   **Recommendation**: If `core/neural_network.py` contains another `ActorCritic` model with duplicated logic for `get_action_and_value` and `evaluate_actions` (as in `ActorCriticResTower`), refactor these into a shared base class.
-   **Affected System(s)/File(s)**: System 4 / `core/neural_network.py`, `training/models/resnet_tower.py`, potentially a new base model file.
-   **Actionable Steps**:
    1.  Identify the common `ActorCritic` implementation in `core/neural_network.py`.
    2.  Create a new base class (e.g., `BaseActorCriticModel`) that inherits `nn.Module` and conceptually implements `ActorCriticProtocol`.
    3.  Move the shared methods (`get_action_and_value`, `evaluate_actions`, potentially `forward` if the structure is very similar before the heads) into this base class.
    4.  Refactor `ActorCriticResTower` and the model in `core/neural_network.py` to inherit from this new base class.
    5.  Ensure all existing tests for both models still pass.
-   **Estimated Effort**: Medium
-   **Impact**: Improves code maintainability, reduces redundancy, ensures consistency between model implementations.

### 2. Model Injection for PPOAgent
-   **Recommendation**: `PPOAgent` should receive the instantiated model (conforming to `ActorCriticProtocol`) as a constructor argument (dependency injection), rather than initializing a default model.
-   **Affected System(s)/File(s)**: System 5 / `core/ppo_agent.py`; System 6 / `training/trainer.py` (or wherever `PPOAgent` is instantiated).
-   **Actionable Steps**:
    1.  Modify the `PPOAgent.__init__` method to accept an `actor_critic: ActorCriticProtocol` argument.
    2.  Remove any default model instantiation within `PPOAgent`.
    3.  Update the `Trainer` (or other instantiation points) to:
        -   Instantiate the model using `ModelManager`.
        -   Pass the instantiated model to the `PPOAgent` constructor.
    4.  Update relevant unit tests for `PPOAgent`.
-   **Estimated Effort**: Small to Medium
-   **Impact**: Decouples `PPOAgent` from specific model implementations, improves testability and flexibility.

### 3. Implement Learning Rate Scheduling
-   **Recommendation**: Implement configurable learning rate scheduling for the PPO optimizer.
-   **Affected System(s)/File(s)**: System 5 / `core/ppo_agent.py`; `config_schema.py` (for new config options).
-   **Actionable Steps**:
    1.  Add configuration options to `TrainingConfig` in `config_schema.py` for LR scheduling (e.g., `lr_schedule_type: Optional[str] = None` (e.g., "linear", "cosine"), `lr_schedule_kwargs: Optional[dict] = None`).
    2.  In `PPOAgent.__init__` or a dedicated setup method:
        -   Based on the configuration, initialize a learning rate scheduler from `torch.optim.lr_scheduler` (e.g., `LambdaLR` for linear decay, `CosineAnnealingLR`).
    3.  In `PPOAgent.learn()`, after `optimizer.step()`, call `scheduler.step()`. The exact point depends on whether stepping is per epoch or per update.
    4.  Ensure the LR scheduler's state is saved and loaded with the optimizer's state dict if training is resumed.
    5.  Add tests for LR scheduling.
-   **Estimated Effort**: Medium
-   **Impact**: Can improve training convergence and final performance.

### 4. Consider Value Function Clipping
-   **Recommendation**: Consider implementing value function loss clipping if value loss instability is observed.
-   **Affected System(s)/File(s)**: System 5 / `core/ppo_agent.py`.
-   **Actionable Steps**:
    1.  Monitor value loss during training runs. If it shows high variance or instability:
    2.  In `PPOAgent.learn()`, when calculating the value loss:
        -   Clip the predicted values (`values_pred_clipped = old_values + torch.clamp(values_pred - old_values, -config.clip_epsilon, config.clip_epsilon)`).
        -   Calculate value loss using `values_pred_clipped` (`loss_vf_clipped`).
        -   The total value loss can be `max(loss_vf_unclipped, loss_vf_clipped)`.
    3.  Add a configuration option to enable/disable this feature.
    4.  Test its impact.
-   **Estimated Effort**: Small to Medium
-   **Impact**: Can stabilize value function training, potentially improving overall PPO performance.

### 5. Optimize Memory Usage for `legal_masks` in Experience Buffer
-   **Recommendation**: Investigate on-the-fly regeneration of `legal_masks` or use sparse storage to reduce memory consumption.
-   **Affected System(s)/File(s)**: System 5 / `core/experience_buffer.py`; System 6 / `training/step_manager.py` (if masks are passed differently).
-   **Actionable Steps**:
    1.  **Profile `ShogiGame.get_legal_moves()`**: Determine if its performance is acceptable for on-the-fly regeneration during minibatch creation.
    2.  **Option A (On-the-fly Regeneration)**:
        -   Remove `legal_masks` storage from `ExperienceBuffer`.
        -   When `ExperienceBuffer.get_batch()` is called, or when `PPOAgent` processes a minibatch, regenerate legal masks for the observations in that minibatch. This might require storing raw game states or having a way to reconstruct them if `get_legal_moves` needs more than just the observation tensor. (This could be complex).
        -   Alternatively, if only the model's `evaluate_actions` needs it for entropy calculation with a mask, the `ExperienceBuffer` might not need to store it at all if the mask isn't used during learning updates for taken actions.
    3.  **Option B (Sparse Storage)**:
        -   Investigate sparse tensor representations (e.g., `torch.sparse_coo_tensor`) for `legal_masks`. This is viable if the number of legal moves is consistently small relative to the total action space.
        -   Modify `ExperienceBuffer.add()` and `get_batch()` to handle sparse tensors.
    4.  Benchmark memory usage and performance implications of the chosen approach.
-   **Estimated Effort**: Medium to Large
-   **Impact**: Reduces memory footprint, allowing for larger batch sizes or longer replay buffers.

### 6. Review Holistic Error Handling and Resilience
-   **Recommendation**: Review the overall strategy for error propagation and system recovery/graceful termination.
-   **Affected System(s)/File(s)**: All core ML systems.
-   **Actionable Steps**:
    1.  Conduct a specific review pass focusing on error handling across manager interactions and critical code paths (e.g., training loop, evaluation loop).
    2.  Ensure that exceptions are caught at appropriate levels, logged informatively, and lead to either graceful recovery (if possible) or clean termination (e.g., saving a final checkpoint, closing W&B runs).
    3.  Document the expected error handling behavior for major components.
-   **Estimated Effort**: Medium (as an ongoing review and refinement task)
-   **Impact**: Increases system stability and makes debugging easier.

### 7. Enhance Testing for Specific Areas
-   **Recommendation**: Address specific areas needing more thorough testing as highlighted in `CODE_MAP.md` (e.g., complex Shogi rules, I/O, model NaN paths).
-   **Affected System(s)/File(s)**: `tests/` directory, relevant game logic and model files.
-   **Actionable Steps**:
    1.  Prioritize writing new tests for:
        -   Edge cases in Shogi rules (`uchi-fu-zume`, `sennichite`, mandatory promotions).
        -   SFEN/KIF conversion correctness and edge cases.
        -   Model behavior with all-false legal masks or logits leading to NaNs (to verify upstream fixes or model fallbacks).
    2.  Increase test coverage for these identified areas.
-   **Estimated Effort**: Medium (as an ongoing task)
-   **Impact**: Improves reliability and catches regressions.

---

## IV. Low Priority / Minor Enhancements

### 1. Add `named_parameters()` to `ActorCriticProtocol`
-   **Recommendation**: Consider adding `named_parameters()` to the protocol for more granular access/logging of model parameters.
-   **Affected System(s)/File(s)**: System 4 / `core/actor_critic_protocol.py`
-   **Actionable Steps**:
    1.  Add `named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]: ...` to the `ActorCriticProtocol` definition.
    2.  Ensure implementing classes satisfy this. (PyTorch `nn.Module` already provides this).
-   **Estimated Effort**: Very Small
-   **Impact**: Minor enhancement for debugging/logging flexibility.

### 2. Add Random Suffix to Session Run Names
-   **Recommendation**: Append a short random alphanumeric suffix to timestamp-based run names to further reduce collision probability in `SessionManager`.
-   **Affected System(s)/File(s)**: System 6 / `training/session_manager.py`
-   **Actionable Steps**:
    1.  In `SessionManager` where run names are generated (e.g., `_generate_run_name` or similar):
        -   If using a timestamp, append a short (e.g., 4-6 character) random string (e.g., using `random.choices` from `string.ascii_lowercase + string.digits`).
-   **Estimated Effort**: Very Small
-   **Impact**: Reduces likelihood of run name collisions, especially in automated/CI environments.

### 3. Centralize Device String Validation
-   **Recommendation**: `EnvManager` or `SetupManager` should validate the `device` string format and check availability (e.g., `torch.cuda.is_available()`).
-   **Affected System(s)/File(s)**: System 6 / `training/env_manager.py` or `training/setup_manager.py`.
-   **Actionable Steps**:
    1.  In the chosen manager (e.g., `SetupManager.initialize_components` or `EnvManager.setup_environment`):
        -   Add logic to parse the `config.env.device` string.
        -   If it starts with "cuda":
            -   Check `torch.cuda.is_available()`. If not, log a warning and fall back to "cpu", or raise an error if CUDA is strictly required.
            -   Validate the format (e.g., "cuda" or "cuda:0").
    2.  Ensure this validation happens early in the setup process.
-   **Estimated Effort**: Small
-   **Impact**: Improves robustness against configuration errors.

### 4. Maintain High-Quality Documentation and Comments
-   **Recommendation**: Maintain high-quality inline code comments alongside the excellent external documentation.
-   **Affected System(s)/File(s)**: All codebase.
-   **Actionable Steps**:
    1.  Encourage developers to write clear, concise comments for complex logic, public APIs, and non-obvious code sections during development and refactoring.
    2.  Periodically review comments for accuracy and relevance.
-   **Estimated Effort**: Ongoing
-   **Impact**: Improves code readability and maintainability.

---

This implementation plan provides a structured approach to addressing the review findings. Priorities and effort estimates can be adjusted based on team capacity and evolving project needs.
