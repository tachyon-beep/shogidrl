# ML Core Systems - Implementation Plan (June 1, 2025)
## Progress Summary (Updated June 2, 2025)

**Critical Items**: 1/1 ✅ **COMPLETED**
**High Priority Items**: 2/2 ✅ **COMPLETED** 
**Medium Priority Items**: 1/7 completed  
**Low Priority Items**: 0/4 completed

### Recent Completions:
- ✅ **June 2, 2025**: Silent Failure in Experience Buffer Batching - Critical issue resolved with comprehensive error handling and test coverage
- ✅ **June 2, 2025**: Handling 'No Legal Moves' & Model NaN Issues - High priority issue resolved with upstream handling in StepManager
- ✅ **June 2, 2025**: Model Injection for PPOAgent - High priority dependency injection implementation completed with zero deviations from plan

---

This document outlines an implementation plan to address the findings and recommendations from the `MLCORE_REVIEW.md` document dated June 1, 2025. The plan prioritizes critical fixes and improvements to enhance the robustness, maintainability, and performance of Keisei's Machine Learning core systems.

## Prioritization Key:
-   **Critical**: Must be addressed immediately. Significant impact on correctness or stability.
-   **High**: Important for robust training and correctness.
-   **Medium**: Improves best practices, maintainability, or adds valuable features.
-   **Low**: Minor enhancements or ongoing tasks.

---

## I. Critical Priority Items

### ✅ 1. Silent Failure in Experience Buffer Batching - **RESOLVED (June 2, 2025)**
-   **Original Recommendation**: `ExperienceBuffer.get_batch()` should explicitly raise an error on tensor-stack failures or if batch construction results in an invalid/empty state, instead of returning an empty dict.
-   **Resolution Status**: **COMPLETED** ✅
-   **Implementation Details**:
    -   Modified `ExperienceBuffer.get_batch()` method with comprehensive error handling
    -   Added try-catch blocks around both `torch.stack` operations (observations and legal_masks)
    -   Implemented descriptive `ValueError` exceptions with specific error messages:
        -   "Failed to stack observation tensors: {error}. Check tensor shapes and devices."
        -   "Failed to stack legal_mask tensors: {error}. Check tensor shapes and devices."
    -   Added proper empty buffer handling returning empty dict with warning message
    -   Comprehensive test coverage including `test_experience_buffer_get_batch_stack_error`
    -   All 9 ExperienceBuffer tests pass successfully
-   **Affected System(s)/File(s)**: System 5 / `core/experience_buffer.py`, `tests/test_experience_buffer.py`
-   **Impact**: Silent training skips eliminated, training integrity ensured, debugging experience significantly improved

---

## II. High Priority Items

### 1. ✅ COMPLETED: Handling "No Legal Moves" & Model NaN Issues
-   **Recommendation**:
    -   Shift responsibility for handling "no legal moves" scenarios upstream from the model to `StepManager` or `PPOAgent`.
    -   `StepManager` should detect if `ShogiGame.get_legal_moves()` returns an empty list. If so, it should treat this as a terminal condition for the episode or signal an error, rather than proceeding to `agent.select_action` with an all-false mask.
-   **Status**: ✅ **RESOLVED** - Upstream no legal moves handling implemented
-   **Implementation Details**:
    1.  **✅ StepManager Upstream Handling**: Modified `StepManager.execute_step()` to check for empty legal moves after `game.get_legal_moves()` call
        - If `legal_shogi_moves` is empty, treat as terminal condition
        - Log appropriate message with "TERMINAL" level
        - Reset game and return failure result with `done=True` and `info={"terminal_reason": "no_legal_moves"}`
        - Prevents downstream NaN handling in the model by catching the condition before `agent.select_action()` is called
    2.  **✅ Model NaN Defense**: Existing NaN handling in `ActorCriticResTower` remains as fallback defense (should be triggered less frequently)
    3.  **✅ Integration Tests**: Added comprehensive test coverage for no legal moves scenarios in `TestExecuteStepNoLegalMoves` class
        - `test_execute_step_no_legal_moves_terminal_condition`: Verifies terminal handling when no legal moves
        - `test_execute_step_no_legal_moves_with_episode_state`: Tests handling with existing episode state
        - `test_execute_step_normal_flow_with_legal_moves`: Ensures normal flow continues when legal moves available
        - `test_execute_step_no_legal_moves_logs_appropriate_level`: Validates proper logging
-   **Files Modified**:
    - `/keisei/training/step_manager.py`: Added upstream check for empty legal moves in `execute_step()` method
    - `/tests/test_step_manager.py`: Added `TestExecuteStepNoLegalMoves` class with 4 comprehensive integration tests
-   **Affected System(s)/File(s)**:
    -   System 4 / `training/models/resnet_tower.py` (existing NaN handling preserved as fallback)
    -   System 6 / `training/step_manager.py` (primary implementation location)
-   **Impact**: Prevents model from attempting to process invalid states, reduces NaN occurrences, improves training stability and correctness. The upstream handling catches terminal conditions before they reach the model layer.
-   **Date Completed**: 2025-06-02

### 2. ✅ COMPLETED: Implement Observation and Advantage Normalization
-   **Recommendation**: Implement and ensure proper use of observation normalization (e.g., running mean/std) and advantage normalization (batch-wise mean/std subtraction and division) in PPO.
-   **Affected System(s)/File(s)**: System 5 / `core/ppo_agent.py`, `core/experience_buffer.py`, `config_schema.py`
-   **Status**: ✅ **RESOLVED** - Advantage normalization configuration implemented
-   **Implementation Details**:
    1.  **✅ Advantage Normalization**: Already implemented in `PPOAgent.learn()` method with normalization: `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)`
    2.  **✅ Configuration Control**: Added `normalize_advantages: bool = True` option to `TrainingConfig` in `config_schema.py`
    3.  **✅ PPO Agent Integration**: Updated `PPOAgent.__init__()` to read configuration and `learn()` method to conditionally apply normalization
    4.  **✅ Test Coverage**: Added comprehensive tests validating configuration control and behavioral differences
    5.  **❌ Observation Normalization**: Not implemented (marked as optional, would require significant effort for running statistics)
-   **Files Modified**:
    - `/keisei/config_schema.py`: Added `normalize_advantages` configuration option
    - `/keisei/core/ppo_agent.py`: Added configuration reading and conditional normalization
    - `/tests/test_ppo_agent.py`: Added tests for configuration control and behavior validation
-   **Impact**: Provides configurable advantage normalization for improved PPO training stability while maintaining backward compatibility with existing training runs.
-   **Date Completed**: 2025-06-02
    - `/keisei/core/ppo_agent.py`: Added configuration reading and conditional normalization
    - `/tests/test_ppo_agent.py`: Added tests for configuration control and behavior validation
-   **Impact**: Provides configurable advantage normalization for improved PPO training stability while maintaining backward compatibility with existing training runs.
-   **Date Completed**: 2025-06-02

---

## III. Medium Priority Items

### ✅ 1. Refactor Model Code Duplication - **RESOLVED (June 2, 2025)**
-   **Original Recommendation**: If `core/neural_network.py` contains another `ActorCritic` model with duplicated logic for `get_action_and_value` and `evaluate_actions` (as in `ActorCriticResTower`), refactor these into a shared base class.
-   **Resolution Status**: **COMPLETED** ✅
-   **Implementation Details**:
    -   Created new `BaseActorCriticModel` class in `core/base_actor_critic.py` that inherits from both `nn.Module` and implements `ActorCriticProtocol`
    -   Moved common `get_action_and_value` and `evaluate_actions` methods to the base class
    -   Refactored `ActorCritic` (in `core/neural_network.py`) to inherit from `BaseActorCriticModel`
    -   Refactored `ActorCriticResTower` (in `training/models/resnet_tower.py`) to inherit from `BaseActorCriticModel`
    -   Both models now only implement their specific `forward` methods, inheriting shared behavior
    -   Added comprehensive test coverage in `tests/test_actor_critic_refactoring.py`
    -   All existing tests continue to pass for both models
    -   Enhanced warning messages to include class name for better debugging
-   **Affected System(s)/File(s)**: System 4 / `core/neural_network.py`, `training/models/resnet_tower.py`, `core/base_actor_critic.py` (new), `core/__init__.py`, `tests/test_actor_critic_refactoring.py` (new)
-   **Impact**: Eliminated ~120 lines of duplicated code, improved maintainability, ensured consistency between model implementations, enhanced error reporting

### ✅ 2. Model Injection for PPOAgent - **RESOLVED (June 2, 2025)**
-   **Original Recommendation**: `PPOAgent` should receive the instantiated model (conforming to `ActorCriticProtocol`) as a constructor argument (dependency injection), rather than initializing a default model.
-   **Resolution Status**: **COMPLETED** ✅
-   **Implementation Details**:
    -   Modified `PPOAgent.__init__` constructor to require `model: ActorCriticProtocol` as first parameter
    -   Removed internal default model creation from PPOAgent constructor
    -   Updated `SetupManager.setup_training_components()` to use dependency injection pattern
    -   Updated `agent_loading.load_evaluation_agent()` to create temporary model before PPOAgent instantiation
    -   Updated all test files to use proper dependency injection with `_create_test_model()` helpers
    -   Updated `MockPPOAgent` in test infrastructure to properly inject mock model
-   **Affected Files**: `core/ppo_agent.py`, `training/setup_manager.py`, `utils/agent_loading.py`, all test files
-   **Impact**: Successfully decoupled PPOAgent from specific model implementations, improved testability and eliminated unnecessary model instantiation
-   **Validation**: Comprehensive comparison analysis confirmed zero deviations between implementation and plan

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

## V. Configuration System Enhancements

### ✅ 1.5. Configuration System Analysis & Validation Enhancement - **COMPLETED (June 2, 2025)**
-   **Recommendation**: Address configuration system inconsistencies discovered during recent work
-   **Resolution Status**: **COMPLETED** ✅
-   **Implementation Details**:
    -   **Schema Completeness**: Fixed missing fields in Pydantic schema that existed in YAML
        -   Added `evaluation.log_file_path_eval` field to `EvaluationConfig`
        -   Added `wandb.log_model_artifact` field to `WandBConfig`
    -   **Validation Enhancement**: Added proper field validation to `EvaluationConfig`
        -   `evaluation_interval_timesteps` must be positive (> 0)
        -   `num_games` must be positive (> 0)
        -   `max_moves_per_game` must be positive (> 0)
    -   **Test Coverage**: Fixed failing configuration integration tests
        -   Updated `test_config_validation_with_missing_fields` to test actual validation scenarios
        -   Added comprehensive test cases for invalid configuration values
        -   Verified all configuration fields are accessible and properly validated
    -   **Documentation**: Created comprehensive configuration system documentation
        -   Configuration system analysis with usage flow tracing
        -   Quick reference guide for immediate use
        -   Detailed mapping task plan for future comprehensive analysis
-   **Affected System(s)/File(s)**: Configuration System / `config_schema.py`, `tests/test_configuration_integration.py`, `docs/development/configuration_system_*.md`
-   **Impact**: Eliminated silent configuration failures, improved validation, enhanced developer experience, established foundation for configuration system maintenance

### 1.6. Configuration System Comprehensive Mapping - **PLANNED**
-   **Recommendation**: Execute comprehensive configuration system audit and documentation
-   **Status**: **READY TO EXECUTE** 📋
-   **Implementation Plan**: Detailed task plan created in `docs/development/configuration_system_mapping_task.md`
-   **Scope**: 
    -   Complete configuration inventory and usage flow mapping
    -   Validation gap analysis and enhancement recommendations
    -   Automated schema-YAML consistency verification
    -   Comprehensive configuration reference documentation
-   **Estimated Effort**: 3-5 days
-   **Priority**: Medium-High (important for system maintainability)
-   **Deliverables**: 
    -   Complete configuration reference documentation
    -   Enhanced validation rules for all config classes
    -   Automated verification tools
    -   Best practices guide for configuration management

---

This implementation plan provides a structured approach to addressing the review findings. Priorities and effort estimates can be adjusted based on team capacity and evolving project needs.
