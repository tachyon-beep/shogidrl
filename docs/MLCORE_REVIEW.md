\
# ML Core Systems Review (Systems 4, 5, 6) - June 1, 2025

This document outlines findings, potential issues, and recommendations for improvement related to the Machine Learning (ML) core systems of the Keisei Shogi project, specifically focusing on:
- System 4: Neural Network Models
- System 5: Reinforcement Learning Core
- System 6: Training Session & Environment Management

This review is based on `CODE_MAP.md`, `DESIGN.md`, component audit documents, and specific code files (`ActorCriticProtocol.py`, `resnet_tower.py`).

## System 4: Neural Network Models

Covers `core/neural_network.py`, `training/models/resnet_tower.py`, and `core/actor_critic_protocol.py`.

### Findings & Recommendations

#### 1. `ActorCriticProtocol.py`
-   **Strength**: The protocol is well-defined and covers essential methods for an actor-critic model, promoting API compatibility.
-   **Recommendation (Minor)**:
    -   Consider adding `named_parameters()` to the protocol for more granular access/logging of model parameters if future needs arise.

#### 2. `training/models/resnet_tower.py` (ActorCriticResTower)
-   **Strength (NaN Handling)**: The model includes defensive checks for `NaN` in probabilities and attempts to fall back to a uniform distribution. This prevents immediate crashes.
-   **Strength (Legal Mask Handling)**: Logic for applying `legal_mask` (setting illegal action logits to `-inf`) is standard and correctly implemented. Shape adjustments for batch size 1 are good.
-   **Strength (Squeeze-and-Excitation)**: SE block implementation is standard.
-   **Strength (Slim Heads)**: Use of 1x1 convolutions for parameter efficiency in policy/value heads is good practice.

-   **Concern/Recommendation (NaN Handling & "No Legal Moves")**:
    -   **Issue**: The `CODE_MAP.md` notes a concern: "NaN Path: Test and handle the −∞ logits → NaN probability path". Current handling (warning + uniform distribution) might mask underlying issues, particularly if *no legal moves* are available. The `pass` statement when `not torch.any(legal_mask)` in `get_action_and_value` is too passive and can lead to NaNs if all logits become `-inf`.
    -   **Recommendation**:
        -   The responsibility for handling "no legal moves" scenarios should ideally reside upstream (e.g., in `PPOAgent` or `StepManager`) rather than the model attempting to recover from NaNs arising from game-specific states.
        -   If `StepManager` detects no legal moves from `ShogiGame.get_legal_moves()`, it should likely terminate the episode or signal an error, preventing the model from receiving an all-false mask or problematic logits.

-   **Concern/Recommendation (Code Duplication - `get_action_and_value`, `evaluate_actions`)**:
    -   **Issue**: `CODE_MAP.md` suggests: "Refactor common methods (`get_action_and_value`, `evaluate_actions`) from both model classes into a shared base." `ActorCriticResTower` implements these. If `core/neural_network.py` contains another `ActorCritic` model with duplicated logic, this should be addressed.
    -   **Recommendation**: If duplication exists, refactor these methods into a base class that both `ActorCriticResTower` and the other `ActorCritic` model inherit from. This base class would itself inherit `nn.Module` and implement the `ActorCriticProtocol`.

-   **Note (BatchNorm Handling)**:
    -   `CODE_MAP.md` mentions: "Handle BatchNorm running stats correctly during save/load and evaluation."
    -   **Observation**: The model uses `nn.BatchNorm2d`. PyTorch generally handles this well. This is more a procedural check for checkpointing and evaluation mode logic.

-   **Note (Input Channel Padding)**:
    -   `CODE_MAP.md` mentions: "`utils/checkpoint.load_checkpoint_with_padding` only handles `stem.weight` input-channel changes; generalise or document."
    -   **Observation**: This is a utility concern but relates to the model's `stem.weight`.

## System 5: Reinforcement Learning Core

Covers `core/ppo_agent.py` and `core/experience_buffer.py`.

### Findings & Recommendations

#### 1. `core/ppo_agent.py` (PPOAgent)
-   **Concern/Recommendation (Model Injection)**:
    -   **Issue**: `CODE_MAP.md` states: "`PPOAgent` initializes a default `ActorCritic` model, later replaced by `Trainer` with one from `ModelManager`." This creates tight coupling.
    -   **Recommendation**: `PPOAgent` should receive the instantiated model (conforming to `ActorCriticProtocol`) as a constructor argument (dependency injection). `ModelManager` would create/load the model, and `Trainer` would pass it to `PPOAgent`.

-   **Note (Hyperparameters)**:
    -   `CODE_MAP.md` advises: "Ensure sanity and appropriate scaling of PPO hyperparameters."
    -   **Observation**: Default values in `TrainingConfig` should be sensible (e.g., `clip_epsilon=0.2`, `gamma=0.99`, `lambda_gae=0.95`).

-   **Recommendation (Learning Rate Scheduling)**:
    -   The `DESIGN.md` mentions "Adaptive learning rate support." Ensure this is implemented (e.g., via `torch.optim.lr_scheduler`) and configurable.

-   **Recommendation (Value Function Clipping)**:
    -   Consider implementing value function loss clipping (similar to policy clipping) if value loss instability is observed. This is a common PPO enhancement.

-   **Recommendation (Normalization)**:
    -   Implement and ensure proper use of observation normalization (e.g., running mean/std) and advantage normalization (batch-wise mean/std subtraction and division). Advantage normalization is particularly crucial for PPO stability.

#### 2. `core/experience_buffer.py` (ExperienceBuffer)
-   **Concern/Recommendation (Memory Usage for `legal_masks`)**:
    -   **Issue**: `CODE_MAP.md` notes: "`ExperienceBuffer` stores ~13k booleans/step for `legal_masks`". (Note: `DESIGN.md` implies 6480 actions, but the concern remains). Storing dense boolean masks is memory-intensive.
    -   **Recommendation**:
        -   **On-the-fly regeneration**: If `ShogiGame.get_legal_moves()` is sufficiently fast, regenerate masks when constructing minibatches. This depends on the performance trade-off.
        -   **Sparse representation**: If regeneration is too slow, consider sparse storage if the number of legal moves is typically much smaller than the total action space.

-   **✅ RESOLVED - Critical Issue (Silent Failure in Batching)**:
    -   **Previous Issue**: `CODE_MAP.md` warned: "`ExperienceBuffer.get_batch()` returns an empty dict on tensor-stack errors, risking silent training skips."
    -   **Resolution (June 2, 2025)**: This critical issue has been **completely resolved**. The `ExperienceBuffer.get_batch()` method now includes:
        -   Comprehensive try-catch blocks around both `torch.stack` operations (observations and legal_masks)
        -   Descriptive `ValueError` exceptions with specific error messages and troubleshooting guidance
        -   Proper empty buffer handling with warning messages
        -   Comprehensive test coverage including `test_experience_buffer_get_batch_stack_error`
    -   **Impact**: Silent failures in the data pipeline have been eliminated, significantly improving training reliability and debugging experience.

-   **Note (GAE Calculation)**:
    -   Ensure Generalized Advantage Estimation (GAE) is calculated correctly, typically iterated backward from the end of trajectories.

-   **Note (Device Handling)**:
    -   The buffer should efficiently move data to the configured device (`config.env.device`), usually during minibatch creation.

## System 6: Training Session & Environment Management

Covers `training/session_manager.py`, `training/env_manager.py`, and `training/step_manager.py`.

### Findings & Recommendations

#### 1. `training/session_manager.py` (SessionManager)
-   **Recommendation (Run Name Collisions)**:
    -   **Issue**: `CODE_MAP.md` notes timestamp-based run names could collide in high-frequency CI.
    -   **Recommendation**: Append a short random alphanumeric suffix to timestamp-based run names to further reduce collision probability.

-   **Note (W&B Integration)**:
    -   Ensure robust `wandb.init()` handling (retries, clear errors) and that `wandb.finish()` is called appropriately (end of training, exceptions).

-   **Note (Configuration Saving)**:
    -   Saving the *effective* configuration (after all overrides) is crucial for reproducibility. `DESIGN.md` confirms this is intended.

#### 2. `training/env_manager.py` (EnvManager)
-   **Strength (Action Space Validation)**:
    -   `CODE_MAP.md` confirms: "`EnvManager` fatally errors on mismatch—good guard-rail." This is excellent.

-   **Note (Seeding)**:
    -   Ensure comprehensive seeding (`random`, `numpy.random`, `torch.manual_seed`) is correctly applied for reproducibility.

-   **Recommendation (Device String Validation)**:
    -   **Issue**: `CODE_MAP.md` suggests: "`EnvManager` could centralise validation for device strings (e.g., `cuda:0` vs `cuda:O`)."
    -   **Recommendation**: `EnvManager` or `SetupManager` should validate the `device` string format and check availability (e.g., `torch.cuda.is_available()`).

#### 3. `training/step_manager.py` (StepManager)
-   **Note (Demo Mode)**:
    -   Per-ply sleep for demo mode is correctly handled by being config-dependent (`demo.enable_demo_mode`).

-   **Concern/Recommendation (Handling "No Legal Moves")**:
    -   **Issue**: If `game.get_legal_moves()` returns an empty list, how does `StepManager` handle this? Passing an all-false `legal_mask` can lead to model-side NaN issues.
    -   **Recommendation**: `StepManager` should detect this scenario. If no legal moves are available, it should likely be treated as a terminal condition for the episode or an error, rather than proceeding to `agent.select_action`.

-   **Note (Episode State Management)**:
    -   The `EpisodeState` dataclass is good. Ensure rewards, dones, and observations are correctly collected.

-   **Note (Logging)**:
    -   `logger_func` is passed to `execute_step`. Ensure logging is informative but not performance-hindering.

## General Cross-Cutting Concerns
-   **Error Handling and Resilience**: Review the holistic strategy for error propagation and system recovery/graceful termination.
-   **Testing**: Address specific areas needing more thorough testing as highlighted in `CODE_MAP.md` (e.g., complex Shogi rules, I/O, model NaN paths).
-   **Documentation and Comments**: Maintain high-quality inline code comments alongside the excellent external documentation.

This review aims to identify areas for further refinement and to ensure the robustness and adherence to best practices within Keisei's ML core.
