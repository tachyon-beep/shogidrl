## AI Programming Agent Tasking Statement: PPO Implementation Refinement

**Project Goal:** Refine and validate the existing Proximal Policy Optimization (PPO) algorithm implementation for the Shogi-playing agent to ensure algorithmic correctness, robustness, and adherence to best practices.

### I. High-Priority Implementation Tasks (Address Concerns):

1.  **Implement Value Function Clipping:**
    * **File:** `keisei/core/ppo_agent.py`
    * **Method:** `learn()`
    * **Task:** Implement value function clipping if `self.config.training.enable_value_clipping` is `True`.
    * **Details:**
        * Fetch the old value predictions (`old_values_minibatch`) for the current minibatch from the experience buffer. These are the `values` stored at collection time ($V_{\theta_{old}}(s_t)$).
        * Calculate the clipped value prediction: `values_pred_clipped = old_values_minibatch + torch.clamp(new_values - old_values_minibatch, -self.clip_epsilon, self.clip_epsilon)`.
        * Calculate two value losses:
            * `value_loss_unclipped = F.mse_loss(new_values.squeeze(), returns_minibatch.squeeze())` (current implementation).
            * `value_loss_clipped = F.mse_loss(values_pred_clipped.squeeze(), returns_minibatch.squeeze())`.
        * The final value loss should be `value_loss = torch.max(value_loss_unclipped, value_loss_clipped)`.
    * **Reference:** Standard PPO implementations with value clipping (e.g., OpenAI Baselines, CleanRL).
    * **Note:** Ensure `old_values_minibatch` are correctly retrieved and aligned with `new_values` and `returns_minibatch`.

2.  **Apply Observation Normalization:**
    * **File:** `keisei/core/ppo_agent.py`
    * **Methods:** `select_action()`, `learn()` (specifically where `obs_minibatch` is used), and potentially `get_value()`.
    * **Task:** Utilize the `self.scaler` (passed during `PPOAgent` initialization) to normalize observations before they are fed into the model.
    * **Details:**
        * If `self.scaler` is not `None` and is intended for observation normalization (e.g., a running mean-std scaler), apply its transformation method (e.g., `self.scaler.transform(obs_tensor)` or `self.scaler(obs_tensor)` depending on its API) to:
            * `obs_tensor` in `select_action()`.
            * `obs_tensor` in `get_value()`.
            * `obs_minibatch` in `learn()`.
        * Ensure the scaler is fitted appropriately (this typically happens elsewhere, e.g., `ModelManager` or during an initial data collection phase, but the agent should *use* the fitted scaler).
    * **Note:** This assumes `self.scaler` is an object with a compatible API for normalizing observation tensors.

3.  **Ensure `torch.no_grad()` in `select_action()` for Experience Collection:**
    * **File:** `keisei/core/ppo_agent.py`
    * **Method:** `select_action()`
    * **Task:** When `is_training` is `True` (i.e., during experience collection), ensure that the model's forward pass for action selection and value estimation is wrapped in `with torch.no_grad():`.
    * **Details:**
        ```python
        # In PPOAgent.select_action()
        if is_training:
            with torch.no_grad():
                (
                    selected_policy_index_tensor,
                    log_prob_tensor,
                    value_tensor,
                ) = self.model.get_action_and_value(
                    obs_tensor, legal_mask=legal_mask, deterministic=not is_training # deterministic will be False
                )
        else: # if not is_training (e.g. evaluation)
             (
                selected_policy_index_tensor,
                log_prob_tensor,
                value_tensor,
            ) = self.model.get_action_and_value(
                obs_tensor, legal_mask=legal_mask, deterministic=not is_training # deterministic will be True
            )
        ```
    * **Rationale:** Prevents unnecessary gradient computations and reduces memory usage during the data collection phase. The `self.model.train(is_training)` call correctly sets batch norm and dropout layers, but `torch.no_grad()` controls gradient calculation.

---

### II. Areas Consistent with Best Practices (No Change Needed): ✅

The following components and practices are well-implemented and align with established PPO theory and best practices:

1.  **Core PPO Loss Calculations (`ppo_agent.py`):**
    * **Policy Loss (Clipped Surrogate Objective):** Correctly implemented using probability ratios and clamping.
    * **Entropy Bonus:** Correctly calculated (based on legal moves) and added to the loss to encourage exploration.
    * **Combined Loss:** Standard combination of policy, value (current MSE form), and entropy losses with appropriate coefficients.

2.  **Experience Buffer (`experience_buffer.py`):**
    * **Data Storage:** Efficient pre-allocation of tensors for all necessary PPO elements.
    * **GAE Calculation:** `compute_advantages_and_returns` correctly implements Generalized Advantage Estimation with proper handling of `last_value` and terminal states (`masks_tensor`).
    * **Return Calculation:** Correctly calculates value function targets as $A_t^{\text{GAE}} + V(s_t)$.
    * **Batching:** `get_batch()` correctly returns the processed data for the learning phase.
    * **Device Handling:** Proper management of tensor devices within the buffer.

3.  **Training Loop Structure (`trainer.py`, `training_loop_manager.py`, `ppo_agent.py`):**
    * **Data Collection & PPO Update Cycle:** The overall cycle of collecting `steps_per_epoch` experiences, computing advantages, and then performing `ppo_epochs` of updates on minibatches is sound.
    * **Minibatch Sampling:** Shuffling data at the start of each PPO epoch and iterating through minibatches is correctly implemented in `PPOAgent.learn()`.
    * **Optimizer and Gradient Handling:** `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` are correctly sequenced. Gradient clipping (`torch.nn.utils.clip_grad_norm_`) is correctly applied. Mixed precision support (`torch.cuda.amp.GradScaler`) is correctly integrated.

4.  **Network Architecture & Forward Pass (`models/resnet_tower.py`, `core/base_actor_critic.py`):**
    * **Outputs:** Produces appropriate policy logits and scalar value predictions.
    * **Shared Backbone:** Efficient use of a shared backbone with separate heads.
    * **Legal Action Masking:** `BaseActorCriticModel` correctly uses `legal_mask` in `get_action_and_value` and `evaluate_actions` to handle logits and probabilities for legal actions, including for entropy calculation. The NaN fallback is a good robustness measure.

5.  **Hyperparameterization (`config_schema.py`):**
    * Key PPO hyperparameters are well-defined and fall within generally accepted ranges.
    * Configuration for learning rate scheduling is flexible.

6.  **Advantage Normalization (`ppo_agent.py`):**
    * Correctly implemented with normalization per batch and stability checks.

7.  **Learning Rate Scheduler (`core/scheduler_factory.py`, `core/ppo_agent.py`):**
    * The factory provides standard schedulers.
    * The calculation of `total_steps` for schedulers and the stepping logic (`lr_schedule_step_on`) in `PPOAgent` appear correct and robust for both "epoch" and "update" modes.

8.  **Code Structure and Clarity:**
    * The modular design with distinct manager classes (`Trainer`, `TrainingLoopManager`, `StepManager`, `ModelManager`, `SessionManager`, `EnvManager`, `MetricsManager`, `CallbackManager`, `DisplayManager`) is good for organization.
    * The `PPOAgent` and `ExperienceBuffer` classes are now clearly defined and generally well-commented.
