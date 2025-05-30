# Software Documentation Template for Subsystems - Experience Buffer

## 📘 experience_buffer.py as of 2025-05-31

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/core/`
**Documentation Version:** `1.0`
**Date:** `2025-05-31`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Implements the experience buffer for storing and processing transitions during reinforcement learning training. This module handles the collection, processing, and batching of experiences for PPO training with Generalized Advantage Estimation (GAE).

* **Key Responsibilities:**
  - Store transitions (observations, actions, rewards, etc.) during episode rollouts
  - Compute advantages and returns using Generalized Advantage Estimation (GAE)
  - Batch experiences for neural network training
  - Handle legal action masks for Shogi-specific constraints
  - Manage buffer lifecycle and memory efficiently

* **Domain Context:**
  Experience replay and advantage estimation in PPO-based deep reinforcement learning for Shogi gameplay, with support for legal action masking and GAE computation.

* **High-Level Architecture / Interaction Summary:**
  The ExperienceBuffer sits between the training loop and the PPO agent, collecting experiences during environment interaction and preparing batched data for policy updates. It interfaces with the training orchestrator to receive transitions and provides processed batches to the PPO agent for learning.

---

### 2. Modules 📦

* **Module Name:** `experience_buffer.py`

  * **Purpose:** Implement experience storage and processing for PPO training with GAE.
  * **Design Patterns Used:** Buffer pattern for experience storage, batch processing for efficient neural network training.
  * **Key Functions/Classes Provided:** 
    - `ExperienceBuffer` - Main buffer class for experience management
  * **Configuration Surface:**
    - `buffer_size`: Maximum number of transitions to store
    - `gamma`: Discount factor for future rewards
    - `lambda_gae`: GAE lambda parameter for bias-variance tradeoff
    - `device`: PyTorch device for tensor operations
  * **Dependencies:**
    * **Internal:** None
    * **External:**
      - `torch`: PyTorch for tensor operations and device management
  * **External API Contracts:**
    - Provides standardized interface for experience collection and batch preparation
    - Compatible with PPO training requirements
  * **Side Effects / Lifecycle Considerations:**
    - Manages memory through tensor operations on specified device
    - Clears buffers after batch processing to prevent memory leaks
  * **Usage Examples:**
    ```python
    from keisei.core.experience_buffer import ExperienceBuffer
    
    buffer = ExperienceBuffer(buffer_size=2048, gamma=0.99, lambda_gae=0.95, device="cuda")
    buffer.add(obs, action, reward, log_prob, value, done, legal_mask)
    buffer.compute_advantages_and_returns(last_value)
    batch_data = buffer.get_batch()
    ```

---

### 3. Classes 🏛️

* **Class Name:** `ExperienceBuffer`

  * **Defined In Module:** `experience_buffer.py`
  * **Purpose:** Store and process experience transitions for PPO training with GAE computation.
  * **Design Role:** Buffer manager implementing experience replay with advantage estimation for policy gradient methods.
  * **Inheritance:**
    * **Extends:** `object` (implicit)
    * **Subclasses (internal only):** None
  * **Key Attributes/Properties:**
    - `buffer_size: int` – Maximum number of transitions to store
    - `gamma: float` – Discount factor for future rewards
    - `lambda_gae: float` – GAE lambda parameter
    - `device: torch.device` – PyTorch device for computations
    - `obs: list[torch.Tensor]` – List of observation tensors
    - `actions: list[int]` – List of action indices
    - `rewards: list[float]` – List of immediate rewards
    - `log_probs: list[float]` – List of action log probabilities
    - `values: list[float]` – List of value function estimates
    - `dones: list[bool]` – List of episode termination flags
    - `legal_masks: list[torch.Tensor]` – List of legal action masks
    - `advantages: list[torch.Tensor]` – Computed advantage estimates
    - `returns: list[torch.Tensor]` – Computed discounted returns
    - `ptr: int` – Current buffer position pointer
  * **Key Methods:**
    - `add()` - Add new transition to buffer
    - `compute_advantages_and_returns()` - Compute GAE advantages and returns
    - `get_batch()` - Return batched data for training
    - `clear()` - Clear all stored experiences
    - `__len__()` - Return current buffer size
  * **Interconnections:**
    * **Internal Class/Module Calls:** Used by training loop and PPO agent
    * **External Systems:** PyTorch tensor operations, CUDA device management
  * **Lifecycle & State:**
    - Initializes empty lists for all experience components
    - Fills incrementally during episode rollouts
    - Processes advantages/returns when buffer is full
    - Cleared after batch extraction for next rollout
  * **Threading/Concurrency:**
    - Not thread-safe; designed for single-threaded training loops
  * **Usage Example:**
    ```python
    buffer = ExperienceBuffer(2048, 0.99, 0.95, "cuda")
    # During rollout
    buffer.add(obs, action, reward, log_prob, value, done, legal_mask)
    # After rollout
    buffer.compute_advantages_and_returns(final_value)
    batch = buffer.get_batch()
    ```

---

### 4. Functions/Methods ⚙️

* **Function/Method Name:** `add`

  * **Defined In:** `experience_buffer.py`
  * **Belongs To:** `ExperienceBuffer`
  * **Purpose:** Add a single transition to the experience buffer.
  * **Parameters:**
    - `obs: torch.Tensor` – Observation tensor (C, H, W) on device
    - `action: int` – Action index taken
    - `reward: float` – Immediate reward received
    - `log_prob: float` – Log probability of the action
    - `value: float` – Value function estimate for the state
    - `done: bool` – Whether episode terminated
    - `legal_mask: torch.Tensor` – Boolean mask of legal actions
  * **Returns:**
    - `None`
  * **Raises/Exceptions:**
    - Prints warning if buffer is full
  * **Side Effects:**
    - Appends transition data to internal lists
    - Increments buffer pointer
  * **Calls To:**
    - List append operations
  * **Preconditions:** Buffer not full (ptr < buffer_size)
  * **Postconditions:** Transition stored in buffer, pointer incremented
  * **Algorithmic Note:**
    - Simple list-based storage with overflow protection
  * **Usage Example:**
    ```python
    buffer.add(obs_tensor, action_idx, reward_val, log_prob_val, value_est, is_done, legal_mask_tensor)
    ```

* **Function/Method Name:** `compute_advantages_and_returns`

  * **Defined In:** `experience_buffer.py`
  * **Belongs To:** `ExperienceBuffer`
  * **Purpose:** Compute GAE advantages and discounted returns for all stored experiences.
  * **Parameters:**
    - `last_value: float` – Value estimate for the state following the last stored transition
  * **Returns:**
    - `None`
  * **Raises/Exceptions:**
    - Prints warning if called on empty buffer
  * **Side Effects:**
    - Populates `advantages` and `returns` lists
    - Performs tensor operations on device
  * **Calls To:**
    - PyTorch tensor operations (stack, tensor creation)
  * **Preconditions:** Buffer contains transitions (ptr > 0)
  * **Postconditions:** Advantages and returns computed and stored
  * **Algorithmic Note:**
    - Implements Generalized Advantage Estimation with configurable lambda
    - Uses reverse iteration for temporal dependency computation
  * **Usage Example:**
    ```python
    buffer.compute_advantages_and_returns(final_state_value)
    ```

* **Function/Method Name:** `get_batch`

  * **Defined In:** `experience_buffer.py`
  * **Belongs To:** `ExperienceBuffer`
  * **Purpose:** Return all stored experiences as batched PyTorch tensors for training.
  * **Parameters:**
    - None
  * **Returns:**
    - `dict` – Dictionary containing batched tensors: obs, actions, log_probs, values, advantages, returns, dones, legal_masks
  * **Raises/Exceptions:**
    - Returns empty dict if buffer is empty or advantages not computed
    - Prints warnings for tensor stacking errors
  * **Side Effects:**
    - Creates large tensors by stacking stored data
  * **Calls To:**
    - PyTorch tensor operations (stack, tensor creation)
  * **Preconditions:** Advantages and returns computed
  * **Postconditions:** Returns structured batch data for training
  * **Algorithmic Note:**
    - Efficiently batches variable-length sequences into fixed tensors
    - Handles device placement for all tensors
  * **Usage Example:**
    ```python
    batch_data = buffer.get_batch()
    obs_batch = batch_data['obs']
    advantages_batch = batch_data['advantages']
    ```

* **Function/Method Name:** `clear`

  * **Defined In:** `experience_buffer.py`
  * **Belongs To:** `ExperienceBuffer`
  * **Purpose:** Clear all stored experiences and reset buffer state.
  * **Parameters:**
    - None
  * **Returns:**
    - `None`
  * **Raises/Exceptions:**
    - None
  * **Side Effects:**
    - Clears all internal lists
    - Resets pointer to 0
  * **Calls To:**
    - List clear() method
  * **Preconditions:** None
  * **Postconditions:** Buffer is empty and ready for new rollout
  * **Algorithmic Note:**
    - Simple state reset for buffer reuse
  * **Usage Example:**
    ```python
    buffer.clear()  # Prepare for next rollout
    ```

---

### 5. Shared or Complex Data Structures 📊

* **Structure Name:** `Batch Data Dictionary`
  * **Type:** `Dict[str, torch.Tensor]`
  * **Purpose:** Structured format for batched experience data returned by get_batch()
  * **Format:** Dictionary with standardized keys
  * **Fields:**
    - `obs: torch.Tensor` – Batched observations (N, C, H, W)
    - `actions: torch.Tensor` – Batched action indices (N,)
    - `log_probs: torch.Tensor` – Batched log probabilities (N,)
    - `values: torch.Tensor` – Batched value estimates (N,)
    - `advantages: torch.Tensor` – Batched GAE advantages (N,)
    - `returns: torch.Tensor` – Batched discounted returns (N,)
    - `dones: torch.Tensor` – Batched episode termination flags (N,)
    - `legal_masks: torch.Tensor` – Batched legal action masks (N, num_actions)
  * **Validation Constraints:**
    - All tensors must be on the same device
    - Batch dimensions must be consistent
    - Legal masks must match action space size
  * **Used In:** PPO agent training, neural network forward passes

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  experience_buffer.py
    └── used by → ppo_agent.py
    └── used by → training/trainer.py
  ```

* **Cross-Folder Imports:**
  - Used by `/training/` modules for experience collection
  - Provides data to neural network modules in `core/`

* **Data Flow Summary:**
  - Training loop → add() transitions during rollout
  - compute_advantages_and_returns() → processes accumulated experiences
  - get_batch() → provides structured data to PPO agent
  - clear() → resets for next training iteration

---

### 7. Non-Functional Aspects 🛠️

* **Performance:**
  - Memory usage scales with buffer_size and observation dimensions
  - GAE computation is O(buffer_size) with efficient tensor operations
  - Device placement optimization for GPU acceleration

* **Security:**
  - No external input validation (assumes trusted internal usage)
  - Tensor operations are memory-safe through PyTorch

* **Error Handling & Logging:**
  - Warning messages for buffer overflow and empty buffer operations
  - Graceful handling of tensor stacking errors

* **Scalability Concerns:**
  - Buffer size limited by available GPU/CPU memory
  - Not designed for distributed training scenarios

* **Testing & Instrumentation:**
  - Unit tests should verify GAE computation correctness
  - Memory usage monitoring recommended for large buffer sizes

---

### 8. Configuration & Environment ♻️

* **Environment Variables:**
  - None directly; device selection may use CUDA environment variables

* **CLI Interfaces / Entry Points:**
  - Not applicable (internal component)

* **Config File Schema:**
  - Buffer parameters typically specified in training configuration:
    ```yaml
    buffer_size: 2048
    gamma: 0.99
    lambda_gae: 0.95
    device: "cuda"
    ```

---

### 9. Glossary 📖

* **GAE (Generalized Advantage Estimation):** Method for estimating advantage function with bias-variance tradeoff control
* **Lambda (λ):** GAE parameter controlling bias-variance tradeoff (0=high bias/low variance, 1=low bias/high variance)
* **Advantage:** Estimate of how much better an action is compared to the average action in a state
* **Return:** Discounted sum of future rewards from a given state
* **Legal Mask:** Boolean tensor indicating which actions are valid in Shogi game state

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:**
  - Buffer overflow handling could be more robust
  - Tensor stacking errors are logged but not gracefully recovered

* **TODOs / Deferred Features:**
  - Support for prioritized experience replay
  - Memory-efficient storage for very large buffers
  - Distributed buffer for multi-worker training

* **Suggested Refactors:**
  - Consider using torch.nn.utils.rnn.pad_sequence for variable-length sequences
  - Add automatic device detection and migration
  - Implement buffer checkpointing for training resumption

---

## Notes for AI/Agent Developers 🧠

1. **GAE Implementation:** The GAE computation uses reverse iteration to maintain temporal dependencies correctly
2. **Memory Management:** Buffer is designed to be cleared after each training iteration to prevent memory accumulation
3. **Device Consistency:** All tensors are maintained on the specified device for efficient GPU utilization
4. **Shogi-Specific:** Legal mask handling is integrated throughout for game-specific constraints
