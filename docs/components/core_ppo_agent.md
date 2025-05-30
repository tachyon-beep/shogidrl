# Software Documentation Template for Subsystems - PPO Agent

## 📘 ppo_agent.py as of 2025-05-31

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/core/`
**Documentation Version:** `1.0`
**Date:** `2025-05-31`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Implements the Proximal Policy Optimization (PPO) agent that serves as the core learning algorithm for the Shogi DRL system. This module orchestrates the training process by managing policy updates, value function training, and experience-based learning with clipped objectives and entropy regularization.

* **Key Responsibilities:**
  - Implement PPO algorithm with clipped surrogate objective
  - Manage actor-critic neural network training and optimization
  - Handle experience batch processing and minibatch training
  - Implement entropy regularization for exploration
  - Provide action selection interface for environment interaction
  - Track training metrics and learning statistics
  - Support both training and evaluation modes

* **Domain Context:**
  Proximal Policy Optimization for deep reinforcement learning in Shogi gameplay, implementing state-of-the-art policy gradient methods with proven stability and sample efficiency characteristics.

* **High-Level Architecture / Interaction Summary:**
  The PPOAgent serves as the central learning component, receiving experiences from the environment through the ExperienceBuffer, updating the ActorCritic neural network using PPO objectives, and providing trained policies for action selection. It interfaces with the training orchestrator for learning coordination and the neural network for policy representation.

---

### 2. Modules 📦

* **Module Name:** `ppo_agent.py`

  * **Purpose:** Implement the PPO learning algorithm for actor-critic policy optimization.
  * **Design Patterns Used:** Agent pattern for RL algorithms, Strategy pattern for different optimization configurations.
  * **Key Functions/Classes Provided:** 
    - `PPOAgent` - Main PPO implementation class
  * **Configuration Surface:**
    - `lr`: Learning rate for optimizer
    - `eps_clip`: PPO clipping parameter
    - `value_coef`: Value function loss coefficient
    - `entropy_coef`: Entropy regularization coefficient
    - `max_grad_norm`: Gradient clipping threshold
    - `ppo_epochs`: Number of PPO training epochs per batch
    - `minibatch_size`: Size of minibatches for training
  * **Dependencies:**
    * **Internal:**
      - `actor_critic_protocol.ActorCriticProtocol`: Interface for neural network
    * **External:**
      - `torch`: PyTorch framework for neural networks and optimization
      - `torch.optim`: Optimization algorithms (Adam optimizer)
      - `torch.nn.utils`: Gradient clipping utilities
      - `typing`: Type annotations (Optional, Dict, Any)
  * **External API Contracts:**
    - Provides RL agent interface for training orchestration
    - Compatible with OpenAI Gym-style environment interaction
    - Supports experience buffer integration
  * **Side Effects / Lifecycle Considerations:**
    - Modifies neural network parameters during training
    - Accumulates training statistics and metrics
    - Manages optimizer state across training steps
  * **Usage Examples:**
    ```python
    from keisei.core.ppo_agent import PPOAgent
    from keisei.core.neural_network import ActorCritic
    
    model = ActorCritic(input_channels=46, num_actions_total=4096)
    agent = PPOAgent(model, lr=3e-4, eps_clip=0.2)
    agent.learn(batch_data)
    action = agent.get_action(obs, legal_mask)
    ```

---

### 3. Classes 🏛️

* **Class Name:** `PPOAgent`

  * **Defined In Module:** `ppo_agent.py`
  * **Purpose:** Implement PPO algorithm for training actor-critic policies in Shogi reinforcement learning.
  * **Design Role:** Central learning agent implementing policy gradient optimization with proven stability guarantees.
  * **Inheritance:**
    * **Extends:** `object` (implicit)
    * **Subclasses (internal only):** None
  * **Key Attributes/Properties:**
    - `actor_critic: ActorCriticProtocol` – Neural network implementing actor-critic interface
    - `optimizer: torch.optim.Adam` – Adam optimizer for parameter updates
    - `lr: float` – Learning rate for optimization
    - `eps_clip: float` – PPO clipping parameter (typically 0.1-0.3)
    - `value_coef: float` – Coefficient for value function loss
    - `entropy_coef: float` – Coefficient for entropy regularization
    - `max_grad_norm: float` – Maximum gradient norm for clipping
    - `ppo_epochs: int` – Number of optimization epochs per batch
    - `minibatch_size: int` – Size of minibatches for training
    - `device: torch.device` – Device for tensor operations
  * **Key Methods:**
    - `learn()` - Main PPO training method
    - `get_action()` - Action selection for environment interaction
    - `get_value()` - Value estimation for advantage computation
    - `_compute_losses()` - Internal loss computation
    - `save_model()` / `load_model()` - Model persistence
  * **Interconnections:**
    * **Internal Class/Module Calls:** Uses ActorCritic for policy and value estimation
    * **External Systems:** PyTorch optimization ecosystem, training orchestrator
  * **Lifecycle & State:**
    - Initialized with neural network and hyperparameters
    - Maintains optimizer state across training iterations
    - Tracks training metrics and loss history
  * **Threading/Concurrency:**
    - Not thread-safe; designed for single-threaded training
    - Can be used in evaluation mode from multiple threads if network is in eval mode
  * **Usage Example:**
    ```python
    agent = PPOAgent(actor_critic_model, lr=3e-4, eps_clip=0.2, value_coef=0.5, entropy_coef=0.01)
    training_stats = agent.learn(experience_batch)
    ```

---

### 4. Functions/Methods ⚙️

* **Function/Method Name:** `__init__`

  * **Defined In:** `ppo_agent.py`
  * **Belongs To:** `PPOAgent`
  * **Purpose:** Initialize PPO agent with neural network and hyperparameters.
  * **Parameters:**
    - `actor_critic: ActorCriticProtocol` – Neural network implementing required interface
    - `lr: float = 3e-4` – Learning rate for optimization
    - `eps_clip: float = 0.2` – PPO clipping parameter
    - `value_coef: float = 0.5` – Value function loss coefficient
    - `entropy_coef: float = 0.01` – Entropy regularization coefficient
    - `max_grad_norm: float = 0.5` – Maximum gradient norm for clipping
    - `ppo_epochs: int = 4` – Number of training epochs per batch
    - `minibatch_size: int = 64` – Minibatch size for training
    - `device: str = "cpu"` – Device for tensor operations
  * **Returns:**
    - `None`
  * **Raises/Exceptions:**
    - PyTorch device errors for invalid device specification
  * **Side Effects:**
    - Creates Adam optimizer for neural network parameters
    - Sets up device and moves model if necessary
  * **Calls To:**
    - `torch.optim.Adam()` - Optimizer creation
    - `torch.device()` - Device specification
  * **Preconditions:** Valid actor-critic model and hyperparameters
  * **Postconditions:** Agent ready for training and action selection
  * **Algorithmic Note:**
    - Standard PPO hyperparameters with sensible defaults
  * **Usage Example:**
    ```python
    agent = PPOAgent(model, lr=3e-4, eps_clip=0.2, device="cuda")
    ```

* **Function/Method Name:** `learn`

  * **Defined In:** `ppo_agent.py`
  * **Belongs To:** `PPOAgent`
  * **Purpose:** Execute PPO training on a batch of experiences with multiple epochs and minibatching.
  * **Parameters:**
    - `batch_data: Dict[str, torch.Tensor]` – Experience batch from ExperienceBuffer
  * **Returns:**
    - `Dict[str, float]` – Training statistics (policy_loss, value_loss, entropy, etc.)
  * **Raises/Exceptions:**
    - Returns empty dict for invalid or empty batch data
  * **Side Effects:**
    - Updates neural network parameters
    - Modifies optimizer state
    - Accumulates gradient updates
  * **Calls To:**
    - `_compute_losses()` - Internal loss computation
    - `optimizer.zero_grad()`, `optimizer.step()` - Optimization steps
    - `torch.nn.utils.clip_grad_norm_()` - Gradient clipping
  * **Preconditions:** Valid batch data with required keys
  * **Postconditions:** Neural network parameters updated according to PPO objectives
  * **Algorithmic Note:**
    - Implements PPO clipped surrogate objective with value function and entropy losses
    - Uses multiple epochs and minibatching for stable learning
  * **Usage Example:**
    ```python
    stats = agent.learn(experience_buffer.get_batch())
    print(f"Policy loss: {stats['policy_loss']}")
    ```

* **Function/Method Name:** `get_action`

  * **Defined In:** `ppo_agent.py`
  * **Belongs To:** `PPOAgent`
  * **Purpose:** Select action for environment interaction using current policy.
  * **Parameters:**
    - `obs: torch.Tensor` – Current observation
    - `legal_mask: Optional[torch.Tensor] = None` – Legal action mask
    - `deterministic: bool = False` – Whether to use deterministic action selection
  * **Returns:**
    - `Tuple[int, float, float]` – (action_index, log_probability, value_estimate)
  * **Raises/Exceptions:**
    - PyTorch tensor operation exceptions
  * **Side Effects:**
    - None (inference only)
  * **Calls To:**
    - `actor_critic.get_action_and_value()` - Neural network forward pass
  * **Preconditions:** Valid observation tensor
  * **Postconditions:** Returns valid action within legal constraints
  * **Algorithmic Note:**
    - Delegates to neural network with optional deterministic mode
  * **Usage Example:**
    ```python
    action, log_prob, value = agent.get_action(obs_tensor, legal_mask, deterministic=True)
    ```

* **Function/Method Name:** `get_value`

  * **Defined In:** `ppo_agent.py`
  * **Belongs To:** `PPOAgent`
  * **Purpose:** Get value estimate for a given observation without action selection.
  * **Parameters:**
    - `obs: torch.Tensor` – Observation tensor
  * **Returns:**
    - `float` – Value estimate for the observation
  * **Raises/Exceptions:**
    - PyTorch tensor operation exceptions
  * **Side Effects:**
    - None (inference only)
  * **Calls To:**
    - `actor_critic.forward()` - Neural network forward pass
  * **Preconditions:** Valid observation tensor
  * **Postconditions:** Returns scalar value estimate
  * **Algorithmic Note:**
    - Extracts only value output from neural network
  * **Usage Example:**
    ```python
    value = agent.get_value(final_obs_tensor)
    ```

* **Function/Method Name:** `_compute_losses`

  * **Defined In:** `ppo_agent.py`
  * **Belongs To:** `PPOAgent`
  * **Purpose:** Compute PPO policy loss, value loss, and entropy for training.
  * **Parameters:**
    - `obs: torch.Tensor` – Observation batch
    - `actions: torch.Tensor` – Action batch
    - `old_log_probs: torch.Tensor` – Old log probabilities from experience
    - `advantages: torch.Tensor` – Advantage estimates
    - `returns: torch.Tensor` – Return targets for value function
    - `legal_masks: Optional[torch.Tensor] = None` – Legal action masks
  * **Returns:**
    - `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` – (policy_loss, value_loss, entropy)
  * **Raises/Exceptions:**
    - PyTorch tensor operation exceptions
  * **Side Effects:**
    - None (computation only)
  * **Calls To:**
    - `actor_critic.evaluate_actions()` - Action evaluation
    - PyTorch loss computations (MSE, clamp operations)
  * **Preconditions:** Valid batch tensors with consistent shapes
  * **Postconditions:** Returns loss tensors for optimization
  * **Algorithmic Note:**
    - Implements PPO clipped surrogate objective with importance sampling ratio
    - Combines policy, value, and entropy losses with configured coefficients
  * **Usage Example:**
    ```python
    policy_loss, value_loss, entropy = agent._compute_losses(obs, actions, old_log_probs, advantages, returns)
    ```

---

### 5. Shared or Complex Data Structures 📊

* **Structure Name:** `Training Statistics Dictionary`
  * **Type:** `Dict[str, float]`
  * **Purpose:** Return training metrics from learning iterations
  * **Format:** Dictionary with standardized metric keys
  * **Fields:**
    - `policy_loss: float` – Average policy loss across minibatches
    - `value_loss: float` – Average value function loss
    - `entropy: float` – Average policy entropy
    - `total_loss: float` – Combined loss (policy + value + entropy)
    - `grad_norm: float` – Gradient norm before clipping
    - `approx_kl: float` – Approximate KL divergence between old and new policies
  * **Validation Constraints:**
    - All values should be finite (not NaN or infinite)
    - Policy loss and value loss should be non-negative
  * **Used In:** Training monitoring, logging, early stopping decisions

* **Structure Name:** `PPO Hyperparameters`
  * **Type:** `Configuration Parameters`
  * **Purpose:** Define PPO algorithm behavior and training dynamics
  * **Format:** Instance attributes with default values
  * **Fields:**
    - `eps_clip: float` – Clipping parameter for surrogate objective (0.1-0.3)
    - `value_coef: float` – Weight for value function loss (0.5-1.0)
    - `entropy_coef: float` – Weight for entropy regularization (0.01-0.1)
    - `max_grad_norm: float` – Gradient clipping threshold (0.5-1.0)
    - `ppo_epochs: int` – Training epochs per batch (3-10)
    - `minibatch_size: int` – Minibatch size for SGD (32-128)
  * **Validation Constraints:**
    - eps_clip should be in (0, 1) range
    - Coefficients should be non-negative
    - Epochs and minibatch size should be positive integers
  * **Used In:** PPO loss computation, optimization scheduling, training stability

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  ppo_agent.py
    ├── uses → actor_critic_protocol.py (interface)
    ├── uses → neural_network.py (implementation)
    ├── receives data from → experience_buffer.py
    └── used by → training/trainer.py
  ```

* **Cross-Folder Imports:**
  - Used by `/training/` orchestrator for learning coordination
  - Receives experiences processed by ExperienceBuffer
  - Interfaces with neural network implementations

* **Data Flow Summary:**
  - Experience batch → learn() → PPO optimization → updated policy
  - Observation → get_action() → policy sampling → environment action
  - Training metrics → logging/monitoring systems
  - Model parameters → save/load → persistent storage

---

### 7. Non-Functional Aspects 🛠️

* **Performance:**
  - Minibatching reduces memory usage and improves gradient estimates
  - Multiple PPO epochs increase sample efficiency
  - Gradient clipping prevents training instability
  - Device placement optimization for GPU acceleration

* **Security:**
  - No external input validation (assumes trusted internal usage)
  - Model file I/O should use secure paths in production

* **Error Handling & Logging:**
  - Graceful handling of empty or invalid batch data
  - Gradient norm monitoring for training stability
  - Loss value validation to detect training divergence

* **Scalability Concerns:**
  - Memory usage scales with batch size and model parameters
  - Training time increases with ppo_epochs and minibatch count
  - Single-agent design limits to single-process training

* **Testing & Instrumentation:**
  - Unit tests for loss computation correctness
  - Integration tests with dummy neural networks
  - Performance profiling for optimization bottlenecks

---

### 8. Configuration & Environment ♻️

* **Environment Variables:**
  - PyTorch CUDA environment variables for GPU usage
  - No direct environment variable dependencies

* **CLI Interfaces / Entry Points:**
  - Not applicable (component used by training orchestrator)

* **Config File Schema:**
  - PPO parameters specified in training configuration:
    ```yaml
    ppo:
      lr: 3e-4
      eps_clip: 0.2
      value_coef: 0.5
      entropy_coef: 0.01
      max_grad_norm: 0.5
      ppo_epochs: 4
      minibatch_size: 64
    ```

---

### 9. Glossary 📖

* **PPO (Proximal Policy Optimization):** Policy gradient algorithm with clipped surrogate objective for stable training
* **Clipping Parameter (eps_clip):** Threshold for limiting policy updates to prevent large changes
* **Surrogate Objective:** Approximation of policy gradient objective using importance sampling
* **Advantage:** Estimate of action quality relative to average action value
* **Entropy Regularization:** Technique to encourage exploration by penalizing low-entropy policies
* **Minibatch:** Small subset of experience batch used for single gradient update

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:**
  - No automatic hyperparameter tuning or adaptation
  - Limited batch validation could allow invalid tensor operations
  - Gradient norm tracking could be more comprehensive

* **TODOs / Deferred Features:**
  - Implement adaptive learning rate scheduling
  - Add support for different optimizers (RMSprop, SGD)
  - Implement early stopping based on KL divergence
  - Add comprehensive training metrics and visualization

* **Suggested Refactors:**
  - Extract loss computation to separate utility module
  - Add configuration validation for hyperparameters
  - Implement model checkpointing and resumption
  - Add support for distributed training scenarios

---

## Notes for AI/Agent Developers 🧠

1. **PPO Stability:** The clipped objective prevents policy updates that are too large, maintaining training stability
2. **Sample Efficiency:** Multiple epochs and minibatching maximize learning from each experience batch
3. **Hyperparameter Sensitivity:** eps_clip and learning rate are critical for performance and should be tuned carefully
4. **Legal Action Integration:** The agent handles Shogi-specific legal action constraints transparently through the neural network interface
