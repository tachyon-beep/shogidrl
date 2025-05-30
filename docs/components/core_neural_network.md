# Software Documentation Template for Subsystems - Neural Network

## 📘 neural_network.py as of 2025-05-31

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/core/`
**Documentation Version:** `1.0`
**Date:** `2025-05-31`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Implements the concrete Actor-Critic neural network architecture for the Shogi DRL agent. This module provides the neural network implementation that follows the ActorCriticProtocol interface, featuring convolutional layers for board state processing and separate policy and value heads.

* **Key Responsibilities:**
  - Implement actor-critic neural network architecture for Shogi board evaluation
  - Process board observations through convolutional feature extraction
  - Generate policy distributions over legal actions with masking support
  - Estimate state values for advantage computation
  - Handle action sampling with deterministic and stochastic modes
  - Provide action evaluation for PPO training updates

* **Domain Context:**
  Convolutional neural network architecture for Shogi game state evaluation, designed specifically for reinforcement learning with legal action constraints and board representation processing.

* **High-Level Architecture / Interaction Summary:**
  The ActorCritic class serves as the primary neural network component in the PPO training system. It processes board observations through convolutional layers and outputs both policy logits and value estimates. The network interfaces with the PPO agent for action selection and training updates, implementing the ActorCriticProtocol interface.

---

### 2. Modules 📦

* **Module Name:** `neural_network.py`

  * **Purpose:** Implement the concrete actor-critic neural network for Shogi gameplay.
  * **Design Patterns Used:** Actor-Critic architecture pattern, Protocol implementation pattern for interface compliance.
  * **Key Functions/Classes Provided:** 
    - `ActorCritic` - Main neural network class implementing ActorCriticProtocol
  * **Configuration Surface:**
    - `input_channels`: Number of input channels for board representation
    - `num_actions_total`: Total number of possible actions in action space
    - Network architecture parameters (kernel sizes, layer dimensions)
  * **Dependencies:**
    * **Internal:**
      - `actor_critic_protocol.ActorCriticProtocol`: Interface implementation
    * **External:**
      - `torch`: PyTorch neural network framework
      - `torch.nn`: Neural network modules and layers
      - `torch.nn.functional`: Activation and utility functions
      - `sys`: Standard error output for warnings
      - `typing`: Type annotations (Optional, Tuple)
  * **External API Contracts:**
    - Implements ActorCriticProtocol interface
    - Compatible with PyTorch nn.Module ecosystem
    - Supports legal action masking for Shogi constraints
  * **Side Effects / Lifecycle Considerations:**
    - Initializes learnable parameters on module creation
    - Warning messages printed to stderr for NaN handling
  * **Usage Examples:**
    ```python
    from keisei.core.neural_network import ActorCritic
    
    model = ActorCritic(input_channels=46, num_actions_total=4096)
    action, log_prob, value = model.get_action_and_value(obs, legal_mask)
    log_probs, entropy, values = model.evaluate_actions(obs_batch, actions_batch)
    ```

---

### 3. Classes 🏛️

* **Class Name:** `ActorCritic`

  * **Defined In Module:** `neural_network.py`
  * **Purpose:** Implement actor-critic neural network for Shogi RL agent with PPO training support.
  * **Design Role:** Concrete implementation of ActorCriticProtocol providing neural network functionality for policy and value estimation.
  * **Inheritance:**
    * **Extends:** `torch.nn.Module`
    * **Implements:** `ActorCriticProtocol` (implicit)
    * **Subclasses (internal only):** None
  * **Key Attributes/Properties:**
    - `conv: nn.Conv2d` – Convolutional layer for feature extraction (input_channels→16, 3x3 kernel)
    - `relu: nn.ReLU` – ReLU activation function
    - `flatten: nn.Flatten` – Flattening layer for transition to linear layers
    - `policy_head: nn.Linear` – Linear layer for policy logits (16*9*9 → num_actions_total)
    - `value_head: nn.Linear` – Linear layer for value estimation (16*9*9 → 1)
  * **Key Methods:**
    - `forward()` - Forward pass returning policy logits and value
    - `get_action_and_value()` - Sample action with value estimate
    - `evaluate_actions()` - Evaluate actions for PPO training
  * **Interconnections:**
    * **Internal Class/Module Calls:** Used by PPOAgent for policy updates
    * **External Systems:** PyTorch training ecosystem, CUDA acceleration
  * **Lifecycle & State:**
    - Initialized with learnable parameters via PyTorch's parameter initialization
    - Training/evaluation modes controlled by nn.Module interface
    - State persisted through PyTorch's state_dict mechanism
  * **Threading/Concurrency:**
    - Thread-safe for inference when in eval mode
    - Training requires single-threaded access for gradient updates
  * **Usage Example:**
    ```python
    model = ActorCritic(input_channels=46, num_actions_total=4096)
    model.to("cuda")
    action, log_prob, value = model.get_action_and_value(obs_tensor, legal_mask, deterministic=False)
    ```

---

### 4. Functions/Methods ⚙️

* **Function/Method Name:** `__init__`

  * **Defined In:** `neural_network.py`
  * **Belongs To:** `ActorCritic`
  * **Purpose:** Initialize the actor-critic network with specified architecture parameters.
  * **Parameters:**
    - `input_channels: int` – Number of input channels for board representation
    - `num_actions_total: int` – Total number of possible actions in action space
  * **Returns:**
    - `None`
  * **Raises/Exceptions:**
    - PyTorch initialization exceptions if invalid parameters
  * **Side Effects:**
    - Creates and initializes neural network layers
    - Registers parameters with PyTorch's parameter system
  * **Calls To:**
    - `super().__init__()` - PyTorch nn.Module initialization
    - Layer constructors for Conv2d, ReLU, Flatten, Linear
  * **Preconditions:** Valid positive integers for input parameters
  * **Postconditions:** Network ready for forward passes and training
  * **Algorithmic Note:**
    - Simple CNN architecture with single convolutional layer followed by linear heads
  * **Usage Example:**
    ```python
    model = ActorCritic(input_channels=46, num_actions_total=4096)
    ```

* **Function/Method Name:** `forward`

  * **Defined In:** `neural_network.py`
  * **Belongs To:** `ActorCritic`
  * **Purpose:** Perform forward pass through the network to get policy logits and value estimate.
  * **Parameters:**
    - `x: torch.Tensor` – Input observation tensor
  * **Returns:**
    - `Tuple[torch.Tensor, torch.Tensor]` – (policy_logits, value_estimate)
  * **Raises/Exceptions:**
    - PyTorch tensor operation exceptions for shape mismatches
  * **Side Effects:**
    - None (pure function)
  * **Calls To:**
    - Convolutional and linear layer forward methods
  * **Preconditions:** Input tensor with correct shape (batch_size, input_channels, 9, 9)
  * **Postconditions:** Returns policy and value outputs
  * **Algorithmic Note:**
    - Sequential processing: conv → relu → flatten → linear heads
  * **Usage Example:**
    ```python
    policy_logits, value = model.forward(obs_tensor)
    ```

* **Function/Method Name:** `get_action_and_value`

  * **Defined In:** `neural_network.py`
  * **Belongs To:** `ActorCritic`
  * **Purpose:** Sample action from policy and return log probability with value estimate, supporting legal action masking.
  * **Parameters:**
    - `obs: torch.Tensor` – Input observation tensor
    - `legal_mask: Optional[torch.Tensor] = None` – Boolean mask for legal actions
    - `deterministic: bool = False` – Whether to use deterministic action selection
  * **Returns:**
    - `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` – (action, log_probability, value)
  * **Raises/Exceptions:**
    - Warnings for NaN probabilities in stderr
  * **Side Effects:**
    - Prints warnings to stderr for probability NaNs
  * **Calls To:**
    - `forward()` method
    - PyTorch softmax, argmax, Categorical distribution
  * **Preconditions:** Valid observation tensor, optional legal mask
  * **Postconditions:** Returns valid action within legal constraints if mask provided
  * **Algorithmic Note:**
    - Applies legal masking by setting illegal action logits to -infinity
    - Handles NaN probabilities with uniform fallback distribution
  * **Usage Example:**
    ```python
    action, log_prob, value = model.get_action_and_value(obs, legal_mask, deterministic=True)
    ```

* **Function/Method Name:** `evaluate_actions`

  * **Defined In:** `neural_network.py`
  * **Belongs To:** `ActorCritic`
  * **Purpose:** Evaluate log probabilities, entropy, and values for given observation-action pairs during PPO training.
  * **Parameters:**
    - `obs: torch.Tensor` – Batch of observation tensors
    - `actions: torch.Tensor` – Batch of actions to evaluate
    - `legal_mask: Optional[torch.Tensor] = None` – Optional legal action masks
  * **Returns:**
    - `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` – (log_probabilities, entropy, values)
  * **Raises/Exceptions:**
    - Warnings for NaN probabilities in stderr
  * **Side Effects:**
    - Prints warnings to stderr for NaN handling
  * **Calls To:**
    - `forward()` method
    - PyTorch softmax, Categorical distribution operations
  * **Preconditions:** Batched observations and actions with consistent dimensions
  * **Postconditions:** Returns evaluation metrics for PPO policy updates
  * **Algorithmic Note:**
    - Applies legal masking if provided for accurate probability calculation
    - Handles NaN probabilities with uniform distribution fallback
    - Computes entropy over the action distribution
  * **Usage Example:**
    ```python
    log_probs, entropy, values = model.evaluate_actions(obs_batch, actions_batch, legal_masks)
    ```

---

### 5. Shared or Complex Data Structures 📊

* **Structure Name:** `Network Architecture`
  * **Type:** `Neural Network Layer Configuration`
  * **Purpose:** Define the structure of the actor-critic network for Shogi board processing
  * **Format:** PyTorch nn.Module layer definitions
  * **Fields:**
    - `Conv2d(input_channels, 16, kernel_size=3, padding=1)` – Feature extraction
    - `ReLU()` – Non-linear activation
    - `Flatten()` – Spatial to vector conversion
    - `Linear(16*9*9, num_actions_total)` – Policy head
    - `Linear(16*9*9, 1)` – Value head
  * **Validation Constraints:**
    - Input must be (batch_size, input_channels, 9, 9) for Shogi board
    - Output policy logits must match action space size
  * **Used In:** Forward pass computation, action sampling, action evaluation

* **Structure Name:** `Legal Action Masking`
  * **Type:** `torch.Tensor (boolean)`
  * **Purpose:** Constrain action selection to legal moves in Shogi
  * **Format:** Boolean tensor with True for legal actions, False for illegal
  * **Fields:**
    - Shape: (batch_size, num_actions_total) or (num_actions_total,)
    - Dtype: torch.bool
    - Device: Same as model parameters
  * **Validation Constraints:**
    - Must have at least one True value (legal action available)
    - Shape must be broadcastable with policy logits
  * **Used In:** Action sampling, probability computation, entropy calculation

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  neural_network.py
    ├── implements → actor_critic_protocol.py
    ├── used by → ppo_agent.py
    └── receives data from → experience_buffer.py
  ```

* **Cross-Folder Imports:**
  - Used by `/training/` modules for policy training
  - Receives observations from `/shogi/` game environment

* **Data Flow Summary:**
  - Observations from Shogi environment → forward() → policy logits and values
  - Legal masks from game rules → get_action_and_value() → legal action sampling
  - Experience batches → evaluate_actions() → training metrics for PPO updates

---

### 7. Non-Functional Aspects 🛠️

* **Performance:**
  - Lightweight CNN architecture for fast inference
  - Single convolutional layer limits computational complexity
  - Efficient tensor operations with GPU acceleration support

* **Security:**
  - Input validation through PyTorch tensor operations
  - No external input processing or file I/O

* **Error Handling & Logging:**
  - NaN detection and fallback mechanisms for numerical stability
  - Warning messages for problematic probability distributions
  - Graceful handling of illegal action masking edge cases

* **Scalability Concerns:**
  - Architecture designed for single-agent training
  - Memory usage scales linearly with batch size
  - Limited feature extraction capacity with simple CNN

* **Testing & Instrumentation:**
  - Unit tests should verify forward pass shapes and legal masking
  - Performance monitoring for inference latency
  - Gradient flow analysis for training stability

---

### 8. Configuration & Environment ♻️

* **Environment Variables:**
  - PyTorch CUDA environment variables for GPU usage
  - No direct environment variable dependencies

* **CLI Interfaces / Entry Points:**
  - Not applicable (neural network component)

* **Config File Schema:**
  - Network parameters specified in training configuration:
    ```yaml
    model:
      input_channels: 46
      num_actions_total: 4096
      architecture: "simple_cnn"
    ```

---

### 9. Glossary 📖

* **Actor-Critic:** Neural network architecture combining policy network (actor) and value function (critic)
* **Policy Logits:** Raw neural network outputs before softmax, representing action preferences
* **Legal Masking:** Technique to constrain action selection to valid moves in game environments
* **Entropy:** Measure of randomness in policy distribution, used for exploration in RL
* **Deterministic Mode:** Action selection using argmax instead of sampling for evaluation

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:**
  - Simple CNN architecture may limit learning capacity for complex Shogi patterns
  - NaN handling uses uniform distribution fallback which may not be optimal
  - No batch normalization or regularization techniques implemented

* **TODOs / Deferred Features:**
  - Implement deeper CNN architecture with residual connections
  - Add attention mechanisms for better board pattern recognition
  - Support for different action space configurations
  - Implement proper legal action masking validation

* **Suggested Refactors:**
  - Extract architecture configuration to separate module
  - Add support for different CNN backbone architectures
  - Implement proper initialization schemes for better training stability
  - Add model introspection and visualization capabilities

---

## Notes for AI/Agent Developers 🧠

1. **Simplicity by Design:** Current architecture prioritizes simplicity and debugging over performance
2. **Legal Masking Integration:** Legal action constraints are handled at the probability level, not logit level initially
3. **Numerical Stability:** NaN detection and fallback mechanisms ensure training doesn't crash on edge cases
4. **Protocol Compliance:** Strict adherence to ActorCriticProtocol ensures modularity and testability
