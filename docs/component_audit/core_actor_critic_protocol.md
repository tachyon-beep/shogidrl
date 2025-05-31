# Software Documentation Template for Subsystems - Actor-Critic Protocol

## 📘 actor_critic_protocol.py as of 2024-12-28

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/core/`
**Documentation Version:** `1.0`
**Date:** `2024-12-28`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Defines the protocol interface for actor-critic neural networks used in the PPO-based deep reinforcement learning system for Shogi. This module establishes the contract that all actor-critic implementations must follow.

* **Key Responsibilities:**
  - Define the interface contract for actor-critic neural networks
  - Specify method signatures for action selection and evaluation
  - Ensure compatibility with PyTorch nn.Module requirements
  - Provide type hints for neural network operations

* **Domain Context:**
  Actor-critic architectures in deep reinforcement learning for Shogi gameplay, specifically designed for Proximal Policy Optimization (PPO) algorithms.

* **High-Level Architecture / Interaction Summary:**
  This protocol module serves as the interface definition that concrete neural network implementations (like `neural_network.py`) must implement. It ensures that the PPO agent can interact with any actor-critic network that follows this contract, providing modularity and extensibility for different network architectures.

---

### 2. Modules 📦

* **Module Name:** `actor_critic_protocol.py`

  * **Purpose:** Define the protocol interface for actor-critic neural networks in the PPO system.
  * **Design Patterns Used:** Protocol pattern for interface definition, providing runtime type checking capabilities.
  * **Key Functions/Classes Provided:** 
    - `ActorCriticProtocol` - Main protocol interface
  * **Configuration Surface:**
    - No direct configuration requirements
    - Relies on implementing classes for configuration handling
  * **Dependencies:**
    * **Internal:** None
    * **External:**
      - `typing.Protocol`: For interface definition
      - `typing.runtime_checkable`: For runtime type checking
      - `typing.Tuple`, `typing.Iterator`, `typing.Dict`, `typing.Any`: Type annotations
      - `torch`: PyTorch tensor operations and neural network modules
  * **External API Contracts:**
    - Provides protocol interface that must be implemented by concrete actor-critic classes
  * **Side Effects / Lifecycle Considerations:**
    - No side effects as this is purely an interface definition
  * **Usage Examples:**
    ```python
    from keisei.core.actor_critic_protocol import ActorCriticProtocol
    
    # Use for type checking
    def train_agent(model: ActorCriticProtocol):
        action, log_prob, value = model.get_action_and_value(obs, legal_mask)
    ```

---

### 3. Classes 🏛️

* **Class Name:** `ActorCriticProtocol`

  * **Defined In Module:** `actor_critic_protocol.py`
  * **Purpose:** Define the interface contract for actor-critic neural networks used in PPO training.
  * **Design Role:** Protocol interface ensuring consistent API across different actor-critic implementations.
  * **Inheritance:**
    * **Extends:** `typing.Protocol`
    * **Subclasses (internal only):** Implemented by `ActorCritic` in `neural_network.py`
  * **Key Attributes/Properties:**
    - No attributes defined (protocol only specifies method signatures)
  * **Key Methods:**
    - `get_action_and_value()` - Sample actions and get value estimates
    - `evaluate_actions()` - Evaluate log probabilities and entropy for given actions
    - PyTorch nn.Module methods (`train()`, `eval()`, `parameters()`, etc.)
  * **Interconnections:**
    * **Internal Class/Module Calls:** Used by `PPOAgent` for type checking
    * **External Systems:** PyTorch neural network ecosystem
  * **Lifecycle & State:**
    - Protocol defines interface only; state management handled by implementing classes
  * **Threading/Concurrency:**
    - No concurrency considerations at protocol level
  * **Usage Example:**
    ```python
    # Type annotation usage
    def create_ppo_agent(actor_critic: ActorCriticProtocol) -> PPOAgent:
        return PPOAgent(actor_critic)
    ```

---

### 4. Functions/Methods ⚙️

* **Function/Method Name:** `get_action_and_value`

  * **Defined In:** `actor_critic_protocol.py`
  * **Belongs To:** `ActorCriticProtocol`
  * **Purpose:** Sample an action from the policy and return value estimate for given observation.
  * **Parameters:**
    - `obs: torch.Tensor` – Input observation tensor
    - `legal_mask: Optional[torch.Tensor] = None` – Boolean mask indicating legal actions
    - `deterministic: bool = False` – Whether to use deterministic action selection
  * **Returns:**
    - `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` – (action, log_probability, value_estimate)
  * **Raises/Exceptions:**
    - Not specified in protocol (implementation-dependent)
  * **Side Effects:**
    - None at protocol level
  * **Calls To:**
    - Implementation-dependent
  * **Preconditions:** Valid observation tensor input
  * **Postconditions:** Returns valid action within legal action space if mask provided
  * **Algorithmic Note:**
    - Implements action sampling from policy distribution with optional legal action masking
  * **Usage Example:**
    ```python
    action, log_prob, value = model.get_action_and_value(obs, legal_mask, deterministic=False)
    ```

* **Function/Method Name:** `evaluate_actions`

  * **Defined In:** `actor_critic_protocol.py`
  * **Belongs To:** `ActorCriticProtocol`
  * **Purpose:** Evaluate log probabilities, entropy, and value estimates for given observation-action pairs.
  * **Parameters:**
    - `obs: torch.Tensor` – Input observation tensor
    - `actions: torch.Tensor` – Actions to evaluate
    - `legal_mask: Optional[torch.Tensor] = None` – Boolean mask for legal actions
  * **Returns:**
    - `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` – (log_probabilities, values, entropy)
  * **Raises/Exceptions:**
    - Implementation-dependent
  * **Side Effects:**
    - None at protocol level
  * **Calls To:**
    - Implementation-dependent
  * **Preconditions:** Valid observation and action tensors
  * **Postconditions:** Returns evaluation metrics for PPO training
  * **Algorithmic Note:**
    - Used during PPO policy updates to evaluate actions from experience buffer
  * **Usage Example:**
    ```python
    log_probs, values, entropy = model.evaluate_actions(obs_batch, actions_batch, legal_masks)
    ```

---

### 5. Shared or Complex Data Structures 📊

* **Structure Name:** `Method Signatures`
  * **Type:** `Protocol method definitions`
  * **Purpose:** Define the interface contract for actor-critic implementations
  * **Format:** Python type hints with Protocol
  * **Fields:**
    - Action sampling interface with optional deterministic mode
    - Action evaluation interface for PPO training
    - PyTorch nn.Module compatibility methods
  * **Validation Constraints:** 
    - Must return tensors of correct shapes
    - Legal mask handling must be consistent
  * **Used In:** PPOAgent, neural network implementations

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  actor_critic_protocol.py (interface)
    └── implemented by → neural_network.py
    └── used by → ppo_agent.py
  ```

* **Cross-Folder Imports:**
  - Used by `/training/` modules for type checking
  - Implemented by concrete neural network classes

* **Data Flow Summary:**
  - Protocol defines interface contract
  - Concrete implementations provide actual functionality
  - PPO agent uses protocol for type safety and modularity

---

### 7. Non-Functional Aspects 🛠️

* **Performance:**
  No performance impact as this is interface definition only

* **Security:**
  No security considerations at protocol level

* **Error Handling & Logging:**
  Error handling delegated to implementing classes

* **Scalability Concerns:**
  Protocol supports scalable implementations through consistent interface

* **Testing & Instrumentation:**
  - Test coverage through implementing classes
  - Protocol enables mockable interfaces for testing

---

### 8. Configuration & Environment ♻️

* **Environment Variables:**
  None required

* **CLI Interfaces / Entry Points:**
  Not applicable (interface definition only)

* **Config File Schema:**
  No configuration required

---

### 9. Glossary 📖

* **Protocol:** Python typing construct that defines interface contracts without inheritance
* **Actor-Critic:** Neural network architecture combining policy (actor) and value function (critic)
* **Legal Mask:** Boolean tensor indicating which actions are valid in current game state

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:**
  None identified

* **TODOs / Deferred Features:**
  - Consider adding protocol methods for model saving/loading interfaces
  - Potential extension for multi-head architectures

* **Suggested Refactors:**
  - Could be extended to support different action space types beyond discrete actions

---

## Notes for AI/Agent Developers 🧠

1. **Interface Contract:** This protocol ensures all actor-critic implementations follow the same interface, enabling modular design
2. **Type Safety:** Provides compile-time and runtime type checking for neural network components
3. **Extensibility:** New neural network architectures can be easily integrated by implementing this protocol
4. **PPO Compatibility:** Specifically designed to support PPO algorithm requirements with legal action masking
