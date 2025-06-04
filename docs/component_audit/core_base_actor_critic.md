# Software Documentation Template for Subsystems - Base Actor Critic

## 📘 base_actor_critic.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/core/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Provides an abstract base class implementing shared logic for Actor-Critic neural network models. Subclasses implement the actual forward network but reuse the action and value calculation helpers defined here.
* **Key Responsibilities:**
  - Define the common interface used by PPO agents
  - Implement `get_action_and_value` and `evaluate_actions` helpers
  - Require subclasses to implement `forward`

### 2. Dependencies 🔗

* **Internal:** `keisei.core.actor_critic_protocol`, `keisei.utils.unified_logger`
* **External:** `torch`, `torch.nn`

---

## Notes for AI/Agent Developers 🧠

Use this base class when creating new network architectures so that they are compatible with the PPO agent without rewriting boilerplate logic.
