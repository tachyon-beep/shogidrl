# Software Documentation Template for Subsystems - Keisei Package

## 📘 __init__.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Provides the package initialization for the top level `keisei` package. It re-exports commonly used classes such as `ShogiGame` and evaluation utilities so that users can import them directly from `keisei`.
* **Key Responsibilities:**
  - Define the public API of the package through `__all__`
  - Re-export central types and helpers from submodules
  - Simplify user imports for evaluation entry points

### 2. Dependencies 🔗

* **Internal:**
  - `keisei.shogi.shogi_core_definitions`
  - `keisei.shogi.shogi_game`
  - `keisei.evaluation.evaluate`
* **External:** None

---

## Notes for AI/Agent Developers 🧠

The package `__init__` keeps the import surface small and friendly. When extending the project, update `__all__` here to expose new high-level utilities.
