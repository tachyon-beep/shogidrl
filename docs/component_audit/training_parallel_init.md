# Software Documentation Template for Subsystems - Parallel Training Package

## 📘 training/parallel/__init__.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/training/parallel/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Initializes the parallel training subsystem and exposes its primary classes. This package enables experience collection across multiple worker processes.
* **Key Responsibilities:**
  - Re-export `ParallelManager`, `SelfPlayWorker`, `ModelSynchronizer`, and helpers
  - Provide package metadata and versioning

### 2. Dependencies 🔗

* **Internal:** other modules in `keisei.training.parallel`
* **External:** `multiprocessing`

---

## Notes for AI/Agent Developers 🧠

Import `ParallelManager` from this package to coordinate self-play workers in your training loop.
