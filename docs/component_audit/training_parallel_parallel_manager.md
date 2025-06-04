# Software Documentation Template for Subsystems - Parallel Manager

## 📘 parallel_manager.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/training/parallel/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Coordinates multiple self-play workers and orchestrates the overall parallel experience collection process.
* **Key Responsibilities:**
  - Spawn and monitor worker processes
  - Collect experiences from a shared queue
  - Trigger model synchronization events

### 2. Dependencies 🔗

* **Internal:** `WorkerCommunicator`, `ModelSynchronizer`, `SelfPlayWorker`
* **External:** `torch`, `multiprocessing`

---

## Notes for AI/Agent Developers 🧠

Use `ParallelManager` as the entry point for multi-process training setups.
