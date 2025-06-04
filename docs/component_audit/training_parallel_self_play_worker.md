# Software Documentation Template for Subsystems - Self Play Worker

## 📘 self_play_worker.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/training/parallel/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Implements the worker process that runs self-play games and sends experience batches to the main process.
* **Key Responsibilities:**
  - Interact with the Shogi environment to generate training data
  - Maintain its own copy of the model for inference
  - Communicate results and receive updates via the communicator

### 2. Dependencies 🔗

* **Internal:** `keisei.shogi`, `keisei.core.experience_buffer`, `WorkerCommunicator`
* **External:** `torch`, `numpy`, `multiprocessing`

---

## Notes for AI/Agent Developers 🧠

Workers should be started and stopped through `ParallelManager` rather than directly instantiated in user code.
