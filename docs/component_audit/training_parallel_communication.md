# Software Documentation Template for Subsystems - Worker Communicator

## 📘 communication.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/training/parallel/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Implements queue-based communication between the main training process and worker processes. Handles experience collection messages and model update notifications.
* **Key Responsibilities:**
  - Provide non-blocking send/receive utilities with timeout handling
  - Support compression of large numpy arrays for efficiency

### 2. Dependencies 🔗

* **Internal:** `.utils` for compression helpers
* **External:** `multiprocessing`, `numpy`, `torch`

---

## Notes for AI/Agent Developers 🧠

This module is central to the parallel training workflow; monitor its queues and log outputs when debugging worker issues.
