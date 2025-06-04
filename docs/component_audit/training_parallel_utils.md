# Software Documentation Template for Subsystems - Parallel Utilities

## 📘 utils.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/training/parallel/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Contains helper functions for compressing and decompressing numpy arrays during inter-process communication.
* **Key Responsibilities:**
  - Provide gzip-based compression routines
  - Record compression ratios for profiling

### 2. Dependencies 🔗

* **External:** `gzip`, `numpy`
* **Internal:** none

---

## Notes for AI/Agent Developers 🧠

These utilities are used by both the communicator and the model synchronizer.
