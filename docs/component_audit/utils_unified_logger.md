# Software Documentation Template for Subsystems - Unified Logger

## 📘 unified_logger.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/utils/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Offers a consistent logging interface used across training and evaluation components. Provides colored console output via Rich and optional timestamping.
* **Key Responsibilities:**
  - Format messages with Rich to enhance readability
  - Send log output to stderr to avoid interfering with tqdm progress bars
  - Provide simple helper methods for info, warning, and error logs

### 2. Dependencies 🔗

* **External:** `rich`
* **Internal:** none

---

## Notes for AI/Agent Developers 🧠

Use this logger instead of plain `print` statements for uniform log styling.
