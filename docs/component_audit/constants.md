# Software Documentation Template for Subsystems - Project Constants

## 📘 constants.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Centralizes numerical constants and static values used throughout the application. By defining them in one place the codebase avoids magic numbers and maintains consistency between modules.
* **Key Responsibilities:**
  - Provide board and action space constants for the Shogi environment
  - Define observation channel indices and normalization factors
  - Serve as a shared reference for other modules

### 2. Dependencies 🔗

None

---

## Notes for AI/Agent Developers 🧠

Update these constants carefully as downstream modules may rely on them for tensor shapes and normalization logic.
