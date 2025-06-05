# Software Documentation Template for Subsystems - utils

## 📘 utils.py as of 2025-06-05

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/training/`
**Documentation Version:** `1.0`
**Date:** `2025-06-05`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

training/utils.py: Helper functions for setup and configuration in the Shogi RL trainer.
### 2. Modules 📦

* **Dependencies:**
- import glob
- import json
- import os
- import pickle
- import random
- import sys
- from typing import Optional
- import numpy as np
- import torch
- import wandb
- from keisei.config_schema import AppConfig
- from keisei.utils.unified_logger import log_error_to_stderr, log_info_to_stderr
### 3. Classes 🏛️

*None*
### 4. Functions/Methods ⚙️

- `_validate_checkpoint`: Validate checkpoint file integrity by attempting to load it.
- `find_latest_checkpoint`: No docstring
- `serialize_config`: Serialize AppConfig to JSON string using Pydantic's built-in serialization.
- `setup_directories`: No docstring
- `setup_seeding`: No docstring
- `setup_wandb`: No docstring
- `apply_wandb_sweep_config`: Apply W&B sweep configuration to override parameters.
- `build_cli_overrides`: Build configuration overrides from command line arguments.
### 5. Shared or Complex Data Structures 📊

* **Structure Name:** `[e.g., TrainingConfigDict]`

  * **Type:** `[e.g., Dict[str, Any]]`
  * **Purpose:** What the structure is meant to hold.
  * **Format:** JSON schema, class, pydantic model, dataclass, etc.
  * **Fields:**

    * `learning_rate: float – Training LR (default 0.001)`
    * `env_name: str – Environment name, must be Gym-compliant`
  * **Validation Constraints:** (Optional)
  * **Used In:** List of modules/classes/functions using this.

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  Visual or list representation of module relationships:

  ```
  training_orchestrator.py
    ├── uses → model_factory.py
    ├── uses → replay_buffer.py
  model_factory.py
    └── uses → architectures/
  ```

* **Cross-Folder Imports:**

  * `[From ../agents]: imports BaseAgent, PPOAgent`
  * `[To /shogi/]: calls validate_board_state()`

* **Data Flow Summary:**

  * Describe the flow of data (esp. structured or recurring payloads).
  * Clarify transformation stages (e.g., raw → validated → batched → logged)

---

### 7. Non-Functional Aspects 🛠️

* **Performance:**
  Targets or expectations (latency, throughput, memory usage)

* **Security:**
  Input sanitisation, secrets handling, any interfaces that cross trust boundaries

* **Error Handling & Logging:**
  Global error handling strategy. Log levels used.

* **Scalability Concerns:**
  Horizontal scaling? Worker pool strategy? Resource contention?

* **Testing & Instrumentation:**

  * Test harness location: `[e.g. tests/test_training_orchestrator.py]`
  * Fakes, mocks, or stubs used
  * Metrics or tracing (e.g., OpenTelemetry, Prometheus)

---

### 8. Configuration & Environment ♻️

* **Environment Variables:**

  * `TRAINING_CONFIG_PATH: str – Path to default config`
  * `GPU_ENABLED: bool – Whether GPU usage is enabled`

* **CLI Interfaces / Entry Points (if any):**

  * `python -m training_orchestrator --config configs/dev.yaml`

* **Config File Schema:**

  * Reference if JSON/YAML schema is defined externally.
  * Inline spec if useful.

---

### 9. Glossary (Optional) 📖

* **\[Term]:** `[Definition]`
  *Include terms specific to the business logic, framework, or internal slang (e.g., “rollout,” “shard,” “trace span”).*

---

### 10. Known Issues, TODOs, Future Work 🧭

*No TODO/FIXME comments found*
