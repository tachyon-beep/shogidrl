# Software Documentation Template for Subsystems - utils

## 📘 utils.py as of 2025-06-05

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/utils/`
**Documentation Version:** `1.0`
**Date:** `2025-06-05`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

utils.py: Contains PolicyOutputMapper and TrainingLogger.
### 2. Modules 📦

* **Dependencies:**
- from __future__ import annotations
- import datetime
- import json
- import logging
- import os
- import sys
- from abc import ABC, abstractmethod
- from typing import (
- import torch
- import yaml
- from pydantic import ValidationError
- from rich.console import Console
- from rich.text import Text
- from keisei.config_schema import AppConfig
- from keisei.shogi.shogi_core_definitions import (
- from keisei.utils.unified_logger import log_error_to_stderr
### 3. Classes 🏛️

- `BaseOpponent`: Abstract base class for game opponents.
- `PolicyOutputMapper`: Maps Shogi moves to/from policy network output indices.
- `TrainingLogger`: Handles logging of training progress to a file and optionally to stdout.
- `EvaluationLogger`: Handles logging of evaluation results to a file and optionally to stdout.
### 4. Functions/Methods ⚙️

- `_load_yaml_or_json`: No docstring
- `_merge_overrides`: No docstring
- `_map_flat_overrides`: No docstring
- `load_config`: Loads configuration from a YAML or JSON file and applies CLI overrides.
- `generate_run_name`: Generates a unique run name based on config and timestamp, or returns the provided run_name if set.
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
