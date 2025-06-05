# Software Documentation Template for Subsystems - shogi_game

## 📘 shogi_game.py as of 2025-06-05

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/shogi/`
**Documentation Version:** `1.0`
**Date:** `2025-06-05`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

shogi_game.py: Main ShogiGame class for DRL Shogi Client.
### 2. Modules 📦

* **Dependencies:**
- import copy  # Added for __deepcopy__
- import logging
- import re  # Added for SFEN parsing
- from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
- import numpy as np
- from . import shogi_game_io, shogi_move_execution, shogi_rules_logic
- from .shogi_core_definitions import BASE_TO_PROMOTED_TYPE  # For SFEN deserialization
- from .shogi_core_definitions import PIECE_TYPE_TO_HAND_TYPE  # Used in add_to_hand
- from .shogi_core_definitions import PROMOTED_TYPES_SET  # For SFEN serialization
- from .shogi_core_definitions import SYMBOL_TO_PIECE_TYPE  # Added for SFEN parsing
- from .shogi_core_definitions import MoveTuple  # Already imported above
- from .shogi_core_definitions import (
### 3. Classes 🏛️

- `ShogiGame`: Represents the Shogi game state, board, and operations.
### 4. Functions/Methods ⚙️

*None*
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
