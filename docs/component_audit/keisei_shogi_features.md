# Software Documentation Template for Subsystems - features

## 📘 features.py as of 2025-06-05

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/shogi/`
**Documentation Version:** `1.0`
**Date:** `2025-06-05`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

features.py: FeatureSpec registry and core46 observation builder for Keisei Shogi.
### 2. Modules 📦

* **Dependencies:**
- from typing import Callable, Dict
- import numpy as np
- from ..constants import MOVE_COUNT_NORMALIZATION_FACTOR
### 3. Classes 🏛️

- `FeatureSpec`: Describes a set of feature planes for Shogi observation tensors.
### 4. Functions/Methods ⚙️

- `register_feature`: No docstring
- `build_core46`: Build the standard 46-plane observation tensor for the given game state.
- `add_check_plane`: No docstring
- `add_repetition_plane`: No docstring
- `add_prom_zone_plane`: No docstring
- `add_last2ply_plane`: No docstring
- `add_hand_onehot_plane`: No docstring
- `build_core46_all`: No docstring
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
