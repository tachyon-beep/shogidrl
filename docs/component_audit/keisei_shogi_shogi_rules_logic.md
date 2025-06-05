# Software Documentation Template for Subsystems - shogi_rules_logic

## 📘 shogi_rules_logic.py as of 2025-06-05

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/shogi/`
**Documentation Version:** `1.0`
**Date:** `2025-06-05`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

Core Shogi game rules, move generation, and validation logic.
### 2. Modules 📦

* **Dependencies:**
- from typing import TYPE_CHECKING, List, Optional, Set, Tuple  # Added Set
- from .shogi_core_definitions import (
### 3. Classes 🏛️

*None*
### 4. Functions/Methods ⚙️

- `find_king`: Finds the king of the specified color on the board.
- `is_in_check`: Checks if the king of 'player_color' is in check.
- `is_piece_type_sliding`: Returns True if the piece type is a sliding piece (Lance, Bishop, Rook or their promoted versions).
- `generate_piece_potential_moves`: Returns a list of (r_to, c_to) tuples for a piece, considering its
- `check_for_nifu`: Checks for two unpromoted pawns of the same color on the given file.
- `check_if_square_is_attacked`: Checks if the square (r_target, c_target) is attacked by any piece of attacker_color.
- `check_for_uchi_fu_zume`: Returns True if dropping a pawn at (drop_row, drop_col) by 'color'
- `is_king_in_check_after_simulated_move`: Checks if the king of 'player_color' is in check on the current board.
- `can_promote_specific_piece`: Checks if a piece *can* be promoted given its type and move.
- `must_promote_specific_piece`: Checks if a piece *must* promote when moving to r_to.
- `can_drop_specific_piece`: Checks if a specific piece_type can be legally dropped by 'color' at (r_to, c_to).
- `generate_all_legal_moves`: No docstring
- `check_for_sennichite`: Returns True if the current board state has occurred four times (Sennichite).
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
