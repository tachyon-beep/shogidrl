# Software Documentation Template for Subsystems - Elo Rating Registry

## 📘 elo_registry.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `keisei/evaluation/elo_registry.py`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `Documentation Agent`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Provide persistent storage and update mechanics for Elo ratings of trained agents. Used during evaluation to track relative strength between models.
* **Key Responsibilities:**
  - Load and save Elo rating data from a JSON file
  - Compute expected scores and update ratings from match results
  - Expose helper to retrieve a model's current rating
* **Domain Context:**
  Part of the evaluation subsystem measuring competitive performance of PPO agents through standard Elo rating calculations.
* **High-Level Architecture / Interaction Summary:**
  Simple dataclass wrapper around a JSON file. The Evaluator instantiates `EloRegistry` to update ratings after evaluation games.

---

### 2. Modules 📦

* **Module Name:** `elo_registry.py`
  * **Purpose:** Maintain a JSON-backed mapping of model identifiers to Elo ratings.
  * **Design Patterns Used:** Lightweight Repository pattern for persistence.
  * **Key Functions/Classes Provided:**
    - `EloRegistry` dataclass
  * **Configuration Surface:**
    - File path to registry JSON (`path` attribute)
    - Default rating and K-factor constants
  * **Dependencies:**
    * **Internal:** None
    * **External:** `json`, `pathlib.Path`, `dataclasses`
  * **External API Contracts:** None – internal use only by evaluation workflow.
  * **Side Effects / Lifecycle Considerations:** Reads and writes JSON file on disk via `load()` and `save()` methods.
  * **Usage Examples:**
    ```python
    registry = EloRegistry(Path("elo.json"))
    registry.update_ratings("agentA", "agentB", ["agent_win", "opponent_win"])
    registry.save()
    ```

---

### 3. Classes 🏛️

* **Class Name:** `EloRegistry`
  * **Defined In Module:** `elo_registry.py`
  * **Purpose:** Persistently track Elo ratings for agents across evaluation runs.
  * **Design Role:** Simple data repository with rating update logic.
  * **Inheritance:**
    * **Extends:** `object` via `dataclass`
    * **Subclasses:** None
  * **Key Attributes/Properties:**
    - `path: Path` – Filesystem location of the JSON registry
    - `default_rating: float` – Starting rating for unknown agents (1500)
    - `k_factor: float` – Elo K-factor applied to rating updates (32)
    - `ratings: Dict[str, float]` – Mapping of model_id to current rating
  * **Key Methods:**
    - `load()` – Read registry from file if it exists
    - `save()` – Persist current ratings to disk
    - `get_rating(model_id)` – Retrieve rating for identifier
    - `update_ratings(model_a_id, model_b_id, results)` – Apply match outcomes
  * **Interconnections:**
    * **Internal Calls:** Used by `keisei.evaluation.evaluate.Evaluator`
    * **External Systems:** Filesystem for persistence
  * **Lifecycle & State:** Automatically loads from disk on initialization via `__post_init__`; callers must invoke `save()` to persist updates.
  * **Threading/Concurrency:** Not thread-safe; external synchronization required if accessed concurrently.
  * **Usage Example:**
    ```python
    elo = EloRegistry(Path("elo.json"))
    elo.update_ratings("agentA", "agentB", ["agent_win", "opponent_win"])
    elo.save()
    ```

---

### 4. Functions/Methods ⚙️

#### `load()`
* **Defined In:** `elo_registry.py`
* **Belongs To:** `EloRegistry`
* **Purpose:** Load ratings from disk if the registry file exists.
* **Parameters:** None
* **Returns:** `None`
* **Raises/Exceptions:** Handles `json.JSONDecodeError` and `OSError` internally, defaulting to empty registry.
* **Side Effects:** Reads the JSON file.

#### `save()`
* **Defined In:** `elo_registry.py`
* **Belongs To:** `EloRegistry`
* **Purpose:** Write current ratings to disk.
* **Parameters:** None
* **Returns:** `None`
* **Raises/Exceptions:** Silently ignores `OSError` when writing fails.
* **Side Effects:** Writes JSON file to configured path.

#### `get_rating(model_id)`
* **Defined In:** `elo_registry.py`
* **Belongs To:** `EloRegistry`
* **Purpose:** Retrieve rating for given agent id, falling back to default rating.
* **Parameters:** `model_id: str`
* **Returns:** `float` rating value

#### `update_ratings(model_a_id, model_b_id, results)`
* **Defined In:** `elo_registry.py`
* **Belongs To:** `EloRegistry`
* **Purpose:** Update ratings for two agents based on a list of result strings (`"agent_win"`, `"opponent_win"`, or other for draw).
* **Parameters:**
  - `model_a_id: str` – Evaluated agent identifier
  - `model_b_id: str` – Opponent identifier
  - `results: List[str]` – Sequence of game results
* **Returns:** `None`
* **Algorithmic Note:** Uses standard Elo expected-score formula and K-factor.

---

### 5. Shared or Complex Data Structures 📊

None – registry stores a simple `Dict[str, float]` loaded from JSON.

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  evaluation/evaluate.py → EloRegistry
  ```
* **Cross-Folder Imports:** None other than `evaluate.py` usage.
* **Data Flow Summary:**
  `Evaluator` loads registry → evaluations run → results list passed to `update_ratings` → new ratings saved back to JSON file.

---

### 7. Non-Functional Aspects 🛠️

* **Performance:** Negligible; operations are simple dictionary manipulations and small file I/O.
* **Security:** Registry file path should be trusted; no sanitization beyond Python's `json` handling.
* **Error Handling & Logging:** Errors while loading/saving are silently ignored to avoid failing evaluation runs.
* **Scalability Concerns:** Suitable for a small number of agents; large registries may need different storage.
* **Testing & Instrumentation:**
  * Tests located at `tests/evaluation/test_elo_registry.py` ensure rating updates work as expected.

---

### 8. Configuration & Environment ♻️

* **Environment Variables:** None.
* **CLI Interfaces / Entry Points:** Not standalone; used by `evaluate.py` which passes path via config.
* **Config File Schema:** Path to Elo registry specified in `EvaluationConfig.elo_registry_path`.

---

### 9. Glossary (Optional) 📖

* **Elo Rating:** Numerical skill rating where 400 points corresponds to a 10x expected win rate difference.

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:** None.
* **TODOs / Deferred Features:**
  - Potential future support for decay or season reset logic.
* **Suggested Refactors:** None.

---

## Notes for AI/Agent Developers 🧠

Use this registry to track long-term agent performance. Always call `save()` after updating to persist results.
