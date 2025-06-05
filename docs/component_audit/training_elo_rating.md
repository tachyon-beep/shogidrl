# Software Documentation Template for Subsystems - Training Elo Rating System

## 📘 elo_rating.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `keisei/training/elo_rating.py`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `Documentation Agent`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Track Elo rating changes between the agent playing black and white during self-play training sessions.
* **Key Responsibilities:**
  - Maintain separate ratings for black and white sides
  - Update ratings after each game using the Elo formula
  - Provide a simple assessment of rating imbalance
* **Domain Context:**
  Used in training to monitor potential bias between player colors and overall progress.
* **High-Level Architecture / Interaction Summary:**
  Independent helper class that can be instantiated by training code or metrics manager to accumulate ratings over time.

---

### 2. Modules 📦

* **Module Name:** `elo_rating.py`
  * **Purpose:** Utility implementing a two-player Elo system.
  * **Design Patterns Used:** Minimal stateful object with statistical update.
  * **Key Functions/Classes Provided:** `EloRatingSystem`
  * **Configuration Surface:** Initial rating and K-factor parameters.
  * **Dependencies:**
    * **Internal:** `keisei.shogi.shogi_core_definitions.Color`
    * **External:** Python standard library only
  * **External API Contracts:** None.
  * **Side Effects / Lifecycle Considerations:** Maintains history of rating updates.
  * **Usage Examples:**
    ```python
    elo = EloRatingSystem()
    elo.update_ratings(Color.BLACK)
    print(elo.get_strength_assessment())
    ```

---

### 3. Classes 🏛️

* **Class Name:** `EloRatingSystem`
  * **Defined In Module:** `elo_rating.py`
  * **Purpose:** Calculate and store Elo ratings for each player colour.
  * **Design Role:** Lightweight statistics tracker.
  * **Inheritance:** `object`
  * **Key Attributes/Properties:**
    - `initial_rating: float` – Starting rating for both sides
    - `k_factor: float` – K-factor for rating adjustments
    - `black_rating: float` – Current rating for black
    - `white_rating: float` – Current rating for white
    - `rating_history: List[Dict[str, float]]` – Log of ratings after each update
  * **Key Methods:**
    - `_expected_score(rating_a, rating_b)` – Helper for Elo expectation formula
    - `update_ratings(winner_color)` – Apply update based on winner
    - `get_strength_assessment()` – Return textual comparison between sides
  * **Interconnections:** Consumed by `MetricsManager` tests in `tests/test_display_infrastructure.py`.
  * **Lifecycle & State:** Mutable ratings updated per game.
  * **Threading/Concurrency:** Not thread-safe.
  * **Usage Example:**
    ```python
    elo = EloRatingSystem(initial_rating=1500)
    result = elo.update_ratings(Color.WHITE)
    ```

---

### 4. Functions/Methods ⚙️

#### `update_ratings(winner_color)`
* **Defined In:** `elo_rating.py`
* **Belongs To:** `EloRatingSystem`
* **Purpose:** Update internal ratings based on winning colour or draw (`None`).
* **Parameters:** `winner_color: Optional[Color]`
* **Returns:** `Dict[str, float]` summary of new ratings
* **Algorithmic Note:** Standard Elo calculation with symmetrical K-factor.

#### `get_strength_assessment()`
* **Purpose:** Provide a qualitative description of rating difference.
* **Returns:** `str`

---

### 5. Shared or Complex Data Structures 📊

* **`rating_history`** – list of dictionaries with keys `black_rating`, `white_rating`, `difference`.

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):** None
* **Data Flow Summary:** Training loop records winner after each game → `update_ratings` is called → ratings stored and optionally displayed.

---

### 7. Non-Functional Aspects 🛠️

* **Performance:** Minimal computations per game.
* **Security:** Not applicable.
* **Error Handling & Logging:** None; assumes valid Color input.
* **Scalability Concerns:** None.
* **Testing & Instrumentation:**
  * Covered by `tests/test_display_infrastructure.py::test_elo_rating_updates`.

---

### 8. Configuration & Environment ♻️

* **Environment Variables:** None.
* **CLI Interfaces / Entry Points:** None.
* **Config File Schema:** Parameter defaults may be overridden when instantiating the class.

---

### 9. Glossary (Optional) 📖

* **K-factor:** Parameter determining Elo rating sensitivity to new results.

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:** None.
* **TODOs / Deferred Features:** Option to decay ratings over time.
* **Suggested Refactors:** None.

---

## Notes for AI/Agent Developers 🧠

Use this utility to monitor colour advantage during self-play. Ratings are not persisted automatically.
