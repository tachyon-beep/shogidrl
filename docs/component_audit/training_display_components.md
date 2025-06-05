# Software Documentation Template for Subsystems - Display Components

## 📘 display_components.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `keisei/training/display_components.py`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `Documentation Agent`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Provide reusable Rich display components for the training UI, including an ASCII board renderer and sparkline visualizer.
* **Key Responsibilities:**
  - Define a protocol for components used by `TrainingDisplay`
  - Render the current Shogi board state with optional move list
  - Generate small sparklines for metric trends
* **Domain Context:**
  Terminal-based monitoring of RL training; visuals aid understanding of agent behaviour and performance.
* **High-Level Architecture / Interaction Summary:**
  `TrainingDisplay` composes these components into panels within the Rich layout. They rely on `rich` primitives and internal policy mappers for move formatting.

---

### 2. Modules 📦

* **Module Name:** `display_components.py`
  * **Purpose:** House auxiliary UI widgets for terminal display.
  * **Design Patterns Used:** Protocol-based interface for renderables.
  * **Key Functions/Classes Provided:**
    - `DisplayComponent` protocol
    - `ShogiBoard` class
    - `Sparkline` class
  * **Configuration Surface:**
    - `ShogiBoard` constructor options (`use_unicode`, `show_moves`, `max_moves`)
    - `Sparkline` width
  * **Dependencies:**
    * **Internal:** `keisei.utils.unified_logger.log_error_to_stderr`
    * **External:** `rich` library components
  * **External API Contracts:** None; objects return Rich `RenderableType`.
  * **Side Effects / Lifecycle Considerations:** None aside from rendering; `ShogiBoard` logs unexpected errors during move formatting.
  * **Usage Examples:**
    ```python
    board = ShogiBoard(show_moves=True)
    panel = board.render(game_state, move_history, policy_mapper)
    spark = Sparkline(width=10)
    panel2 = spark.render(values=[0.1, 0.2, 0.3])
    ```

---

### 3. Classes 🏛️

* **Class Name:** `ShogiBoard`
  * **Defined In Module:** `display_components.py`
  * **Purpose:** Render an ASCII representation of the Shogi board with optional recent move list.
  * **Design Role:** View component in the UI architecture.
  * **Inheritance:** `object`
  * **Key Attributes/Properties:**
    - `use_unicode: bool` – Whether to use Japanese piece symbols
    - `show_moves: bool` – Whether to append recent move history panel
    - `max_moves: int` – How many recent moves to display
  * **Key Methods:**
    - `_piece_to_symbol(piece)` – Map board piece object to text symbol
    - `_generate_ascii_board(board_state)` – Create multi-line ASCII board
    - `_move_to_usi(move_tuple, policy_mapper)` – Convert move tuple to USI string
    - `render(board_state, move_history, policy_mapper)` – Return Rich panel/group
  * **Interconnections:**
    * **Internal Calls:** Uses `log_error_to_stderr` on unexpected formatting errors.
  * **Lifecycle & State:** Stateless aside from constructor flags.
  * **Threading/Concurrency:** Not thread-safe but used in single-threaded display loop.
  * **Usage Example:**
    ```python
    board = ShogiBoard(use_unicode=True, show_moves=True)
    panel = board.render(game, history, mapper)
    ```

* **Class Name:** `Sparkline`
  * **Defined In Module:** `display_components.py`
  * **Purpose:** Generate compact sparkline strings to visualise metric trends.
  * **Design Role:** Small visual component.
  * **Inheritance:** `object`
  * **Key Attributes/Properties:**
    - `width: int` – Number of characters in the sparkline
    - `chars: str` – Unicode block characters used for visualisation
  * **Key Methods:**
    - `generate(values)` – Produce sparkline string from numeric sequence
    - `render(values, title)` – Return Rich panel with sparkline
  * **Interconnections:** None beyond `rich` rendering.
  * **Lifecycle & State:** Stateless except for configured width.
  * **Threading/Concurrency:** N/A
  * **Usage Example:**
    ```python
    spark = Sparkline(width=5)
    panel = spark.render([1,2,3,4,5], "Policy Loss")
    ```

---

### 4. Functions/Methods ⚙️

Focus is on methods outlined above; there are no standalone functions.

---

### 5. Shared or Complex Data Structures 📊

None.

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  training/display.py → ShogiBoard, Sparkline
  ```
* **Data Flow Summary:** Game state and metric history are passed from `TrainingDisplay`/`Trainer` into these components, which produce Rich renderables for display.

---

### 7. Non-Functional Aspects 🛠️

* **Performance:** Lightweight rendering; minimal overhead.
* **Security:** No external input other than sanitized board/move data.
* **Error Handling & Logging:** Uses `log_error_to_stderr` on unexpected failures in move formatting.
* **Scalability Concerns:** None.
* **Testing & Instrumentation:**
  * See tests in `tests/test_display_infrastructure.py` (`test_shogi_board_basic_render`, `test_sparkline_generation`).

---

### 8. Configuration & Environment ♻️

* **Environment Variables:** None.
* **CLI Interfaces / Entry Points:** None.
* **Config File Schema:** Controlled indirectly via `DisplayConfig` passed from application config.

---

### 9. Glossary (Optional) 📖

* **USI:** Universal Shogi Interface notation used for representing moves.

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:** Unicode rendering depends on terminal font support.
* **TODOs / Deferred Features:** Potential colorised board or piece highlighting.
* **Suggested Refactors:** None.

---

## Notes for AI/Agent Developers 🧠

These components are intended for console output only and rely on Rich for rendering. Keep their usage lightweight within the training loop to avoid slowing down performance.
