# Software Documentation Template for Subsystems - Adaptive Display Manager

## 📘 adaptive_display.py as of 2025-06-01

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `keisei/training/adaptive_display.py`
**Documentation Version:** `1.0`
**Date:** `2025-06-01`
**Responsible Author or Agent ID:** `Documentation Agent`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Decide which terminal user interface layout should be used based on terminal size and configuration. Helps `TrainingDisplay` select between compact and enhanced layouts.
* **Key Responsibilities:**
  - Inspect terminal width/height using Rich or `os.get_terminal_size`
  - Detect Unicode rendering capability
  - Choose layout variant (`"compact"` or `"enhanced"`)
* **Domain Context:**
  Terminal UI for monitoring training progress. Layout choice impacts how much information can be displayed.
* **High-Level Architecture / Interaction Summary:**
  `AdaptiveDisplayManager` is instantiated by `TrainingDisplay`. It queries terminal characteristics and consults `DisplayConfig` to return the preferred layout string.

---

### 2. Modules 📦

* **Module Name:** `adaptive_display.py`
  * **Purpose:** Utility to pick an appropriate TUI layout based on runtime environment.
  * **Design Patterns Used:** Simple strategy/heuristic pattern.
  * **Key Functions/Classes Provided:**
    - `TerminalInfo` dataclass
    - `AdaptiveDisplayManager` class
  * **Configuration Surface:**
    - Reads settings from `DisplayConfig` (e.g., `enable_enhanced_layout`)
  * **Dependencies:**
    * **Internal:** `keisei.config_schema.DisplayConfig`
    * **External:** `os`, `rich.console.Console`
  * **External API Contracts:** None – used internally by display manager.
  * **Side Effects / Lifecycle Considerations:** None beyond terminal size queries.
  * **Usage Examples:**
    ```python
    manager = AdaptiveDisplayManager(display_cfg)
    layout = manager.choose_layout(Console())
    ```

---

### 3. Classes 🏛️

* **Class Name:** `TerminalInfo`
  * **Defined In Module:** `adaptive_display.py`
  * **Purpose:** Simple container describing terminal width, height and Unicode capability.
  * **Design Role:** Lightweight data transfer object.
  * **Inheritance:** Extends `object` via `dataclass`.
  * **Key Attributes/Properties:**
    - `width: int` – Terminal column count
    - `height: int` – Terminal row count
    - `unicode_ok: bool` – Whether Unicode characters render properly
  * **Key Methods:** None (dataclass only)
  * **Interconnections:** Used exclusively by `AdaptiveDisplayManager`.
  * **Lifecycle & State:** Immutable once created.
  * **Threading/Concurrency:** N/A

* **Class Name:** `AdaptiveDisplayManager`
  * **Defined In Module:** `adaptive_display.py`
  * **Purpose:** Inspect console and configuration to select layout type.
  * **Design Role:** Decision helper for the UI subsystem.
  * **Inheritance:** `object`
  * **Key Attributes/Properties:**
    - `config: DisplayConfig` – Display related configuration options
  * **Key Methods:**
    - `_get_terminal_size(console)` – Internal helper to read terminal dimensions
    - `get_terminal_info(console)` – Return `TerminalInfo` with Unicode check
    - `choose_layout(console)` – Return layout string
  * **Interconnections:** Called by `TrainingDisplay` inside the `training.display` module.
  * **Lifecycle & State:** Stateless aside from stored config.
  * **Threading/Concurrency:** Not thread-safe but no mutable state.
  * **Usage Example:**
    ```python
    adaptive = AdaptiveDisplayManager(cfg.display)
    if adaptive.choose_layout(console) == "enhanced":
        layout = create_enhanced()
    ```

---

### 4. Functions/Methods ⚙️

#### `_get_terminal_size(console)`
* **Defined In:** `adaptive_display.py`
* **Belongs To:** `AdaptiveDisplayManager`
* **Purpose:** Obtain terminal dimensions using Rich console or fallback to `os.get_terminal_size`.
* **Parameters:** `console: Console`
* **Returns:** `Tuple[int, int]`
* **Raises/Exceptions:** Handles `OSError` when size cannot be determined, returning default (80,24).
* **Side Effects:** None

#### `get_terminal_info(console)`
* **Purpose:** Return a `TerminalInfo` describing terminal size and Unicode ability.
* **Parameters:** `console: Console`
* **Returns:** `TerminalInfo`

#### `choose_layout(console)`
* **Purpose:** Decide between `"compact"` and `"enhanced"` layout strings.
* **Parameters:** `console: Console`
* **Returns:** `str`
* **Algorithmic Note:** Uses width/height thresholds (`120x25`) and the `enable_enhanced_layout` flag.

---

### 5. Shared or Complex Data Structures 📊

* **`TerminalInfo`** – small dataclass used as return type from `get_terminal_info`.

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  training/display.py → AdaptiveDisplayManager
  ```
* **Cross-Folder Imports:** Uses configuration models from `keisei.config_schema`.
* **Data Flow Summary:**
  `TrainingDisplay` asks manager for layout → manager queries console → returns layout string → display creates layout accordingly.

---

### 7. Non-Functional Aspects 🛠️

* **Performance:** Negligible; simple attribute checks.
* **Security:** None; does not process user input.
* **Error Handling & Logging:** Gracefully handles terminal size query failures.
* **Scalability Concerns:** None.
* **Testing & Instrumentation:**
  * Tested in `tests/test_display_infrastructure.py` (`test_adaptive_layout_choice`).

---

### 8. Configuration & Environment ♻️

* **Environment Variables:** None.
* **CLI Interfaces / Entry Points:** None; used programmatically.
* **Config File Schema:** Parameters provided by `DisplayConfig`.

---

### 9. Glossary (Optional) 📖

Not applicable.

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:** None.
* **TODOs / Deferred Features:** Consider dynamic adjustment when terminal is resized during runtime.
* **Suggested Refactors:** None.

---

## Notes for AI/Agent Developers 🧠

Use `AdaptiveDisplayManager` to automatically select layouts that fit the user's terminal; keep configuration in sync with available space.
