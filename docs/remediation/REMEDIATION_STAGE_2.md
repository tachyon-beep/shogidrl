### **Executable Plan: Critical Fixes & Technical Debt Reduction**

#### **Task 2.1: Resolve Non-Functional Configuration Override**

- ✅ **COMPLETED**
    - The `--override` CLI flag is now fully functional and tested.
    - The `apply_config_overrides` function supports nested keys and is covered by unit and end-to-end tests.

#### **Task 2.2: Correct Configuration Inconsistencies**

- ✅ **COMPLETED**
    - `NUM_ACTIONS_TOTAL` is set to 13,527 in `config.py` and is the single source of truth.
    - All hardcoded values in the codebase (including `evaluate.py` and legacy scripts) have been replaced with config references.
    - The test suite for `PolicyOutputMapper` passes, confirming correctness.
    - Code search confirms no stray hardcoded values remain.

#### **Task 2.3: Implement Deterministic Environment Seeding**

- ✅ **COMPLETED**
    - Added a `seed()` method to `ShogiGame` (no-op for now).
    - Called `game.seed(cfg.SEED)` after instantiating `ShogiGame` in the training setup (`trainer.py`).
    - (Optional) Add a test or manual check for reproducibility.

#### **Task 2.4: Prune Unused Dependencies**

- ✅ **COMPLETED**
    - Removed `networkx`, `sympy`, `GitPython`, `tqdm`, and `sentry-sdk` from `requirements.txt`.
    - Recreated and tested the environment to ensure nothing breaks.
    - Ran the full test suite to confirm no missing dependencies (all functional tests pass; only lint warnings remain).

---

**Summary Table:**

| Task                                      | Status     |
|-------------------------------------------|------------|
| 2.1 Config override                      | ✅ Complete|
| 2.2 Config consistency                   | ✅ Complete|
| 2.3 Deterministic seeding                | ✅ Complete|
| 2.4 Prune dependencies                   | ✅ Complete|

---

**Next Steps:**  
- Proceed to Stage 3: Modernisation & Automation.