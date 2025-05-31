# Test Remediation Plan

**Project:** Keisei Shogi AI  
**Date:** June 1, 2025  
**Based on:** TEST_AUDIT.md findings  
**Status:** Implementation Ready

## Executive Summary

This document outlines a systematic approach to address the critical issues identified in the test suite audit. The plan is organized by priority and provides specific implementation steps for each issue.

## Priority Classification

- **Critical (P0)**: Issues that provide false coverage or block CI/CD
- **High (P1)**: Issues that impact test reliability and portability  
- **Medium (P2)**: Issues that affect maintainability and best practices
- **Low (P3)**: Minor improvements and style issues

---

## Phase 1: Critical Issues (P0) - COMPLETED ✅

All Phase 1 critical issues have been successfully resolved.

### 1.1 Fix Placeholder/Empty Tests (P0) - ✅ COMPLETED
**File:** `tests/test_model_manager_init.py`  
**Issue:** All test methods contained only `pass` statements, providing false positive coverage.

#### Implementation Steps - COMPLETED:
1. ✅ **Analyzed ModelManager API** - Reviewed `keisei/training/model_manager.py` to understand expected behavior
2. ✅ **Implemented test bodies** for all placeholder tests:
   - `test_initialization_success` - Tests basic ModelManager creation and initialization
   - `test_initialization_with_args_override` - Tests command-line argument overrides  
   - `test_mixed_precision_cuda_enabled` - Tests CUDA mixed precision setup
   - `test_mixed_precision_cpu_warning` - Tests CPU mixed precision warnings
   - `test_get_model_info` - Tests model information retrieval
   - `test_model_creation_and_agent_instantiation` - Tests model creation and device placement

#### Acceptance Criteria - ALL MET:
- ✅ All test methods have meaningful assertions
- ✅ Tests verify actual ModelManager behavior
- ✅ Mocks are properly configured for isolated testing
- ✅ Tests pass with valid assertions

### 1.2 Address Skipped Critical Game Logic Tests (P0) - ✅ COMPLETED
**File:** `tests/test_shogi_game_core_logic.py`  
**Issue:** Critical game termination and move legality tests were commented out/skipped.

#### Implementation Steps - COMPLETED:
1. ✅ **Investigated Skip Reasons**:
   - Found commented skip decorators for pinned piece tests
   - Reviewed "SFEN/outcome needs review" issues  
   - Analyzed "Stalemate logic might be affected" concerns
2. ✅ **Validated Underlying Logic**:
   - All pinned piece move generation works correctly
   - SFEN parsing/generation is functioning properly
   - Stalemate detection logic is working as expected
3. ✅ **Confirmed Tests Active and Passing**:
   - All previously skipped tests are now active
   - All game termination tests pass successfully
   - Pinned piece move legality tests pass

#### Acceptance Criteria - ALL MET:
- ✅ All previously skipped game logic tests are active
- ✅ Game termination conditions are properly tested  
- ✅ Pinned piece move legality is verified
- ✅ SFEN-related test cases work correctly

### 1.3 Fix Incomplete Fixture (P0) - ✅ COMPLETED
**File:** `tests/conftest.py`  
**Issue:** `sample_board_state` fixture returned `None`.

#### Implementation Steps - COMPLETED:
1. ✅ **Analyzed Fixture Usage** - Found all references to `sample_board_state`
2. ✅ **Removed Unused Fixture** - Fixture was not used by any tests, so removed entirely

#### Acceptance Criteria - ALL MET:
- ✅ Unused fixture removed, no tests affected

### 1.4 Remove Duplicate Test Files (P0) - ✅ COMPLETED
**Files:** `test_shogi_rules_logic.py` and `test_shogi_rules_and_validation.py`  
**Issue:** Identical files causing redundant test execution.

#### Implementation Steps - COMPLETED:
1. ✅ **Verified Identity** - Confirmed files were truly identical
2. ✅ **Chose Canonical File** - Kept `test_shogi_rules_and_validation.py`
3. ✅ **Removed Duplicate** - Deleted `test_shogi_rules_logic.py`
4. ✅ **Validated References** - No imports or references affected

#### Acceptance Criteria - ALL MET:
- ✅ Only one copy of the test file exists
- ✅ No broken imports or references  
- ✅ Test execution time reduced

---

## Phase 2: High Priority Issues (P1) - IN PROGRESS
- [ ] No tests fail due to `None` fixture values

#### Estimated Effort: 1-2 hours

### 1.4 Remove Duplicate Test Files (P0)
**Files:** `test_shogi_rules_logic.py` and `test_shogi_rules_and_validation.py`  
**Issue:** Identical files causing redundant test execution.

#### Implementation Steps:
1. **Verify Identity** - Confirm files are truly identical (already confirmed)
2. **Choose Canonical File** - Keep `test_shogi_rules_and_validation.py`
3. **Remove Duplicate** - Delete `test_shogi_rules_logic.py`
4. **Update References** - Check for any imports or references to the deleted file

#### Acceptance Criteria:
- [ ] Only one copy of the test file exists
- [ ] No broken imports or references
- [ ] Test execution time reduced

#### Estimated Effort: 30 minutes

---

## Phase 2: High Priority Issues (P1) - Portability & Reliability

### 2.1 Eliminate Hardcoded Absolute Paths (P1)
**Files:** `test_dependencies.py`, `test_remediation_integration.py`  
**Issue:** Hardcoded paths like `/home/john/keisei/...` break portability.

#### Implementation Steps:
1. **Identify All Hardcoded Paths**:
   - `/home/john/keisei/pyproject.toml`
   - `/home/john/keisei/docs/...`
   - Any other absolute path references
2. **Implement Dynamic Path Resolution**:
   ```python
   from pathlib import Path
   
   # Replace hardcoded paths with:
   project_root = Path(__file__).resolve().parents[1]  # Adjust as needed
   pyproject_path = project_root / "pyproject.toml"
   ```
3. **Create Path Utilities** if needed for complex path calculations
4. **Validate on Different Systems** - Test path resolution works correctly

#### Acceptance Criteria:
- [ ] No hardcoded absolute paths in test files
- [ ] Tests work on different machines/containers
- [ ] Path resolution is reliable and maintainable

#### Estimated Effort: 2-3 hours

### 2.2 Fix Fragile Configuration File Parsing (P1)
**File:** `test_dependencies.py`  
**Issue:** String-based TOML parsing instead of proper parser.

#### Implementation Steps:
1. **Replace String Searches** with proper TOML parsing:
   ```python
   import tomllib  # Python 3.11+, or use 'toml' library
   
   # Replace string searches like:
   # assert "[project]" in content
   # With:
   with open(pyproject_path, "rb") as f:
       data = tomllib.load(f)
   assert "project" in data
   ```
2. **Update All TOML-Related Assertions**
3. **Add Proper Error Handling** for malformed TOML

#### Acceptance Criteria:
- [ ] All TOML parsing uses proper parser
- [ ] Tests are resilient to formatting changes
- [ ] Clear error messages for malformed files

#### Estimated Effort: 2-3 hours

### 2.3 Fix Flawed Checkpoint Path Logic (P1) ✅ **COMPLETED**
**File:** `test_train.py`  
**Test:** `test_train_resume_autodetect`  
**Issue:** Checkpoint save location doesn't match expected resume logic.

#### ✅ Completed Implementation:
1. **✅ Analyzed ModelManager checkpoint search logic** - Confirmed parent directory search works correctly
2. **✅ Improved test robustness**:
   - Added explicit docstring explaining parent directory search functionality
   - Made assertions more specific about expected checkpoint copy behavior
   - Added verification that training progresses from resumed timestep
   - Added explicit timestep parameter to checkpoint save
3. **✅ Enhanced verification**:
   - Test now verifies checkpoint is copied from parent to run directory
   - Test verifies exact resume log message matches copied checkpoint path
   - Test verifies training progression from resumed timestep

#### ✅ Results:
- **Test clarity improved**: Now explicitly documents the parent directory search functionality being tested
- **Assertions strengthened**: More specific verification of correct checkpoint resume behavior
- **Robustness enhanced**: Test will fail if wrong checkpoint is resumed or training doesn't progress properly
- **All tests pass**: No regressions introduced

---

## Phase 3: Medium Priority Issues (P2) - Best Practices & Maintainability

### 3.1 Separate Unit and Integration Tests (P2)
**Files:** Multiple test files mixing unit and integration tests  
**Issue:** Unclear distinction between test types affects maintainability.

#### Implementation Steps:
1. **Audit All Test Files** and classify tests as unit vs integration
2. **Add Pytest Markers**:
   ```python
   @pytest.mark.integration
   def test_full_system_integration():
       # Integration test
   
   @pytest.mark.unit  # Optional, can be default
   def test_isolated_component():
       # Unit test
   ```
3. **Update pytest.ini Configuration**:
   ```ini
   [tool:pytest]
   markers =
       unit: Unit tests (fast, isolated)
       integration: Integration tests (slower, multiple components)
       slow: Slow tests
   ```
4. **Document Test Categories** in testing documentation

#### Acceptance Criteria:
- [ ] All tests properly marked
- [ ] Can run unit tests separately from integration tests
- [ ] CI pipeline can execute different test categories

#### Estimated Effort: 3-4 hours

### 3.2 Fix Flaky Assertions - Magic Numbers (P2)
**File:** `test_dependencies.py`  
**Issue:** Arbitrary threshold `assert dep_issues <= 15`.

#### Implementation Steps:
1. **Analyze Dependency Issues**:
   - Understand what constitutes a "dependency issue"
   - Determine if issues can be eliminated
2. **Configure Ignorable Issues** or fix underlying problems
3. **Replace Magic Number** with:
   - Expected exact count (ideally 0)
   - Configuration-based threshold
   - Well-documented reasoning for any threshold

#### Acceptance Criteria:
- [ ] No arbitrary magic numbers in assertions
- [ ] Dependency checks are deterministic
- [ ] Clear documentation for any thresholds used

#### Estimated Effort: 2-3 hours

### 3.3 Investigate Manual State Adjustment (P2)
**File:** `test_shogi_game_core_logic.py`  
**Test:** `test_undo_move_multiple_moves`  
**Issue:** Manual `game.set_piece(5, 1, None)` suggests undo logic problem.

#### Implementation Steps:
1. **Analyze the Test Scenario**:
   - Understand the complex multi-step scenario
   - Identify why manual adjustment is needed
2. **Investigate Undo Logic**:
   - Verify `undo_move` is perfect inverse of `make_move`
   - Check state restoration accuracy
3. **Fix Root Cause**:
   - Either fix undo logic or improve test setup
   - Remove manual workaround if possible

#### Acceptance Criteria:
- [ ] Undo/redo logic works correctly without manual adjustments
- [ ] Test accurately represents expected behavior
- [ ] No workarounds in test logic

#### Estimated Effort: 3-4 hours

### 3.4 Improve Warning/Log Verification (P2)
**File:** `test_experience_buffer.py`  
**Issue:** Tests imply warnings are printed but don't verify emission.

#### Implementation Steps:
1. **Add Explicit Warning Checks**:
   ```python
   import pytest
   
   def test_warning_emission():
       with pytest.warns(UserWarning, match="expected warning pattern"):
           # Code that should emit warning
   ```
2. **Use Appropriate Capture Methods**:
   - `pytest.warns()` for Python warnings
   - `caplog` for logging module warnings
   - `capsys`/`capfd` for stdout/stderr

#### Acceptance Criteria:
- [ ] All warning tests explicitly verify warning emission
- [ ] Proper warning capture mechanisms used
- [ ] Clear assertion failure messages

#### Estimated Effort: 1-2 hours

---

## Phase 4: Low Priority Issues (P3) - Code Quality & Style

### 4.1 Standardize W&B Patching (P3)
**Issue:** Inconsistent approach to mocking wandb across tests.

#### Implementation Steps:
1. **Audit W&B Usage** across all test files
2. **Create Standard Approach**:
   - Consistent patching strategy
   - Central fixture for W&B mocking
3. **Apply Consistently** across all tests

#### Estimated Effort: 2-3 hours

### 4.2 Improve Test Parameterization (P3)
**Issue:** Opportunities to reduce boilerplate with better parameterization.

#### Implementation Steps:
1. **Identify Repetitive Test Patterns**
2. **Apply `@pytest.mark.parametrize`** where beneficial
3. **Balance Clarity vs. Conciseness**

#### Estimated Effort: 2-3 hours

### 4.3 Style and Best Practice Improvements (P3)
**Various small improvements**:
- Use `isinstance()` for type checking
- Add `@pytest.mark.slow` markers
- Avoid broad exception catching
- Import constants from canonical sources
- Minimize protected member access

#### Estimated Effort: 2-3 hours

---

## Implementation Timeline

### Week 1: Critical Issues (P0)
- Day 1-2: Fix placeholder tests in `test_model_manager_init.py`
- Day 3-4: Address skipped game logic tests
- Day 5: Fix incomplete fixture and remove duplicate files

### Week 2: High Priority Issues (P1)  
- Day 1-2: Eliminate hardcoded paths
- Day 3: Fix fragile TOML parsing
- Day 4-5: Fix checkpoint path logic and investigate state adjustment

### Week 3: Medium Priority Issues (P2)
- Day 1-2: Separate unit and integration tests
- Day 3: Fix flaky assertions
- Day 4-5: Improve warning verification

### Week 4: Low Priority Issues (P3)
- Day 1-5: Code quality improvements and standardization

## Risk Mitigation

### High Risk Items:
1. **Game Logic Changes**: Fixing skipped tests might reveal actual bugs in game logic
   - **Mitigation**: Thorough testing and validation against known game positions
2. **Breaking Changes**: Path refactoring might break other parts of the system
   - **Mitigation**: Comprehensive testing after each change

### Dependencies:
- Some issues may require understanding of domain-specific Shogi rules
- ModelManager fixes require understanding of the training system architecture

## Success Criteria

### Phase 1 (Critical) Success:
- [ ] All placeholder tests have meaningful implementations
- [ ] No skipped critical game logic tests
- [ ] No false positive test coverage
- [ ] Duplicate files removed

### Phase 2 (High Priority) Success:
- [ ] Tests run successfully on any development machine
- [ ] No hardcoded paths in test suite
- [ ] Robust configuration file parsing
- [ ] Accurate checkpoint resume testing

### Phase 3 (Medium Priority) Success:
- [ ] Clear separation between unit and integration tests
- [ ] No arbitrary magic numbers in assertions
- [ ] Proper warning verification

### Overall Success:
- [ ] Test suite provides accurate coverage metrics
- [ ] Tests are portable across different environments
- [ ] CI/CD pipeline is reliable and maintainable
- [ ] Test execution time is optimized
- [ ] Code quality meets project standards

## Validation Plan

After each phase:
1. **Run Full Test Suite** - Ensure no regressions
2. **Check Coverage Reports** - Verify coverage accuracy
3. **Test Portability** - Run tests in clean environment
4. **Performance Check** - Ensure no significant slowdown
5. **CI/CD Validation** - Verify pipeline still works

## Notes and Considerations

- This plan assumes the core application logic is correct and issues are primarily in the test suite
- Some fixes may reveal actual bugs in the application code, which should be addressed separately
- The plan prioritizes test reliability and accuracy over extensive feature additions
- Regular communication with the development team is essential during implementation

---

**Document Owner:** Test Remediation Team  
**Next Review:** After Phase 1 completion  
**Related Documents:** TEST_AUDIT.md, CI_CD.md
