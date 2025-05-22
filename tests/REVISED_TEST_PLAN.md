# Revised Test Remediation Plan

## Executive Summary

After analyzing the current test suite, I've identified that while many test files exist, the low coverage is primarily due to:

1. Redundancy and duplication across test files
2. Lack of proper mock pattern usage in most test files
3. Complete absence of tests for key files (especially shogi_game_io.py)

This revised plan addresses these issues with a clear prioritization.

## Current Status

- Overall test coverage: 10%
- `shogi_core_definitions.py`: 49% coverage
- `shogi_game.py`: 1% coverage (423 out of 423 statements untested)
- `shogi_game_io.py`: 3% coverage (149 out of 154 statements untested)

## Implementation Plan

### Phase 1: Game I/O Coverage (HIGH PRIORITY - COMPLETED)

I've created a comprehensive test file `test_shogi_game_io.py` that provides extensive test coverage for `shogi_game_io.py`, including:

- Neural network observation generation
- Text representation of game states
- KIF file export
- SFEN move parsing
- Helper functions

**Expected Outcome:** Increase coverage of `shogi_game_io.py` from 3% to at least 70%.

### Phase 2: Mock Pattern Integration (HIGH PRIORITY - IN PROGRESS)

To address the extremely low coverage of `shogi_game.py`, I've created `test_shogi_game_updated_with_mocks.py` to convert the existing tests to use the mock pattern. This enables:

- More reliable testing without PyTorch dependencies
- Better coverage metrics
- Increased consistency in the test suite

**Expected Outcome:** Increase coverage of `shogi_game.py` from 1% to at least 50%.

### Phase 3: Test Consolidation (MEDIUM PRIORITY)

I've created `TEST_MAPPING.md` to document the current test coverage and identify duplication. Next steps:

1. Consolidate the three different undo move test files
2. Consolidate observation tests between test_shogi_game.py and test_observation_constants.py
3. Remove redundant tests or mark them with meaningful names to indicate their uniqueness

**Expected Outcome:** Cleaner, more maintainable test suite with reduced duplication.

### Phase 4: Neural Network Integration Testing (MEDIUM PRIORITY)

Once the core game tests are consolidated:

1. Expand the observation plane constants test with different game states
2. Add tests for policy mask generation from legal moves
3. Test handling of terminal states in the neural network integration

**Expected Outcome:** Comprehensive testing of the neural network integration points.

### Phase 5: Parametrized Tests for Edge Cases (LOW PRIORITY)

Add parameterized tests for:

1. Middle game positions
2. Endgame positions
3. Special rule edge cases
4. Rare promotion scenarios
5. Complex capture sequences

**Expected Outcome:** Improved robustness through edge case testing.

## Execution Plan

1. First run the new tests to verify they pass and check coverage improvement:
   ```bash
   pytest tests/test_shogi_game_io.py -v
   pytest tests/test_shogi_game_updated_with_mocks.py -v
   ```

2. Generate a coverage report to measure progress:
   ```bash
   pytest --cov=keisei.shogi.shogi_game_io tests/test_shogi_game_io.py
   pytest --cov=keisei.shogi.shogi_game tests/test_shogi_game_updated_with_mocks.py
   ```

3. Address any failing tests or coverage gaps by expanding the test files

4. Begin consolidation of the redundant tests

5. Run the full test suite to ensure overall compatibility

## Test Coverage Targets

| Module                      | Current | After Phase 1 | After Phase 2 | Final Target |
|-----------------------------|---------|--------------|--------------|--------------|
| shogi/__init__.py           | 67%     | 67%          | 67%          | 90%          |
| shogi_core_definitions.py   | 49%     | 49%          | 60%          | 80%          |
| shogi_game.py               | 1%      | 1%           | 50%          | 70%          |
| shogi_game_io.py            | 3%      | 70%          | 70%          | 80%          |
| OVERALL                     | 10%     | 30%          | 50%          | 75%          |

## Best Practices for Mock Pattern

All test files should:

1. Import the mock utility: `from tests.mock_utilities import setup_pytorch_mock_environment`
2. Use the context manager for imports and test code:
   ```python
   def test_something():
       with setup_pytorch_mock_environment():
           # Import modules that depend on PyTorch
           from keisei.shogi.shogi_game import ShogiGame
           
           # Test code here
   ```
3. Add proper path handling for direct execution when needed

## Conclusion

This revised plan addresses the specific issues identified in the current test suite, with a clear focus on improving coverage of the most critical and undertested files. The implementation of `test_shogi_game_io.py` and `test_shogi_game_updated_with_mocks.py` will immediately improve coverage, while the consolidation phases will make the test suite more maintainable.
