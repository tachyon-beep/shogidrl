# DRL Shogi Client: Test Policy

## 1. Test-Driven Development (TDD)
- Every new function and class must have a corresponding unit test written immediately after implementation.
- No code is considered complete until it is covered by tests and passes linting.

## 2. Test Organization
- All tests are placed in the `tests/` directory, mirroring the structure of the `keisei/` codebase.
- Shared fixtures, mocks, and test utilities are placed only in `tests/conftest.py`.
- No test scaffolding or fixtures should be duplicated across test files.

## 3. Test Coverage
- All public methods and critical private methods must be tested.
- Tests must cover normal cases, edge cases, and error handling.
- Parameterized tests should be used for repeated logic.
- Mock external dependencies where appropriate.

## 4. Test Naming and Documentation
- Use descriptive test function names (e.g., `test_make_move_valid`, `test_get_legal_moves_nifu`).
- Each test should have a docstring explaining its purpose.

## 5. Test Isolation and Independence
- Tests must be independent and not rely on the state of other tests.
- Use fixtures to set up and tear down state as needed.

## 6. Linting and Formatting
- Run linting (flake8, black) after every implementation and test addition.
- Resolve all linting errors before proceeding.

## 7. Test Evaluation and Maintenance
- Read all relevant code before making a decision on the best way to proceed with a test or fix.
- Do not change working live code to resolve a failing test; instead, evaluate if the test is testing meaningful functionality and follows best practice.
- Remove or refactor tests that are not meaningful or do not align with the code's intent.

## 8. Continuous Integration (Optional)
- Set up CI to run tests and linting on every commit for early detection of issues.

## 9. Test Execution
- Use `pytest` as the test runner.
- Run all tests before every commit and before merging branches.

## 10. Documentation
- Keep this policy up to date as the project evolves.
