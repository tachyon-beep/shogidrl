# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) setup for the Keisei Shogi project, implementing Task 4.3 from the remediation plan.

## Overview

The CI/CD pipeline is designed to ensure code quality, catch regressions early, and validate that the parallel system (when implemented) works correctly. It includes multiple layers of testing and quality checks.

## Pipeline Structure

### 1. Main CI Workflow (`.github/workflows/ci.yml`)

Triggered on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

#### Jobs:

**test** (Matrix: Python 3.9, 3.10, 3.11, 3.12)
- Code linting with flake8
- Type checking with mypy (informational)
- Unit test execution with pytest
- Test coverage reporting
- Upload coverage to Codecov

**integration-test** (Python 3.11, push to main/develop only)
- Run integration smoke tests
- Verify the full system can initialize and run

**parallel-system-test** (Python 3.11, push to main only)
- Run parallelism smoke tests
- Verify parallel system design and multiprocessing capabilities
- Upload test artifacts on failure

**performance-check** (Python 3.11, push to main only)
- Run performance profiling on a short training session
- Generate profiling reports
- Upload profiling results as artifacts

**security-scan** (All events)
- Run Bandit security scanner
- Check for known vulnerabilities with Safety

### 2. Release Workflow (`.github/workflows/release.yml`)

Triggered on:
- Tags matching `v*.*.*` pattern
- Manual workflow dispatch

#### Jobs:

**test**
- Full test suite execution
- Integration tests

**build**
- Build Python package
- Validate package with twine
- Upload build artifacts

**release** (on tags only)
- Generate changelog from git commits
- Create GitHub release
- Upload release assets

## Local Development

### Running CI Checks Locally

Use the provided script to run the same checks as CI:

```bash
./scripts/run_local_ci.sh
```

This script runs:
1. Dependency installation
2. Linting checks (flake8)
3. Type checking (mypy)
4. Security scanning (bandit)
5. Unit tests with coverage
6. Integration smoke tests

### Pre-commit Hooks

Install pre-commit hooks to catch issues before committing:

```bash
pip install pre-commit
pre-commit install
```

The hooks will run:
- Code formatting (black, isort)
- Linting (flake8)
- Security scanning (bandit)
- Type checking (mypy)
- Basic file checks (trailing whitespace, etc.)

## Test Categories

### 1. Unit Tests (`tests/test_*.py`)
- Fast, isolated tests of individual components
- Mock external dependencies
- Run on all Python versions

### 2. Integration Smoke Tests (`tests/test_integration_smoke.py`)
- Test that the full training pipeline can initialize and run
- Test configuration system end-to-end
- Test evaluation system
- Run on push to main/develop branches

### 3. Parallel System Smoke Tests (`tests/test_parallel_smoke.py`)
- Test multiprocessing capabilities in CI environment
- Validate parallel system interface design
- Test future SelfPlayWorker and VecEnv interfaces
- Run only on push to main branch

## Performance Profiling

### Automated Profiling
The CI pipeline includes automated performance profiling that:
- Runs a short training session (500 timesteps) with cProfile
- Generates detailed profiling reports
- Uploads results as artifacts

### Manual Profiling
Use the profiling script for detailed analysis:

```bash
# Run profiling with default settings (2048 timesteps)
python scripts/profile_training.py

# Run shorter profiling session with report
python scripts/profile_training.py --timesteps 1000 --report

# Analyze existing profile
python scripts/profile_training.py --analyze-only profile.prof

# View interactive flame graph
pip install snakeviz
snakeviz profile.prof
```

## Security Scanning

The pipeline includes multiple security checks:

1. **Bandit** - Scans for common security issues in Python code
2. **Safety** - Checks for known vulnerabilities in dependencies
3. **Pre-commit hooks** - Catch issues during development

## Artifacts and Reports

The CI pipeline generates several artifacts:

- **Test Coverage Report** - HTML coverage report (`htmlcov/`)
- **Performance Profiles** - cProfile output and analysis reports
- **Security Scan Results** - Bandit JSON reports
- **Test Logs** - Detailed logs from failed tests

## Configuration

### Required Secrets
- `GITHUB_TOKEN` - Automatically provided by GitHub
- `CODECOV_TOKEN` - For coverage reporting (optional)

### Environment Variables
No special environment variables required for basic CI functionality.

## Troubleshooting

### Common Issues

1. **Test Failures in CI but not Locally**
   - Ensure you're using the same Python version
   - Check for platform-specific issues
   - Verify all dependencies are installed

2. **Multiprocessing Issues**
   - Some CI environments have limitations on process creation
   - Tests are designed to be robust against these limitations

3. **Performance Test Timeouts**
   - Performance tests have generous timeouts
   - May need adjustment based on CI runner performance

### Debugging Failed Tests

1. Check the CI job logs for detailed error messages
2. Download test artifacts for more information
3. Run the same tests locally with `./scripts/run_local_ci.sh`
4. Use the profiling script to investigate performance issues

## Future Enhancements

When the parallel system is implemented (Task 4.2), the CI pipeline will:

1. **Expand Parallel Tests**
   - Test actual parallel experience collection
   - Validate model synchronization
   - Test deadlock prevention

2. **Performance Benchmarks**
   - Compare parallel vs. serial performance
   - Track performance regressions over time
   - Automated performance alerts

3. **Integration with W&B**
   - Automated experiment tracking in CI
   - Performance regression detection
   - Model artifact validation

## References

- [Task 4.3 - Enhance CI for Parallel System](../remediation/REMEDIATION_STAGE_4.md#task-43-enhance-ci-for-a-parallel-system)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Documentation](https://pre-commit.com/)
