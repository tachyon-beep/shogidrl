"""
Shared fixtures and utilities for E2E tests.
"""

import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Generator

import pytest
import torch


@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """Provides isolated temporary directory for each test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_isolation():
    """Ensures each test starts with clean state."""
    # Clear any existing PyTorch caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reset any global state
    torch.manual_seed(42)

    yield

    # Cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# E2E-specific monitoring fixtures with appropriate limits

@pytest.fixture
def performance_monitor(request):
    """Monitor test performance with E2E-appropriate limits."""
    import time

    start_time = time.perf_counter()

    yield

    execution_time = time.perf_counter() - start_time
    
    # E2E tests get generous timeouts due to model compilation and training overhead
    timeout = 180.0  # 3 minute timeout for E2E tests

    assert (
        execution_time < timeout
    ), f"E2E test took {execution_time:.3f}s, should be under {timeout}s"


@pytest.fixture
def memory_monitor(request):
    """Monitor memory usage with E2E-appropriate limits."""
    import gc
    import psutil

    # Force garbage collection before monitoring
    gc.collect()

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    yield

    # Force garbage collection after test
    gc.collect()

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    # E2E tests get generous memory limits due to PyTorch model loading and training
    limit = 500  # E2E tests get 500MB

    assert (
        memory_increase < limit
    ), f"Memory increased by {memory_increase:.1f} MB, E2E limit is {limit} MB"


# Cleanup utilities to ensure test isolation

@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically clean up test environment after each test."""
    yield

    # Clean up any remaining temporary files
    import tempfile
    import os

    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith("pytest_"):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except PermissionError:
                pass  # Ignore cleanup errors