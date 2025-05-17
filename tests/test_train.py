"""
Unit tests for train.py (smoke test for main loop).
"""

import importlib


def test_train_main_runs():
    """Smoke test: train.py main() runs without error."""
    train = importlib.import_module("train")
    train.main()
