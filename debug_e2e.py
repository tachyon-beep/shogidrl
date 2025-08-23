#!/usr/bin/env python3
"""Debug script to isolate E2E test hanging issue."""

import sys
import os
sys.path.insert(0, '/home/john/keisei')

print("1. Testing basic imports...")
try:
    from keisei.evaluation.performance_manager import EvaluationPerformanceManager
    print("   ✓ EvaluationPerformanceManager imported")
except Exception as e:
    print(f"   ✗ Failed to import EvaluationPerformanceManager: {e}")
    sys.exit(1)

print("2. Testing EvaluationPerformanceManager creation...")
try:
    perf_manager = EvaluationPerformanceManager()
    print("   ✓ EvaluationPerformanceManager created")
except Exception as e:
    print(f"   ✗ Failed to create EvaluationPerformanceManager: {e}")
    sys.exit(1)

print("3. Testing ResourceMonitor lazy access...")
try:
    resource_monitor = perf_manager.resource_monitor
    print("   ✓ ResourceMonitor accessed successfully")
except Exception as e:
    print(f"   ✗ Failed to access ResourceMonitor: {e}")
    sys.exit(1)

print("4. Testing EvaluationManager import...")
try:
    from keisei.evaluation.core_manager import EvaluationManager
    print("   ✓ EvaluationManager imported")
except Exception as e:
    print(f"   ✗ Failed to import EvaluationManager: {e}")
    sys.exit(1)

print("5. Testing train.py import...")
try:
    from keisei.training.train import main
    print("   ✓ train.py imported")
except Exception as e:
    print(f"   ✗ Failed to import train.py: {e}")
    sys.exit(1)

print("All imports successful!")