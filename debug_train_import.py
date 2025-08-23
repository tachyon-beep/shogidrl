#!/usr/bin/env python3
"""Debug what happens when we import and run train.py components step by step."""

import sys
import os
sys.path.insert(0, '/home/john/keisei')

print("1. Basic imports test...")
try:
    import argparse
    print("   ✓ argparse imported")
    
    import tempfile
    print("   ✓ tempfile imported")
    
    from pathlib import Path
    print("   ✓ pathlib imported")
    
except Exception as e:
    print(f"   ✗ Basic import failed: {e}")
    sys.exit(1)

print("2. Keisei imports test...")
try:
    from keisei.config_schema import AppConfig
    print("   ✓ ConfigSchema imported")
    
    from keisei.training.trainer import Trainer
    print("   ✓ Trainer imported")
    
except Exception as e:
    print(f"   ✗ Keisei import failed: {e}")
    sys.exit(1)

print("3. Test loading default config...")
try:
    from keisei.utils import load_config
    config = load_config()
    print("   ✓ Default config loaded")
    
except Exception as e:
    print(f"   ✗ Config creation failed: {e}")
    sys.exit(1)

print("4. Test creating Trainer (this might hang)...")
print("   Starting Trainer creation...")

try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(config, savedir=tmp_dir)
        print("   ✓ Trainer created successfully!")
        
except Exception as e:
    print(f"   ✗ Trainer creation failed: {e}")
    sys.exit(1)

print("All tests passed!")