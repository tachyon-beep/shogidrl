#!/usr/bin/env python3
"""Test minimal training to isolate hanging."""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Set environment
env = os.environ.copy()
env['WANDB_DISABLED'] = 'true'

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    
    print("Starting minimal training test...")
    
    result = subprocess.run([
        sys.executable,
        "/home/john/keisei/train.py",
        "train",
        "--total-timesteps", "1",  # Just 1 timestep
        "--savedir", str(tmp_path),
        "--seed", "42",
        "--device", "cpu",
    ], capture_output=True, text=True, env=env, timeout=30)
    
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")