#!/usr/bin/env python3
"""Test with torch.compile fully disabled."""

import subprocess
import sys
import tempfile
import time
import os
from pathlib import Path

TRAIN_PATH = Path(__file__).parent / "train.py"

def test_fully_disabled():
    """Test with torch.compile and benchmarking both disabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        env = os.environ.copy()
        env['WANDB_DISABLED'] = 'true'
        
        print(f"Starting subprocess with fully disabled compile...")
        
        start_time = time.time()
        
        result = subprocess.run(
            [
                sys.executable,
                str(TRAIN_PATH),
                "train",
                "--total-timesteps", "1",
                "--savedir", str(tmp_path),
                "--seed", "42",
                "--device", "cpu",
                "--override", "training.enable_torch_compile=false",  # Disable torch.compile
                "--override", "training.enable_compilation_benchmarking=false",  # Disable benchmarking  
                "--override", "logging.wandb_enabled=false",  # Disable WandB
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,  # Should be much faster now
        )
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s")
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

if __name__ == "__main__":
    test_fully_disabled()