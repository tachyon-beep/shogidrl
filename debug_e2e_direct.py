#!/usr/bin/env python3
"""Debug E2E test directly."""

import subprocess
import sys
import tempfile
import time
import os
from pathlib import Path

TRAIN_PATH = Path(__file__).parent / "train.py"

def test_train_runs_minimal():
    """Test that train.py runs with minimal configuration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        env = os.environ.copy()
        env['WANDB_DISABLED'] = 'true'
        
        print(f"Starting subprocess with args: {[
            sys.executable,
            str(TRAIN_PATH),
            'train',
            '--total-timesteps', '1',  # Just 1 timestep
            '--savedir', str(tmp_path),
            '--seed', '42',
            '--device', 'cpu',
        ]}")
        
        print("Environment variables:")
        for key in ['WANDB_DISABLED', 'PYTHONPATH']:
            print(f"  {key}={env.get(key, 'NOT SET')}")
        
        start_time = time.time()
        
        # Start the process
        process = subprocess.Popen(
            [
                sys.executable,
                str(TRAIN_PATH),
                "train",
                "--total-timesteps", "1",
                "--savedir", str(tmp_path),
                "--seed", "42",
                "--device", "cpu",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        
        # Wait with timeout and periodic status updates
        timeout = 60
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"Process timed out after {elapsed:.1f}s")
                process.terminate()
                process.wait(timeout=5)
                break
                
            # Check if process finished
            if process.poll() is not None:
                print(f"Process finished after {elapsed:.1f}s")
                break
                
            # Print status every 10 seconds
            if int(elapsed) % 10 == 0 and elapsed > 0:
                print(f"Still running... {elapsed:.1f}s elapsed")
            
            time.sleep(1)
        
        stdout, stderr = process.communicate()
        
        print(f"Return code: {process.returncode}")
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")

if __name__ == "__main__":
    test_train_runs_minimal()