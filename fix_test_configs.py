#!/usr/bin/env python3
"""
Script to fix all test configurations by adding the missing ParallelConfig field.
"""

import os
import re

# List of test files that need fixing
test_files = [
    "/home/john/keisei/tests/test_ppo_agent.py",
    "/home/john/keisei/tests/test_env_manager.py", 
    "/home/john/keisei/tests/test_evaluate.py",
    "/home/john/keisei/tests/test_session_manager.py",
    "/home/john/keisei/tests/test_trainer_resume_state.py",
    "/home/john/keisei/tests/test_trainer_session_integration.py",
    "/home/john/keisei/tests/test_trainer_training_loop_integration.py",
    "/home/john/keisei/tests/test_wandb_integration.py",
    "/home/john/keisei/tests/test_model_manager_checkpoint_and_artifacts.py"
]

def fix_test_file(file_path):
    """Fix a single test file by adding ParallelConfig import and usage."""
    print(f"Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if ParallelConfig is already imported
    if 'ParallelConfig' not in content:
        # Add ParallelConfig to the imports
        import_pattern = r'(from keisei\.config_schema import \([^)]*)\)'
        match = re.search(import_pattern, content, re.DOTALL)
        if match:
            imports = match.group(1)
            if 'ParallelConfig' not in imports:
                # Add ParallelConfig to the import list
                new_imports = imports.rstrip() + ',\n    ParallelConfig'
                content = content.replace(match.group(1), new_imports)
    
    # Check if parallel field is missing from AppConfig instantiation
    if 'parallel=' not in content:
        # Find AppConfig instantiation and add parallel field
        # Look for the pattern where demo= is the last field before closing )
        demo_pattern = r'(\s+demo=DemoConfig\([^)]*\),)\s*\)'
        match = re.search(demo_pattern, content)
        if match:
            replacement = match.group(1) + '\n        parallel=ParallelConfig(),\n    )'
            content = content.replace(match.group(0), replacement)
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed {file_path}")

def main():
    for file_path in test_files:
        if os.path.exists(file_path):
            fix_test_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
