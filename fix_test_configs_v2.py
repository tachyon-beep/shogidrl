import re
import os

# Define the files to be processed
FILES_TO_PROCESS = [
    "tests/test_model_manager_init.py",
    "tests/test_ppo_agent.py",
    "tests/test_env_manager.py",
    "tests/test_evaluate.py",
    "tests/test_session_manager.py",
    "tests/test_trainer_resume_state.py",
    "tests/test_trainer_session_integration.py",
    "tests/test_trainer_training_loop_integration.py",
    "tests/test_wandb_integration.py",
    "tests/test_model_manager_checkpoint_and_artifacts.py",
    "tests/test_neural_network.py",
    "tests/test_seeding.py",
    "tests/test_model_save_load.py",
    "tests/test_train.py",
    "tests/test_trainer_config.py",
    "keisei/utils/agent_loading.py" 
]

BASE_DIR = "/home/john/keisei"

IMPORT_STRING = "from keisei.config_schema import ParallelConfig"
APPCONFIG_PATTERN = r"AppConfig\("
# Regex to find AppConfig instantiations, ensuring not to match if parallel is already there.
APPCONFIG_INSERT_PATTERN = r"AppConfig\((?!.*parallel=)([^)]*)\)"
PARALLEL_CONFIG_DEFAULT = "parallel=ParallelConfig(enabled=False, num_workers=4, batch_size=32, sync_interval=100, compression_enabled=True, timeout_seconds=10.0, max_queue_size=1000, worker_seed_offset=1000)"

def update_file_content(filepath):
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return

    original_content = content # Keep a copy for comparison
    modified_in_step = False # Track if any modification happens

    # Ensure ParallelConfig is imported
    if IMPORT_STRING not in content:
        config_schema_import_pattern = r"(from keisei\.config_schema import .*)"
        match = re.search(config_schema_import_pattern, content)
        if match:
            existing_import_line = match.group(1)
            if "ParallelConfig" not in existing_import_line: # Add only if not already part of the line
                content = content.replace(existing_import_line, f"{existing_import_line}, ParallelConfig")
                modified_in_step = True
        else:
            # Add the import string on a new line, typically at the top with other imports
            # This is a simplistic approach; ideally, it would find the import block
            content = IMPORT_STRING + "\n" + content
            modified_in_step = True

    # Update AppConfig instantiations
    # Iteratively find and replace to handle multiple AppConfig instances in a file
    # Use a temporary variable for modifications in this loop to avoid issues with changing content being searched
    current_iter_content = content
    final_content_parts = []
    last_end = 0

    for match in re.finditer(APPCONFIG_INSERT_PATTERN, content):
        start, end = match.span(0)
        final_content_parts.append(content[last_end:start]) # Add content before the match

        existing_args_str = match.group(1).strip()
        
        new_args_list = []
        if PARALLEL_CONFIG_DEFAULT not in existing_args_str: # Check to prevent double adding if script is rerun
            new_args_list.append(PARALLEL_CONFIG_DEFAULT)
        
        if existing_args_str:
            new_args_list.append(existing_args_str)
            
        new_args_combined = ", ".join(filter(None, new_args_list))
        
        final_content_parts.append(f"AppConfig({new_args_combined})")
        last_end = end
        modified_in_step = True

    final_content_parts.append(content[last_end:]) # Add any remaining content
    content = "".join(final_content_parts)

    # Check for AppConfig.parse_obj calls and print info
    if "AppConfig.parse_obj" in content:
        print(f"INFO: 'AppConfig.parse_obj' found in {filepath}. "
              f"Ensure the dictionary passed to parse_obj includes the 'parallel' field. "
              f"Example: {{..., 'parallel': {{'enabled': False, 'start_method': 'fork', 'num_envs': 1, 'base_port': 50000}}}}")

    if content != original_content or modified_in_step: # Check if content actually changed
        try:
            with open(filepath, "w") as f:
                f.write(content)
            print(f"Updated: {filepath}")
        except IOError as e:
            print(f"Error writing to file {filepath}: {e}")
    elif "AppConfig.parse_obj" not in content : # Avoid "No changes" if only parse_obj was found
        print(f"No changes needed: {filepath}")

if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory '{BASE_DIR}' does not exist.")
        exit(1)

    for rel_path in FILES_TO_PROCESS:
        abs_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(abs_path):
            print(f"Warning: File '{abs_path}' not found, skipping.")
            continue
        print(f"Processing: {abs_path}")
        update_file_content(abs_path)

    print("\nScript finished. Review changes and INFO messages.")

