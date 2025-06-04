#!/usr/bin/env python3
"""
Quick validation test for the simplified serialize_config function.
"""

import json
import sys
import os

from keisei.config_schema import AppConfig
from keisei.utils import load_config
from keisei.training.utils import serialize_config

def test_serialize_config():
    """Test that the simplified serialize_config function works correctly."""
    print("Testing simplified serialize_config function...")
    
    # Load default config
    config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
    config = load_config(config_path)
    
    # Test serialization
    try:
        result = serialize_config(config)
        print("✅ serialize_config executed successfully")
        
        # Verify it's valid JSON
        parsed = json.loads(result)
        print("✅ Output is valid JSON")
        
        # Check that it contains expected top-level keys
        expected_keys = {'env', 'training', 'evaluation', 'logging', 'wandb', 'parallel', 'demo'}
        actual_keys = set(parsed.keys())
        
        if expected_keys.issubset(actual_keys):
            print("✅ All expected configuration sections present")
        else:
            missing = expected_keys - actual_keys
            print(f"❌ Missing sections: {missing}")
            return False
            
        # Check that output is properly formatted (indented)
        if '\n' in result and '    ' in result:
            print("✅ Output is properly indented")
        else:
            print("❌ Output lacks proper indentation")
            return False
            
        print(f"📊 Serialized config size: {len(result)} characters")
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during serialization: {e}")
        return False

if __name__ == "__main__":
    success = test_serialize_config()
    sys.exit(0 if success else 1)
