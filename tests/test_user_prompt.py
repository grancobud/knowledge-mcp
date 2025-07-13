#!/usr/bin/env python3
"""
Test script for user_prompt functionality.
Tests backward compatibility and functionality of configurable user prompts.
"""

import logging
import tempfile
import yaml
from pathlib import Path
from knowledge_mcp.knowledgebases import load_kb_query_config, DEFAULT_QUERY_PARAMS

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def test_empty_user_prompt_backward_compatibility():
    """Test 5.1: Test with empty user_prompt (backward compatibility)"""
    print("=== Test 5.1: Empty user_prompt (backward compatibility) ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = Path(temp_dir) / "test_kb"
        kb_path.mkdir()
        
        # Create config.yaml WITHOUT user_prompt field (simulating old config)
        config_content = {
            "mode": "local",
            "top_k": 20,
            "description": "Test KB without user_prompt"
        }
        
        config_file = kb_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        print(f"Created config without user_prompt: {config_content}")
        
        # Load config and verify user_prompt is handled
        result = load_kb_query_config(kb_path)
        
        print(f"Loaded config result: {result}")
        
        # Verify user_prompt exists and is empty string
        assert 'user_prompt' in result, "user_prompt should exist in result"
        assert result['user_prompt'] == '', f"user_prompt should be empty string, got: {result['user_prompt']}"
        assert result['mode'] == 'local', "mode should be preserved from config"
        assert result['top_k'] == 20, "top_k should be preserved from config"
        
        print("✅ Test 5.1 PASSED: Backward compatibility works correctly")
        return True

def test_configured_user_prompt_functionality():
    """Test 5.2: Test with configured user_prompt (functionality)"""
    print("\n=== Test 5.2: Configured user_prompt (functionality) ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = Path(temp_dir) / "test_kb"
        kb_path.mkdir()
        
        # Create config.yaml WITH user_prompt field
        test_prompt = "Please provide a concise answer in bullet points."
        config_content = {
            "mode": "hybrid",
            "top_k": 30,
            "user_prompt": test_prompt,
            "description": "Test KB with user_prompt"
        }
        
        config_file = kb_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        print(f"Created config with user_prompt: {config_content}")
        
        # Load config and verify user_prompt is preserved
        result = load_kb_query_config(kb_path)
        
        print(f"Loaded config result: {result}")
        
        # Verify user_prompt is correctly loaded
        assert 'user_prompt' in result, "user_prompt should exist in result"
        assert result['user_prompt'] == test_prompt, f"user_prompt should match, got: {result['user_prompt']}"
        assert result['mode'] == 'hybrid', "mode should be preserved from config"
        assert result['top_k'] == 30, "top_k should be preserved from config"
        
        print("✅ Test 5.2 PASSED: User prompt functionality works correctly")
        return True

def test_default_parameters():
    """Test 5.3: Verify logging output and default parameter changes"""
    print("\n=== Test 5.3: Default parameter changes ===")
    
    print(f"DEFAULT_QUERY_PARAMS: {DEFAULT_QUERY_PARAMS}")
    
    # Verify the default parameter changes from Task 1.0
    assert DEFAULT_QUERY_PARAMS['mode'] == 'hybrid', f"Default mode should be 'hybrid', got: {DEFAULT_QUERY_PARAMS['mode']}"
    assert DEFAULT_QUERY_PARAMS['top_k'] == 40, f"Default top_k should be 40, got: {DEFAULT_QUERY_PARAMS['top_k']}"
    assert DEFAULT_QUERY_PARAMS['user_prompt'] == '', f"Default user_prompt should be empty string, got: {DEFAULT_QUERY_PARAMS['user_prompt']}"
    
    print("✅ Test 5.3 PASSED: Default parameters are correctly updated")
    return True

def main():
    """Run all tests"""
    print("Starting user_prompt implementation tests...\n")
    
    try:
        # Run all tests
        test1_passed = test_empty_user_prompt_backward_compatibility()
        test2_passed = test_configured_user_prompt_functionality() 
        test3_passed = test_default_parameters()
        
        if test1_passed and test2_passed and test3_passed:
            print("\n🎉 ALL TESTS PASSED! User prompt implementation is working correctly.")
            return True
        else:
            print("\n❌ Some tests failed.")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
