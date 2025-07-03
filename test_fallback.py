#!/usr/bin/env python3
"""
Test script to verify that the dummy data fallback works correctly
"""
import os
import sys
import tempfile
import shutil

# Add current directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import generate_dummy_test_data, load_test_data

def test_dummy_data_generation():
    """Test that dummy data generation works"""
    print("Testing dummy data generation...")
    try:
        dummy_data = generate_dummy_test_data()
        print(f"✅ Generated {len(dummy_data['images'])} dummy images")
        print(f"✅ Generated {len(dummy_data['labels'])} dummy labels")
        print(f"✅ Generated {len(dummy_data['image_arrays'])} dummy arrays")
        
        # Check that all data is consistent
        assert len(dummy_data['images']) == len(dummy_data['labels']) == len(dummy_data['image_arrays'])
        assert len(dummy_data['images']) == 10  # Should be 10 digits
        
        # Check that labels are 0-9
        assert set(dummy_data['labels']) == set(range(10))
        
        print("✅ Dummy data generation test passed!")
        return True
    except Exception as e:
        print(f"❌ Dummy data generation test failed: {e}")
        return False

def test_fallback_scenario():
    """Test the fallback scenario by temporarily hiding static files"""
    print("\nTesting fallback scenario...")
    
    # Save current working directory
    original_cwd = os.getcwd()
    
    try:
        # Create a temporary directory without static files
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            print(f"Changed to temporary directory: {temp_dir}")
            
            # Reset global test_data
            import app
            app.test_data = None
            
            # Try to load test data (should fall back to dummy data)
            result = load_test_data()
            
            if result and app.test_data is not None:
                print(f"✅ Fallback successful! Loaded {len(app.test_data['images'])} test images")
                print("✅ Fallback scenario test passed!")
                return True
            else:
                print("❌ Fallback scenario test failed!")
                return False
                
    except Exception as e:
        print(f"❌ Fallback scenario test failed with exception: {e}")
        return False
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    print("Running fallback tests...\n")
    
    test1_passed = test_dummy_data_generation()
    test2_passed = test_fallback_scenario()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The fallback mechanism should work on Railway.")
    else:
        print("\n❌ Some tests failed. There may be issues with the fallback mechanism.")
    
    sys.exit(0 if (test1_passed and test2_passed) else 1)
