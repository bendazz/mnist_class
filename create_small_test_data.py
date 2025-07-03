#!/usr/bin/env python3
"""
Create a smaller test data file for Railway deployment
"""
import json
import os

def create_small_test_data():
    """Create a smaller version of test_data.json with fewer images"""
    
    # Load the original test data
    original_path = 'static/test_data.json'
    if not os.path.exists(original_path):
        print(f"Original test data not found at {original_path}")
        return False
    
    print("Loading original test data...")
    with open(original_path, 'r') as f:
        original_data = json.load(f)
    
    original_count = len(original_data['images'])
    print(f"Original test data has {original_count} images")
    
    # Create smaller version with only 200 images (should be ~1.3MB instead of 13MB)
    small_count = 200
    print(f"Creating smaller version with {small_count} images...")
    
    small_data = {
        'images': original_data['images'][:small_count],
        'labels': original_data['labels'][:small_count],
        'image_arrays': original_data['image_arrays'][:small_count]
    }
    
    # Save the smaller version
    small_path = 'static/test_data_small.json'
    with open(small_path, 'w') as f:
        json.dump(small_data, f)
    
    small_size = os.path.getsize(small_path) / (1024 * 1024)  # Size in MB
    print(f"Small test data saved to {small_path}")
    print(f"File size: {small_size:.1f} MB")
    print(f"Contains {len(small_data['images'])} images")
    
    return True

if __name__ == "__main__":
    create_small_test_data()
