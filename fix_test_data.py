#!/usr/bin/env python3
"""
Fix test_data.json format to match what the Flask app expects.
Converts the saved test data to the correct format with base64 encoded images.
"""

import json
import numpy as np
import base64
import io
from PIL import Image

def array_to_base64_image(image_array):
    """Convert numpy array to base64 encoded PNG image"""
    # Ensure the array is in the right format (0-255, uint8)
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(image_array, mode='L')  # 'L' for grayscale
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def fix_test_data():
    """Load the existing test data and convert it to the correct format"""
    
    # Load existing test data
    try:
        with open('static/test_data.json', 'r') as f:
            old_data = json.load(f)
        print("Loaded existing test data")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading test data: {e}")
        print("Generating new test data from MNIST dataset...")
        return generate_new_test_data()
    
    # Check if data is already in correct format
    if 'images' in old_data and 'labels' in old_data and 'image_arrays' in old_data:
        print("Test data is already in correct format!")
        return True
    
    # Convert old format to new format
    print("Converting test data to correct format...")
    
    x_test_viz = np.array(old_data['x_test_viz'])
    y_test_viz = np.array(old_data['y_test_viz'])
    x_test_flat = np.array(old_data['x_test_flat'])
    
    print(f"Converting {len(x_test_viz)} test images...")
    
    # Convert images to base64
    images_base64 = []
    for i, img_array in enumerate(x_test_viz):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(x_test_viz)}")
        
        base64_img = array_to_base64_image(img_array)
        images_base64.append(base64_img)
    
    # Create new data structure
    new_data = {
        'images': images_base64,
        'labels': y_test_viz.tolist(),
        'image_arrays': x_test_flat.tolist()
    }
    
    # Save the corrected data
    with open('static/test_data.json', 'w') as f:
        json.dump(new_data, f)
    
    print(f"✅ Test data converted successfully!")
    print(f"   - {len(new_data['images'])} images converted to base64")
    print(f"   - {len(new_data['labels'])} labels preserved")
    print(f"   - {len(new_data['image_arrays'])} flattened arrays for prediction")
    
    return True

def generate_new_test_data():
    """Generate new test data from scratch if needed"""
    try:
        import tensorflow as tf
        print("Loading MNIST dataset...")
        
        # Load MNIST data
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Sample subset for web display
        test_sample_size = 2000
        indices = np.random.choice(len(x_test), test_sample_size, replace=False)
        x_test_sample = x_test[indices]
        y_test_sample = y_test[indices]
        
        # Normalize for display
        x_test_normalized = x_test_sample.astype('float32') / 255.0
        
        print(f"Converting {len(x_test_sample)} test images...")
        
        # Convert images to base64
        images_base64 = []
        for i, img_array in enumerate(x_test_sample):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(x_test_sample)}")
            
            base64_img = array_to_base64_image(img_array)
            images_base64.append(base64_img)
        
        # Create flattened arrays for prediction
        x_test_flat = x_test_normalized.reshape(len(x_test_normalized), 784)
        
        # Create data structure
        new_data = {
            'images': images_base64,
            'labels': y_test_sample.tolist(),
            'image_arrays': x_test_flat.tolist()
        }
        
        # Save the data
        with open('static/test_data.json', 'w') as f:
            json.dump(new_data, f)
        
        print(f"✅ New test data generated successfully!")
        print(f"   - {len(new_data['images'])} images converted to base64")
        print(f"   - {len(new_data['labels'])} labels")
        print(f"   - {len(new_data['image_arrays'])} flattened arrays for prediction")
        
        return True
        
    except Exception as e:
        print(f"Error generating new test data: {e}")
        return False

if __name__ == '__main__':
    print("Fixing test_data.json format...")
    print("=" * 50)
    
    success = fix_test_data()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ Test data format fixed successfully!")
        print("The Flask app should now work properly.")
    else:
        print("\n" + "=" * 50)
        print("❌ Failed to fix test data format.")
