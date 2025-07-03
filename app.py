from flask import Flask, jsonify, render_template_string, request, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

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

# Global variables for pre-trained model and test data
model = None
test_data = None
model_info = None

def load_pretrained_model():
    """Load the pre-trained model from static directory"""
    global model, model_info
    try:
        model_path = os.path.join('static', 'mnist_model.h5')
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False
            
        print("Loading pre-trained model...")
        model = tf.keras.models.load_model(model_path)
        print("Pre-trained model loaded successfully!")
        
        # Load model info
        model_info_path = os.path.join('static', 'model_info.json')
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            print("Model info loaded successfully!")
        else:
            print("Warning: Model info file not found")
            model_info = {}
            
        return True
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return False

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files including the training plot"""
    return send_from_directory('static', filename)

@app.route('/')
def home():
    """Serve the main HTML page"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({
            "error": "index.html not found",
            "message": "Make sure index.html is in the same directory as app.py"
        }), 404

@app.route('/api/debug/test-data-loading')
def debug_test_data_loading():
    """Debug endpoint to test the test data loading process"""
    debug_info = {
        "current_test_data_status": test_data is not None,
        "loading_attempts": []
    }
    
    # Try loading test data files and record the process
    test_files = [
        ('static/test_data.json', 'full'),
        ('static/test_data_small.json', 'small (200 images)')
    ]
    
    for test_data_path, description in test_files:
        attempt_info = {
            "file": test_data_path,
            "description": description,
            "exists": os.path.exists(test_data_path)
        }
        
        if attempt_info["exists"]:
            try:
                file_size = os.path.getsize(test_data_path)
                attempt_info["size_bytes"] = file_size
                attempt_info["size_mb"] = round(file_size / (1024 * 1024), 2)
                
                # Try to load and parse
                with open(test_data_path, 'r') as f:
                    data = json.load(f)
                
                attempt_info["json_valid"] = True
                attempt_info["keys"] = list(data.keys())
                
                if 'images' in data:
                    attempt_info["num_images"] = len(data['images'])
                    attempt_info["loadable"] = True
                else:
                    attempt_info["loadable"] = False
                    attempt_info["error"] = "Missing 'images' key"
                    
            except Exception as e:
                attempt_info["loadable"] = False
                attempt_info["error"] = str(e)
        else:
            attempt_info["loadable"] = False
            attempt_info["error"] = "File not found"
            
        debug_info["loading_attempts"].append(attempt_info)
    
    # Check if we can generate dummy data
    debug_info["dummy_data_generation"] = {
        "available": True,
        "description": "Can generate 10 dummy MNIST-like images in memory"
    }
    
    return jsonify(debug_info)

def generate_dummy_test_data():
    """Generate dummy test data in memory as a last resort"""
    print("🚨 Generating dummy test data as fallback...")
    
    dummy_images = []
    dummy_labels = []
    dummy_arrays = []
    
    for i in range(10):  # Generate 10 dummy images (one for each digit)
        # Create a simple pattern for each digit
        img_array = np.zeros((28, 28), dtype=np.float32)
        
        # Create simple patterns for each digit
        if i == 0:  # Circle for 0
            center = 14
            for x in range(28):
                for y in range(28):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    if 8 <= dist <= 12:
                        img_array[x, y] = 1.0
        elif i == 1:  # Vertical line for 1
            img_array[6:22, 13:15] = 1.0
        elif i == 2:  # Horizontal lines for 2
            img_array[8:10, 6:22] = 1.0
            img_array[13:15, 6:22] = 1.0
            img_array[18:20, 6:22] = 1.0
        else:  # Simple patterns for other digits
            # Create a simple rectangular pattern
            start_x = 6 + (i % 3) * 2
            start_y = 6 + (i % 3) * 2
            img_array[start_x:start_x + 16, start_y:start_y + 16] = 0.3
            img_array[start_x + 2:start_x + 14, start_y + 2:start_y + 14] = 0.7
            img_array[start_x + 4:start_x + 12, start_y + 4:start_y + 12] = 1.0
        
        # Convert to base64 image
        image_base64 = array_to_base64_image(img_array)
        
        dummy_images.append(image_base64)
        dummy_labels.append(i)
        dummy_arrays.append(img_array.flatten().tolist())  # Flatten for model input
    
    return {
        'images': dummy_images,
        'labels': dummy_labels,
        'image_arrays': dummy_arrays
    }

def load_test_data():
    """Load test data from static directory with ultimate fallback to dummy data"""
    global test_data
    
    # Try loading the full test data first, then fall back to small version
    test_files = [
        ('static/test_data.json', 'full'),
        ('static/test_data_small.json', 'small (200 images)')
    ]
    
    for test_data_path, description in test_files:
        try:
            print(f"Trying to load {description} test data from: {test_data_path}")
            print(f"Current working directory: {os.getcwd()}")
            
            if os.path.exists('static'):
                print(f"Static directory exists. Contents: {os.listdir('static')}")
            else:
                print("Static directory does not exist!")
                continue
                
            if not os.path.exists(test_data_path):
                print(f"Test data file not found at {test_data_path}")
                continue
                
            file_size = os.path.getsize(test_data_path)
            print(f"Test data file found. Size: {file_size} bytes ({file_size/(1024*1024):.1f} MB)")
            
            if file_size == 0:
                print("Error: Test data file is empty!")
                continue
                
            print(f"Loading {description} test data...")
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
            
            print(f"Test data JSON loaded. Keys: {list(test_data.keys())}")
            
            if 'images' in test_data:
                print(f"✅ {description.capitalize()} test data loaded successfully! {len(test_data['images'])} test images available.")
                return True
            else:
                print(f"Error: Test data missing 'images' key. Available keys: {list(test_data.keys())}")
                continue
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error loading {description} test data: {e}")
            continue
        except Exception as e:
            print(f"Error loading {description} test data: {e}")
            print(f"Error type: {type(e).__name__}")
            continue
    
    # Ultimate fallback: generate dummy test data
    print("❌ Failed to load any test data files - generating dummy data as fallback")
    try:
        test_data = generate_dummy_test_data()
        print(f"✅ Generated {len(test_data['images'])} dummy test images as fallback!")
        return True
    except Exception as e:
        print(f"❌ Failed to generate dummy test data: {e}")
        return False

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "MNIST Neural Network - Pre-trained Model Deployment",
        "endpoints": {
            "/api/dataset/info": "Get dataset information",
            "/api/model/info": "Get pre-trained model information",
            "/api/training/plots": "Get training plot image URL",
            "/api/test/random": "Get random test image",
            "/api/test/image/<index>": "Get test image by index",
            "/api/test/predict/<index>": "Predict test image label"
        },
        "model_loaded": model is not None,
        "test_data_loaded": test_data is not None
    })

@app.route('/api/dataset/info')
def dataset_info():
    """Get information about the test dataset"""
    if test_data is None:
        return jsonify({"error": "Test dataset not loaded"}), 500
    
    return jsonify({
        "test_samples": len(test_data['images']),
        "input_shape": [784],  # Flattened 28x28 images
        "image_shape": [28, 28],
        "num_classes": 10,
        "class_names": list(range(10)),
        "note": "Using pre-saved test data from local training"
    })

@app.route('/api/model/info')
def model_info_endpoint():
    """Get pre-trained model information"""
    if model is None:
        return jsonify({"error": "Pre-trained model not loaded"}), 500
    
    # Get model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    response_data = {
        "model_summary": '\n'.join(model_summary),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "layers": [
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(layer.output.shape) if hasattr(layer, 'output') and layer.output is not None else "Not built"
            }
            for layer in model.layers
        ]
    }
    
    # Add training info if available
    if model_info:
        response_data.update({
            "test_accuracy": model_info.get("test_accuracy"),
            "test_loss": model_info.get("test_loss"),
            "training_info": model_info.get("training_info", {}),
            "note": "Pre-trained model loaded from local training"
        })
    
    return jsonify(response_data)

@app.route('/api/training/plots')
def training_plots():
    """Get the URL for the static training plots image"""
    plot_path = os.path.join('static', 'training_plots.png')
    if os.path.exists(plot_path):
        return jsonify({
            "plot_url": "/static/training_plots.png",
            "available": True,
            "note": "Static training plots from local training"
        })
    else:
        return jsonify({
            "plot_url": None,
            "available": False,
            "error": "Training plots not found"
        }), 404

@app.route('/api/test/random')
def get_random_test_image():
    """Get a random test image for evaluation"""
    if test_data is None:
        return jsonify({"error": "Test dataset not loaded"}), 500
    
    # Get random index
    max_index = len(test_data['images']) - 1
    index = np.random.randint(0, max_index + 1)
    
    return get_test_image_by_index(index)

@app.route('/api/test/image/<int:index>')
def get_test_image_by_index(index):
    """Get test image by specific index with bounds checking"""
    if test_data is None:
        return jsonify({"error": "Test dataset not loaded"}), 500
    
    # Check bounds
    max_index = len(test_data['images']) - 1
    if index < 0 or index > max_index:
        return jsonify({"error": f"Index out of range. Valid range: 0-{max_index}"}), 400
    
    try:
        # Get image data (already base64 encoded from saved test data)
        image_base64 = test_data['images'][index]
        true_label = test_data['labels'][index]
        
        return jsonify({
            "index": index,
            "true_label": true_label,
            "image": image_base64,
            "image_shape": [28, 28],
            "note": f"Test image from pre-saved dataset (max index: {max_index})"
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing test image: {str(e)}"}), 500

@app.route('/api/test/predict/<int:index>')
def predict_test_image(index):
    """Predict label for a test image using the pre-trained model"""
    if test_data is None:
        return jsonify({"error": "Test dataset not loaded"}), 500
    
    if model is None:
        return jsonify({"error": "Pre-trained model not loaded"}), 500
    
    # Check bounds
    max_index = len(test_data['images']) - 1
    if index < 0 or index > max_index:
        return jsonify({"error": f"Index out of range. Valid range: 0-{max_index}"}), 400
    
    try:
        # Get the flattened image data for prediction
        image_data = np.array(test_data['image_arrays'][index], dtype=np.float32)
        image_data = image_data.reshape(1, 784)  # Reshape for model input
        true_label = test_data['labels'][index]
        
        # Make prediction
        predictions = model.predict(image_data, verbose=0)
        predicted_probs = predictions[0]
        predicted_label = int(np.argmax(predicted_probs))
        confidence = float(predicted_probs[predicted_label])
        
        # Get all class probabilities
        all_probs = {str(i): float(predicted_probs[i]) for i in range(10)}
        
        return jsonify({
            "index": index,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": predicted_label == true_label,
            "all_probabilities": all_probs,
            "note": f"Prediction using pre-trained model (max index: {max_index})"
        })
        
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

@app.route('/api/debug/filesystem')
def debug_filesystem():
    """Debug endpoint to check filesystem status"""
    try:
        debug_info = {
            "working_directory": os.getcwd(),
            "directory_contents": os.listdir('.'),
            "static_exists": os.path.exists('static'),
            "test_data_exists": os.path.exists('static/test_data.json'),
            "test_data_small_exists": os.path.exists('static/test_data_small.json'),
            "model_exists": os.path.exists('static/mnist_model.h5'),
        }
        
        if os.path.exists('static'):
            debug_info["static_contents"] = os.listdir('static')
            
        if os.path.exists('static/test_data.json'):
            debug_info["test_data_size"] = os.path.getsize('static/test_data.json')
            
        if os.path.exists('static/test_data_small.json'):
            debug_info["test_data_small_size"] = os.path.getsize('static/test_data_small.json')
            
        if os.path.exists('static/mnist_model.h5'):
            debug_info["model_size"] = os.path.getsize('static/mnist_model.h5')
            
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": f"Debug filesystem error: {str(e)}"}), 500

@app.route('/api/debug/force-reload', methods=['POST'])
def force_reload_data():
    """Force reload model and test data"""
    global model, test_data, model_info
    
    result = {
        "model_reload_success": False,
        "test_data_reload_success": False,
        "errors": []
    }
    
    try:
        # Force reload model
        print("🔄 Force reloading pre-trained model...")
        model_success = load_pretrained_model()
        result["model_reload_success"] = model_success
        
        if not model_success:
            result["errors"].append("Failed to reload pre-trained model")
        
        # Force reload test data
        print("🔄 Force reloading test data...")
        test_data_success = load_test_data()
        result["test_data_reload_success"] = test_data_success
        
        if test_data_success:
            if test_data and 'images' in test_data:
                result["test_images_count"] = len(test_data['images'])
                
                # Determine source
                if result["test_images_count"] == 10:
                    result["test_data_source"] = "dummy data (fallback)"
                elif result["test_images_count"] == 200:
                    result["test_data_source"] = "small test data file"
                elif result["test_images_count"] == 2000:
                    result["test_data_source"] = "full test data file"
                else:
                    result["test_data_source"] = f"unknown ({result['test_images_count']} images)"
        else:
            result["errors"].append("Failed to reload test data")
        
        return jsonify(result)
        
    except Exception as e:
        result["errors"].append(f"Exception during force reload: {str(e)}")
        return jsonify(result), 500

def load_assets_with_retry(max_retries=3):
    """Load assets with retry logic for Railway deployment reliability"""
    import time
    
    for attempt in range(max_retries):
        print(f"Asset loading attempt {attempt + 1}/{max_retries}")
        
        # Load pre-trained model
        model_loaded = load_pretrained_model()
        
        # Load test data with retry logic
        test_data_loaded = load_test_data()
        
        print(f"Attempt {attempt + 1} results:")
        print(f"  - Model loaded: {model_loaded}")
        print(f"  - Test data loaded: {test_data_loaded}")
        
        if model_loaded and test_data_loaded:
            print("✅ All assets loaded successfully!")
            return True, True
        elif attempt < max_retries - 1:
            print(f"⚠️ Retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
        else:
            print("⚠️ Max retries reached, starting server with available assets")
            
    return model_loaded, test_data_loaded

if __name__ == '__main__':
    print("Starting MNIST Neural Network Deployment Server...")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    print(f"Server will run on port: {port}")
    print(f"Environment PORT variable: {os.environ.get('PORT', 'Not set')}")
    
    # Load assets with retry logic for Railway deployment reliability
    model_loaded, test_data_loaded = load_assets_with_retry(max_retries=3)
    
    # Always start the server, even if some assets fail to load
    # This allows debugging endpoints to work
    print(f"Final asset loading status:")
    print(f"  - Model loaded: {model_loaded}")
    print(f"  - Test data loaded: {test_data_loaded}")
    
    if model_loaded and test_data_loaded:
        print("✅ All assets loaded successfully! Server ready for full functionality!")
    elif model_loaded:
        print("⚠️ Model loaded but test data failed - server has limited functionality")
        print("💡 Tip: Use the 'Force Reload Data' button in the web interface to retry loading")
    elif test_data_loaded:
        print("⚠️ Test data loaded but model failed - server has limited functionality")
    else:
        print("❌ Both model and test data failed to load - server running for debugging only")
        print("💡 Tip: Use the debugging buttons in the web interface to troubleshoot")
    
    print(f"Starting Flask app on 0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
