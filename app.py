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

def generate_dummy_test_data():
    """Generate minimal dummy test data if files are not available"""
    print("🔧 Generating dummy test data as fallback...")
    
    # Create 10 simple test images (one for each digit)
    dummy_images = []
    dummy_labels = []
    dummy_arrays = []
    
    for digit in range(10):
        # Create a simple 28x28 array with the digit pattern
        # This is just a placeholder - not real MNIST data
        img_array = np.zeros((28, 28), dtype=np.float32)
        
        # Add some simple pattern to represent the digit
        if digit == 0:  # Circle-like pattern for 0
            img_array[10:18, 10:18] = 0.8
            img_array[12:16, 12:16] = 0.0
        elif digit == 1:  # Vertical line for 1
            img_array[5:23, 13:15] = 0.8
        else:  # Simple pattern for other digits
            img_array[5+digit:15+digit, 5:15] = 0.8
        
        # Convert to base64 image
        img_base64 = array_to_base64_image(img_array)
        dummy_images.append(img_base64)
        dummy_labels.append(digit)
        
        # Flatten for predictions
        flat_array = img_array.flatten().tolist()
        dummy_arrays.append(flat_array)
    
    # Create the test data structure
    dummy_data = {
        'images': dummy_images,
        'labels': dummy_labels,
        'image_arrays': dummy_arrays
    }
    
    print(f"✅ Generated {len(dummy_images)} dummy test images")
    return dummy_data

def load_test_data():
    """Load test data from static directory with ultimate fallback"""
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
    
    # If all file loading failed, generate dummy data
    print("❌ Failed to load any test data files")
    print("🔧 Falling back to generated dummy test data...")
    
    try:
        test_data = generate_dummy_test_data()
        print("✅ Dummy test data generated successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to generate dummy test data: {e}")
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
    """Debug endpoint to test data loading step by step"""
    debug_steps = []
    
    try:
        debug_steps.append(f"Working directory: {os.getcwd()}")
        debug_steps.append(f"Directory contents: {os.listdir('.')}")
        
        # Check static directory
        if os.path.exists('static'):
            static_contents = os.listdir('static')
            debug_steps.append(f"Static directory exists. Contents: {static_contents}")
            
            # Check each test data file
            test_files = ['test_data.json', 'test_data_small.json']
            for filename in test_files:
                filepath = os.path.join('static', filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    debug_steps.append(f"{filename}: EXISTS, size={size} bytes")
                    
                    # Try to read just the first few characters
                    try:
                        with open(filepath, 'r') as f:
                            first_chars = f.read(100)
                            debug_steps.append(f"{filename} first 100 chars: {repr(first_chars)}")
                    except Exception as e:
                        debug_steps.append(f"{filename} read error: {str(e)}")
                else:
                    debug_steps.append(f"{filename}: NOT FOUND")
        else:
            debug_steps.append("Static directory does not exist!")
        
        # Try loading each file manually
        test_files = [
            ('static/test_data.json', 'full'),
            ('static/test_data_small.json', 'small')
        ]
        
        for filepath, description in test_files:
            try:
                debug_steps.append(f"Attempting to load {description} from {filepath}")
                
                if not os.path.exists(filepath):
                    debug_steps.append(f"  - File not found: {filepath}")
                    continue
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                keys = list(data.keys())
                debug_steps.append(f"  - SUCCESS: {description} loaded with keys: {keys}")
                
                if 'images' in data:
                    debug_steps.append(f"  - Images count: {len(data['images'])}")
                    return jsonify({
                        "status": "SUCCESS", 
                        "loaded_file": description,
                        "image_count": len(data['images']),
                        "debug_steps": debug_steps
                    })
                else:
                    debug_steps.append(f"  - ERROR: No 'images' key in {description}")
                    
            except json.JSONDecodeError as e:
                debug_steps.append(f"  - JSON decode error in {description}: {str(e)}")
            except Exception as e:
                debug_steps.append(f"  - Error loading {description}: {str(e)}")
        
        return jsonify({
            "status": "FAILED",
            "error": "No test data files could be loaded",
            "debug_steps": debug_steps
        })
        
    except Exception as e:
        debug_steps.append(f"Critical error: {str(e)}")
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "debug_steps": debug_steps
        }), 500

@app.route('/api/debug/filesystem')
def debug_filesystem():
    """Debug endpoint to check filesystem status"""
    try:
        debug_info = {
            "working_directory": os.getcwd(),
            "directory_contents": os.listdir('.'),
            "static_exists": os.path.exists('static'),
            "test_data_exists": os.path.exists('static/test_data.json'),
            "model_exists": os.path.exists('static/mnist_model.h5'),
        }
        
        if os.path.exists('static'):
            debug_info["static_contents"] = os.listdir('static')
            
        if os.path.exists('static/test_data.json'):
            debug_info["test_data_size"] = os.path.getsize('static/test_data.json')
            
        if os.path.exists('static/mnist_model.h5'):
            debug_info["model_size"] = os.path.getsize('static/mnist_model.h5')
            
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": f"Debug filesystem error: {str(e)}"}), 500

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

if __name__ == '__main__':
    print("Starting MNIST Neural Network Deployment Server...")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    print(f"Server will run on port: {port}")
    print(f"Environment PORT variable: {os.environ.get('PORT', 'Not set')}")
    
    # Load pre-trained model and test data on startup
    model_loaded = load_pretrained_model()
    test_data_loaded = load_test_data()
    
    if model_loaded and test_data_loaded:
        print("Pre-trained model and test data loaded successfully!")
        print("Server ready for predictions!")
        print(f"Starting Flask app on 0.0.0.0:{port}")
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Failed to load required assets:")
        if not model_loaded:
            print("  - Pre-trained model could not be loaded")
        if not test_data_loaded:
            print("  - Test data could not be loaded")
        print("Server not started.")
