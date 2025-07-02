from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import threading
import time
from datetime import datetime
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

# Global variables for training state
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'training_history': {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    },
    'start_time': None,
    'model_summary': None,
    'stop_requested': False
}

# Global variables for data and model
mnist_data = None
model = None

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    global mnist_data
    try:
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Split training data into train and validation sets
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        # Store original (non-flattened) test images for visualization
        x_test_original = x_test.copy()
        y_test_original = y_test.copy()
        
        # Normalize pixel values to 0-1 range
        x_train = x_train.astype('float32') / 255.0
        x_val = x_val.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_test_original = x_test_original.astype('float32') / 255.0
        
        # Flatten the images for feedforward network
        x_train = x_train.reshape(x_train.shape[0], 28 * 28)
        x_val = x_val.reshape(x_val.shape[0], 28 * 28)
        x_test = x_test.reshape(x_test.shape[0], 28 * 28)
        
        # Convert labels to categorical (one-hot encoding) for training
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_val = tf.keras.utils.to_categorical(y_val, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        mnist_data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test,
            'x_test_original': x_test_original,  # For visualization
            'y_test_original': y_test_original   # Original labels (not one-hot)
        }
        
        print(f"MNIST dataset loaded successfully!")
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Input shape: {x_train.shape[1:]}")
        return True
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        return False

def create_model():
    """Create a feedforward neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class TrainingCallback(tf.keras.callbacks.Callback):
    """Custom callback to update training progress"""
    
    def on_epoch_end(self, epoch, logs=None):
        global training_status
        
        training_status['current_epoch'] = epoch + 1
        training_status['training_history']['epoch'].append(epoch + 1)
        training_status['training_history']['loss'].append(float(logs.get('loss', 0)))
        training_status['training_history']['accuracy'].append(float(logs.get('accuracy', 0)))
        training_status['training_history']['val_loss'].append(float(logs.get('val_loss', 0)))
        training_status['training_history']['val_accuracy'].append(float(logs.get('val_accuracy', 0)))
        
        print(f"Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, "
              f"accuracy={logs.get('accuracy', 0):.4f}, "
              f"val_loss={logs.get('val_loss', 0):.4f}, "
              f"val_accuracy={logs.get('val_accuracy', 0):.4f}")
    
    def on_epoch_begin(self, epoch, logs=None):
        global training_status
        # Check if stop was requested
        if training_status['stop_requested']:
            print("Training stop requested by user")
            self.model.stop_training = True

def train_model_async(epochs, batch_size):
    """Train the model in a separate thread"""
    global training_status, model, mnist_data
    
    try:
        # Reset training status
        training_status['is_training'] = True
        training_status['current_epoch'] = 0
        training_status['total_epochs'] = epochs
        training_status['start_time'] = datetime.now().isoformat()
        training_status['stop_requested'] = False
        training_status['training_history'] = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Create and compile model
        model = create_model()
        
        # Get model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        training_status['model_summary'] = '\n'.join(model_summary)
        
        # Create callback
        callback = TrainingCallback()
        
        # Train the model
        history = model.fit(
            mnist_data['x_train'], 
            mnist_data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(mnist_data['x_val'], mnist_data['y_val']),
            callbacks=[callback],
            verbose=1
        )
        
    except Exception as e:
        print(f"Training error: {e}")
        training_status['error'] = str(e)
    finally:
        training_status['is_training'] = False
        if training_status['stop_requested']:
            print("Training stopped by user")
            training_status['stop_requested'] = False
        else:
            print("Training completed successfully!")

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

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "MNIST Neural Network Training API",
        "endpoints": {
            "/api/dataset/info": "Get dataset information",
            "/api/model/info": "Get model architecture information",
            "/api/training/start": "Start model training (POST)",
            "/api/training/stop": "Stop current training (POST)",
            "/api/training/status": "Get current training status",
            "/api/training/history": "Get training history for plotting",
            "/api/test/random": "Get random test image",
            "/api/test/image/<index>": "Get test image by index",
            "/api/test/predict/<index>": "Predict test image label"
        },
        "dataset_loaded": mnist_data is not None
    })

@app.route('/api/dataset/info')
def dataset_info():
    """Get information about the loaded dataset"""
    if mnist_data is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    return jsonify({
        "training_samples": len(mnist_data['x_train']),
        "validation_samples": len(mnist_data['x_val']),
        "test_samples": len(mnist_data['x_test']),
        "input_shape": list(mnist_data['x_train'].shape[1:]),
        "num_classes": 10,
        "class_names": list(range(10))
    })

@app.route('/api/model/info')
def model_info():
    """Get model architecture information"""
    # Create a temporary model to get architecture info
    temp_model = create_model()
    
    # Build the model by calling it with sample input
    sample_input = tf.zeros((1, 784))
    temp_model(sample_input)
    
    model_summary = []
    temp_model.summary(print_fn=lambda x: model_summary.append(x))
    
    # Count parameters
    total_params = temp_model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in temp_model.trainable_weights])
    
    return jsonify({
        "model_summary": '\n'.join(model_summary),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "layers": [
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(layer.output.shape) if hasattr(layer, 'output') and layer.output is not None else "Not built"
            }
            for layer in temp_model.layers
        ]
    })

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    if mnist_data is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    if training_status['is_training']:
        return jsonify({"error": "Training is already in progress"}), 400
    
    try:
        data = request.get_json() or {}
        epochs = data.get('epochs', 10)
        batch_size = data.get('batch_size', 32)
        
        # Validate parameters
        if epochs < 1 or epochs > 15:
            return jsonify({"error": "Epochs must be between 1 and 15"}), 400
        
        if batch_size < 16 or batch_size > 512:
            return jsonify({"error": "Batch size must be between 16 and 512"}), 400
        
        # Start training in a separate thread
        training_thread = threading.Thread(
            target=train_model_async, 
            args=(epochs, batch_size)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "message": "Training started successfully",
            "epochs": epochs,
            "batch_size": batch_size
        })
        
    except Exception as e:
        return jsonify({"error": f"Error starting training: {str(e)}"}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop current training"""
    if not training_status['is_training']:
        return jsonify({"error": "No training in progress"}), 400
    
    training_status['stop_requested'] = True
    return jsonify({"message": "Training stop requested"})

@app.route('/api/training/status')
def get_training_status():
    """Get current training status"""
    status = {
        "is_training": training_status['is_training'],
        "current_epoch": training_status['current_epoch'],
        "total_epochs": training_status['total_epochs'],
        "start_time": training_status['start_time'],
        "stop_requested": training_status['stop_requested']
    }
    
    if 'error' in training_status:
        status['error'] = training_status['error']
    
    if training_status['model_summary']:
        status['model_summary'] = training_status['model_summary']
    
    return jsonify(status)

@app.route('/api/training/history')
def get_training_history():
    """Get training history for plotting"""
    return jsonify(training_status['training_history'])

@app.route('/api/test/random')
def get_random_test_image():
    """Get a random test image for evaluation"""
    if mnist_data is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    # Get random index from test set
    index = np.random.randint(0, len(mnist_data['x_test']))
    
    return get_test_image_by_index(index)

@app.route('/api/test/image/<int:index>')
def get_test_image_by_index(index):
    """Get test image by specific index"""
    if mnist_data is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    if index < 0 or index >= len(mnist_data['x_test']):
        return jsonify({"error": f"Index out of range. Valid range: 0-{len(mnist_data['x_test'])-1}"}), 400
    
    try:
        # Get original (28x28) image for visualization
        image_2d = mnist_data['x_test_original'][index]
        true_label = int(mnist_data['y_test_original'][index])
        
        # Convert to base64 image
        image_base64 = array_to_base64_image(image_2d)
        
        return jsonify({
            "index": index,
            "true_label": true_label,
            "image": image_base64,
            "image_shape": image_2d.shape
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing test image: {str(e)}"}), 500

@app.route('/api/test/predict/<int:index>')
def predict_test_image(index):
    """Predict label for a test image using trained model"""
    if mnist_data is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    if model is None:
        return jsonify({"error": "No trained model available"}), 400
    
    if index < 0 or index >= len(mnist_data['x_test']):
        return jsonify({"error": f"Index out of range. Valid range: 0-{len(mnist_data['x_test'])-1}"}), 400
    
    try:
        # Get the flattened image for prediction
        image_flat = mnist_data['x_test'][index:index+1]  # Keep batch dimension
        true_label = int(mnist_data['y_test_original'][index])
        
        # Make prediction
        predictions = model.predict(image_flat, verbose=0)
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
            "all_probabilities": all_probs
        })
        
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting MNIST Neural Network Training Server...")
    
    # Load MNIST data on startup
    if load_mnist_data():
        print("Server ready!")
        # Use environment port or fallback to 5000
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Failed to load MNIST dataset. Server not started.")
