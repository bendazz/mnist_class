#!/usr/bin/env python3
"""
Local training script for MNIST neural network.
This script trains the model once and saves:
- The trained model
- Training history plots
- Model architecture information
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

# Configure matplotlib for headless operation
plt.switch_backend('Agg')

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Store original test images for visualization (sample subset)
    test_sample_size = 2000  # Reasonable subset for web display
    indices = np.random.choice(len(x_test), test_sample_size, replace=False)
    x_test_viz = x_test[indices].copy()
    y_test_viz = y_test[indices].copy()
    
    # Normalize pixel values to 0-1 range
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_test_viz = x_test_viz.astype('float32') / 255.0
    
    # Flatten the images for feedforward network
    x_train_flat = x_train.reshape(x_train.shape[0], 28 * 28)
    x_val_flat = x_val.reshape(x_val.shape[0], 28 * 28)
    x_test_flat = x_test.reshape(x_test.shape[0], 28 * 28)
    
    # Convert labels to categorical (one-hot encoding) for training
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Visualization samples: {len(x_test_viz)}")
    
    return {
        'x_train': x_train_flat,
        'y_train': y_train_cat,
        'x_val': x_val_flat,
        'y_val': y_val_cat,
        'x_test': x_test_flat,
        'y_test': y_test_cat,
        'x_test_viz': x_test_viz,  # Original 28x28 for visualization
        'y_test_viz': y_test_viz,  # Original labels for visualization
        'test_indices': indices
    }

def create_model():
    """Create the neural network model"""
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

def train_model(data, epochs=15, batch_size=32):
    """Train the model and return history"""
    print("Creating model...")
    model = create_model()
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        data['x_train'], 
        data['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data['x_val'], data['y_val']),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(data['x_test'], data['y_test'], verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return model, history, test_accuracy, test_loss

def save_training_plots(history, output_dir='static'):
    """Save training history plots as static images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MNIST Neural Network Training Results', fontsize=16)
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    ax2.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss comparison (final values)
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    ax3.bar(['Training Loss', 'Validation Loss'], [final_train_loss, final_val_loss], 
            color=['blue', 'red'], alpha=0.7)
    ax3.set_title('Final Loss Comparison', fontsize=14)
    ax3.set_ylabel('Loss')
    for i, v in enumerate([final_train_loss, final_val_loss]):
        ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # Plot 4: Accuracy comparison (final values)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    ax4.bar(['Training Accuracy', 'Validation Accuracy'], [final_train_acc, final_val_acc], 
            color=['blue', 'red'], alpha=0.7)
    ax4.set_title('Final Accuracy Comparison', fontsize=14)
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim(0, 1)
    for i, v in enumerate([final_train_acc, final_val_acc]):
        ax4.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_path = os.path.join(output_dir, 'training_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_path}")
    return plot_path

def save_model_info(model, history, test_accuracy, test_loss, output_dir='static'):
    """Save model information and training results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model summary as string
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    # Prepare model info
    model_info = {
        'model_summary': '\n'.join(model_summary),
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_info': {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'epochs_trained': len(history.history['loss']),
            'training_date': datetime.now().isoformat()
        },
        'layers': [
            {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output.shape) if hasattr(layer, 'output') and layer.output is not None else "Not built"
            }
            for layer in model.layers
        ]
    }
    
    # Save model info as JSON
    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model info saved to: {info_path}")
    return info_path

def main():
    """Main training function"""
    print("MNIST Neural Network Local Training")
    print("=" * 50)
    
    # Load data
    data = load_mnist_data()
    
    # Train model
    model, history, test_accuracy, test_loss = train_model(data, epochs=15, batch_size=32)
    
    # Create output directory
    os.makedirs('static', exist_ok=True)
    
    # Save model
    model_path = 'static/mnist_model.h5'
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save test data for web app
    test_data = {
        'x_test_viz': data['x_test_viz'].tolist(),  # Convert to list for JSON serialization
        'y_test_viz': data['y_test_viz'].tolist(),
        'test_indices': data['test_indices'].tolist(),
        'x_test_flat': data['x_test'][data['test_indices']].tolist()  # Flattened for prediction
    }
    
    test_data_path = 'static/test_data.json'
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f)
    print(f"Test data saved to: {test_data_path}")
    
    # Save training plots
    plot_path = save_training_plots(history)
    
    # Save model info
    info_path = save_model_info(model, history, test_accuracy, test_loss)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nFiles created:")
    print(f"  - Model: {model_path}")
    print(f"  - Test data: {test_data_path}")
    print(f"  - Training plots: {plot_path}")
    print(f"  - Model info: {info_path}")
    print("\nYou can now run the web app with the pre-trained model!")

if __name__ == '__main__':
    main()
