# MNIST Neural Network - Pre-trained Model Deployment

An interactive web application showcasing a pre-trained feedforward neural network for MNIST handwritten digit classification. This deployment-optimized version uses a locally trained model with static training visualizations for efficient cloud hosting.

## 🚀 **Live Demo Deployment (Railway)**

This app is perfectly optimized for Railway deployment with the following features:
- ✅ **Pre-trained Model**: No training required - uses a high-performance pre-trained model (98.16% test accuracy)
- ✅ **Static Assets**: Training plots and model info generated locally for fast loading
- ✅ **Memory Efficient**: Minimal resource usage with pre-saved test data
- ✅ **Instant Testing**: Immediate predictions on handwritten digits
- ✅ **Educational Ready**: Perfect for demonstrating neural network capabilities

### **Quick Railway Deployment:**

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Ready for Railway deployment with pre-trained model"
   git push origin main
   ```

2. **Deploy to Railway**:
   - Go to [railway.app](https://railway.app)
   - Sign up with your GitHub account
   - Click "New Project" → "Deploy from GitHub repo"
   - Select this repository
   - Railway will automatically detect the Python app and deploy!

3. **Automatic Setup**:
   - Railway reads `Procfile` and `requirements.txt`
   - Loads pre-trained model (`static/mnist_model.h5`)
   - Serves static training plots and assets
   - Assigns a public URL (e.g., `https://mnist-classifier-production.up.railway.app`)

4. **Share with Students**:
   - Copy the Railway URL and share with your class
   - Students can immediately test the neural network on handwritten digits
   - Perfect for demonstrating machine learning concepts!

## Features

- **Pre-trained Neural Network**: 
  - 98.16% test accuracy on MNIST dataset
  - 3-layer feedforward architecture (128→64→10 neurons)
  - 109,386 trainable parameters
- **Static Training Visualizations**: 
  - Training and validation loss/accuracy plots (generated locally)
  - Model architecture information
  - Training history and performance metrics
- **Interactive Testing**: 
  - Test the model on 2,000 handwritten digit images
  - Real-time predictions with confidence scores
  - Probability distributions for all 10 digit classes
- **Educational Interface**: 
  - Model summary and architecture details
  - Test accuracy and loss information
  - Clean, responsive design optimized for learning
- **Deployment Optimized**: 
  - Fast startup (no training required)
  - Minimal memory footprint
  - Static asset serving for training plots

## Neural Network Architecture

The application uses a feedforward neural network with:
- Input layer: 784 neurons (28x28 flattened MNIST images)
- Hidden layer 1: 128 neurons with ReLU activation + Dropout (0.2)
- Hidden layer 2: 64 neurons with ReLU activation + Dropout (0.2)
- Output layer: 10 neurons with softmax activation (digit classes 0-9)
- **Total Parameters**: 109,386

## Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python app.py
   ```

3. **Open the Application**:
   - Navigate to `http://localhost:5000`
   - Click "Start Training" to begin training

## API Endpoints

- `GET /` - Main application interface
- `GET /api/dataset/info` - Dataset statistics
- `GET /api/model/info` - Model architecture information
- `POST /api/training/start` - Start model training
- `POST /api/training/stop` - Stop current training
- `GET /api/training/status` - Get training status
- `GET /api/training/history` - Get training history for plotting
- `GET /api/test/random` - Get random test image
- `GET /api/test/predict/<index>` - Predict test image label

## Educational Use

Perfect for:
- Machine learning courses
- Neural network demonstrations  
- Understanding training dynamics (loss curves, overfitting, etc.)
- Interactive exploration of deep learning concepts
- Real-time visualization of model training
- Hands-on prediction testing

## Technical Details

- **Backend**: Python Flask + TensorFlow
- **Frontend**: HTML/CSS/JavaScript + Chart.js
- **Training**: Asynchronous with real-time updates
- **Data**: MNIST dataset (auto-downloaded)
- **Architecture**: Feedforward neural network
- **Deployment**: Railway-ready with environment-based port handling

## Files Overview

**Essential for Deployment:**
- `app.py` - Main Flask application
- `index.html` - Frontend interface
- `requirements.txt` - Python dependencies
- `Procfile` - Railway deployment configuration
- `README.md` - Documentation

**Development Only:**
- `start.sh` - Local development script (not needed for Railway)

## License

Educational use. MNIST dataset is public domain.