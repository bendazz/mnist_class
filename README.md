# MNIST Neural Network Trainer

An interactive web application for training feedforward neural networks on the MNIST handwritten digit dataset. Students can start training with a button click and watch real-time plots of loss and accuracy during training.

## 🚀 **Live Demo Deployment (Railway)**

This app is optimized for Railway deployment with the following features:
- ✅ **Smart port handling**: Uses Railway's assigned PORT environment variable
- ✅ **Resource optimized**: Limited to 15 epochs for efficient cloud usage
- ✅ **Fast startup**: ~10-15 seconds for MNIST dataset loading
- ✅ **Educational ready**: Perfect for classroom use

### **Quick Railway Deployment:**

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Ready for Railway deployment"
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
   - Installs all dependencies automatically
   - Assigns a public URL (e.g., `https://mnist-trainer-production.up.railway.app`)

4. **Share with Students**:
   - Copy the Railway URL and share with your class
   - Students can access immediately - no setup required!

## Features

- **Interactive Training**: Start neural network training with a simple button click
- **Real-time Visualization**: 
  - Training and validation loss plots updated in real-time
  - Training and validation accuracy plots updated in real-time
  - Progress tracking with epoch counters
- **Model Testing**: After training, test individual images and see predictions
- **Educational Interface**: 
  - Model architecture display
  - Dataset information (51K train, 9K validation, 10K test samples)
  - Training parameter controls (epochs: 1-15, batch size: 16-512)
- **Modern UI**: Clean, responsive design with real-time charts using Chart.js

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