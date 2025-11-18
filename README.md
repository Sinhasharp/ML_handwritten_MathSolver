# ML Handwritten Math Solver

A machine learning-based web application that recognizes handwritten mathematical equations and solves them. This project utilizes a Convolutional Neural Network (CNN) to classify handwritten characters and OpenCV for image segmentation, wrapped in a Flask web interface.

## 📝 Description

This project bridges the gap between computer vision and mathematical problem solving. It takes an image of a handwritten equation as input, processes the image to extract individual characters, recognizes them using a trained deep learning model, and then computes the result. It supports both basic arithmetic and simple linear equations involving variables.

## ✨ Key Features

- **Handwritten Character Recognition**: Accurately identifies digits (0-9) and mathematical symbols (+, -, x) and variables (a, b).

- **Image Segmentation**: Uses OpenCV to detect contours and isolate individual characters from a single image.

- **Dual Solving Modes**:
  - Basic Mode: Evaluates standard arithmetic expressions (e.g., 2 + 2, 3 x 5).
  - Linear Equation Mode: Solves simple linear equations for variables 'a' or 'b' (e.g., 2a + 5 = 15).

- **Web Interface**: A user-friendly Flask-based frontend to upload images and view results.

## 🛠️ Built With

- **Python 3.x**
- **Flask**: For the backend web server.
- **TensorFlow / Keras**: For building and training the Convolutional Neural Network.
- **OpenCV**: For image pre-processing and character segmentation.
- **NumPy & Pandas**: For data manipulation.


## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sinhasharp/ml_handwritten_mathsolver.git
   cd ml_handwritten_mathsolver
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Web Application

To use the solver interface:

1. Ensure the trained model files (`model.json` and `model_weights.h5`) are located in the `model/` directory.

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

4. Upload an image containing a handwritten equation and select the mode (Basic or Linear Equation).

### Training the Model (Optional)

If you wish to retrain the model or train it on a new dataset:

1. Prepare your dataset. The code expects a directory structure where images are sorted into folders by label (e.g., `data/extracted_images/0`, `data/extracted_images/+`, etc.).

2. Update the `train_data_dir` path in `train.py` if necessary.

3. Run the training script:
   ```bash
   python train.py
   ```

4. This will save the new model and weights to the designated path.

## 🧠 Model Architecture

The project uses a Sequential CNN model with the following structure:

- **Convolutional Layers**: Extract features from the 28x28 input images.
- **Max Pooling Layers**: Reduce spatial dimensions.
- **Dropout Layers**: Prevent overfitting.
- **Dense (Fully Connected) Layers**: Perform classification based on extracted features.

The model handles classes for digits 0-9, operations +, -, x, and variables a, b.

## 📂 Project Structure

```
ML_handwritten_MathSolver/
├── app.py              # Main Flask application entry point
├── train.py            # Script to train the CNN model
├── requirements.txt    # List of python dependencies
├── README.md           # Project documentation
├── model/              # Directory containing trained model files
│   ├── model.json
│   └── model_weights.h5
└── templates/          # HTML templates for the web app (index.html)
```

## 📄 License

Distributed under the MIT License. See LICENSE for more information.
