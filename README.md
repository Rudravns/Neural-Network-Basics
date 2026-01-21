# Neural Network Handwritten Digit Recognition

A fully-implemented neural network from scratch that recognizes handwritten digits (0-9) from the MNIST dataset with an interactive Pygame visualization interface.

## Features

- **Custom Neural Network Implementation**: Built entirely from scratch using NumPy
  - Multi-layer perceptron architecture
  - Configurable hidden layers
  - ReLU activation for hidden layers
  - Softmax activation for output layer
  - Backpropagation for training

- **Interactive Visualization**: Real-time Pygame interface for drawing and prediction
  - Draw digits directly on canvas
  - Instant predictions with confidence scores
  - Visual grid representation (28x28 pixels)

- **MNIST Dataset Support**: 
  - Automatic dataset loading and extraction
  - Training and testing sets included
  - Supports 60,000 training samples and 10,000 test samples

- **Model Persistence**: Save and load trained models using JSON

## Project Structure

```
Machine-Learning/
├── Main.py              # Main application with Pygame UI
├── network.py           # Neural network implementation
├── load_data.py         # MNIST dataset loading utilities
├── helper.py            # Helper functions and model persistence
├── model_data.json      # Saved trained model weights
├── README.md            # This file
└── data/                # EMNIST dataset
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Pygame
- Matplotlib (optional, for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rudravns/Machine-Learning.git
cd Machine-Learning
```

2. Install dependencies:
```bash
pip install numpy scipy pygame matplotlib
```

3. Download MNIST dataset files and place them in the `data/` directory

## Usage

### Running the Application

```bash
python Main.py
```

#### Command-line Arguments:

- `--learning` or `-l`: Train a new model from scratch (removes old model)
- `--test_acc` or `-t`: Evaluate model accuracy on test set
- `--times N`: Repeat training N times

### Examples

**Train a new model:**
```bash
python Main.py --learning
```

**Train and display test accuracy:**
```bash
python Main.py --learning --test_acc
```

**Use pre-trained model for inference:**
```bash
python Main.py
```

### In-Application Controls

- **Draw** on the canvas to create a digit
- **Predictions appear in real-time** at the top of the window
- **Clear** to erase the canvas and start over
- **Close window** to exit

## Neural Network Architecture

### Default Configuration

- **Input Layer**: 784 nodes (28×28 pixels flattened)
- **Hidden Layers**: 128 → 100 → 40 nodes
- **Output Layer**: 11 nodes (digits 0-9)
- **Activation Functions**: 
  - ReLU for hidden layers
  - Softmax for output layer
- **Training Parameters**:
  - Epochs: 10
  - Batch Size: 32
  - Learning Rate: 0.1

## Model Training

The network is trained using:
- **Forward propagation**: Compute predictions
- **Backpropagation**: Calculate gradients
- **Gradient descent**: Update weights and biases
- **Batch processing**: Process 32 samples at a time

## API Overview

### Neural_Network Class

```python
# Initialize network
nn = Neural_Network(
    Input_layer_size=784,
    Output_layer_size=11,
    Hidden_Layers_Sizes=(128, 100, 40)
)

# Forward pass
predictions = nn.forward(input_data)

# Train on batch
nn.train(X_batch, y_batch, learning_rate=0.1)

# Evaluate accuracy
accuracy = nn.evaluate(X_test, y_test)
```

### Helper Functions

```python
# Save model
helper.save(model_data)

# Load model
model_data = helper.load()

# Time function execution
@helper.timer
def my_function():
    pass
```

## Performance

- Expected test accuracy: 85-95% (depending on training iterations)
- Training time: ~2-5 minutes per epoch (10,000 samples)
- Inference: Real-time predictions on canvas input

## Dataset Information

The MNIST dataset contains:
- **Training Set**: 60,000 handwritten digit images
- **Test Set**: 10,000 handwritten digit images
- **Image Size**: 28×28 pixels (784 features)
- **Labels**: 0-9 (10 classes)

## Future Improvements

- [ ] Convolutional Neural Network (CNN) implementation
- [ ] Data augmentation techniques
- [ ] Batch normalization
- [ ] Additional optimization algorithms (Adam, RMSprop)
- [ ] Web interface
- [ ] Model export to ONNX format
- [ ] Dropout regularization

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

MIT License - Feel free to use this project for learning and development.

## Author

Rudransh (Rudravns)

## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation)
- [Neural Networks Basics](https://en.wikipedia.org/wiki/Artificial_neural_network)
