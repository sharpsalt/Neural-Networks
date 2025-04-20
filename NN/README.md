A simple yet powerful neural network implementation from scratch using NumPy. This repository contains a fully customizable neural network designed for classification tasks, trained on a spiral dataset for demonstration.
 Key Features

Pure NumPy Implementation – No deep learning frameworks (like TensorFlow/PyTorch) required.
Flexible Architecture – Supports customizable hidden layers and activation functions.
Training & Evaluation – Implements forward/backward propagation, cross-entropy loss, and L2 regularization.
Visualization – Includes Matplotlib plots to visualize training data and decision boundaries.
How It Works

    Forward Propagation: Computes predictions using ReLU (hidden layer) and Softmax (output).

    Backpropagation: Updates weights via gradient descent.

    Loss Calculation: Uses Categorical Cross-Entropy (CCE) for multi-class classification.

    Regularization: Adds L2 penalty to prevent overfitting.

Repository Structure

Neural-Network/
├── spiral.csv               # Sample dataset (2D spiral classification)
├── Neural_Network.ipynb     # Jupyter Notebook with full implementation

Quick Start

    Clone the repo
    bash

git clone https://github.com/sharpsalt/Neural-Network.git
cd Neural-Network

Run the Notebook
bash

jupyter notebook Neural_Network.ipynb

Train the Model
python

    nn = NN(n_features=2, n_hidden=100, n_classes=3)
    nn.fit(X, y, lr=0.1, reg=1e-3, max_iters=10000)
    print(f"Accuracy: {np.mean(nn.predict(X) == y) * 100:.2f}%")

 Sample Output

Decision Boundary Plot
🔧 Dependencies
    Python 3.8+
    NumPy (pip install numpy)
    Pandas(pip install pandas)
    Matplotlib (pip install matplotlib)
    Jupyter Notebook (pip install notebook)
