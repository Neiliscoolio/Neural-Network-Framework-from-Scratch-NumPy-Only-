This is my first ever programming project. I built a small neural network library written in pure python with the help of NumPy while refraining from touching any machine learning libraries such as PyTorch or TensorFlow. The goal of this project was to understand what frameworks like PyTorch and TensorFlow are doing under the hood by implementing the core pieces myself.The library supports building and training simple feedforward neural networks with backpropagation and stochastic gradient descent. I utilized Claude in order to debug some issues that I was having with downloading the MNIST library and to teach me the fundementals of machine learning and the applications of NumPy.

Features I completely implemented from scratch:

-Fully connected layers (LinearLayer) with forward and backward passes

-ReLU activation

-Softmax output layer

-CrossEntropyLoss with backpropagation

-Sequential model container (similar to PyTorch)

-SGD optimizer

-Automatic gradient propagation through the network

Everything from forward passes, gradient computation, to parameter updates are implemented manually using NumPy.

Results:

-The model was trained on the MNIST handwritten digit dataset.

Network architecture:

784 → 256 → 128 → 10

After training on 60,000 images, the model achieved:

~97.35% accuracy on the 10,000 image test set.

To Run the Project

Install dependencies:

pip install numpy pillow requests

Train and evaluate the model:

python MNIST.py

You can also test the network on your own digit:

Draw a number (0–9)

Save it as my_digit.png in the project folder

Run:

python MNIST.py

Why I Built This:

I wanted to  understand how neural networks actually work rather than just using existing libraries. Building everything manually  from forward propagation to backpropagation helped me understand how gradients flow through a network and how optimizers update weights.

Along the way I also implemented things like Xavier weight initialization and numerical stability tricks for the softmax / cross entropy computation.

Thank you! I hope you have as much fun reading about and using this project as I did building it!!!


