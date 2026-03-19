import numpy as np
import requests
import os
from PIL import Image
from Neuron import LinearLayer, ReLU, Sequential, CrossEntropyLoss, SGD, softmax, train

def download_mnist():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    if not os.path.exists("mnist.npz"):
        print("Downloading MNIST...")
        r = requests.get(url)
        open("mnist.npz", "wb").write(r.content)
    data = np.load("mnist.npz")
    x_train = data["x_train"].reshape(-1, 784) / 255.0
    y_train = data["y_train"].astype(int)
    x_test = data["x_test"].reshape(-1, 784) / 255.0
    y_test = data["y_test"].astype(int)
    return x_train, y_train, x_test, y_test
    
def predict_image(model, image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(784)
    output = model.forward(img_array)
    probs = softmax(output)
    prediction = np.argmax(probs)
    confidence = probs[prediction] * 100
    print(f"Predicted: {prediction} ({confidence:.1f}% confident)")

# load data
x_train, y_train, x_test, y_test = download_mnist()

# build network
layer1 = LinearLayer(784, 256)
relu1 = ReLU()
layer2 = LinearLayer(256, 128)
relu2 = ReLU()
layer3 = LinearLayer(128, 10)
model = Sequential([layer1, relu1, layer2, relu2, layer3])
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.layers, learning_rate=0.01)

# load or train
if os.path.exists("model_weights.npz"):
    print("Loading saved weights...")
    data = np.load("model_weights.npz")
    layer1.weights = data['w1']
    layer1.bias = data['b1']
    layer2.weights = data['w2']
    layer2.bias = data['b2']
    layer3.weights = data['w3']
    layer3.bias = data['b3']
else:
    print("Training...")
    train(model, loss_fn, optimizer, x_train, y_train, epochs=5)
    np.savez('model_weights.npz',
        w1=layer1.weights, b1=layer1.bias,
        w2=layer2.weights, b2=layer2.bias,
        w3=layer3.weights, b3=layer3.bias)
    print("Weights saved!")

# test accuracy
correct = 0
for i in range(len(x_test)):
    output = model.forward(x_test[i])
    prediction = np.argmax(softmax(output))
    if prediction == y_test[i]:
        correct += 1
accuracy = correct / len(x_test) * 100
print(f"Test accuracy: {accuracy:.2f}%")

# predict your own image
def predict_image(model, image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(784)
    output = model.forward(img_array)
    probs = softmax(output)
    prediction = np.argmax(probs)
    confidence = probs[prediction] * 100
    print(f"Predicted: {prediction} ({confidence:.1f}% confident)")
    print("\nAll probabilities:")
    for digit, prob in enumerate(probs):
        bar = "█" * int(prob * 30)
        print(f"  {digit}: {bar} {prob*100:.1f}%")
predict_image(model, "my_digit.png")
