import numpy as np



class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)

        self.bias = np.zeros(output_size)
        
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, inputs) + self.bias
    
    def backward(self, gradient_in):
        self.gradient_weights =np.outer(gradient_in, self.inputs)
        self.gradient_bias = gradient_in
        return np.dot(self.weights.T, gradient_in)
    
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, gradient_in):
        return gradient_in * (self.inputs > 0)
    

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self,inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, gradient_in):
        for layer in reversed(self.layers):
            gradient_in = layer.backward(gradient_in)
        return gradient_in
    
def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

class CrossEntropyLoss:
    def forward(self, predictions, target):
        self.predictions = softmax(predictions)
        self.target = target
        loss = -np.log(self.predictions[target] + 1e-15)
        return loss
    
    def backward(self):
        one_hot = np.zeros(len(self.predictions))
        one_hot[self.target] = 1
        return self.predictions - one_hot
    
class SGD:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
    
    def update(self):
        for layer in self.layers:
            if hasattr(layer, "gradient_weights"):
                layer.weights -= self.learning_rate * layer.gradient_weights
                layer.bias -= self.learning_rate * layer.gradient_bias



def train(model, loss_fn, optimizer, inputs, targets, epochs=1000):
    print(f"TRAIN CALLED: {len(inputs)} samples, {epochs} epochs")
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(inputs)):
            output = model.forward(inputs[i])
            loss = loss_fn.forward(output, targets[i])
            total_loss += loss
            gradient = loss_fn.backward()
            model.backward(gradient)
            optimizer.update()
            
            if i % 500 == 0:
                print(f"  Epoch {epoch}, sample {i}/{len(inputs)}, Loss so far: {total_loss/(i+1):.4f}")
        
        print(f"Epoch {epoch} complete, Loss: {total_loss/len(inputs):.4f}")


if __name__ == "__main__":
    inputs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    targets = np.array([0, 1, 1, 0])
    
    layer1 = LinearLayer(784, 256)
    relu1 = ReLU()
    layer2 = LinearLayer(256, 128)
    relu2 = ReLU()
    layer3 = LinearLayer(128, 10)

    model = Sequential([layer1, relu1, layer2, relu2, layer3])
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.layers, learning_rate=0.01)


