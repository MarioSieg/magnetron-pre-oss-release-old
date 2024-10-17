from msml.core import *


class MultilayerPerceptron:
    """A simple multilayer perceptron (feedforward network)."""
    def __init__(self, layout: list[int]):
        self.layout = layout
        self.weights = []
        self.biases = []
        self.cache = []

        for i in range(0, len(layout)-1):
            self.weights.append(Tensor.random([layout[i+1], layout[i]], name=f'Weight {i}'))
            self.biases.append(Tensor.random([layout[i+1], 1], name=f'Bias {i}'))

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward propagate the input through the network."""
        prev: Tensor = inputs
        self.cache.clear()
        self.cache.append(prev.clone())
        for i in range(0, len(self.weights)):
            x: Tensor = prev if i == 0 else prev.relu()
            prev = ((self.weights[i] @ x).transpose().clone()) + self.biases[i]
            self.cache.append(prev.clone())
        return prev.sigmoid()

    def backward(self, outputs: Tensor, targets: Tensor, rate: Tensor):
        errors: Tensor = outputs - targets
        gradients: Tensor = outputs.sigmoid(derivative=True)
        for i in reversed(range(0, len(self.weights))):
            gradients = gradients * errors * rate
            self.weights[i] = self.weights[i] + ((gradients @ self.cache[i].transpose().clone()).transpose().clone())
            self.biases[i] = self.biases[i] + gradients
            errors = self.weights[i].transpose().clone() @ errors
            gradients = self.cache[i].sigmoid(derivative=True)

    def train(self, inputs: list[Tensor], targets: list[Tensor], epochs: int, learning_rate: float):
        assert len(inputs) == len(targets)
        rate = Tensor.full([1], fill_value=learning_rate)
        for _ in range(1, epochs+1):
            for i in range(0, len(inputs)):
                input_tensor: Tensor = inputs[i]
                target_tensor: Tensor = targets[i]
                output: Tensor = self.forward(input_tensor)
                self.backward(output, target_tensor, rate)
