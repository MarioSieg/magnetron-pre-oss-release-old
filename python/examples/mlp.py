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
        prev = inputs
        self.cache.clear()
        self.cache.append(prev)
        for i in range(0, len(self.layout)-1):
            prev = (self.weights[i] @ prev + self.biases[i]).sigmoid()
            self.cache.append(prev)
        return prev

    def backward(self, outputs: Tensor, targets: Tensor, rate: Tensor):
        delta = (outputs - targets) * outputs.sigmoid(derivative=True)
        for i in reversed(range(0, len(self.layout) - 1)):
            self.weights[i] = self.weights[i] - (delta @ self.cache[i].transpose()) * rate
            self.biases[i] = self.biases[i] - delta * rate
            if i != 0:
                delta = (self.weights[i].transpose() @ delta) * self.cache[i].sigmoid(derivative=True)

    def train(self, inputs: list[Tensor], targets: list[Tensor], epochs: int, learning_rate: float):
        assert len(inputs) == len(targets)
        rate = Tensor.full([1], fill_value=learning_rate)
        for e in range(1, epochs+1):
            if e % 1000 == 0:
                print(f'Epoch: {e}')
            for i in range(0, len(inputs)):
                self.backward(self.forward(inputs[i]), targets[i], rate)
