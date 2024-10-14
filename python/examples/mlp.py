from msml.core import *


class MultilayerPerceptron:
    """A simple multilayer perceptron (feedforward network)."""
    def __init__(self, ctx: Context, layout: list[int]):
        self.ctx = ctx
        self.layout = layout
        self.weights = []
        self.biases = []
        self.cache = []

        for i in range(0, len(layout)-1):
            self.weights.append(Tensor.random(ctx, [layout[i+1], layout[i]]))
            self.biases.append(Tensor.random(ctx, [layout[i+1], 1]))

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward propagate the input through the network."""
        assert len(self.weights) == len(self.biases)
        prev: Tensor = inputs
        self.cache.clear()
        self.cache.append(prev.clone()())
        for i in range(0, len(self.weights)):
            x: Tensor = prev if i == 0 else prev.relu()
            prev = (self.weights[i] @ x) + self.biases[i]
            self.cache.append(prev.clone()())
        return prev.sigmoid()

    def backward(self, outputs: Tensor, targets: Tensor, learning_rate: float):
        errors: Tensor = outputs - targets
        gradients = outputs.sigmoid(derivative=True)
        rate = Tensor.full(self.ctx, [1], fill_value=learning_rate)
        for i in reversed(range(0, len(self.weights)-1)):
            gradients = gradients * errors * rate
            self.weights[i] = (self.weights[i].clone() + (gradients @ self.cache[i].transpose()))()
            self.biases[i] = (self.biases[i] + gradients)()
            errors = (self.weights[i].transpose() @ errors)()
            gradients = self.cache[i].sigmoid(derivative=True)()

    def train(self, inputs: list[Tensor], targets: list[Tensor], epochs: int, learning_rate: float):
        assert len(inputs) == len(targets)
        for _ in range(1, epochs+1):
            for i in range(0, len(inputs)):
                input_tensor: Tensor = inputs[i]
                target_tensor: Tensor = targets[i]
                output: Tensor = self.forward(input_tensor)()
                self.backward(output, target_tensor, learning_rate)

