# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from msml.core import *


class Linear:
    """Affine linear transformation on data. A = A@B^T"""

    def __init__(self, in_features: int, out_features: int, activation: Op = Op.SIGMOID):
        self.weight = Tensor.random(shape=[out_features, in_features])
        self.bias = Tensor.random(shape=[out_features, 1])
        self.activation = activation
        self.cache = None

    def forward(self, prev: Tensor) -> Tensor:
        prev = (self.weight @ prev + self.bias)
        return Tensor.operator(self.activation, None, prev)

    def backward(self, is_in: bool, cache: Tensor, delta: Tensor, rate: Tensor) -> Tensor:
        self.weight = self.weight - (delta @ cache.transpose().clone()) * rate
        self.bias = self.bias - delta * rate
        if is_in:
            return (self.weight.transpose() @ delta) * cache.sigmoid(derivative=True)
        else:
            return delta


class Model:
    """A simple multilayer perceptron (feedforward network)."""

    def __init__(self, layers: list[Linear]):
        assert len(layers) > 0
        self.layers = layers
        self.cache = []

    def forward(self, inputs: Tensor) -> Tensor:
        prev = inputs
        self.cache.clear()
        self.cache.append(prev)
        for i in range(0, len(self.layers)):
            prev = self.layers[i].forward(prev)
            self.cache.append(prev)
        return prev

    def backward(self, outputs: Tensor, targets: Tensor, rate: Tensor):
        delta = (outputs - targets) * outputs.sigmoid(derivative=True)
        for i in reversed(range(0, len(self.layers))):
            delta = self.layers[i].backward(i > 0, self.cache[i], delta, rate)

    def train(self, inputs: list[Tensor], targets: list[Tensor], epochs: int, learning_rate: float):
        assert len(inputs) == len(targets)
        rate = Tensor.full([1], fill_value=learning_rate)
        for e in range(1, epochs + 1):
            if e % 1000 == 0:
                print(f'Epoch: {e}')
            for i in range(0, len(inputs)):
                self.backward(self.forward(inputs[i]), targets[i], rate)
