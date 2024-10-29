# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements high level model classes for neural networks based on the msml.core module.

from msml.core import *


def mse(y: Tensor, y_hat: Tensor) -> float:
    e = y - y_hat
    mse_value = (e * e).mean()
    return mse_value.unpack_scalar()


class DenseLayer:
    def __init__(self, in_features: int, out_features: int, activation: Op = Op.SIGMOID):
        self.weight = Tensor.random(shape=[out_features, in_features])
        self.bias = Tensor.random(shape=[out_features, 1])
        self.activation = activation
        self.cache = None

    def forward(self, prev: Tensor, activation: Op | None = None) -> Tensor:
        prev = (self.weight @ prev + self.bias)
        return Tensor.operator(activation if activation is not None else self.activation, None, prev)

    def backward(self, is_in: bool, cache: Tensor, delta: Tensor, rate: Tensor) -> Tensor:
        self.weight = self.weight - (delta @ cache.transpose().clone()) * rate
        self.bias = self.bias - delta * rate
        if is_in:
            return (self.weight.transpose() @ delta) * cache.sigmoid(derivative=True)
        else:
            return delta


class SequentialModel:
    def __init__(self, layers: list[DenseLayer]):
        assert len(layers) > 0
        self.layers = layers
        self.cache = []
        self.loss_epoch_step = 1000

    def forward(self, inputs: Tensor, activation: Op | None = None) -> Tensor:
        prev = inputs
        self.cache.clear()
        self.cache.append(prev)
        for i in range(0, len(self.layers)):
            prev = self.layers[i].forward(prev, activation)
            self.cache.append(prev)
        return prev

    def backward(self, outputs: Tensor, targets: Tensor, rate: Tensor):
        delta = (outputs - targets) * outputs.sigmoid(derivative=True)
        for i in reversed(range(0, len(self.layers))):
            delta = self.layers[i].backward(i > 0, self.cache[i], delta, rate)

    def train(self, inputs: list[Tensor], targets: list[Tensor], epochs: int, learning_rate: float):
        assert len(inputs) == len(targets)
        print(f'Training started {epochs} epochs with learning rate {learning_rate}')
        losses = []
        rate = Tensor.full([1], fill_value=learning_rate)
        for e in range(0, epochs - 1):
            total_mse: float = 0
            for i in range(0, len(inputs)):
                pred: Tensor = self.forward(inputs[i])
                self.backward(pred, targets[i], rate)
                total_mse += mse(pred, targets[i])
            avg_mse = total_mse / len(inputs)
            losses.append(avg_mse)
            if e % self.loss_epoch_step == 0:
                print(f'Epoch: {e}, Loss: {avg_mse}')
        print(f'Training finished')
        return losses

    def summary(self):
        trainable_params = 0
        for layer in self.layers:
            trainable_params += layer.weight.num_elements + layer.bias.num_elements
        layers: int = len(self.layers)
        print(f'---- Model Summary ----')
        print(f'Trainable Parameters: {trainable_params}')
        print(f'Layers: {layers}')
        print(f'-----------------------')
