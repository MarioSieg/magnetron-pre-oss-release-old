# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements high level model classes for neural networks based on the wavelet.core module.

import time

import wavelet.core as wl


def mse(y: wl.Tensor, y_hat: wl.Tensor) -> float:
    """Mean Squared Error"""
    return (y - y_hat).sqr().mean().scalar()


def cross_entropy(y: wl.Tensor, y_hat: wl.Tensor) -> float:
    """Cross Entropy Loss"""
    return -(y * y_hat.log()).sum().scalar()


class DenseLayer:
    def __init__(self, in_features: int, out_features: int, activation: wl.Operator = wl.Operator.SIGMOID):
        self.weight = wl.Tensor.rand(shape=(out_features, in_features))
        self.bias = wl.Tensor.rand(shape=(out_features, 1))
        self.activation = activation
        self.cache = None

    def forward(self, prev: wl.Tensor, activation: wl.Operator | None = None) -> wl.Tensor:
        prev = (self.weight @ prev + self.bias)
        return wl.Tensor.operator(activation if activation is not None else self.activation, True,  None, prev)

    def backward(self, is_in: bool, cache: wl.Tensor, delta: wl.Tensor, rate: wl.Tensor) -> wl.Tensor:
        self.weight -= (delta @ cache.transpose().clone()) * rate
        self.bias -= delta * rate
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

    def forward(self, inputs: wl.Tensor, activation: wl.Operator | None = None) -> wl.Tensor:
        prev = inputs
        self.cache.clear()
        self.cache.append(prev)
        for i in range(0, len(self.layers)):
            prev = self.layers[i].forward(prev, activation)
            self.cache.append(prev)
        return prev

    def backward(self, outputs: wl.Tensor, targets: wl.Tensor, rate: wl.Tensor):
        delta = (outputs - targets) * outputs.sigmoid(derivative=True)
        for i in reversed(range(0, len(self.layers))):
            delta = self.layers[i].backward(i > 0, self.cache[i], delta, rate)

    def train(self, inputs: list[wl.Tensor], targets: list[wl.Tensor], epochs: int, learning_rate: float):
        assert len(inputs) == len(targets)
        print(f'Training started {epochs} epochs with learning rate {learning_rate}')
        now = time.time_ns()
        losses = []
        rate = wl.Tensor.full((1,), fill_value=learning_rate)
        for epoch in range(0, epochs - 1):
            total_loss: float = 0
            for i in range(0, len(inputs)):
                pred: wl.Tensor = self.forward(inputs[i])
                self.backward(pred, targets[i], rate)
                total_loss += mse(pred, targets[i])
            mean_loss = total_loss / len(inputs)
            losses.append(mean_loss)
            if epoch % self.loss_epoch_step == 0:
                print(f'Epoch: {epoch}, Loss: {mean_loss}')
        print(f'Training finished in {(time.time_ns() - now) / 1e9} seconds')
        return losses

    def summary(self):
        trainable_params = 0
        for layer in self.layers:
            trainable_params += layer.weight.num_elements + layer.bias.num_elements
        layers: int = len(self.layers)
        print('---- Model Summary ----')
        print(f'Trainable Parameters: {trainable_params}')
        print(f'Layers: {layers}')
        print('-----------------------')
