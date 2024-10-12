from msml.core import *


class MultilayerPerceptron:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.weights = []
        self.biases = []
        self.weights.append(Tensor.with_data(ctx, [2, 3], [
            -2.482079, 2.482152, 0.335368, 0.549910, 2.440562, -2.440396
        ]))
        self.biases.append(Tensor.with_data(ctx, [3], [
            -0.000055, 1.328214, -0.000007
        ]))
        self.weights.append(Tensor.with_data(ctx, [3], [
            3.430627, -1.212265, 3.379288
        ]))
        self.biases.append(Tensor.with_data(ctx, [1], [
            -1.847787
        ]))

    def forward(self, inputs: Tensor) -> Tensor:
        assert len(self.weights) == len(self.biases)
        prev: Tensor = inputs
        for i in range(0, len(self.weights)):
            x: Tensor = prev if i == 0 else prev.relu()
            prev = (self.weights[i] @ x) + self.biases[i]
        return prev.hard_sigmoid()
