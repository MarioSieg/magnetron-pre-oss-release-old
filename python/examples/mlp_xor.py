import sys

from msml.core import *
from mlp import MultilayerPerceptron

# Define the XOR input and target data manually
inputs = [
    Tensor.with_data([2], [0.0, 0.0], name='Input 1'),
    Tensor.with_data([2], [0.0, 1.0], name='Input 2'),
    Tensor.with_data([2], [1.0, 0.0], name='Input 3'),
    Tensor.with_data([2], [1.0, 1.0], name='Input 4')
]

targets = [
    Tensor.with_data([1], [0.0], name='Target 1'),
    Tensor.with_data([1], [1.0], name='Target 2'),
    Tensor.with_data([1], [1.0], name='Target 3'),
    Tensor.with_data([1], [0.0], name='Target 4')
]

mlp = MultilayerPerceptron(layout=[2, 3, 1])
mlp.train(inputs, targets, epochs=10000, learning_rate=0.1)
for input_tensor in inputs:
    output: float = mlp.forward(input_tensor).f32_data()[0]
    print(f'Output: {output}')
