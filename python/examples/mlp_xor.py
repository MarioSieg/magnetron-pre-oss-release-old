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

mlp = MultilayerPerceptron(layout=[2, 2, 1])
#mlp.train(inputs, targets, epochs=1000, learning_rate=0.1)
mlp.weights.clear()
mlp.biases.clear()
mlp.weights.append(Tensor.with_data([2, 3], [
    -2.482079, 2.482152, 0.335368, 0.549910, 2.440562, -2.440396
]))
mlp.biases.append(Tensor.with_data([3], [
    -0.000055, 1.328214, -0.000007
]))
mlp.weights.append(Tensor.with_data([3], [
    3.430627, -1.212265, 3.379288
]))
mlp.biases.append(Tensor.with_data([1], [
    -1.847787
]))

for input_tensor in inputs:
    output: float = mlp.forward(input_tensor).f32_data()[0]
    print(f'Output: {output}')
