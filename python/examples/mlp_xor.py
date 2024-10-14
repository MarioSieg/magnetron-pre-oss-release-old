from msml.core import *
from mlp import MultilayerPerceptron

# Initialize MSML context
ctx = Context()

# Define the XOR input and target data manually
inputs = [
    Tensor.with_data(ctx, [2], [0.0, 0.0]),
    Tensor.with_data(ctx, [2], [0.0, 1.0]),
    Tensor.with_data(ctx, [2], [1.0, 0.0]),
    Tensor.with_data(ctx, [2], [1.0, 1.0])
]

targets = [
    Tensor.with_data(ctx, [1], [0.0]),
    Tensor.with_data(ctx, [1], [1.0]),
    Tensor.with_data(ctx, [1], [1.0]),
    Tensor.with_data(ctx, [1], [0.0])
]

mlp = MultilayerPerceptron(ctx, [2, 3, 1])
mlp.train(inputs, targets, epochs=3, learning_rate=0.1)

for input_tensor in inputs:
    output: float = mlp.forward(input_tensor)().f32_data()[0]
    print(f'Output: {output}')
