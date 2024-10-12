from msml.core import *
from mlp import MultilayerPerceptron

# Initialize MSML context
ctx = Context()

# Define the XOR input and target data manually
inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

targets = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]

mlp = MultilayerPerceptron(ctx)
for input in inputs:
    input_tensor = Tensor.with_data(ctx, [2], input)
    output: float = mlp.forward(input_tensor)().f32_data()[0]
    print(f'Input: {input}, Output: {output}')
