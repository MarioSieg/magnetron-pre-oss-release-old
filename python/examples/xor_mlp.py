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

mlp = MultilayerPerceptron(ctx, input_size=2, hidden_size=2, output_size=1)

# Predict (inference) the output for each input pair
for pair in inputs:
    output = mlp.forward(Tensor.with_data(ctx, [2], data=pair))
    print(f'Input: {pair} Output: {output().f32_data()[0]}')
