from msml.core import *

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

input_tensor = Tensor.with_data(ctx, [4, 2], [elem for sublist in inputs for elem in sublist], name='Input')
target_tensor = Tensor.with_data(ctx, [4, 1], [elem for sublist in targets for elem in sublist], name='Target')
