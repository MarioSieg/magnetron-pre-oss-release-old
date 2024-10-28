from msml.core import *
from msml.model import Model, Linear

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

mlp = Model([
    Linear(2, 4),
    Linear(4, 1)
])
mlp.train(inputs, targets, epochs=10000, learning_rate=0.5)
for input_tensor in inputs:
    input_data = input_tensor.data_as_f32()
    output: float = mlp.forward(input_tensor).data_as_f32()[0]
    print(f'{input_data[0]} ^ {input_data[1]} = {output}')
