from msml.core import *
from msml.model import Model, Linear
import matplotlib.pyplot as plt

# Define the XOR input and target data manually
inputs = [
    Tensor.const([0.0, 0.0], name='Input 1'),
    Tensor.const([0.0, 1.0], name='Input 2'),
    Tensor.const([1.0, 0.0], name='Input 3'),
    Tensor.const([1.0, 1.0], name='Input 4')
]

targets = [
    Tensor.const([0.0], name='Target 1'),
    Tensor.const([1.0], name='Target 2'),
    Tensor.const([1.0], name='Target 3'),
    Tensor.const([0.0], name='Target 4')
]

mlp = Model([
    Linear(2, 4),
    Linear(4, 1)
])
EPOCHS: int = 10000
LEARNING_RATE: float = 0.8
losses = mlp.train(inputs, targets, EPOCHS, LEARNING_RATE)
for input_tensor in inputs:
    input_data = input_tensor.data_as_f32()
    output: float = mlp.forward(input_tensor).data_as_f32()[0]
    print(f'{input_data[0]} ^ {input_data[1]} = {output}')
y_epochs = list(range(0, EPOCHS - 1))
plt.plot(losses, y_epochs)
plt.show()
