# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from msml.core import *


# Define the perceptron function (McCulloch–Pitts neuron)
def perceptron(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return (w @ x + b).step()


# Negating perceptron
def p_not(xx: int) -> int:
    x: Tensor = Tensor.full([1], fill_value=float(xx))
    w: Tensor = Tensor.full([1], fill_value=-1)
    b: Tensor = Tensor.full([1], fill_value=0.5)
    r: Tensor = perceptron(x, w, b)
    flag: int = int(r.get_scalar_virtual_index(0))
    assert flag & ~1 == 0
    return flag


truth_table = [
    0,
    1,
    0,
    1,
]

for bit in truth_table:
    print(f'NOT {bit} = {p_not(bit)}')
