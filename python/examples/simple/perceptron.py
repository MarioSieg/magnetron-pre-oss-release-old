# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag


# Define the perceptron function (McCulloch–Pitts neuron)
def perceptron(x: mag.Tensor, w: mag.Tensor, b: mag.Tensor) -> mag.Tensor:
    return (w @ x + b).step()


# Negating perceptron
def p_not(xx: int) -> float:
    x: mag.Tensor = mag.Tensor.full((1,), fill_value=float(xx))
    w: mag.Tensor = mag.Tensor.full((1,), fill_value=-1)
    b: mag.Tensor = mag.Tensor.full((1,), fill_value=0.5)
    r: mag.Tensor = perceptron(x, w, b)
    return r.scalar()


truth_table = [
    0,
    1,
    0,
    1,
]

for bit in truth_table:
    print(f'NOT {bit} = {p_not(bit)}')
