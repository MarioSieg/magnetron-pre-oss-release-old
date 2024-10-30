# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl


# Define the perceptron function (McCulloch–Pitts neuron)
def perceptron(x: wl.Tensor, w: wl.Tensor, b: wl.Tensor) -> wl.Tensor:
    return (w @ x + b).step()


# Negating perceptron
def p_not(xx: int) -> int:
    x: wl.Tensor = wl.Tensor.full([1], fill_value=float(xx))
    w: wl.Tensor = wl.Tensor.full([1], fill_value=-1)
    b: wl.Tensor = wl.Tensor.full([1], fill_value=0.5)
    r: wl.Tensor = perceptron(x, w, b)
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
