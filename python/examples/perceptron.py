from msml.core import *

ctx = Context()


# Define the perceptron function (McCulloch–Pitts neuron)
def perceptron(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return (w @ x + b).step()()


# Negating perceptron
def p_not(xx: int) -> int:
    x: Tensor = Tensor.full(ctx, [1], fill_value=float(xx))
    w: Tensor = Tensor.full(ctx, [1], fill_value=-1)
    b: Tensor = Tensor.full(ctx, [1], fill_value=0.5)
    r: Tensor = perceptron(x, w, b)
    flag: int = int(r.get_scalar_virtual_index(0))
    assert flag & ~1 == 0
    return flag


truth_table = [
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0],
]

for pair in truth_table:
    print(f'NOT {pair[0]} = {p_not(pair[0])}')
