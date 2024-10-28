# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from msml.core import Tensor, Op
import matplotlib.pyplot as plt

def plot_approximation_error(name: str, exact_func: callable, approx_op: Op, domain: (float, float), step: float = 0.00001):
    x_values = [i * step for i in range(int(domain[0] / step), int(domain[1] / step))]
    exact = [exact_func(x) for x in x_values]
    approx = Tensor.operator(approx_op, None, Tensor.const(x_values)).data_as_f32()
    error = [abs(exact[i] - approx[i]) for i in range(len(exact))]
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, exact, label=f'Exact {name}', color='blue')
    plt.plot(x_values, approx, label=f'Approx {name}', color='orange', linestyle='--')
    plt.title(f'Exact vs Approx {name}')
    plt.xlabel('x')
    plt.ylabel(f'{name}(x)')
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, error, label='Absolute Error', color='red')
    plt.title(f'Error in {name} Approximation')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.legend()

    plt.show()

