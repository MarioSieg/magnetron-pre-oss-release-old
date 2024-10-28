# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import approx
import math

from msml.core import Op

def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))

approx.plot_approximation_error('silu', silu, Op.SILU, domain=(-2, 2))