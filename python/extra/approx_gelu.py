# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import approx
import math

from msml.core import Op

approx.plot_approximation_error('gelu', math.tanh, Op.TANH, domain=(-2, 2))