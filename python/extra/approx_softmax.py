# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import approx
import math

from msml.core import Op

approx.plot_approximation_error('softmax', math.exp, Op.SOFTMAX, domain=(-10, 10))
