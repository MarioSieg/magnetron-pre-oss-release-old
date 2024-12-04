# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import approx
import math

import wavelet as wl

approx.plot_approximation_error('softmax', math.exp, wl.Operator.SOFTMAX, domain=(-10, 10))
