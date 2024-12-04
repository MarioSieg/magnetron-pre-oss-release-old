# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import approx
import math

import wavelet as wl

approx.plot_approximation_error('tanh', math.tanh, wl.Operator.TANH, domain=(-2, 2))
