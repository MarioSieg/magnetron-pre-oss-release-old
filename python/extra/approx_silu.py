# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import approx
import math

import wavelet.core as wl


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


approx.plot_approximation_error('silu', silu, wl.Operator.SILU, domain=(-2, 2))
