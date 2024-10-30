# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from msml.core import *
import time

N = 1024
A = Tensor.random([N, N], name='A')
B = Tensor.random([N, N], name='B')

flop = N * N * 2 * N
avg = 0
I = 10
for _ in range(I):
    st = time.monotonic()
    C = A + B
    et = time.monotonic()
    s = et - st
    print(f'{flop/s * 1e-12} TFLOP/s')
    avg += flop/s

print(f'Average: {avg/I * 1e-12} TFLOP/s')