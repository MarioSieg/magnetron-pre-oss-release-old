# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
import os

# Enable logging from the wavelet runtime, must be before importing and loading wavelet
os.environ['WAVELET_LOG'] = '1'

import wavelet.core as wl
import time

# Constants
ITERATIONS: int = 10
DIM: int = 1024
FLOP = 2 * DIM ** 3
EXPORT_CSV: str | None = None  # Set to a file path to export the profiler data as CSV

# Create two random matrices
A = wl.Tensor.rand((DIM, DIM), name='A')
B = wl.Tensor.rand((DIM, DIM), name='B')

# Start the profiler
wl.Context.active.start_profiler()

# Perform #ITERATIONS matrix multiplications
avg = 0
for _ in range(ITERATIONS):
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s = et - st
    print(f'{FLOP / s * 1e-12} TFLOP/s')
    avg += FLOP / s

print(f'Average: {avg / ITERATIONS * 1e-12} TFLOP/s')

# Stop the profiler and print the results
wl.Context.active.stop_profiler(EXPORT_CSV)
