# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import numpy as np

M = 4
N = 8
A = np.random.randn(M, N).astype(dtype=np.float32)
B = np.random.randn(N, M).astype(dtype=np.float32)
print(A)
print(B)
print((A @ B).shape)
