# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import numpy as np

A = np.random.rand(3, 4, 2)
print(A)
exit(0)
B = np.random.rand(2)
print(B)
print(np.matmul(A, B))
