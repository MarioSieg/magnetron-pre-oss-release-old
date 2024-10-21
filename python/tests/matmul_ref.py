# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import numpy as np

data1 = [
    [1.6354027, -1.3607267],
    [1.8556793, 1.1689897]
]

data2 = [
    [-0.6105532, 0.10695228],
    [-1.0069681, -0.40955952]
]

A = np.array(data1, dtype=np.float32)
B = np.array(data2, dtype=np.float32)
print(np.matmul(A, B))
