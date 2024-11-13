# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl
import numpy as np

ricc = wl.Tensor.full(shape=(2, 2), fill_value=1.0)
print(ricc)
ricc += ricc
print(ricc)

data = [
    [3,4,2,4], [3,4,2,4]
]
xxxx = np.array(data).astype(np.float32)
print(xxxx.shape)
print(xxxx.strides)
print(xxxx)

hyper_tensor_a: wl.Tensor = wl.Tensor.const(data)
print(hyper_tensor_a)
print(hyper_tensor_a.num_rows)
print(hyper_tensor_a.num_cols)
