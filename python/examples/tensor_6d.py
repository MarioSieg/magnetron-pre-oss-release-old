# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl

hyper_tensor_a: wl.Tensor = wl.Tensor.random(shape=[2, 3, 4, 5, 6, 7])
hyper_tensor_b: wl.Tensor = wl.Tensor.random(shape=[2, 3, 4, 5, 6, 7])
r = hyper_tensor_a + hyper_tensor_b
r.print(True, False)

# Check result
data_a = hyper_tensor_a.data_as_f32()
data_b = hyper_tensor_b.data_as_f32()
data_r = r.data_as_f32()
for i in range(0, sum(hyper_tensor_a.shape)):
    assert data_r[i] == data_a[i] + data_b[i]
print('OK')
