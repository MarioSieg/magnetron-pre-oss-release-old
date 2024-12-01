# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl

wl.GlobalConfig.verbose = True
#wl.GlobalConfig.compute_device = wl.ComputeDevice.CUDA

test = wl.Tensor.rand(shape=(4, 4))
r = test + test
print(r)
