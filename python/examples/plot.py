# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl
from wavelet.util import *

plot_tensor_scatter(wl.Tensor.rand(shape=(10, 10, 10, 10)), '4D Tensor')
plot_tensor_scatter(wl.Tensor.load_image('../../test_data/car.jpg', resize_to=(32, 32)))
