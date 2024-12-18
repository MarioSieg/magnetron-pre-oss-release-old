# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
from .. import utils

utils.plot_tensor_scatter(mag.Tensor.rand(shape=(10, 10, 10, 10)), '4D Tensor')
utils.plot_tensor_scatter(mag.Tensor.load_image('../../../test_data/car.jpg', resize_to=(32, 32)))
