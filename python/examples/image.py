# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl
from wavelet.util import *

cat = wl.Tensor.load_image('../../test_data/car.jpg')
cat.image_draw_box((250, 300), (500, 600), 15, 0xff0000)
plot_tensor_image(cat)

cat2 = wl.Tensor.load_image('../../test_data/car.jpg', resize_to=(256, 256))
cat2.image_draw_text((10, 10), 150, f'{cat2.width} x {cat2.height}', 0xffffff)
cat2.image_draw_text((10, 30), 150, f'{cat2.data_size / 1024} kiB', 0x00ff00)
plot_tensor_image(cat2)
