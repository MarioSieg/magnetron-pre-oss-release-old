# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl
from wavelet.util import *

cat = wl.Tensor.load_image('../../test_data/car.jpg')
cat.image_draw_box((250, 300), (500, 600), 15, 0xff0000)
cat.image_draw_text((250, 1500), 150, f'{cat.width} x {cat.height}', 0xffffff)
cat.image_draw_text((250, 1800), 150, 'Cat Alarm!', 0xff0000)
plot_tensor_image(cat)
