# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl
from wavelet.util import *

cat = wl.Tensor.load_image('../../test_data/car.jpg', resize_to_dims=(128, 128))
cat.image_draw_box((250, 300), (500, 600), 15, wl.pack_color(255, 0, 0))
plot_tensor_image(cat)
