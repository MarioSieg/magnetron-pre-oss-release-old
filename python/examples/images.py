from msml.core import *
from msml.util import plot_tensor_as_image

cat = Tensor.load_image('../../test_data/car.jpg', resize_to_dims=(256, 256))
cat.image_draw_box((20, 20), (40, 40))
plot_tensor_as_image(cat)
