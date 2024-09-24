from msml.core import *
from msml.util import plot_tensor

ctx = Context()

# Generate a random tensor as image and plot it (width x height x channels)
random_img = Tensor.random(ctx, [256, 256, 3], interval=(0, 1))
plot_tensor(random_img, f'Noise Tensor {random_img.dims} - Generator: {ctx.prng_algorithm}')
